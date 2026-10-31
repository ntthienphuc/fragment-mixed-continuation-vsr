"""Single-GPU training with deterministic samples, atomic checkpoints and exact-step resume."""

import argparse
import math
import time
import os
from pathlib import Path
import torch
from .common import config_file, resolve, save_json, read_json, seed_all, code_identity, digest
from .data import PreparedData, check_disjoint
from .model import Recognizer, METHODS
from .objectives import sample_loss
from .evaluation import evaluate
from .runtime import autocast, rng_state, restore_rng, samples


def checkpoint(path, model, optimizer, step, best, identity, scaler):
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
               'best': best, 'identity': identity, 'rng': rng_state(), 'scaler': scaler.state_dict()}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def learning_rate(step, steps, base):
    warmup = min(500, max(1, steps // 10))
    if step <= warmup:
        return base * step / warmup
    return base * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def train(args):
    if args.cohort == 'full' and (args.method != 'T0' or args.selection_view != 'full'):
        raise ValueError('Full-corpus continuation requires T0 and full-utterance selection')
    config = config_file(args.config)
    if args.method == 'B0':
        raise ValueError('B0 is an inference-only reference')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    autocast(args.precision, args.device)
    if args.fused_adam and not args.device.startswith('cuda'):
        raise ValueError('Fused Adam is a CUDA optimization in this runner')
    train_data = PreparedData(resolve(config, 'prepared') / 'train', training=True, cohort=args.cohort)
    validation = PreparedData(resolve(config, 'prepared') / 'val')
    if any(data.receipt['config_sha256'] != config['config_sha256'] for data in [train_data, validation]):
        raise ValueError('Prepared data belongs to another config')
    overlap = check_disjoint(train_data, validation)
    if args.selection_view == 'window' and not any(r['window_eligible'] for r in validation.selection(args.validation_limit)):
        raise ValueError('No validation target windows')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'environment.json', {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'device': torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device})
    identity = {'config_sha256': config['config_sha256'], 'code_sha256': code_identity(),
                'train_manifest_sha256': digest(train_data.manifest),
                'val_manifest_sha256': digest(validation.manifest),
                'method': args.method, 'scope': args.scope, 'seed': args.seed, 'budget': args.budget,
                'steps': args.steps, 'accumulation': args.accumulation, 'lr': args.lr,
                'eval_every': args.eval_every, 'validation_limit': args.validation_limit,
                'precision': args.precision, 'fused_adam': args.fused_adam, 'threads': args.threads}
    identity.update(cohort=args.cohort, selection_view=args.selection_view)
    lock = out / 'identity.json'
    if lock.exists():
        if not args.resume or read_json(lock) != identity:
            raise ValueError('Run already exists or resume identity changed')
    else:
        if args.resume:
            raise FileNotFoundError('No existing run to resume')
        save_json(lock, identity)
    model = Recognizer(config, args.method, args.device, args.scope)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=.01, fused=args.fused_adam)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16', init_scale=256.)
    step, best = 0, float('inf')
    if args.resume:
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Checkpoint identity differs from run')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, best = state['step'], state['best']
        restore_rng(state['rng'])
        scaler.load_state_dict(state['scaler'])
        del state
        log = out / 'train.jsonl'
        if log.exists():
            import json
            kept = [line for line in log.read_text(encoding='utf-8').splitlines()
                    if json.loads(line)['step'] <= step]
            log.write_text('\n'.join(kept) + ('\n' if kept else ''), encoding='utf-8')
    began = time.perf_counter()
    stream = samples(train_data, step * args.accumulation, args.steps * args.accumulation,
                     args.seed, args.workers)
    for step in range(step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(step, args.steps, args.lr)
        for group in optimizer.param_groups:
            group['lr'] = rate
        total = 0.
        for micro in range(args.accumulation):
            sample = (step - 1) * args.accumulation + micro
            row, frames = next(stream)
            with autocast(args.precision, args.device):
                loss, details = sample_loss(model, frames, row, sample, args.seed, args.budget)
            scaler.scale(loss / args.accumulation).backward()
            total += float(loss.detach()) / args.accumulation
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            _, metrics = evaluate(model, validation, context=args.budget, limit=args.validation_limit,
                                  view=args.selection_view)
            if metrics['inference_failed']:
                raise RuntimeError('Validation failure: repair run before selecting a checkpoint')
            value = metrics['metrics']['word']['rate']
            if value is None:
                raise ValueError('Empty validation denominator')
            improved = value < best
            best = min(best, value)
            checkpoint(out / 'last.pt', model, optimizer, step, best, identity, scaler)
            if improved:
                checkpoint(out / 'best.pt', model, optimizer, step, best, identity, scaler)
            save_json(out / f'validation_{step:06}.json', metrics)
        with (out / 'train.jsonl').open('a', encoding='utf-8') as handle:
            import json
            handle.write(json.dumps({'step': step, 'loss': total, 'lr': rate,
                                     'grad_norm': float(norm), 'elapsed_seconds': time.perf_counter()-began}) + '\n')
        print(f'{args.method} step={step}/{args.steps} loss={total:.5f}', flush=True)
        if args.stop_after and step >= args.stop_after and step < args.steps:
            checkpoint(out / 'last.pt', model, optimizer, step, best, identity, scaler)
            stream.close()
            return
    save_json(out / 'completed.json', {'status': 'completed', 'updates': step, 'identity': identity,
                                     'parameters': model.counts(), 'overlap': overlap,
                                     'seconds_this_invocation': time.perf_counter()-began,
                                     'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20
                                     if args.device.startswith('cuda') else None,
                                     'best_checkpoint_sha256': digest(out / 'best.pt')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--validation-limit', type=int, default=0)
    parser.add_argument('--cohort', choices=['window', 'full'], default='window')
    parser.add_argument('--selection-view', choices=['window', 'full'], default='window')
    parser.add_argument('--scope', choices=['all', 'heads'], default='all')
    parser.add_argument('--budget', choices=['past', 'both'], default='both')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.eval_every, args.threads) < 1 or args.lr <= 0:
        parser.error('Training counts and learning rate must be positive')
    if args.workers < 0 or args.stop_after < 0:
        parser.error('Worker count and stop step must be nonnegative')
    train(args)


if __name__ == '__main__':
    main()
