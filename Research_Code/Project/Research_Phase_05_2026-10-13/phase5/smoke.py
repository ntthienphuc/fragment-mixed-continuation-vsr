"""Real native checkpoint parity and per-method optimization diagnostic."""

import argparse
import gc
import time
import numpy as np
import torch
from .common import config_file, resolve, save_json, seed_all
from .data import PreparedData
from .model import Recognizer
from .objectives import sample_loss
from .decoding import decode


def smoke(config, methods, device, output):
    seed_all(17)
    torch.set_num_threads(4)
    data = PreparedData(resolve(config, 'prepared') / 'train', training=True)
    row = min(data.rows, key=lambda r: r['frames'])
    frames = data.frames(row['clip_id'])
    results = []
    for method in methods:
        began = time.perf_counter()
        model = Recognizer(config, method, device, 'heads')
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        model.eval()
        with torch.no_grad():
            short = frames[:48]
            features = model.encode(short)
            tensor = torch.tensor(np.ascontiguousarray(short[:, 4:92, 4:92]), device=device).float()
            tensor = ((tensor / 255 - .421) / .165)[None, :, None]
            native_features = model.native.encoder(
                model.native.proj_encoder(model.native.frontend(tensor)),
                torch.ones(1, 1, 48, dtype=torch.bool, device=device))[0]
            error = float((native_features - features).abs().max())
            if not torch.allclose(native_features, features, atol=1e-5, rtol=1e-5):
                raise AssertionError(f'Native representation mismatch: {error}')
            predictions = {name: decode(model, features, name) for name in
                           ['greedy', 'joint01', 'joint03', 'joint07', 'ctc_only', 'bounded']}
        del features, native_features, tensor
        model.train()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-5)
        before = model.native.ctc.ctc_lo.weight.detach().clone()
        losses = []
        for index in [0, 1]:
            optimizer.zero_grad(set_to_none=True)
            loss, details = sample_loss(model, frames, row, index, 17, 'both')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append({'loss': float(loss.detach()), 'grad_norm': float(norm), **details})
        changed = not torch.equal(before, model.native.ctc.ctc_lo.weight)
        if not changed:
            raise AssertionError('Native CTC parameters did not update')
        results.append({'method': method, 'clip_id': row['clip_id'], 'frames': len(frames),
                        'scope': 'heads', 'optimizer': 'SGD diagnostic', 'counts': model.counts(),
                        'native_feature_max_error': error, 'decoders': predictions, 'objectives': losses,
                        'parameters_updated': changed, 'elapsed_seconds': time.perf_counter()-began,
                        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20
                        if device.startswith('cuda') else None})
        print(results[-1], flush=True)
        save_json(output, {'status': 'partial' if method != methods[-1] else 'passed',
                           'device': device, 'torch': torch.__version__, 'results': results,
                           'limits': 'Heads-only engineering smoke; no estimate of recognition improvement.'})
        del model, optimizer, loss, before
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--methods', nargs='+', default=['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'A5'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='validation/real_model_smoke.json')
    args = parser.parse_args()
    smoke(config_file(args.config), args.methods, args.device, args.output)
