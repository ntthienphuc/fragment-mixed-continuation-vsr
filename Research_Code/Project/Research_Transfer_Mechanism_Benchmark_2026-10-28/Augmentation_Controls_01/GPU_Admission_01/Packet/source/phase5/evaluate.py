"""CLI for validation and frozen-plan test scoring."""

import argparse
from pathlib import Path
import torch
from .common import config_file, resolve, digest, write_rows, save_json, code_identity, read_json
from .data import PreparedData, check_disjoint
from .model import Recognizer
from .evaluation import evaluate
from .selection import verify_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='native')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--view', choices=['full', 'window'], default='window')
    parser.add_argument('--context', choices=['base', 'past', 'both'], default='both')
    parser.add_argument('--decoder', choices=['greedy', 'joint01', 'joint03', 'joint07', 'ctc_only', 'bounded'],
                        default='joint01')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test-plan')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = config_file(args.config)
    if args.split == 'test':
        if not args.test_plan or args.limit:
            raise ValueError('Test requires a frozen plan and full selected population')
        verify_plan(args.test_plan, config, args.checkpoint, args.decoder, args.context, args.view)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Evaluation output already exists')
    data = PreparedData(resolve(config, 'prepared') / args.split)
    if data.receipt['config_sha256'] != config['config_sha256']:
        raise ValueError('Prepared data belongs to another config')
    train = PreparedData(resolve(config, 'prepared') / 'train')
    check_disjoint(train, data)
    if args.split == 'test':
        check_disjoint(PreparedData(resolve(config, 'prepared') / 'val'), data)
    payload = None
    method, scope, identity = 'B0', 'all', None
    if args.checkpoint != 'native':
        identity = read_json(Path(args.checkpoint).parent / 'identity.json')
        if identity['config_sha256'] != config['config_sha256'] or identity['code_sha256'] != code_identity():
            raise ValueError('Training identity differs from evaluation config/code')
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if payload['identity'] != identity:
            raise ValueError('Run and checkpoint identities differ')
        method, scope = identity['method'], identity['scope']
    model = Recognizer(config, method, args.device, scope)
    if payload is not None:
        model.load_state_dict(payload['model'], strict=True)
        del payload
    rows, summary = evaluate(model, data, args.context, args.decoder, args.view, args.limit)
    summary.update(dataset=config['dataset'], method=method, scope=scope, config_sha256=config['config_sha256'],
                   training_seed=identity['seed'] if identity else None,
                   training_identity=identity,
                   code_sha256=code_identity(), manifest_sha256=digest(data.manifest),
                   checkpoint_sha256=config['checkpoint_sha256'] if args.checkpoint == 'native'
                   else digest(args.checkpoint))
    output.mkdir(parents=True)
    write_rows(output / 'predictions.jsonl', rows)
    save_json(output / 'summary.json', summary)
    print(summary)
    if summary['inference_failed']:
        raise RuntimeError('Evaluation includes failures; inspect coverage before interpretation')


if __name__ == '__main__':
    main()
