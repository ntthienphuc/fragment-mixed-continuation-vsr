"""Freeze an explicit evaluation plan before test preprocessing or inference."""

import argparse
from pathlib import Path
from .common import config_file, save_json, read_json, digest, code_identity, resolve


def verify_plan(path, config, run=None, decoder=None, context=None, view=None):
    plan = read_json(path)
    if plan['config_sha256'] != config['config_sha256'] or plan['code_sha256'] != code_identity():
        raise ValueError('Frozen test plan does not match current data config and code')
    for item in plan['runs']:
        if item['checkpoint'] == 'native' and digest(resolve(config, 'checkpoint')) != item['sha256']:
            raise ValueError('Native initializer changed after freeze')
        if item['checkpoint'] != 'native' and digest(item['checkpoint']) != item['sha256']:
            raise ValueError('Selected checkpoint changed after plan freeze')
    for split, expected in plan['development_manifests'].items():
        if digest(resolve(config, 'prepared') / split / 'manifest.jsonl') != expected:
            raise ValueError('Development population changed after freeze')
    selected = run if run == 'native' else str(Path(run).resolve()) if run else None
    if selected is not None and selected not in [r['checkpoint'] for r in plan['runs']]:
        raise ValueError('Checkpoint not in frozen test plan')
    for value, key in [(decoder, 'decoder'), (context, 'context'), (view, 'view')]:
        if value is not None and value != plan[key]:
            raise ValueError('Evaluation setting differs from test plan')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--decoder', choices=['greedy', 'joint01', 'joint03', 'joint07', 'ctc_only', 'bounded'],
                        default='joint01')
    parser.add_argument('--context', choices=['base', 'past', 'both'], default='both')
    parser.add_argument('--view', choices=['full', 'window'], default='window')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Test plan already exists')
    config = config_file(args.config)
    manifests = {split: digest(resolve(config, 'prepared') / split / 'manifest.jsonl')
                 for split in ['train', 'val']}
    for name in args.checkpoints:
        if name == 'native':
            continue
        directory = Path(name).parent
        completed = read_json(directory / 'completed.json')
        identity = completed['identity']
        for split in ['train', 'val']:
            if identity[f'{split}_manifest_sha256'] != manifests[split]:
                raise ValueError('Current prepared population differs from selected training run')
        if identity['config_sha256'] != config['config_sha256'] or identity['code_sha256'] != code_identity():
            raise ValueError('Selected run used a different config/code')
        if Path(name).name != 'best.pt' or digest(name) != completed['best_checkpoint_sha256']:
            raise ValueError('Select the validation-best checkpoint from a completed run')
    runs = [{'checkpoint': str(Path(p).resolve()), 'sha256': digest(p)} for p in args.checkpoints if p != 'native']
    if 'native' in args.checkpoints:
        runs.append({'checkpoint': 'native', 'sha256': config['checkpoint_sha256']})
    save_json(args.output, {'config_sha256': config['config_sha256'], 'code_sha256': code_identity(),
                           'runs': runs, 'decoder': args.decoder, 'context': args.context, 'view': args.view,
                           'development_manifests': manifests,
                           'purpose': 'Frozen evaluation settings; not proof of historically unseen test data.'})


if __name__ == '__main__':
    main()
