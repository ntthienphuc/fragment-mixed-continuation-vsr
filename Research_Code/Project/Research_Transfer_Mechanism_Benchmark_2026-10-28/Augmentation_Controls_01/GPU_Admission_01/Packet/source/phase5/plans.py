"""Generate explicit training queues without launching them."""
import argparse
from pathlib import Path
from .common import save_json


def make(stage, datasets, precision, output, methods=None):
    methods = methods or ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'A5']
    if len(methods) != len(set(methods)) or not set(methods) <= {'T0','T1','T2','T3','T4','T5','A5'}:
        raise ValueError('Invalid or duplicate methods')
    seeds, steps, interval = ([17], 200, 50) if stage == 'pilot' else ([17, 29, 43], 6000, 500)
    jobs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                name = f'{stage}_{dataset}_{method}_{seed}'
                args = ['--config', f'configs/{dataset}.json', '--method', method,
                        '--output', f'runs/{name}', '--steps', str(steps), '--seed', str(seed),
                        '--accumulation', '8', '--eval-every', str(interval),
                        '--precision', precision, '--scope', 'all', '--workers', '2', '--threads', '4']
                jobs.append({'id': name, 'module': 'phase5.training', 'args': args})
    if Path(output).exists():
        raise FileExistsError('Plan exists; preserve it and choose a new name')
    save_json(output, {'stage': stage, 'jobs': jobs, 'selection': 'Explicit methods; no test-based selection'})
    print({'jobs': len(jobs), 'updates_per_run': steps, 'effective_clips_per_update': 8})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot', 'main'], required=True)
    parser.add_argument('--datasets', nargs='+', choices=['vlr', 'lrs2'], default=['vlr','lrs2'])
    parser.add_argument('--precision', choices=['fp32','fp16','bf16'], default='fp32')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    make(a.stage, a.datasets, a.precision, a.output, a.methods)
