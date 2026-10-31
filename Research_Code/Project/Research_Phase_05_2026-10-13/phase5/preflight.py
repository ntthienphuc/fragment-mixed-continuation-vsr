"""Check assets and data before allocating a training run."""
import argparse
import importlib.metadata
import shutil
import torch
from .common import ROOT, config_file, resolve, read_json, digest, save_json


def inspect(config):
    errors = []
    for name, expected in read_json(ROOT / 'assets/Provenance.json')['files'].items():
        path = ROOT / name
        if not path.exists() or digest(path) != expected:
            errors.append('Vendor/tokenizer drift: ' + name)
    checkpoint = resolve(config, 'checkpoint')
    if not checkpoint.exists():
        errors.append('Missing initializer')
    elif digest(checkpoint) != config.get('checkpoint_sha256'):
        errors.append('Initializer not pinned or hash mismatch')
    if not resolve(config, 'data_root').exists():
        errors.append('Missing dataset root')
    if not torch.cuda.is_available():
        errors.append('CUDA unavailable')
    return {'status': 'blocked' if errors else 'asset_checks_passed', 'errors': errors,
            'dataset': config['dataset'], 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'free_disk_gib': shutil.disk_usage(ROOT).free / 2**30,
            'limits': 'Assets only; run smoke and pilot to validate model compatibility and capacity.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = inspect(config_file(args.config))
    save_json(args.output, result)
    print(result)
    raise SystemExit(1 if result['errors'] else 0)
