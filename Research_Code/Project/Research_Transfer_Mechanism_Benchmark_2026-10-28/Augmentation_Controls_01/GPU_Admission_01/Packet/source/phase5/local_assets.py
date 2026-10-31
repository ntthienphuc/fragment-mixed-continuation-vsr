"""One-time export from the existing workspace; never modifies archived research."""

import argparse
import shutil
from pathlib import Path
from .common import ROOT, digest, save_json


def build(workspace):
    source = workspace / (
        'experiments/_archive_sources/lip_final_2026_07/'
        '02_PROJECT_MAIN/main_code_results_phase/baselines/auto_avsr')
    vendor = ROOT / 'vendor/auto_avsr'
    vendor.mkdir(parents=True, exist_ok=True)
    for folder in ['espnet', 'preparation/detectors/mediapipe']:
        shutil.copytree(source / folder, vendor / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(source / 'LICENSE', vendor / 'LICENSE')
    tokens = ROOT / 'assets/tokenizers'
    tokens.mkdir(parents=True, exist_ok=True)
    vi = workspace / 'experiments/_archive_sources/pavir_2026_08/04_artifacts/tokenizer'
    pairs = [(vi / 'vi_unigram_2000.model', 'vi.model'),
             (vi / 'vi_unigram_2000.auto_avsr_units.txt', 'vi_units.txt'),
             (source / 'spm/unigram/unigram5000.model', 'en.model'),
             (source / 'spm/unigram/unigram5000_units.txt', 'en_units.txt')]
    for original, name in pairs:
        shutil.copy2(original, tokens / name)
    original = workspace / (
        'experiments/_archive_sources/lip_final_2026_07/06_CHECKPOINT/'
        'benchmark_v1/auto_avsr/epoch=13-loss_val=53.8436.ckpt')
    expected = '57a6e8086c7b3a9452ab92f97e76265914ce670e920f6598b9e2f2a44cc79816'
    if digest(original) != expected:
        raise ValueError('Historical initializer hash changed')
    exported = ROOT / 'assets/checkpoints/vlr_offline.pt'
    exported.parent.mkdir(parents=True, exist_ok=True)
    if not exported.exists():
        import torch
        payload = torch.load(original, map_location='cpu', weights_only=True)
        state = {key[6:]: value for key, value in payload['state_dict'].items() if key.startswith('model.')}
        if not state:
            raise ValueError('No model parameters to export')
        torch.save(state, exported)
    vlr = {'dataset': 'vlr', 'language': 'vi', 'source': 'vendor/auto_avsr',
           'tokenizer': 'assets/tokenizers/vi.model', 'units': 'assets/tokenizers/vi_units.txt',
           'checkpoint': 'assets/checkpoints/vlr_offline.pt', 'checkpoint_sha256': digest(exported),
           'state_field': 'raw', 'data_root': 'data/vlr/cloud_dataset_signer_split',
           'prepared': 'prepared/vlr'}
    en = {'dataset': 'lrs2', 'language': 'en', 'source': 'vendor/auto_avsr',
          'tokenizer': 'assets/tokenizers/en.model', 'units': 'assets/tokenizers/en_units.txt',
          'checkpoint': 'assets/checkpoints/vsr_trlrs3_base.pth', 'checkpoint_sha256': None,
          'state_field': 'raw', 'data_root': 'data/lrs2/mvlrs_v1', 'prepared': 'prepared/lrs2',
          'train_list': 'data/lrs2/train.txt', 'val_list': 'data/lrs2/val.txt',
          'test_list': 'data/lrs2/test.txt'}
    save_json(ROOT / 'configs/vlr.json', vlr)
    save_json(ROOT / 'configs/lrs2.json', en)
    local = {**vlr, 'data_root': str(workspace / 'data/official_raw/vlr/extracted/cloud_dataset_signer_split'),
             'prepared': 'validation/prepared/vlr'}
    save_json(ROOT / 'configs/local_vlr.json', local)
    archive = workspace / 'data/official_raw/vlr/cloud_dataset_signer_split.zip'
    save_json(ROOT / 'assets/Provenance.json', {
        'source': 'Existing archived Auto-AVSR; Apache-2.0 code; model/data terms remain separate.',
        'initializer_original_sha256': expected, 'initializer_export_sha256': digest(exported),
        'vlr_archive': {'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'files': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                  for directory in [vendor, tokens] for p in directory.rglob('*') if p.is_file()}})
    print('Source, tokenizers, exported VLR initializer and dataset configs prepared.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    build(parser.parse_args().workspace)
