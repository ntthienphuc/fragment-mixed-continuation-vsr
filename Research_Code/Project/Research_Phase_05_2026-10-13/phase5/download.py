"""Download declared assets on the server; never stores access passwords."""

import argparse
from .common import ROOT, read_json, save_json, digest
from .transfer import download, extract

LRS2 = 'https://thor.robots.ox.ac.uk/lip_reading/data2/'
LISTS = {'train': 'ae240cd86c8432afb5a6b8935c863f24', 'val': '3b7502de5c102519d62477f209c4cd35',
         'test': 'e76bb897a2141f3581266daa850966bc'}
ENGLISH = '12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd'
VLR = '10wnNcD0uUD8eYV6UqtZyiTOSMPG2gp-z'


def acquire(asset, unpack=False):
    observed = {}
    if asset == 'vlr':
        metadata = read_json(ROOT / 'assets/Provenance.json')['vlr_archive']
        target = ROOT / 'downloads/cloud_dataset_signer_split.zip'
        url = f'https://drive.usercontent.google.com/download?id={VLR}&export=download&confirm=t'
        observed['sha256'] = download(url, target, metadata['sha256'])
        if unpack:
            extract(target, ROOT / 'data/vlr')
    elif asset == 'english-model':
        target = ROOT / 'assets/checkpoints/vsr_trlrs3_base.pth'
        config_path = ROOT / 'configs/lrs2.json'
        config = read_json(config_path)
        url = f'https://drive.usercontent.google.com/download?id={ENGLISH}&export=download&confirm=t'
        observed['sha256'] = download(url, target, config.get('checkpoint_sha256'))
        # Upstream publishes only a short MD5 fingerprint, not a full trusted SHA-256.
        if not digest(target, 'md5').startswith('c00a7'):
            raise ValueError('English model differs from upstream short MD5 fingerprint')
        config['checkpoint_sha256'] = observed['sha256']
        save_json(config_path, config)
        observed['verification'] = 'Upstream five-character MD5 fingerprint; local SHA-256 pinned on acquisition'
    else:
        for split, checksum in LISTS.items():
            target = ROOT / f'data/lrs2/{split}.txt'
            observed[split] = download(LRS2 + split + '.txt', target, checksum, 'md5', True)
        if asset == 'lrs2':
            target = ROOT / 'downloads/lrs2_v1.tar'
            observed['archive'] = download(LRS2 + 'lrs2_v1.tar', target,
                                           'bba75f658d8172497baec63eec4f4cdd', 'md5', True)
            if unpack:
                extract(target, ROOT / 'data/lrs2/media')
                # The upstream tar's top-level directory is detected, not guessed.
                candidates = [p.parent for p in (ROOT / 'data/lrs2/media').rglob('main') if p.is_dir()]
                if len(candidates) != 1:
                    raise ValueError('Expected exactly one extracted main directory; inspect archive layout')
                config_path = ROOT / 'configs/lrs2.json'
                config = read_json(config_path)
                config['data_root'] = candidates[0].relative_to(ROOT).as_posix()
                save_json(config_path, config)
    save_json(ROOT / f'downloads/{asset}_receipt.json', observed)
    print(observed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=['vlr', 'english-model', 'lrs2-lists', 'lrs2'])
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    acquire(args.asset, args.extract)
