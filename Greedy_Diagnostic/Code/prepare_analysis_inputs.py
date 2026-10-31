"""Freeze portable, checked metadata for the separate greedy diagnostic recount."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'Analysis_Inputs_01'
PAPER = ROOT / 'PatternRecognition_CrossCorpus_Draft_2026-10-29'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def jsonl(payload):
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    lock = read(HERE / 'Code_Lock.json')
    for relative, digest in lock['files'].items():
        assert sha(ROOT / relative) == digest, relative
    plan = read(ROOT / 'Research_Short_Only_Control_2026-10-30/Assessment/Plan.json')
    provenance = {}
    destination = plan['destination']
    manifest = ROOT / destination['assessment']['manifest']
    pixel_index = ROOT / destination['pixel_index']
    assert sha(manifest) == destination['assessment']['sha256']
    assert sha(pixel_index) == destination['pixel_index_sha256']
    rows = jsonl(manifest.read_bytes())
    pixels = {item['clip_id']: item for item in jsonl(pixel_index.read_bytes()) if item['split'] == 'assessment'}
    expected = []
    for position, row in enumerate(rows):
        pixel = pixels[row['clip_id']]
        assert pixel['frames'] == row['frames'] and pixel['source_id'] == row['source_id']
        assert pixel['media_sha256'] == row['input']['sha256']
        expected.append({'position': position, **{key: row[key] for key in
            ('clip_id', 'source_id', 'frames', 'reference')}, 'pixel_sha256': pixel['gray_sha256']})
    provenance['assessment'] = {'manifest_sha256': sha(manifest), 'pixel_index_sha256': sha(pixel_index)}
    cohorts = {'assessment': expected}
    ready = ROOT / 'Research_Transfer_Mechanism_Benchmark_2026-10-28/Source_Retention_Readiness_01'
    prepared_plan = ready / 'Prepared_Plan.json'
    assert sha(prepared_plan) == '57043031647da4c158d8b83b5680d4e2031dfbd8ec1d85af4d21c7521851e084'
    source = read(prepared_plan)['corpora']['vlr']
    source_pixels = ready / 'vlr_Pixel_Manifest.jsonl'
    assert sha(source_pixels) == source['pixel_manifest_sha256']
    pixels = {item['clip_id']: item for item in jsonl(source_pixels.read_bytes())}
    archive_path = ROOT / source['archive']
    assert archive_path.stat().st_size == source['archive_bytes']
    assert sha(archive_path) == source['archive_sha256']
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read('prepared/manifest.jsonl')
        assert hashlib.sha256(payload).hexdigest() == source['prepared_manifest_sha256']
    expected = []
    for position, row in enumerate(jsonl(payload)):
        pixel = pixels[row['clip_id']]
        assert row['split'] == 'test' and pixel['frames'] == row['frames']
        expected.append({'position': position, 'clip_id': row['clip_id'], 'source_id': row['source_id'],
                         'frames': row['frames'], 'reference': row['text'], 'pixel_sha256': pixel['pixel_sha256']})
    cohorts['source'] = expected
    provenance['source'] = {'plan_sha256': sha(prepared_plan), **source}
    for stage, rows in cohorts.items():
        target = OUT / (stage + '_Cohort.jsonl')
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        shutil.copy2(PAPER / 'Data/Short_Only' / stage / 'Per_Clip_Counts.csv', OUT / (stage + '_Common.csv'))
    runtime_fields = ['selected_training_step', 'training_budget_steps', 'classes', 'blank_id', 'eos_id',
                      'precision', 'tf32', 'deterministic_algorithms', 'device', 'threads', 'interop_threads',
                      'torch', 'numpy', 'reference_used_for_inference', 'source_files', 'matched_bridge_sha256',
                      'native_graph_binding', 'training_identity_modified']
    identities = {}
    for model in ('MATCHED_FC17', 'MATCHED_WW17', 'MATCHED_SO17'):
        path = HERE / 'Collected/assessment' / model / 'Identity.json'
        identity = read(path)
        registry = HERE / 'Bindings' / model / 'model_registry.json'
        assert sha(registry) == identity['registry_sha256']
        assert read(registry)['models'][model] == identity['registry_entry']
        for relative, digest in identity['source_files'].items():
            assert sha(ROOT / relative) == digest, relative
        identities[model] = {key: identity[key] for key in runtime_fields}
        identities[model]['registry_sha256'] = sha(registry)
        identities[model]['registry_entry'] = identity['registry_entry']
        assert identity['selected_training_step'] == 6000
    save(OUT / 'Expected_Identities.json', identities)
    for name in ('Protocol.md', 'Code_Lock.json', 'Admission.json', 'Assets_Complete.json'):
        shutil.copy2(HERE / name, OUT / name)
    shutil.copy2(ROOT / 'Research_Phase_05_2026-10-13/assets/tokenizers/vi_units.txt', OUT / 'vi_units.txt')
    save(OUT / 'Provenance.json', provenance)
    files = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    save(OUT / 'Manifest.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'original_frozen_metadata_and_runtime_sources_freshly_hash_checked', 'files': files,
        'source_archive_freshly_rehashed': True, 'source_frames_redecoded_locally': False,
        'original_pixels_verified_by_frozen_inference_loader': True})
    print(json.dumps({'status': 'prepared', 'files': len(files), 'manifest_sha256': sha(OUT / 'Manifest.json')}))


if __name__ == '__main__':
    main()
