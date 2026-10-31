"""One sequential GPU queue for the pre-admitted six complete populations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    assert not (HERE / 'Population_Start.json').exists(), 'Preserve the existing attempt'
    admission = json.loads((HERE / 'Admission.json').read_text(encoding='utf-8'))
    save('Population_Start.json', {'utc': datetime.now(timezone.utc).isoformat(),
                                  'admission': admission, 'python': sys.executable})
    completed = []
    try:
        for stage in admission['stages']:
            for model in admission['models']:
                assert time.time() < admission['deadline_unix']
                target = HERE / 'Population_01' / stage / model
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-u', '-X', 'utf8', str(HERE / 'run.py'), '--model', model,
                           '--stage', stage, '--mode', 'population', '--output', str(target),
                           '--deadline', str(admission['deadline_unix'])]
                with (target.parent / (model + '.stdout.log')).open('x', encoding='utf-8') as stdout:
                    with (target.parent / (model + '.stderr.log')).open('x', encoding='utf-8') as stderr:
                        result = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=HERE,
                                                timeout=max(1, admission['deadline_unix'] - time.time() + 60))
                assert result.returncode == 0, (stage, model, result.returncode)
                terminal = json.loads((target / 'Complete.json').read_text(encoding='utf-8'))
                assert terminal['clips'] == admission['expected_clips'][stage]
                completed.append({'stage': stage, 'model': model,
                                  'complete_sha256': hashlib.sha256((target / 'Complete.json').read_bytes()).hexdigest()})
                save('Population_Progress.json', {'completed': completed})
        save('Population_Complete.json', {'utc': datetime.now(timezone.utc).isoformat(),
            'status': 'six_populations_complete_pending_local_backup_and_independent_recount',
            'completed': completed})
    except BaseException:
        save('Population_Failure.json', {'utc': datetime.now(timezone.utc).isoformat(),
                                        'completed': completed, 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
