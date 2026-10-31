"""Run independent experiments across GPUs; never change effective batch size."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT, read_json, save_json, digest


def run_queue(plan, gpus, execute=False):
    jobs = read_json(plan)['jobs']
    ids = [job['id'] for job in jobs]
    if len(ids) != len(set(ids)) or not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('Job IDs and GPU IDs must be unique and nonempty')
    for job in jobs:
        if job['module'] not in {'phase5.training', 'phase5.evaluate'}:
            raise ValueError('Queue accepts training and evaluation modules only')
        if '--output' not in job['args'] or '--device' in job['args']:
            raise ValueError('Each job needs an output and uses one visible GPU')
        if any(str(job['id']).find(c) >= 0 for c in ['/','\\', ':', '..']):
            raise ValueError('Unsafe job ID')
    outputs = [str(Path(j['args'][j['args'].index('--output') + 1]).resolve()) for j in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Queue jobs must have separate output directories')
    if not execute:
        print({'jobs': len(jobs), 'gpus': gpus, 'mode': 'preview', 'plan_sha256': digest(plan)})
        for job in jobs:
            print(job['id'], sys.executable, '-m', job['module'], *job['args'])
        return
    logs = ROOT / 'runs/queue_logs' / Path(plan).stem
    logs.mkdir(parents=True, exist_ok=True)
    pending, active, done = list(jobs), {}, []
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                args = list(job['args'])
                out = Path(args[args.index('--output') + 1])
                if (out / 'completed.json').exists() or (job['module'] == 'phase5.evaluate' and (out / 'summary.json').exists()):
                    raise ValueError('Completed output exists; remove that job from the queue after review')
                if (out / 'identity.json').exists():
                    if not (out / 'last.pt').exists():
                        raise ValueError('Interrupted before first checkpoint; use a fresh output directory')
                    args.append('--resume')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                handle = (logs / f"{job['id']}.log").open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-u', '-m', job['module'], *args, '--device', 'cuda'],
                                           cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[gpu] = (process, handle, job)
                print(f"START {job['id']} GPU={gpu}", flush=True)
            for gpu, (process, handle, job) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                handle.close()
                del active[gpu]
                done.append({'id': job['id'], 'gpu': gpu, 'exit_code': status})
                save_json(logs / 'status.json', {'plan_sha256': digest(plan), 'finished': done})
                if status:
                    raise RuntimeError(f"Job {job['id']} exited {status}; inspect its log")
                print(f"DONE {job['id']}", flush=True)
            if active:
                time.sleep(1)
    finally:
        for process, handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run_queue(args.plan, args.gpus, args.execute)
