"""Adjust concurrency for the existing long-dataset sweep without changing inference.

Adopts workers recorded during an explicit supervisor handoff. The original
manifest and worker source stay unchanged; concurrency events are append-only.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tracking'))
import sweep_memory_world_long_datasets as base

OUT = base.OUT
CONTROL = OUT / 'concurrency.json'


def process_state(pid):
    try:
        state = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()[0]
        return None if state == 'Z' else state
    except FileNotFoundError:
        return None


def job_from_pid(pid):
    raw = Path('/proc/{}/cmdline'.format(pid)).read_bytes()
    args = [part.decode() for part in raw.split(b'\0') if part]
    if len(args) < 3 or Path(args[1]).name != 'sweep_memory_world_long_datasets.py' or args[2] != 'worker':
        raise ValueError('Not a sweep worker: {}'.format(pid))
    def option(name):
        return args[args.index(name) + 1]
    job = (option('--dataset'), option('--variant'), option('--model'), int(option('--shard')))
    return job


def worker_done(job):
    ds, variant, model, shard = job
    p = OUT / ds / variant / 'workers' / '{}_{}.json'.format(model, shard)
    if not p.exists():
        return False
    row = json.loads(p.read_text())
    return row.get('status') == 'completed'


def read_caps():
    data = json.loads(CONTROL.read_text())
    caps = {gpu: int(data['workers_per_gpu'].get(str(gpu), 4)) for gpu in base.GPUS}
    if any(n < 1 or n > 8 for n in caps.values()):
        raise ValueError('Concurrency must be 1..8 workers per GPU')
    return caps


def record_event(row):
    row = dict(at=base.benchmark.now(), **row)
    with (OUT / 'concurrency_events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def main(args):
    if not (OUT / 'manifest.json').exists():
        raise ValueError('Original experiment manifest missing')
    lock = (OUT / 'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another supervisor still holds the lock')
    saved = json.loads((OUT / 'manifest.json').read_text())
    actual = json.loads(json.dumps(base.identity()))
    if saved['identity'] != actual:
        raise ValueError('Original worker/config/checkpoint/parameter identity changed')
    if not CONTROL.exists():
        base.benchmark.write(CONTROL, {'workers_per_gpu': {str(g): 4 for g in base.GPUS}})
    caps = read_caps()
    running = {}
    adopted = set()
    if args.adopt:
        handoff = json.loads(Path(args.adopt).read_text())
        for row in handoff['workers']:
            pid, gpu = row['pid'], row['gpu']
            job = tuple(row['job'])
            if process_state(pid) is None:
                continue
            if job_from_pid(pid) != job:
                raise ValueError('Adopted worker PID changed identity')
            used = {slot[1] for slot in running if slot[0] == gpu}
            index = next(i for i in range(16) if i not in used)
            running[(gpu, index)] = dict(pid=pid, process=None, log=None, job=job, adopted=True)
            adopted.add(job)
    jobs = [(ds, v, m, shard) for v in base.VARIANTS for shard in range(base.SHARDS)
            for ds in base.DATASETS for m in base.MODELS]
    if len(adopted) != len(running):
        raise ValueError('Duplicate adopted worker job')
    queue = deque(j for j in jobs if j not in adopted and not worker_done(j))
    retries = {}
    cache_file = OUT / 'completed_groups.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    previous = json.loads((OUT / 'status.json').read_text())
    status = dict(status='starting', pid=os.getpid(), started_at=previous.get('started_at', base.benchmark.now()),
                  scheduling_started_at=base.benchmark.now(), retry_count=previous.get('retry_count', 0))
    record_event(dict(event='supervisor_started', pid=os.getpid(), adopted_workers=len(running),
                      script_sha256=base.sha(Path(__file__)), workers_per_gpu=caps))
    last_report = -1
    last_caps = caps
    try:
        while queue or running:
            caps = read_caps()
            if caps != last_caps:
                record_event(dict(event='concurrency_updated', workers_per_gpu=caps))
                last_caps = caps
            for slot, item in list(running.items()):
                p = item['process']
                if p is None:
                    if process_state(item['pid']) is not None:
                        continue
                    code = 0 if worker_done(item['job']) else -1
                else:
                    code = p.poll()
                    if code is None:
                        continue
                    item['log'].close()
                del running[slot]
                if code:
                    job = item['job']
                    retries[job] = retries.get(job, 0) + 1
                    record_event(dict(event='worker_retry', job=job, exit_code=code, attempt=retries[job]))
                    if retries[job] > 2:
                        raise RuntimeError('Worker failed after retries: {}'.format(job))
                    queue.append(job)
            for index in range(max(caps.values())):
                for gpu in base.GPUS:
                    slot = (gpu, index)
                    if index >= caps[gpu] or slot in running or not queue:
                        continue
                    job = queue.popleft()
                    ds, variant, model, shard = job
                    log_path = OUT / 'logs' / 'accelerated-{}-{}-{}-{}-attempt{}.log'.format(ds, variant, model, shard, retries.get(job, 0))
                    log = log_path.open('a')
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MVLM_DEVICE='cuda', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
                    cmd = [sys.executable, str(ROOT / 'tracking/sweep_memory_world_long_datasets.py'), 'worker',
                           '--dataset', ds, '--variant', variant, '--model', model, '--shard', str(shard)]
                    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    running[slot] = dict(pid=process.pid, process=process, log=log, job=job, adopted=False)
            base.collect_complete(cache)
            counts = {ds: {v: {m: len(list(base.result_path(ds, v, m).glob('*.json'))) for m in base.MODELS} for v in base.VARIANTS} for ds in base.DATASETS}
            status.update(status='running', updated_at=base.benchmark.now(), queued_jobs=len(queue), workers_per_gpu=caps,
                          running=[dict(gpu=g, slot=i, pid=x['pid'], dataset=x['job'][0], variant=x['job'][1], model=x['job'][2], shard=x['job'][3], adopted=x['adopted']) for (g, i), x in running.items()],
                          completed=counts, completed_video_evaluations=sum(n for d in counts.values() for v in d.values() for n in v.values()),
                          expected_video_evaluations=23520, retry_count=previous.get('retry_count', 0) + sum(retries.values()))
            base.benchmark.write(OUT / 'status.json', status)
            if len(cache) != last_report:
                base.benchmark.write(cache_file, cache)
                base.write_report(status, cache)
                last_report = len(cache)
            if running:
                time.sleep(10)
        base.collect_complete(cache)
        if len(cache) != len(base.DATASETS) * len(base.VARIANTS) * len(base.MODELS):
            raise ValueError('Missing complete groups')
        base.benchmark.write(OUT / 'summary.json', cache)
        status.update(status='completed', updated_at=base.benchmark.now(), completed_at=base.benchmark.now(), running=[], queued_jobs=0)
    except BaseException as e:
        for item in running.values():
            if item['process'] is not None:
                item['process'].terminate()
                item['log'].close()
        status.update(status='failed', error=repr(e), traceback=traceback.format_exc(), updated_at=base.benchmark.now())
        raise
    finally:
        base.benchmark.write(OUT / 'status.json', status)
        base.write_report(status, cache)
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adopt', help='Explicit handoff snapshot of still-running original workers')
    main(p.parse_args())
