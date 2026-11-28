"""Parallel supervisor for sweep_mvlm_generic.py with a live Markdown report."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'tracking/sweep_mvlm_generic.py'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def safe_name(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name):
        raise ValueError('Unsafe model or variant name: ' + repr(name))
    return name


def make_jobs(spec):
    if not spec.get('models') or not spec.get('variants'):
        raise ValueError('At least one model and one variant are required')
    jobs = []
    for model, model_spec in spec['models'].items():
        safe_name(model)
        for variant, values in spec['variants'].items():
            safe_name(variant)
            # Separate path components avoid ambiguous concatenated names.
            child = dict(spec)
            child['models'] = {model: model_spec}
            child['variants'] = {variant: values}
            jobs.append({'model': model, 'variant': variant, 'spec': child})
    return jobs


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def job_dir(output, job):
    return output / 'jobs' / job['model'] / job['variant']


def job_status(output, job):
    return read_json(job_dir(output, job) / 'status.json') or {}


def render(output, jobs, running, outcomes, state, scope):
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# MVLM 参数搜索实时记录', '', '更新时间：' + now, '状态：' + state,
             '评测范围：' + scope, '',
             '仅完整结束的组合显示结果。AUC、PRE、NPR 为数据集内逐视频等权；参数和协议见实验清单。', '',
             '| 模型 | 参数组 | GPU | 状态 | 已完成/总数 | 各数据集 AUC / PRE / NPR |',
             '|---|---|---:|---|---:|---|']
    for job in jobs:
        key = (job['model'], job['variant'])
        status = job_status(output, job)
        active = next((slot for slot, value in running.items() if value[1] == job), None)
        label = '运行中' if active is not None else outcomes.get(key, '排队中')
        score = '—'
        if label == '完成':
            summary = read_json(job_dir(output, job) / 'summary.json')
            try:
                datasets = summary[job['model']][job['variant']]['per_dataset']
                score = '; '.join('{}: {:.3f} / {:.3f} / {:.3f}'.format(ds, row['AUC'], row['PRE'], row['NPR']) for ds, row in sorted(datasets.items()))
            except (KeyError, TypeError):
                label = '结果缺失'
        progress = '{}/{}'.format(status.get('completed', 0), status.get('total', '?'))
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(job['model'], job['variant'], active[0] if active else '—', label, progress, score))
    atomic_write(output / 'LIVE.md', '\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', required=True, help='Comma-separated physical GPU indices, e.g. 5,6')
    p.add_argument('--workers-per-gpu', type=int, default=1)
    p.add_argument('--interval', type=float, default=10)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-frames', type=int, help='Diagnostic only; truncate every video')
    args = p.parse_args()
    gpus = args.gpus.split(',')
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        p.error('GPU indices must be unique nonnegative integers')
    if args.workers_per_gpu < 1 or args.interval <= 0 or (args.smoke_frames is not None and args.smoke_frames < 2):
        p.error('workers-per-gpu and interval must be positive; smoke-frames must be at least 2')
    source = Path(args.spec).expanduser().resolve()
    spec = json.loads(source.read_text())
    jobs = make_jobs(spec)
    output = Path(args.output).expanduser().resolve()
    models = {}
    for name, model in spec['models'].items():
        paths = {}
        for key in ('config', 'checkpoint'):
            path = Path(model[key]).expanduser()
            path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
            paths[key] = {'path': str(path), 'sha256': digest(path)}
        models[name] = paths
    data_manifest = None
    if spec['data']['kind'] == 'heldout':
        path = Path(spec['data']['manifest']).expanduser()
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        data_manifest = {'path': str(path), 'sha256': digest(path)}
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary', '--', 'lib', 'tracking', 'experiments'], cwd=ROOT)
    identity = {'spec': spec, 'gpus': gpus, 'workers_per_gpu': args.workers_per_gpu,
                'smoke_frames': args.smoke_frames, 'supervisor_sha256': digest(__file__),
                'worker_sha256': digest(WORKER), 'models': models, 'data_manifest': data_manifest,
                'tracked_code_diff_sha256': hashlib.sha256(diff).hexdigest(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    if args.resume:
        if not (output / 'parallel_manifest.json').is_file():
            raise ValueError('Resume requires an existing parallel manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / 'parallel.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / 'parallel_manifest.json'
        if args.resume:
            if read_json(manifest)['identity'] != identity:
                raise ValueError('Cannot resume: specification, GPU slots, code or Git revision changed')
        else:
            write_json(manifest, {'identity': identity, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        for job in jobs:
            write_json(output / 'specs' / job['model'] / (job['variant'] + '.json'), job['spec'])
        pending = []
        outcomes = {}
        for job in jobs:
            key = (job['model'], job['variant'])
            status = job_status(output, job)
            if args.resume and status.get('status') == 'completed' and (job_dir(output, job) / 'summary.json').is_file():
                outcomes[key] = '完成'
            else:
                pending.append(job)
        slots = [(gpu, slot) for slot in range(args.workers_per_gpu) for gpu in gpus]
        running = {}
        failure = None
        scope = ('诊断短片段，不是完整视频结果' if args.smoke_frames else
                 ('标准测试集搜索：探索性选参' if spec['data']['kind'] == 'benchmark' else '训练来源留出验证'))
        try:
            while pending or running:
                for slot, (proc, job, log) in list(running.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    log.close()
                    del running[slot]
                    status = job_status(output, job)
                    key = (job['model'], job['variant'])
                    if code == 0 and status.get('status') == 'completed' and (job_dir(output, job) / 'summary.json').is_file():
                        outcomes[key] = '完成'
                    else:
                        outcomes[key] = '失败（退出码 {}）'.format(code)
                        failure = '{} / {} failed: {}'.format(*key, code)
                for slot in slots:
                    if failure or slot in running or not pending:
                        continue
                    job = pending.pop(0)
                    folder = job_dir(output, job)
                    logpath = output / 'logs' / job['model'] / (job['variant'] + '.log')
                    logpath.parent.mkdir(parents=True, exist_ok=True)
                    log = logpath.open('a')
                    cmd = [sys.executable, '-u', str(WORKER), '--spec', str(output / 'specs' / job['model'] / (job['variant'] + '.json')),
                           '--output', str(folder), '--device', 'cuda:0']
                    if args.smoke_frames is not None:
                        cmd += ['--smoke-frames', str(args.smoke_frames)]
                    if folder.exists():
                        if not args.resume or not (folder / 'manifest.json').is_file():
                            raise ValueError('Existing incomplete job without resumable manifest: ' + str(folder))
                        cmd.append('--resume')
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=slot[0], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    running[slot] = (proc, job, log)
                state = '失败，等待其他任务结束：' + failure if failure else '运行中'
                render(output, jobs, running, outcomes, state, scope)
                write_json(output / 'parallel_status.json', {'status': 'failed_waiting' if failure else 'running',
                           'pending': len(pending), 'running': [{'gpu': slot[0], 'slot': slot[1], 'pid': value[0].pid,
                           'model': value[1]['model'], 'variant': value[1]['variant']} for slot, value in running.items()],
                           'completed': len([x for x in outcomes.values() if x == '完成']), 'error': failure})
                if running:
                    time.sleep(args.interval)
                elif failure:
                    break
            if failure:
                render(output, jobs, running, outcomes, '失败：' + failure, scope)
                write_json(output / 'parallel_status.json', {'status': 'failed', 'error': failure})
                return 1
            combined = {}
            for job in jobs:
                summary = read_json(job_dir(output, job) / 'summary.json')
                combined.setdefault(job['model'], {})[job['variant']] = summary[job['model']][job['variant']]
            write_json(output / 'summary.json', combined)
            render(output, jobs, running, outcomes, '全部完成', scope)
            write_json(output / 'parallel_status.json', {'status': 'completed', 'completed': len(jobs)})
            return 0
        except BaseException as exc:
            for proc, _, log in running.values():
                if proc.poll() is None:
                    proc.terminate()
                log.close()
            write_json(output / 'parallel_status.json', {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()})
            render(output, jobs, {}, outcomes, '监督器异常：' + repr(exc), scope)
            raise


if __name__ == '__main__':
    sys.exit(main())
