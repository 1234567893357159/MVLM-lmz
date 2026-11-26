"""Exploratory full-video LaSOT/TNL2K parameter search for fixed B weights.

Uses the existing benchmark worker and metrics unchanged. Test-selected maxima
are exploratory; reference self-trained E66 scores are historical, not a new
matched-parameter control. No W&B logging.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import deque
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tracking'))
import evaluate_memory_world_benchmark as benchmark

OUT = ROOT / 'output/memory_world_b_lasot_tnl2k_sweep_20261112_run01'
REPORT = ROOT / 'reports/reproduction/MEMORY_WORLD_LASOT_TNL2K_SWEEP_20261112.md'
DATASETS = {'lasot': 280, 'tnl2k': 700}
MODELS = ('b1025', 'b4000')
SHARDS = 16
GPUS = tuple(range(8))
WORKERS_PER_GPU = 2
ANCHORS = {
    'lasot': dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.1, LAMBDA=0.4, PSI_OUT=0.1, QUALITY_WEIGHT=0.25),
    'tnl2k': dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.1, LAMBDA=0.4, PSI_OUT=0.12, QUALITY_WEIGHT=0.25),
}
VARIANTS = {
    'historical': {},
    'alpha03': {'ALPHA_CORR': 0.3},
    'alpha07': {'ALPHA_CORR': 0.7},
    'alpha09': {'ALPHA_CORR': 0.9},
    'tau001': {'TAU': 0.01},
    'tau005': {'TAU': 0.05},
    'tau010': {'TAU': 0.10},
    'omega04_lambda01': {'OMEGA': 0.4, 'LAMBDA': 0.1},
    'psi005': {'PSI_OUT': 0.05},
    'psi020': {'PSI_OUT': 0.20},
    'quality000': {'QUALITY_WEIGHT': 0.0},
    'quality050': {'QUALITY_WEIGHT': 0.5},
}
E66_REFERENCE = {'lasot': {'AUC': 59.09, 'PRE': 63.24, 'NPR': 66.67},
                 'tnl2k': {'AUC': 56.64, 'PRE': 59.41, 'NPR': 72.88}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parameters(dataset, variant):
    return dict(ANCHORS[dataset], **VARIANTS[variant])


def worker(args):
    from lib.test.evaluation.tracker import Tracker
    values = parameters(args.dataset, args.variant)
    original = Tracker.get_parameters
    destination = (OUT.parent / (OUT.name + '_smoke') if args.smoke else OUT) / args.dataset / args.variant

    def get_parameters(self):
        params = original(self)
        for key, value in values.items():
            section = params.cfg.MODEL.WORLD if key == 'QUALITY_WEIGHT' else params.cfg.TEST.MVLM
            setattr(section, key, value)
            if getattr(section, key) != value:
                raise ValueError('Failed to apply ' + key)
        audit = dict(dataset=args.dataset, variant=args.variant, model=args.model,
                     shard=args.shard, parameters=values, checkpoint=str(params.checkpoint))
        benchmark.write(destination / 'effective_parameters' / '{}_{}.json'.format(args.model, args.shard), audit)
        return params

    Tracker.get_parameters = get_parameters
    benchmark.DATASETS = {args.dataset: DATASETS[args.dataset]}
    benchmark.OUT = destination
    print('EFFECTIVE_PARAMETERS', json.dumps(values, sort_keys=True), flush=True)
    benchmark.worker(SimpleNamespace(model=args.model, shard=args.shard, shards=SHARDS, smoke=args.smoke))


def identity():
    paths = [Path(__file__).resolve(), ROOT / 'tracking/evaluate_memory_world_benchmark.py',
             ROOT / 'experiments/mvlm/mvlm_TF_memory_world_tracker_b_2gpu.yaml']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(git_revision=revision, source={str(p.relative_to(ROOT)): sha(p) for p in paths},
                models={m: dict(checkpoint=str(benchmark.MODELS[m][0]), sha256=sha(benchmark.MODELS[m][0])) for m in MODELS},
                datasets=DATASETS, shards=SHARDS, gpus=GPUS, workers_per_gpu=WORKERS_PER_GPU,
                variants={ds: {v: parameters(ds, v) for v in VARIANTS} for ds in DATASETS})


def result_path(ds, variant, model):
    return OUT / ds / variant / 'results' / model / ds


def collect_complete(cache):
    for ds, expected in DATASETS.items():
        for variant in VARIANTS:
            for model in MODELS:
                key = ds + '/' + variant + '/' + model
                if key in cache:
                    continue
                paths = list(result_path(ds, variant, model).glob('*.json'))
                if len(paths) < expected:
                    continue
                if len(paths) != expected:
                    raise ValueError(('Unexpected video count', key, len(paths)))
                rows = [json.loads(p.read_text()) for p in paths]
                if len({r['sequence'] for r in rows}) != expected:
                    raise ValueError(('Duplicate videos', key))
                for row in rows:
                    if row['dataset'] != ds or len(row['boxes']) != row['frames']:
                        raise ValueError(('Invalid completed result', key, row['sequence']))
                cache[key] = dict(sequences=expected, frames=sum(r['frames'] for r in rows),
                                  **{k: sum(r[k] for r in rows) / expected for k in ('AUC', 'PRE', 'NPR')})
    return cache


def write_report(status, cache):
    lines = ['# LaSOT / TNL2K 多参数搜索（探索性）', '',
             '更新时间：' + status.get('updated_at', benchmark.now()) + '。状态：' + status['status'] + '。', '',
             '固定 B1025 和 B4000，不训练新权重；每个数据集12组参数，每组完整视频评测。LaSOT每组280条，TNL2K每组700条，共23,520次视频测试。此处最高值是在测试集上探索选出的有限组合最高值，不是独立最终测试，也不保证全局最优。', '',
             '参考的是用户自训复现 E66 的历史参数与成绩：LaSOT 59.09/63.24/66.67，TNL2K 56.64/59.41/72.88（AUC/PRE/NPR）。其他参数下没有同步重测E66，坐标路径也可能不同，不作严格同参数模型差距结论。', '',
             '搜索脚本：`tracking/sweep_memory_world_long_datasets.py`；独立输出：`' + str(OUT) + '`。manifest记录权重和源码哈希及完整参数；status记录进度，日志和逐视频预测持续写出。中断后以相同脚本运行 `suite --resume` 可跳过完整分片；单分片最多重试两次。无W&B上传。', '',
             '## 参考参数与候选', '', '| 数据集 | α | τ | ω | λ | ψ | 质量重排权重 |', '|---|---:|---:|---:|---:|---:|---:|']
    for ds, vals in ANCHORS.items():
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(ds, *[vals[k] for k in ('ALPHA_CORR','TAU','OMEGA','LAMBDA','PSI_OUT','QUALITY_WEIGHT')]))
    lines.extend(['', '| 参数组 | 相对本数据集参考配置的变化 |', '|---|---|'])
    for name, values in VARIANTS.items():
        lines.append('| {} | {} |'.format(name, ', '.join('{}={}'.format(k, v) for k, v in values.items()) or '不变'))
    lines.extend(['', '## 已完整完成的结果', '', '| 数据集 | 组别 | 模型 | AUC | PRE | NPR |', '|---|---|---|---:|---:|---:|'])
    for key, row in sorted(cache.items()):
        ds, variant, model = key.split('/')
        lines.append('| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} |'.format(ds, variant, model, row['AUC'], row['PRE'], row['NPR']))
    for ds in DATASETS:
        for model in MODELS:
            candidates = [(key.split('/')[1], row) for key, row in cache.items() if key.startswith(ds + '/') and key.endswith('/' + model)]
            if candidates:
                best, row = max(candidates, key=lambda x: (x[1]['AUC'], x[1]['PRE']))
                lines.extend(['', '{} / {}：已完成组合中的最高AUC {:.2f}（{}）；完整参数 `{}`。'.format(ds, model, row['AUC'], best, json.dumps(parameters(ds, best), ensure_ascii=False, sort_keys=True))])
    lines.extend(['', '部分完成的视频不参与参数排名。所有组完整结束后才确定本轮搜索最高值。当前报告按实际完成进度更新，不覆盖旧实验。', ''])
    temporary = REPORT.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(REPORT)


def suite(args):
    if args.resume:
        if not (OUT / 'manifest.json').exists():
            raise ValueError('Resume requires an existing manifest')
    else:
        OUT.mkdir(parents=True, exist_ok=False)
    lock = (OUT / 'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A supervisor is already running')
    ident = identity()
    # JSON normalizes the GPU tuple to a list.
    ident = json.loads(json.dumps(ident))
    manifest_path = OUT / 'manifest.json'
    if args.resume:
        saved = json.loads(manifest_path.read_text())
        if saved['identity'] != ident:
            raise ValueError('Cannot resume changed code/config/checkpoints/search grid')
    else:
        benchmark.write(manifest_path, dict(started_at=benchmark.now(), identity=ident,
                        protocol='main first-frame GT scoring; test-set exploratory search',
                        E66_historical_reference=E66_REFERENCE, expected_video_evaluations=23520))
    cache_path = OUT / 'completed_groups.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    # Interleave datasets and models so progress is available on both test sets.
    queue = deque((ds, variant, model, shard) for variant in VARIANTS for shard in range(SHARDS)
                  for ds in DATASETS for model in MODELS)
    pending = deque()
    for job in queue:
        ds, v, model, shard = job
        path = OUT / ds / v / 'workers' / '{}_{}.json'.format(model, shard)
        if args.resume and path.exists() and json.loads(path.read_text()).get('status') == 'completed':
            continue
        pending.append(job)
    queue = pending
    slots = [(g, slot) for slot in range(WORKERS_PER_GPU) for g in GPUS]
    running = {}
    retries = {}
    status = dict(status='starting', pid=os.getpid(), started_at=benchmark.now())
    last_report_size = -1
    try:
        while queue or running:
            for slot, (process, log, job) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[slot]
                if code:
                    retries[job] = retries.get(job, 0) + 1
                    if retries[job] > 2:
                        raise RuntimeError('Worker failed after retries: {} exit={}'.format(job, code))
                    queue.append(job)
            for slot in slots:
                if slot in running or not queue:
                    continue
                job = queue.popleft()
                ds, variant, model, shard = job
                log_path = OUT / 'logs' / '{}-{}-{}-{}-attempt{}.log'.format(ds, variant, model, shard, retries.get(job, 0))
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open('a')
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(slot[0]), MVLM_DEVICE='cuda', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
                cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--dataset', ds,
                       '--variant', variant, '--model', model, '--shard', str(shard)]
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                running[slot] = (process, log, job)
            collect_complete(cache)
            counts = {ds: {v: {m: len(list(result_path(ds, v, m).glob('*.json'))) for m in MODELS} for v in VARIANTS} for ds in DATASETS}
            status.update(status='running', updated_at=benchmark.now(), queued_jobs=len(queue),
                          running=[dict(gpu=slot[0], slot=slot[1], pid=p.pid, dataset=j[0], variant=j[1], model=j[2], shard=j[3]) for slot, (p, log, j) in running.items()],
                          completed=counts, completed_video_evaluations=sum(n for ds in counts.values() for v in ds.values() for n in v.values()),
                          expected_video_evaluations=23520, retry_count=sum(retries.values()))
            benchmark.write(OUT / 'status.json', status)
            if len(cache) != last_report_size:
                benchmark.write(cache_path, cache)
                write_report(status, cache)
                last_report_size = len(cache)
            if running:
                time.sleep(20)
        collect_complete(cache)
        if len(cache) != len(DATASETS) * len(VARIANTS) * len(MODELS):
            raise ValueError('Missing complete parameter groups')
        benchmark.write(OUT / 'summary.json', cache)
        status.update(status='completed', updated_at=benchmark.now(), completed_at=benchmark.now(), running=[], queued_jobs=0)
    except BaseException as error:
        for process, log, job in running.values():
            process.terminate()
            log.close()
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc(), updated_at=benchmark.now())
        raise
    finally:
        benchmark.write(OUT / 'status.json', status)
        write_report(status, cache)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('suite')
    run.add_argument('--resume', action='store_true')
    work = sub.add_parser('worker')
    work.add_argument('--dataset', choices=DATASETS, required=True)
    work.add_argument('--variant', choices=VARIANTS, required=True)
    work.add_argument('--model', choices=MODELS, required=True)
    work.add_argument('--shard', type=int, choices=range(SHARDS), required=True)
    work.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    worker(args) if args.command == 'worker' else suite(args)
