"""Explore E66 inference parameters using the B benchmark's evaluation path."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tracking'))
import sweep_memory_world_long_datasets as sweep

OUT = ROOT / 'output/e66_parameter_sweep_20261113_run01'
REPORT = ROOT / 'reports/reproduction/E66_PARAMETER_SWEEP_20261113.md'
PREVIOUS = ROOT / 'output/memory_world_b_lasot_tnl2k_sweep_20261112_run01/status.json'
CHECKPOINT = ROOT / 'output/paper_continuation_e60_e70_20261102_run01/checkpoints/train/mvlm/mvlm_TF_continue_e60_e70/MVLM_ep0066.pth.tar'

sweep.OUT = OUT
sweep.REPORT = REPORT
sweep.DATASETS = {'otb99_lang': 48, 'lasot': 280, 'tnl2k': 700}
sweep.MODELS = ('baseline',)
sweep.WORKERS_PER_GPU = 4
sweep.benchmark.MODELS['baseline'] = (CHECKPOINT, list(range(8)))
sweep.ANCHORS = {
    'otb99_lang': dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.4, LAMBDA=0.1, PSI_OUT=0.1),
    'lasot': dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.1, LAMBDA=0.4, PSI_OUT=0.1),
    'tnl2k': dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.1, LAMBDA=0.4, PSI_OUT=0.12),
}
sweep.VARIANTS = {
    'historical': {}, 'alpha03': {'ALPHA_CORR': 0.3},
    'alpha07': {'ALPHA_CORR': 0.7}, 'alpha09': {'ALPHA_CORR': 0.9},
    'tau001': {'TAU': 0.01}, 'tau005': {'TAU': 0.05}, 'tau010': {'TAU': 0.10},
    'omega01_lambda04': {'OMEGA': 0.1, 'LAMBDA': 0.4},
    'omega04_lambda01': {'OMEGA': 0.4, 'LAMBDA': 0.1},
    'omega02_lambda04': {'OMEGA': 0.2, 'LAMBDA': 0.4},
    'psi005': {'PSI_OUT': 0.05}, 'psi020': {'PSI_OUT': 0.20},
}
EXPECTED = sum(sweep.DATASETS.values()) * len(sweep.VARIANTS)
original_write = sweep.benchmark.write
def write(path, value):
    if Path(path).parent == OUT and Path(path).name in ('status.json', 'manifest.json'):
        value = dict(value)
        if 'expected_video_evaluations' in value:
            value['expected_video_evaluations'] = EXPECTED
        value.pop('E66_historical_reference', None)
        if Path(path).name == 'manifest.json':
            value['description'] = 'E66 exploratory parameter search with common temporal baseline evaluation path'
    original_write(path, value)
sweep.benchmark.write = write

# Original worker subprocesses execute the original file, so replace its
# subprocess executable argument with this wrapper, preserving its scheduler.
original_popen = sweep.subprocess.Popen
def popen(command, *args, **kwargs):
    if isinstance(command, list) and len(command) > 2 and command[1] == str(Path(sweep.__file__).resolve()):
        command = list(command)
        command[1] = str(Path(__file__).resolve())
    return original_popen(command, *args, **kwargs)
sweep.subprocess.Popen = popen

original_identity = sweep.identity
def identity():
    result = original_identity()
    result['source'][str(Path(__file__).resolve().relative_to(ROOT))] = sweep.sha(Path(__file__).resolve())
    return result
sweep.identity = identity

def report(status, cache):
    lines = ['# E66 参数探索（测试集探索性结果）', '',
             '更新时间：' + status.get('updated_at', sweep.benchmark.now()) + '；状态：' + status['status'], '',
             '固定自训 E66 权重，使用联合模型测试的基线评测路径；每卡四个进程，八张卡。OTB 48 条、LaSOT 280 条、TNL2K 700 条，每集 12 组参数，共 12336 次视频评测。参数和检查点涉及测试集选择，最高值不构成独立最终结果。', '',
             '五个推理参数按历史配置逐项探索；质量重排是联合模型特有功能，不适用于 E66。历史 OTB 配置使 omega04_lambda01 与 historical 相同，LaSOT/TNL2K 的 omega01_lambda04 同理，保留以对齐搜索网格。', '',
             '输出目录：`' + str(OUT) + '`；manifest 记录配置、源码及权重哈希，status 记录进度，逐视频结果支持续跑。', '',
             '| 数据集 | 参数组 | 完成视频 | AUC | PRE | NPR |', '|---|---|---:|---:|---:|---:|']
    for key, row in sorted(cache.items()):
        ds, variant, _ = key.split('/')
        lines.append('| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} |'.format(ds, variant, row['sequences'], row['AUC'], row['PRE'], row['NPR']))
    for ds in sweep.DATASETS:
        rows = [(k.split('/')[1], v) for k, v in cache.items() if k.startswith(ds + '/')]
        if rows:
            v, row = max(rows, key=lambda item: item[1]['AUC'])
            lines.extend(['', '{} 已完成组最高 AUC {:.2f}：{}，参数 `{}`。'.format(ds, row['AUC'], v, json.dumps(sweep.parameters(ds, v), sort_keys=True))])
    REPORT.write_text('\n'.join(lines) + '\n')
sweep.write_report = report

def wait_previous():
    wait_path = OUT.parent / (OUT.name + '.waiting.json')
    while True:
        status = json.loads(PREVIOUS.read_text())
        old = status.get('status')
        if old == 'completed':
            sweep.benchmark.write(wait_path, {'status': 'previous_completed', 'updated_at': sweep.benchmark.now()})
            return
        pid = status.get('pid')
        try:
            stat = Path('/proc/{}/stat'.format(pid)).read_text().split(') ', 1)[1]
            alive = stat[0] != 'Z'
        except (OSError, IndexError):
            alive = False
        if old not in ('running', 'starting') or not alive:
            sweep.benchmark.write(wait_path, {'status': 'blocked_previous', 'previous_status': old, 'previous_pid': pid, 'updated_at': sweep.benchmark.now()})
            raise RuntimeError('Previous sweep is not running successfully; inspect its status before starting E66.')
        sweep.benchmark.write(wait_path, {'status': 'waiting_previous', 'pid': os.getpid(), 'previous_pid': pid, 'previous_progress': status.get('completed_video_evaluations'), 'updated_at': sweep.benchmark.now()})
        time.sleep(30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('suite')
    run.add_argument('--resume', action='store_true')
    run.add_argument('--wait-previous', action='store_true')
    work = sub.add_parser('worker')
    work.add_argument('--dataset', choices=sweep.DATASETS, required=True)
    work.add_argument('--variant', choices=sweep.VARIANTS, required=True)
    work.add_argument('--model', choices=sweep.MODELS, required=True)
    work.add_argument('--shard', type=int, choices=range(sweep.SHARDS), required=True)
    work.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'worker':
        sweep.worker(args)
    else:
        if args.wait_previous:
            wait_previous()
        sweep.suite(args)
