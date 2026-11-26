"""Update only the marked long-dataset section of the experiment overview."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/memory_world_b_lasot_tnl2k_sweep_20261112_run01'
OVERVIEW = ROOT / 'doc/EXPERIMENTS.md'
BEGIN = '<!-- LONG_SWEEP_OVERVIEW_BEGIN -->'
END = '<!-- LONG_SWEEP_OVERVIEW_END -->'
ORDER = ('historical','alpha03','alpha07','alpha09','tau001','tau005','tau010',
         'omega04_lambda01','psi005','psi020','quality000','quality050')


def build_section(status, manifest, completed):
    reference = manifest['identity']['variants']
    total = status['expected_video_evaluations']
    lines = [BEGIN, '### LaSOT / TNL2K 多参数搜索（探索性）', '',
             '截至 {}，状态 `{}`，已完成 **{} / {} 次视频测试**；此段为带时间的快照。'.format(status.get('updated_at', '未记录'), status['status'], status.get('completed_video_evaluations', 0), total), '',
             'B1025 / B4000 分别测试12组参数；每组为 LaSOT 280 条、TNL2K 700 条完整视频。以下分数为 AUC / PRE / NPR；只汇总每个模型已完整完成的视频集。参数在测试集上探索，不是独立最终测试；目前自训 E66 未按每一组参数与同一路径重新评测。', '',
             '参考配置 `(α, τ, ω, λ, ψ, QUALITY_WEIGHT)`：LaSOT 为 `(0.5, 0.03, 0.1, 0.4, 0.1, 0.25)`；TNL2K 为 `(0.5, 0.03, 0.1, 0.4, 0.12, 0.25)`。表中仅列出相对各数据集参考配置的改动。', '',
             '| 参数组 | 参数改动 | LaSOT B1025 | LaSOT B4000 | TNL2K B1025 | TNL2K B4000 |',
             '|---|---|---|---|---|---|']
    def cell(ds, variant, model):
        row = completed.get('/'.join((ds, variant, model)))
        if row:
            return ' / '.join('{:.2f}'.format(row[k]) for k in ('AUC', 'PRE', 'NPR'))
        n = status.get('completed', {}).get(ds, {}).get(variant, {}).get(model, 0)
        return '进行中 {}/{}'.format(n, 280 if ds == 'lasot' else 700)
    anchor = reference['lasot']['historical']
    for variant in ORDER:
        params = reference['lasot'][variant]
        changes = ', '.join('{}={}'.format(k, v) for k, v in params.items() if v != anchor[k]) or '参考配置'
        cells = [cell(ds, variant, model) for ds in ('lasot', 'tnl2k') for model in ('b1025', 'b4000')]
        lines.append('| {} | {} | {} |'.format(variant, changes, ' | '.join(cells)))
    lines.extend(['', '自训复现 E66 历史测试 AUC：LaSOT 59.09、TNL2K 56.64；这些历史结果使用各数据集历史参数，且与当前部分候选的参数或坐标路径不同，仅作参考。搜索进度和完整参数见[自动更新的搜索报告](../reports/reproduction/MEMORY_WORLD_LASOT_TNL2K_SWEEP_20261112.md)及 `output/memory_world_b_lasot_tnl2k_sweep_20261112_run01/status.json`。', '', END])
    return '\n'.join(lines)


def update():
    status = json.loads((OUT / 'status.json').read_text())
    manifest = json.loads((OUT / 'manifest.json').read_text())
    summary = OUT / ('summary.json' if status['status'] == 'completed' and (OUT / 'summary.json').exists() else 'completed_groups.json')
    completed = json.loads(summary.read_text()) if summary.exists() else {}
    section = build_section(status, manifest, completed)
    for _ in range(5):
        stat = OVERVIEW.stat()
        text = OVERVIEW.read_text()
        if text.count(BEGIN) == text.count(END) == 1:
            start, finish = text.index(BEGIN), text.index(END) + len(END)
            updated = text[:start] + section + text[finish:]
        elif BEGIN not in text and END not in text:
            marker = '## 8. 读表时最重要的限制'
            if text.count(marker) != 1:
                raise ValueError('Overview insertion point is ambiguous')
            updated = text.replace(marker, section + '\n\n' + marker, 1)
        else:
            raise ValueError('Overview markers are missing or duplicated')
        if updated == text:
            return status
        tmp = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=OVERVIEW.parent, prefix='.experiment-overview-', delete=False) as f:
                tmp = Path(f.name)
                f.write(updated)
            current = OVERVIEW.stat()
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (current.st_ino, current.st_size, current.st_mtime_ns):
                continue
            os.chmod(tmp, stat.st_mode & 0o777)
            tmp.replace(OVERVIEW)
            tmp = None
            return status
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    raise RuntimeError('Overview changed concurrently; update was not applied')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Wait until the sweep finishes, then update once')
    args = parser.parse_args()
    if args.watch:
        while True:
            status = json.loads((OUT / 'status.json').read_text())
            if status['status'] in ('completed', 'failed'):
                break
            time.sleep(30)
    result = update()
    print('Updated', OVERVIEW, 'status=', result['status'], 'at', result.get('updated_at'), flush=True)
