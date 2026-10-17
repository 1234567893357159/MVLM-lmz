"""Run one TAU=0.0 experiment and append its configuration and results."""
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/otb_parameter_trials'
CONFIG = ROOT / 'experiments/mvlm/mvlm_TF.yaml'
RECORDS = OUT / 'experiments.jsonl'


def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def record(row):
    with RECORDS.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    events = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    rows = list({event['exp_id']: event for event in events}.values())
    lines = ['# OTB99-Lang 参数实验记录', '',
             '测试集：仓库加载器的 48 条序列、29,077 帧。权重：models/MVLM_TF.pth.tar。',
             '这是测试集上的探索性调参，不能作为独立的论文复现证据。基线为此前运行结果，非本次重新运行。', '',
             '| 实验 | 状态 | αcorr | τ | ω | λ | ψout | PRE (%) | NPR (%) | AUC (%) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        p = r['config']['TEST']['MVLM']
        m = r.get('metrics', {})
        values = [r['exp_id'], r['status']] + [str(p[k]) for k in
                  ['ALPHA_CORR', 'TAU', 'OMEGA', 'LAMBDA', 'PSI_OUT']] + [str(m.get(k, '—')) for k in ['PRE', 'NPR', 'AUC']]
        lines.append('| ' + ' | '.join(values) + ' |')
    for r in rows:
        lines += ['', '## ' + r['exp_id'], '', '记录时间：' + r['timestamp'], '',
                  '完整配置快照：[' + Path(r['config_snapshot']).name + '](' + r['config_snapshot'] + ')', '',
                  '结果目录：' + r['results_dir'], '',
                  '完整 TEST 参数：', '', '```yaml', yaml.safe_dump(r['config']['TEST'], sort_keys=False).rstrip(), '```']
        for k in ['test_log', 'metrics_file', 'eval_data', 'error']:
            if k in r:
                lines += ['', k + ': ' + r[k]]
        if 'rationale' in r:
            lines += ['', '调参依据：' + r['rationale']]
        for command in r.get('commands', []):
            lines += ['', '```text', command, '```']
    (OUT / '实验记录.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = CONFIG.read_text()
    baseline_config = yaml.safe_load(original)
    assert baseline_config['TEST']['MVLM']['TAU'] == 0.1, 'Expected original TAU=0.1'
    baseline_snapshot = OUT / 'otb99_language_baseline.yaml'
    if not baseline_snapshot.exists():
        baseline_snapshot.write_text(original)
    if not RECORDS.exists():
        metrics_file = ROOT / 'output/otb99_language_baseline_metrics.txt'
        match = re.search(r'mvlm_TF\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', metrics_file.read_text())
        assert match, 'Missing baseline scores'
        record(dict(exp_id='otb99_language_baseline', status='completed', timestamp=timestamp(),
                    config=baseline_config, config_snapshot=str(baseline_snapshot),
                    results_dir=str(ROOT / 'output/test/tracking_results/mvlm/otb99_language_baseline/otb99_lang'),
                    metrics=dict(zip(['PRE', 'NPR', 'AUC'], map(float, match.groups()))),
                    metrics_file=str(metrics_file), provenance='Previously completed baseline; not rerun'))
    changed, count = re.subn(r'(?m)^(    TAU:) 0\.1\s*$', r'\1 0.0', original)
    assert count == 1
    trial_config = yaml.safe_load(changed)
    expected = json.loads(json.dumps(baseline_config))
    expected['TEST']['MVLM']['TAU'] = 0.0
    assert trial_config == expected, 'Unexpected configuration changes'
    CONFIG.write_text(changed)
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp = 'otb99_tau_0_' + tag
    snapshot = OUT / (exp + '.yaml')
    snapshot.write_text(changed)
    result_dir = ROOT / 'output/test/tracking_results/mvlm' / exp / 'otb99_lang'
    log = OUT / (exp + '_test.log')
    metrics_file = OUT / (exp + '_metrics.txt')
    base = [sys.executable, '-u', '-m']
    test = base + ['tracking.test', 'mvlm', 'mvlm_TF', '--dataset_name', 'otb99_lang',
                  '--weight_path', str(ROOT / 'models/MVLM_TF.pth.tar'), '--exp_id', exp,
                  '--mode', 'mp', '--threads', '8', '--num_gpus', '8']
    analysis = base + ['tracking.analysis_results', '--tracker_name', 'mvlm',
                       '--tracker_param', 'mvlm_TF', '--exp_id', exp, '--dataset', 'otb99_lang']
    row = dict(exp_id=exp, timestamp=timestamp(), status='running', config=trial_config,
               config_snapshot=str(snapshot), results_dir=str(result_dir), test_log=str(log),
               metrics_file=str(metrics_file), commands=[' '.join(test), ' '.join(analysis)])
    record(row)
    print('STARTED', exp, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        with log.open('w') as f:
            subprocess.run(test, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        from lib.test.evaluation import get_dataset
        dataset = get_dataset('otb99_lang')
        for seq in dataset:
            boxes = np.loadtxt(result_dir / (seq.name + '.txt'))
            times = np.loadtxt(result_dir / (seq.name + '_time.txt'))
            assert boxes.shape == (len(seq.frames), 4), seq.name
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all(), seq.name
            assert len(times) == len(seq.frames) and np.isfinite(times).all(), seq.name
        with metrics_file.open('w') as f:
            subprocess.run(analysis, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        match = re.search(r'mvlm_TF\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', metrics_file.read_text())
        assert match, 'Missing trial scores'
        row.update(status='completed', timestamp=timestamp(), validated_sequences=len(dataset),
                   metrics=dict(zip(['PRE', 'NPR', 'AUC'], map(float, match.groups()))))
        cache = OUT / (exp + '_eval_data.pkl')
        shutil.copy2(ROOT / 'output/test/result_plots/otb99_lang/eval_data.pkl', cache)
        row['eval_data'] = str(cache)
        print('COMPLETED', row['metrics'], flush=True)
    except Exception as exc:
        row.update(status='failed', timestamp=timestamp(), error=repr(exc))
        raise
    finally:
        record(row)


if __name__ == '__main__':
    main()
