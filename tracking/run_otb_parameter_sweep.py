"""Five sequential, adaptive full-dataset parameter trials; preserve every run."""
import copy
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

from tracking.run_otb_parameter_trial import ROOT, OUT, CONFIG, RECORDS, record, timestamp


def completed():
    events = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    rows = list({r['exp_id']: r for r in events}.values())
    return [r for r in rows if r['status'] == 'completed']


def best():
    return max(completed(), key=lambda r: (r['metrics']['AUC'], r['metrics']['PRE']))


def install_config(parameters):
    text = (OUT / 'otb99_language_baseline.yaml').read_text()
    expected = yaml.safe_load(text)
    expected['TEST']['MVLM'] = copy.deepcopy(parameters)
    for key, value in parameters.items():
        text, n = re.subn(r'(?m)^(    ' + re.escape(key) + r':)[^\n]*$',
                          lambda m: m[1] + ' ' + str(value), text)
        assert n == 1, key
    assert yaml.safe_load(text) == expected
    CONFIG.write_text(text)
    return text, expected


def run_trial(label, parameters, rationale, batch):
    text, config = install_config(parameters)
    exp = 'otb99_' + batch + '_' + label
    snapshot = OUT / (exp + '.yaml')
    assert not snapshot.exists(), exp
    snapshot.write_text(text)
    result_dir = ROOT / 'output/test/tracking_results/mvlm' / exp / 'otb99_lang'
    log = OUT / (exp + '_test.log')
    metrics = OUT / (exp + '_metrics.txt')
    test = [sys.executable, '-u', '-m', 'tracking.test', 'mvlm', 'mvlm_TF',
            '--dataset_name', 'otb99_lang', '--weight_path', str(ROOT / 'models/MVLM_TF.pth.tar'),
            '--exp_id', exp, '--mode', 'mp', '--threads', '8', '--num_gpus', '8']
    analysis = [sys.executable, '-m', 'tracking.analysis_results', '--tracker_name', 'mvlm',
                '--tracker_param', 'mvlm_TF', '--exp_id', exp, '--dataset', 'otb99_lang']
    row = dict(exp_id=exp, timestamp=timestamp(), status='running', config=config,
               config_snapshot=str(snapshot), results_dir=str(result_dir), test_log=str(log),
               metrics_file=str(metrics), rationale=rationale, batch=batch,
               commands=[' '.join(test), ' '.join(analysis)])
    record(row)
    print('STARTED', label, json.dumps(parameters), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        with log.open('w') as f:
            subprocess.run(test, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        from lib.test.evaluation import get_dataset
        ds = get_dataset('otb99_lang')
        for s in ds:
            boxes = np.loadtxt(result_dir / (s.name + '.txt'))
            times = np.loadtxt(result_dir / (s.name + '_time.txt'))
            assert boxes.shape == (len(s.frames), 4), s.name
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all(), s.name
            assert len(times) == len(s.frames) and np.isfinite(times).all() and (times >= 0).all(), s.name
        with metrics.open('w') as f:
            subprocess.run(analysis, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        match = re.search(r'mvlm_TF\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', metrics.read_text())
        assert match, 'Missing metrics'
        cache = OUT / (exp + '_eval_data.pkl')
        shutil.copy2(ROOT / 'output/test/result_plots/otb99_lang/eval_data.pkl', cache)
        row.update(status='completed', timestamp=timestamp(), validated_sequences=len(ds),
                   validated_frames=sum(len(s.frames) for s in ds), eval_data=str(cache),
                   metrics=dict(zip(['PRE', 'NPR', 'AUC'], map(float, match.groups()))))
        print('COMPLETED', label, row['metrics'], flush=True)
    except Exception as exc:
        row.update(status='failed', timestamp=timestamp(), error=repr(exc))
        print('FAILED', label, repr(exc), flush=True)
        raise
    finally:
        record(row)


def main():
    batch = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    initial = copy.deepcopy(best()['config']['TEST']['MVLM'])
    plans = [
        ('01_tau003', {'TAU': 0.03}, '在此前 TAU=0.0 的改进配置上，测试较温和的正阈值，平衡干扰框过滤与漏检。', False),
        ('02_tau006', {'TAU': 0.06}, '固定其他参数，测试介于 0.03 与原始 0.1 之间的阈值，观察筛选强度趋势。', False),
        ('03_psi01', {'PSI_OUT': 0.1}, '从当前最高 AUC 配置出发，允许干扰候选框有少量重叠，检验差值计算的干扰池影响。', True),
        ('04_memory', {'OMEGA': 0.6, 'LAMBDA': 0.7}, '从当前最高 AUC 配置出发，采用仓库另一配置中的较强历史记忆，检查时间平滑效果。', True),
        ('05_alpha07', {'ALPHA_CORR': 0.7}, '从当前最高 AUC 配置出发，加入 30% 分类差值，检验纯图文差值之外的置信度信息。', True),
    ]
    for label, changes, rationale, adaptive in plans:
        parameters = copy.deepcopy(best()['config']['TEST']['MVLM'] if adaptive else initial)
        parameters.update(changes)
        run_trial(label, parameters, rationale, batch)
    winner = best()
    install_config(winner['config']['TEST']['MVLM'])
    # Keep the shared analysis cache consistent with the selected configuration.
    if winner.get('eval_data'):
        shutil.copy2(winner['eval_data'], ROOT / 'output/test/result_plots/otb99_lang/eval_data.pkl')
    summary = dict(batch=batch, completed_trials=5, selected_exp_id=winner['exp_id'],
                   selected_parameters=winner['config']['TEST']['MVLM'], metrics=winner['metrics'],
                   criterion='Highest full-test-set AUC, PRE as tie breaker; exploratory tuning', timestamp=timestamp())
    (OUT / (batch + '_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print('BATCH_COMPLETED', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
