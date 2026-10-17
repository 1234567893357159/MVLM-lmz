"""Run one isolated LaSOT baseline, validate all outputs, and archive metrics."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import cv2
    import numpy as np
    import yaml
    from lib.test.evaluation import get_dataset

    runtime_path = ROOT / 'tracking/lasot_runtime.yaml'
    runtime = yaml.safe_load(runtime_path.read_text())
    num_gpus = int(runtime['num_gpus'])
    workers = num_gpus * int(runtime['workers_per_gpu'])
    cpu_threads = str(int(runtime['cpu_threads_per_worker']))
    assert 1 <= num_gpus <= 8 and workers >= num_gpus and int(cpu_threads) > 0
    tag = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    exp = 'lasot_language_original_mp' + str(workers) + '_' + tag
    out = ROOT / 'output/lasot_evaluations' / exp
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runtime.yaml').write_text(runtime_path.read_text())
    result = ROOT / 'output/test/tracking_results/mvlm' / exp / 'lasot'
    result.mkdir(parents=True, exist_ok=False)
    state = dict(exp_id=exp, dataset='lasot', started_at=now(), status='validating_data',
                 parameter_source='HEAD:experiments/mvlm/mvlm_TF.yaml (original published parameters)',
                 results_dir=str(result), artifact_dir=str(out), runtime=runtime, workers=workers,
                 note='Single baseline evaluation; no LaSOT tuning. Uses repository language-only evaluation protocol, including its first-frame handling.')

    def save():
        state['updated_at'] = now()
        tmp = out / 'status.json.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(out / 'status.json')
        print(now(), state['status'], flush=True)

    save()
    try:
        raw = subprocess.check_output(['git', 'show', 'HEAD:experiments/mvlm/mvlm_TF.yaml'], cwd=ROOT, text=True)
        config = yaml.safe_load(raw)
        from lib.config.mvlm.config import cfg as default_cfg
        assert config['MODEL']['ENCODER'].get('TEMPLATE_FREE', default_cfg.MODEL.ENCODER.TEMPLATE_FREE) is True
        assert config['TEST']['MVLM'] == dict(ALPHA_CORR=1.0, TAU=0.1, OMEGA=0.4, LAMBDA=0.4, PSI_OUT=0.0)
        param = exp
        config_path = ROOT / 'experiments/mvlm' / (param + '.yaml')
        with config_path.open('x') as f:
            f.write(raw)
        (out / 'config.yaml').write_text(raw)
        (out / 'working_config_at_start.yaml').write_text((ROOT / 'experiments/mvlm/mvlm_TF.yaml').read_text())
        (out / 'code_changes.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
        state.update(config_name=param, config_path=str(config_path), config=config,
                     git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                     checkpoint=str(ROOT / 'models/MVLM_TF.pth.tar'),
                     checkpoint_sha256=sha256(ROOT / 'models/MVLM_TF.pth.tar'))
        save()
        metadata = json.loads((ROOT / 'downloads/lasot_test_tree.json').read_text())

        def verify_archive(row):
            p = ROOT / 'downloads' / row['path']
            assert p.stat().st_size == row['size'], str(p)
            digest = sha256(p)
            assert digest == row['lfs']['oid'], 'SHA256 mismatch: ' + str(p)
            return dict(file=row['path'], size=row['size'], sha256=digest)

        archives = [r for r in metadata if r['path'].endswith('.tar.gz')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            checks = list(pool.map(verify_archive, archives))
        (out / 'archive_checksums.json').write_text(json.dumps(checks, indent=2) + '\n')
        dataset = get_dataset('lasot')
        assert len(dataset) == 280
        sequences = []
        for seq in dataset:
            n = len(seq.frames)
            assert n > 0 and seq.ground_truth_rect.shape == (n, 4), seq.name
            assert np.isfinite(seq.ground_truth_rect).all(), seq.name
            assert len(seq.target_visible) == n and seq.language_query.strip(), seq.name
            base = Path(seq.frames[0]).parent
            present = {f.name for f in base.iterdir() if f.is_file() and f.stat().st_size > 0}
            missing = [Path(f).name for f in seq.frames if Path(f).name not in present]
            assert not missing, (seq.name, missing[:5])
            for i in sorted({0, n // 2, n - 1}):
                assert cv2.imread(seq.frames[i]) is not None, seq.frames[i]
            sequences.append(dict(name=seq.name, frames=n, language=seq.language_query))
        (out / 'dataset_manifest.json').write_text(json.dumps(sequences, ensure_ascii=False, indent=2) + '\n')
        state.update(validated_sequences=len(dataset), total_frames=sum(len(s.frames) for s in dataset),
                     validation='All six archive SHA256 match source metadata; all expected frames nonempty; annotations/language/visibility lengths checked; first/middle/last images decoded per sequence.')
        env = dict(os.environ, OMP_NUM_THREADS=cpu_threads, MKL_NUM_THREADS=cpu_threads,
                   OPENBLAS_NUM_THREADS=cpu_threads, NUMEXPR_NUM_THREADS=cpu_threads,
                   OPENCV_FOR_THREADS_NUM=cpu_threads, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
        test = [sys.executable, '-u', '-m', 'tracking.test', 'mvlm', param,
                '--dataset_name', 'lasot', '--weight_path', state['checkpoint'], '--exp_id', exp,
                '--mode', 'mp', '--threads', str(workers), '--num_gpus', str(num_gpus)]
        analysis = [sys.executable, '-u', '-m', 'tracking.analysis_results', '--tracker_name', 'mvlm',
                    '--tracker_param', param, '--exp_id', exp, '--dataset', 'lasot']
        state.update(commands=[test, analysis], status='testing', test_started_at=now())
        (out / 'README.md').write_text(
            '# LaSOT 原始参数测试（提高并发）\n\n实验：' + exp +
            '\n\n280 条序列，685,360 帧；8 张 GPU，每卡 2 个进程，共 16 个进程。' +
            '\n\n模型和评分参数沿用原始发布配置，runtime.yaml 只控制并发。' +
            '\n\n状态见 status.json，推理日志见 test.log，完成后指标见 metrics.txt 和 结果说明.md。' +
            '\n\n预测框及耗时：' + str(result) + '\n')
        (out / 'commands.json').write_text(json.dumps(state['commands'], indent=2) + '\n')
        save()
        with (out / 'test.log').open('w') as f:
            subprocess.run(test, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        state['status'] = 'validating_predictions'
        save()
        errors = []
        for seq in dataset:
            try:
                boxes = np.loadtxt(result / (seq.name + '.txt'))
                times = np.loadtxt(result / (seq.name + '_time.txt'))
                assert boxes.shape == (len(seq.frames), 4)
                assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
                assert times.shape == (len(seq.frames),) and np.isfinite(times).all() and (times >= 0).all()
            except Exception as exc:
                errors.append(dict(sequence=seq.name, error=repr(exc)))
        (out / 'prediction_validation.json').write_text(json.dumps(dict(checked=len(dataset), errors=errors), indent=2) + '\n')
        assert not errors, 'Missing or invalid predictions; see prediction_validation.json'
        state['status'] = 'computing_metrics'
        save()
        with (out / 'metrics.txt').open('w') as f:
            subprocess.run(analysis, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        text = (out / 'metrics.txt').read_text()
        match = re.search(re.escape(param) + r'\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', text)
        assert match, 'Metrics not found'
        cache = ROOT / 'output/test/result_plots/lasot/eval_data.pkl'
        (out / 'eval_data.pkl').write_bytes(cache.read_bytes())
        state.update(status='completed', completed_at=now(), metrics_percent=dict(zip(['PRE', 'NPR', 'AUC'], map(float, match.groups()))))
        save()
        m = state['metrics_percent']
        (out / '结果说明.md').write_text('# LaSOT 官方权重原始参数测试\n\n实验：' + exp + '\n\n280 条测试序列，' + str(state['total_frames']) + ' 帧。\n\nPRE：' + str(m['PRE']) + '%；NPR：' + str(m['NPR']) + '%；AUC：' + str(m['AUC']) + '%。\n\n使用原始发布参数，没有使用 OTB 调参配置。配置见 config.yaml，预测框及耗时目录见 status.json，原始指标见 metrics.txt。\n')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
