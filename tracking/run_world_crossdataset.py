"""Fixed-checkpoint cross-dataset tests, with fresh matched MVLM baselines."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import _init_paths
import numpy as np
import torch
import yaml
from lib.config.mvlm.config import load_config
from lib.test.evaluation import get_dataset
from train_world_model import sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/root/MVLM/output/test/tracking_results/mvlm')
DATASETS = ('otb99_lang', 'lasot', 'tnl2k')
SIZES = (48, 280, 700)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def validate_predictions(dataset, folder):
    errors = []
    for sequence in dataset:
        try:
            boxes = np.loadtxt(str(folder / (sequence.name + '.txt')), ndmin=2)
            times = np.loadtxt(str(folder / (sequence.name + '_time.txt')), ndmin=1)
            assert boxes.shape == (len(sequence.frames), 4), boxes.shape
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            assert times.shape == (len(sequence.frames),), times.shape
            assert np.isfinite(times).all() and (times >= 0).all()
        except Exception as error:
            errors.append(dict(sequence=sequence.name, error=repr(error)))
    return dict(checked_sequences=len(dataset), frames=sum(len(s.frames) for s in dataset), errors=errors)


def read_metrics(path, expected_sequences):
    with path.open('rb') as handle:
        data = pickle.load(handle)
    assert len(data['sequences']) == expected_sequences and all(data['valid_sequence'])
    success = np.asarray(data['ave_success_rate_plot_overlap'], dtype=np.float64).mean(0)[0]
    precision = np.asarray(data['ave_success_rate_plot_center'], dtype=np.float64).mean(0)[0]
    normalized = np.asarray(data['ave_success_rate_plot_center_norm'], dtype=np.float64).mean(0)[0]
    assert len(success) == 21
    return dict(AUC=float(success.mean() * 100), PRE=float(precision[20] * 100), NPR=float(normalized[20] * 100))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline', default='/root/MVLM/models/MVLM_TF.pth.tar')
    parser.add_argument('--tag', default='20261026_run01')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-entity', default='2896287267-hit')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    status = dict(status='preparing', started_at=now(), pid=os.getpid(), completed_jobs=0, total_jobs=6)
    state_path = output / 'status.json'
    write_json(state_path, status)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    assert checkpoint['world_stage'] == 'B' and checkpoint['step'] == 1000
    config = json.loads(json.dumps(checkpoint['config']))
    assert config['MODEL']['WORLD']['ENABLED']
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=checkpoint['step'],
        checkpoint_sha256=sha256(args.checkpoint), baseline=str(Path(args.baseline).resolve()),
        baseline_sha256=sha256(args.baseline), tag=args.tag, datasets=list(DATASETS),
        workers=args.workers, gpus=args.gpus, started_at=now(),
        protocol='standard complete test sequences; original MVLM metric rules and integer saved boxes',
        selection='B step 1000 chosen using VastTrack validation loss before external tests; no test tuning',
        known_annotation_issue='TNL2K Fight_video_6-Done has empty language; included in both standard 700-sequence runs')
    del checkpoint
    previous = json.loads((output / 'run.json').read_text()) if args.resume and (output / 'run.json').exists() else None
    if previous:
        for key in ('checkpoint_sha256', 'baseline_sha256', 'tag', 'datasets'):
            assert previous[key] == metadata[key], key
    write_json(output / 'run.json', metadata)
    names = {}
    for variant in ('baseline', 'world'):
        value = json.loads(json.dumps(config))
        value['MODEL']['WORLD']['ENABLED'] = variant == 'world'
        name = 'world_cross_' + variant + '_' + args.tag
        path = ROOT / 'experiments/mvlm' / (name + '.yaml')
        text = yaml.safe_dump(value, sort_keys=False)
        if path.exists():
            assert path.read_text() == text, 'Existing config differs: ' + str(path)
        else:
            path.write_text(text)
        (output / (variant + '_config.yaml')).write_text(text)
        assert load_config(path)['TEST'] == config['TEST']
        names[variant] = name
    # Preserve operational code/configs as well as the checkpoint selection.
    for relative in ('tracking/run_world_crossdataset.py', 'tracking/test.py', 'lib/test/evaluation/running.py',
                     'lib/test/evaluation/tracker.py', 'lib/test/analysis/extract_results.py', 'lib/test/parameter/mvlm.py',
                     'lib/test/tracker/mvlm.py'):
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(ROOT / relative), str(target))
    shutil.copytree(str(ROOT / 'lib/models/mvlm'), str(output / 'source/lib/models/mvlm'), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    tracking = None
    if args.wandb_project:
        try:
            import wandb
            prior = json.loads((output / 'wandb_run.json').read_text()) if args.resume and (output / 'wandb_run.json').exists() else None
            tracking = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                name='world-crossdataset-evaluation-' + args.tag, config=metadata, dir=str(output),
                id=prior['id'] if prior else None, resume='must' if prior else None, mode='online',
                settings=wandb.Settings(init_timeout=90))
            write_json(output / 'wandb_run.json', dict(id=tracking.id, url=tracking.url))
            print('W&B evaluation URL: ' + tracking.url, flush=True)
        except Exception as error:
            status['wandb_error'] = repr(error)
            print('Online monitoring unavailable; continuing local evaluations: ' + repr(error), flush=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               OPENCV_FOR_THREADS_NUM='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    summaries = []
    try:
        for dataset_name, size in zip(DATASETS, SIZES):
            dataset = get_dataset(dataset_name)
            assert len(dataset) == size
            for variant in ('baseline', 'world'):
                artifact = output / dataset_name / variant
                artifact.mkdir(parents=True, exist_ok=True)
                exp_id = names[variant] + '_' + dataset_name
                result = RESULTS / exp_id / dataset_name
                weight = args.baseline if variant == 'baseline' else args.checkpoint
                test = [sys.executable, '-u', '-m', 'tracking.test', 'mvlm', names[variant],
                        '--exp_id', exp_id, '--dataset_name', dataset_name, '--weight_path', weight,
                        '--mode', 'mp', '--threads', str(args.workers), '--num_gpus', str(args.gpus)]
                analysis = [sys.executable, '-u', '-m', 'tracking.analysis_results', '--tracker_name', 'mvlm',
                            '--tracker_param', names[variant], '--exp_id', exp_id, '--dataset', dataset_name]
                write_json(artifact / 'commands.json', dict(test=test, analysis=analysis, result_dir=str(result)))
                status.update(status='testing', dataset=dataset_name, variant=variant, result_dir=str(result),
                              expected_sequences=size, expected_frames=sum(len(s.frames) for s in dataset), updated_at=now())
                write_json(state_path, status)
                print('TEST ' + dataset_name + ' ' + variant, flush=True)
                if not (artifact / 'summary.json').exists():
                    with (artifact / 'test.log').open('a') as log:
                        process = subprocess.Popen(test, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                        last_report = 0.
                        while process.poll() is None:
                            tick = time.monotonic()
                            if tick - last_report >= 15:
                                count = sum((result / (s.name + '_time.txt')).is_file() for s in dataset)
                                status.update(completed_sequences=count, child_pid=process.pid, updated_at=now())
                                write_json(state_path, status)
                                if tracking:
                                    tracking.log({'progress/job': len(summaries) + 1, 'progress/completed_sequences': count,
                                                  'progress/expected_sequences': size})
                                last_report = tick
                            time.sleep(2)
                        if process.returncode:
                            raise RuntimeError('Test process failed: ' + str(process.returncode))
                    validation = validate_predictions(dataset, result)
                    write_json(artifact / 'prediction_validation.json', validation)
                    if validation['errors']:
                        raise RuntimeError('Missing/invalid predictions: ' + repr(validation['errors'][:3]))
                    with (artifact / 'metrics.txt').open('w') as log:
                        subprocess.run(analysis, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    cache = Path('/root/MVLM/output/test/result_plots') / dataset_name / 'eval_data.pkl'
                    shutil.copyfile(str(cache), str(artifact / 'eval_data.pkl'))
                    metrics = read_metrics(artifact / 'eval_data.pkl', size)
                    summary = dict(dataset=dataset_name, variant=variant, sequences=size, frames=validation['frames'],
                                   metrics=metrics, result_dir=str(result), checkpoint_sha256=metadata[
                                       'baseline_sha256' if variant == 'baseline' else 'checkpoint_sha256'], completed_at=now())
                    write_json(artifact / 'summary.json', summary)
                summary = json.loads((artifact / 'summary.json').read_text())
                summaries.append(summary)
                status.update(completed_jobs=len(summaries), completed_sequences=size, updated_at=now())
                write_json(state_path, status)
                print('RESULT ' + json.dumps(summary, ensure_ascii=False), flush=True)
                if tracking:
                    tracking.log({'metrics/' + dataset_name + '/' + variant + '/' + k: v for k, v in summary['metrics'].items()})
            base, world = summaries[-2:]
            comparison = dict(dataset=dataset_name, baseline=base['metrics'], world=world['metrics'],
                              delta={k: world['metrics'][k] - base['metrics'][k] for k in ('AUC', 'PRE', 'NPR')})
            write_json(output / dataset_name / 'comparison.json', comparison)
            if tracking:
                tracking.log({'comparison/' + dataset_name + '/' + k + '_delta': v for k, v in comparison['delta'].items()})
        comparisons = [json.loads((output / name / 'comparison.json').read_text()) for name in DATASETS]
        write_json(output / 'summary.json', dict(status='completed', completed_at=now(), comparisons=comparisons, runs=summaries))
        lines = ['# 世界模型跨数据集测试', '', '| 数据集 | 原始 AUC | 世界模型 AUC | AUC 差值 | 原始 PRE | 世界模型 PRE | 原始 NPR | 世界模型 NPR |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in comparisons:
            b, w, delta = row['baseline'], row['world'], row['delta']
            lines.append('| {} | {:.2f} | {:.2f} | {:+.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(
                row['dataset'], b['AUC'], w['AUC'], delta['AUC'], b['PRE'], w['PRE'], b['NPR'], w['NPR']))
        lines.extend(['', '固定采用 VastTrack 验证选出的 B 第 1000 步，不在测试集调参。AUC 差值单位为百分点。',
                      '标准加载器：OTB 48、LaSOT 280、TNL2K 700；TNL2K 已知空语言序列在两组中均保留。',
                      '采用仓库原指标与整型预测保存规则；含首帧评分/可见性处理等既有协议。',
                      '比较的是完整新增系统与官方模型，包含 CENTER 微调和世界模型坐标路径变化，不是单模块因果消融。'])
        (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
        status.update(status='completed', completed_at=now(), completed_jobs=6)
        write_json(state_path, status)
        if tracking:
            tracking.summary['status'] = 'completed'
            tracking.finish()
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc(), updated_at=now())
        write_json(state_path, status)
        if tracking:
            tracking.summary['status'] = 'failed'
            tracking.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
