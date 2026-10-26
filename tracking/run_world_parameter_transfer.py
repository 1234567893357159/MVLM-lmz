"""Evaluate fixed world-model weights under previously tested inference parameters."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import _init_paths
import yaml
import torch
from lib.test.evaluation import get_dataset
from run_world_crossdataset import now, read_metrics, validate_predictions, write_json
from train_world_model import sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/root/MVLM/output/test/tracking_results/mvlm')
CHECKPOINT = Path('/root/MVLM/output/world_stage_b_vasttrack_8gpu_20261026_run01/best_clip_loss.pth.tar')
BASE_CONFIG = ROOT / 'experiments/mvlm/world_cross_world_20261026_run01.yaml'
COMMON = dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.1, LAMBDA=0.4, PSI_OUT=0.1)
CASES = {
    'otb_common': ('otb99_lang', 48, COMMON),
    'lasot_common': ('lasot', 280, COMMON),
    'tnl2k_common': ('tnl2k', 700, COMMON),
    'otb_previous_best': ('otb99_lang', 48, dict(COMMON, OMEGA=0.4, LAMBDA=0.1)),
    'tnl2k_previous_best': ('tnl2k', 700, dict(COMMON, PSI_OUT=0.12)),
    'otb_alpha03': ('otb99_lang', 48, dict(COMMON, ALPHA_CORR=0.3)),
    'otb_alpha07': ('otb99_lang', 48, dict(COMMON, ALPHA_CORR=0.7)),
    'otb_tau001': ('otb99_lang', 48, dict(COMMON, TAU=0.01)),
    'otb_tau005': ('otb99_lang', 48, dict(COMMON, TAU=0.05)),
    'otb_psi0': ('otb99_lang', 48, dict(COMMON, PSI_OUT=0.0)),
    'otb_alpha09': ('otb99_lang', 48, dict(COMMON, ALPHA_CORR=0.9)),
    'otb_alpha07_tau005': ('otb99_lang', 48, dict(COMMON, ALPHA_CORR=0.7, TAU=0.05)),
    'lasot_selected': ('lasot', 280, COMMON),
    'tnl2k_selected': ('tnl2k', 700, COMMON),
}
PAPER = {'otb99_lang': dict(AUC=60.7, PRE=84.3, NPR=75.4),
         'lasot': dict(AUC=60.7, PRE=65.5, NPR=68.3),
         'tnl2k': dict(AUC=57.8, PRE=60.9, NPR=74.1)}
BASELINE_TUNED = {'otb_common': dict(AUC=60.52, PRE=83.98, NPR=75.29),
                  'lasot_common': dict(AUC=61.06, PRE=65.93, NPR=68.72),
                  'tnl2k_common': dict(AUC=57.59, PRE=60.48, NPR=73.77),
                  'otb_previous_best': dict(AUC=60.56, PRE=84.05, NPR=75.33),
                  'tnl2k_previous_best': dict(AUC=57.62, PRE=60.52, NPR=73.81)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=['otb_common', 'lasot_common', 'tnl2k_common'])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--parameter-file', type=Path,
                        help='Completed candidate summary to transfer without further target-set tuning')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-entity', default='2896287267-hit')
    args = parser.parse_args()
    selected = None
    if args.parameter_file:
        selected = json.loads(args.parameter_file.read_text())['parameters']
        assert set(selected) == set(COMMON)
        assert all(0 <= float(v) <= 1 for v in selected.values())
        assert all(label.endswith('_selected') for label in args.cases)
    else:
        assert not any(label.endswith('_selected') for label in args.cases), '--parameter-file required'
    out = Path(args.output).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    status = dict(status='preparing', started_at=now(), pid=os.getpid(), completed_cases=0, total_cases=len(args.cases))
    write_json(status_path, status)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    assert checkpoint['world_stage'] == 'B' and checkpoint['step'] == 1000
    del checkpoint
    config = yaml.safe_load(BASE_CONFIG.read_text())
    assert config['MODEL']['WORLD']['ENABLED'] is True
    metadata = dict(cases=args.cases, checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT),
                    base_config=str(BASE_CONFIG), tag=args.tag, workers=args.workers, gpus=args.gpus,
                    paper=PAPER, old_test_tuned_baseline=BASELINE_TUNED,
                    note='Exploratory comparison using old test-set-tuned inference parameters; not an independent paper reproduction',
                    started_at=now())
    metadata['parameter_source'] = str(args.parameter_file.resolve()) if args.parameter_file else None
    metadata['parameter_source_sha256'] = sha256(args.parameter_file) if args.parameter_file else None
    if args.resume:
        previous = json.loads((out / 'run.json').read_text())
        for key in ('cases', 'checkpoint_sha256', 'tag', 'workers', 'gpus'):
            assert previous[key] == metadata[key], key
        assert previous.get('parameter_source_sha256') == metadata['parameter_source_sha256']
        metadata['resumed_from'] = previous['started_at']
    write_json(out / 'run.json', metadata)
    shutil.copyfile(__file__, out / ('script_resumed.py' if args.resume else 'script.py'))
    shutil.copyfile(BASE_CONFIG, out / 'base_config.yaml')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               OPENCV_FOR_THREADS_NUM='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    tracking = None
    if args.wandb_project:
        try:
            import wandb
            tracking = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                  name='world-parameter-transfer-' + args.tag, config=metadata,
                                  dir=str(out), mode='online', settings=wandb.Settings(init_timeout=90))
            write_json(out / 'wandb_run.json', dict(id=tracking.id, url=tracking.url))
            print('W&B URL:', tracking.url, flush=True)
        except Exception as error:
            status['wandb_error'] = repr(error)
            write_json(status_path, status)
            print('W&B unavailable:', repr(error), flush=True)
    summaries = []
    try:
        for label in args.cases:
            dataset_name, expected, params = CASES[label]
            if selected is not None:
                params = selected
            dataset = get_dataset(dataset_name)
            assert len(dataset) == expected
            case = out / label
            case.mkdir(exist_ok=True)
            value = copy.deepcopy(config)
            value['TEST']['MVLM'].update(params)
            name = 'world_transfer_' + args.tag + '_' + label
            config_path = ROOT / 'experiments/mvlm' / (name + '.yaml')
            text = yaml.safe_dump(value, sort_keys=False)
            if config_path.exists() and not args.resume:
                raise FileExistsError(config_path)
            if config_path.exists():
                assert config_path.read_text() == text, config_path
            else:
                config_path.write_text(text)
            shutil.copyfile(config_path, case / 'config.yaml')
            result = RESULTS / name / dataset_name
            if args.resume and (case / 'summary.json').exists():
                validation = validate_predictions(dataset, result)
                assert not validation['errors'], validation['errors'][:3]
                summaries.append(json.loads((case / 'summary.json').read_text()))
                status.update(completed_cases=len(summaries), updated_at=now())
                write_json(status_path, status)
                continue
            if args.resume and result.exists():
                # The original runner skips whenever a bbox file exists. Archive
                # incomplete pairs so a killed write cannot silently skip a video.
                validation = validate_predictions(dataset, result)
                for item in validation['errors']:
                    for suffix in ('.txt', '_time.txt'):
                        path = result / (item['sequence'] + suffix)
                        if path.exists():
                            backup = case / 'incomplete_before_resume'
                            backup.mkdir(exist_ok=True)
                            shutil.move(str(path), str(backup / path.name))
            test = [sys.executable, '-u', '-m', 'tracking.test', 'mvlm', name,
                    '--exp_id', name, '--dataset_name', dataset_name, '--weight_path', str(CHECKPOINT),
                    '--mode', 'mp', '--threads', str(args.workers), '--num_gpus', str(args.gpus)]
            analysis = [sys.executable, '-u', '-m', 'tracking.analysis_results', '--tracker_name', 'mvlm',
                        '--tracker_param', name, '--exp_id', name, '--dataset', dataset_name]
            write_json(case / 'commands.json', dict(test=test, analysis=analysis, result_dir=str(result)))
            status.update(status='testing', case=label, dataset=dataset_name, completed_sequences=0,
                          expected_sequences=expected, updated_at=now())
            write_json(status_path, status)
            print('TEST', label, flush=True)
            with (case / 'test.log').open('a') as log:
                process = subprocess.Popen(test, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                last_report = 0
                while process.poll() is None:
                    tick = time.monotonic()
                    if tick - last_report >= 15:
                        completed = sum((result / (s.name + '_time.txt')).is_file() for s in dataset)
                        status.update(completed_sequences=completed, child_pid=process.pid, updated_at=now())
                        write_json(status_path, status)
                        if tracking:
                            tracking.log({'progress/case': len(summaries) + 1,
                                          'progress/completed_sequences': completed,
                                          'progress/expected_sequences': expected})
                        last_report = tick
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError('Test process failed: ' + str(process.returncode))
            validation = validate_predictions(dataset, result)
            write_json(case / 'prediction_validation.json', validation)
            if validation['errors']:
                raise RuntimeError('Invalid predictions: ' + repr(validation['errors'][:3]))
            with (case / 'metrics.txt').open('w') as log:
                subprocess.run(analysis, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            cache = Path('/root/MVLM/output/test/result_plots') / dataset_name / 'eval_data.pkl'
            shutil.copyfile(cache, case / 'eval_data.pkl')
            metrics = read_metrics(case / 'eval_data.pkl', expected)
            summary = dict(case=label, dataset=dataset_name, parameters=params, sequences=expected,
                           frames=validation['frames'], metrics=metrics, paper=PAPER[dataset_name],
                           delta_vs_paper={k: metrics[k] - PAPER[dataset_name][k] for k in metrics},
                           old_test_tuned_baseline=BASELINE_TUNED.get(label),
                           delta_vs_old_tuned_baseline=({k: metrics[k] - BASELINE_TUNED[label][k] for k in metrics}
                                                       if label in BASELINE_TUNED else None),
                           completed_at=now())
            write_json(case / 'summary.json', summary)
            summaries.append(summary)
            write_json(out / 'summary.json', dict(status='running', cases=summaries))
            status.update(completed_cases=len(summaries), completed_sequences=expected, updated_at=now())
            write_json(status_path, status)
            print('RESULT', json.dumps(summary, ensure_ascii=False), flush=True)
            if tracking:
                tracking.log({'metrics/' + label + '/' + k: v for k, v in metrics.items()})
                tracking.log({'vs_paper/' + label + '/' + k: v for k, v in summary['delta_vs_paper'].items()})
        write_json(out / 'summary.json', dict(status='completed', completed_at=now(), cases=summaries))
        status.update(status='completed', completed_at=now())
        write_json(status_path, status)
        if tracking:
            tracking.summary['status'] = 'completed'
            tracking.finish()
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc(), updated_at=now())
        write_json(status_path, status)
        if tracking:
            tracking.summary['status'] = 'failed'
            tracking.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
