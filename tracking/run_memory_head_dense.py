"""Run 4k joint fine-tuning with saved 200-step checkpoints and fixed-subset validation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE = Path('/root/MVLM/output/memory_three_custom_20261018_run01')
CONFIG = 'mvlm_TF_memory_head_dense_20261019'
STEPS = tuple(range(200, 4001, 200))
GPU_COUNT = 8


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def subset_manifest(out):
    source = REFERENCE / 'split/manifest.json'
    manifest = json.loads(source.read_text())
    selected = []
    for dataset, count in [('otb99_lang', 4), ('lasot', 8), ('tnl2k', 12)]:
        records = sorted((r for r in manifest['records'] if r['split'] == 'val' and r['dataset'] == dataset),
                         key=lambda r: (r['frames'], r['key']))
        selected.extend(records[min(len(records) - 1, int((i + .5) * len(records) / count))]
                        for i in range(count))
    assert len(selected) == 24 and len({r['key'] for r in selected}) == 24
    dest = out / 'validation/subset_manifest.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        assert [r['key'] for r in old['records']] == [r['key'] for r in selected]
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_json(dest, {'source_manifest': str(source),
                         'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'selection': 'fixed length-stratified quantiles: OTB 4, LaSOT 8, TNL2K 12',
                         'records': selected})
    return dest, selected


def available_gpus(env):
    code = ('import torch\n'
            'assert torch.cuda.is_available() and torch.cuda.device_count() == 8\n'
            'for i in range(8): torch.empty(1, device="cuda:%d" % i)\n')
    try:
        result = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT), env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_evaluator(env, out, weights, manifest, destination, scope):
    destination.mkdir(parents=True, exist_ok=True)
    metrics_file = destination / 'metrics.json'
    if not metrics_file.exists():
        command = [sys.executable, '-u', str(ROOT / 'tracking/evaluate_memory_custom.py'),
                   '--manifest', str(manifest), '--weights', str(weights), '--out', str(destination),
                   '--config', CONFIG, '--gpus', str(GPU_COUNT), '--workers', '16', '--scope', scope]
        with (destination / 'evaluate.log').open('a') as log:
            subprocess.run(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(metrics_file.read_text())
    expected = 24 if scope == 'fixed_subset' else 159
    assert result['sequences'] == expected and result['evaluation_scope'] == scope
    return result


def run(out):
    import torch
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_file = out / 'training_status.json'
        if state_file.exists():
            raise RuntimeError('Existing run status; refusing to start a second training run')
        state = {'status': 'waiting_gpus', 'planned_optimizer_steps': 4000,
                 'checkpoint_interval_steps': 200, 'supervisor_pid': os.getpid(),
                 'created_at': utc_now(), 'config': CONFIG,
                 'world_size': GPU_COUNT, 'per_gpu_batch': 2, 'effective_batch': 16,
                 'evaluation_plan': '24 fixed validation sequences per 200-step checkpoint; full 159 on top 3 subset checkpoints and step 2000'}
        metadata_path = out / 'wandb_tracking/train/mvlm' / CONFIG / 'run.json'
        if metadata_path.exists():
            state['monitor_url'] = json.loads(metadata_path.read_text()).get('url')
        def save():
            save_json(state_file, state)
        save()
        wb = None
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', MVLM_DISTRIBUTED_TIMEOUT_SECONDS='10800')
        try:
            subset, records = subset_manifest(out)
            # The host currently reports a failed GPU. Queue the authorized run and
            # start only when all eight original devices are usable again.
            deadline = time.monotonic() + 6 * 3600
            while not available_gpus(env):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Eight CUDA devices did not recover within six hours')
                state['last_gpu_check_at'] = utc_now(); save(); time.sleep(60)
            state.update(status='starting', gpu_ready_at=utc_now()); save()
            command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node=8', '--module', 'lib.train.run_training',
                       '--script', 'mvlm', '--config', CONFIG, '--save_dir', str(out), '--seed', '20261018']
            with (out / 'train.log').open('a') as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                state.update(status='running', training_pid=process.pid, command=command,
                             training_started_at=utc_now()); save()
                code = process.wait()
            if code:
                raise RuntimeError('Training subprocess exit ' + str(code))
            checkpoints = out / 'checkpoints/train/mvlm' / CONFIG
            final = checkpoints / 'MVLM_step004000.pth.tar'
            checkpoint = torch.load(final, map_location='cpu')
            assert checkpoint['global_step'] == 4000
            baseline_weights = torch.load('/root/MVLM/models/MVLM_TF.pth.tar', map_location='cpu')['net']
            unchanged = [k for k, v in baseline_weights.items() if not k.startswith('head.')
                         and not torch.equal(v, checkpoint['net'][k])]
            assert not unchanged, unchanged[:5]
            changed_head = [k for k, v in baseline_weights.items() if k.startswith('head.')
                            and not torch.equal(v, checkpoint['net'][k])]
            assert changed_head
            assert checkpoint['net']['memory_fusion.output_proj.weight'].norm() > 0
            assert all(bool(torch.isfinite(v).all()) for v in checkpoint['net'].values() if v.is_floating_point())
            assert all((checkpoints / ('MVLM_step%06d.pth.tar' % s)).exists() for s in STEPS)
            state.update(status='validating_subset', completed_optimizer_steps=4000,
                         training_finished_at=utc_now(), final_checkpoint=str(final),
                         changed_head_tensors=len(changed_head), frozen_encoder_text_changed=0)
            save()
            del checkpoint, baseline_weights

            import wandb
            metadata = out / 'wandb_tracking/train/mvlm' / CONFIG / 'run.json'
            saved = json.loads(metadata.read_text())
            state['monitor_url'] = saved['url']; save()
            wb = wandb.init(project=saved['project'], entity=saved['entity'], id=saved['id'],
                            resume='must', dir=str(metadata.parent))
            wb.summary['status'] = 'validating_subset'
            wb.define_metric('val_subset/step')
            wb.define_metric('val_subset/*', step_metric='val_subset/step')
            wb.define_metric('val_full/step')
            wb.define_metric('val_full/*', step_metric='val_full/step')

            baseline = json.loads((REFERENCE / 'validation/baseline/metrics.json').read_text())
            subset_keys = {r['key'] for r in records}
            baseline_by_dataset = {}
            for dataset, rows in baseline['per_sequence'].items():
                scores = [r['AUC'] for r in rows if dataset + '/' + r['name'] in subset_keys]
                baseline_by_dataset[dataset] = float(np.mean(scores))
            baseline_mean = float(np.mean(list(baseline_by_dataset.values())))
            save_json(out / 'validation/subset_baseline_reference.json',
                      {'source': str(REFERENCE / 'validation/baseline/metrics.json'),
                       'AUC': baseline_by_dataset, 'mean_AUC': baseline_mean})
            subset_results = []
            for step in STEPS:
                state.update(validating_step=step); save()
                weights = checkpoints / ('MVLM_step%06d.pth.tar' % step)
                result = run_evaluator(env, out, weights, subset, out / 'validation/subset' / ('step_%06d' % step), 'fixed_subset')
                entry = {'global_step': step, 'mean_AUC': result['mean_AUC'],
                         'metrics': result['metrics'], 'checkpoint': str(weights)}
                subset_results.append(entry)
                save_json(out / 'validation/subset_summary.json',
                          {'baseline_mean_AUC': baseline_mean, 'checkpoints': subset_results,
                           'scope': 'fixed 24-sequence subset; exploratory checkpoint screening only'})
                payload = {'global_step': 4000, 'val_subset/step': step,
                           'val_subset/mean_AUC': result['mean_AUC'],
                           'val_subset/baseline_mean_AUC': baseline_mean}
                for dataset, metrics in result['metrics'].items():
                    payload['val_subset/' + dataset + '/AUC'] = metrics['AUC']
                wb.log(payload)

            candidates = {2000}
            candidates.update(x['global_step'] for x in sorted(subset_results,
                              key=lambda x: x['mean_AUC'], reverse=True)[:3])
            state.update(status='validating_full', full_validation_steps=sorted(candidates)); save()
            wb.summary['status'] = 'validating_full'
            full_results = []
            for step in sorted(candidates):
                state.update(validating_step=step); save()
                weights = checkpoints / ('MVLM_step%06d.pth.tar' % step)
                result = run_evaluator(env, out, weights, REFERENCE / 'split/manifest.json',
                                       out / 'validation/full' / ('step_%06d' % step), 'full')
                entry = {'global_step': step, 'mean_AUC': result['mean_AUC'],
                         'metrics': result['metrics'], 'checkpoint': str(weights)}
                full_results.append(entry)
                best = max(full_results, key=lambda x: x['mean_AUC'])
                save_json(out / 'validation/full_summary.json',
                          {'baseline_mean_AUC': baseline['mean_AUC'], 'checkpoints': full_results,
                           'best': best, 'selected_using': 'fixed subset top 3 plus step 2000'})
                save_json(out / 'validation/best.json', best)
                if best is entry:
                    shutil.copy2(weights, out / 'validation/best.pth.tar')
                payload = {'global_step': 4000, 'val_full/step': step,
                           'val_full/mean_AUC': result['mean_AUC'],
                           'val_full/baseline_mean_AUC': baseline['mean_AUC']}
                for dataset, metrics in result['metrics'].items():
                    payload['val_full/' + dataset + '/AUC'] = metrics['AUC']
                wb.log(payload)
                wb.summary['best_val_full_mean_AUC'] = best['mean_AUC']
                wb.summary['best_val_full_step'] = best['global_step']
            state.update(status='completed', best=best, finished_at=utc_now()); save()
            wb.summary['status'] = 'completed'; wb.finish(); wb = None
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), finished_at=utc_now()); save()
            if wb:
                wb.summary['status'] = 'failed'; wb.finish(exit_code=1)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    run(parser.parse_args().run_dir.resolve())
