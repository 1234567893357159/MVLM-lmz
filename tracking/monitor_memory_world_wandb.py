"""Backfill and stream an existing memory/world run to W&B without restarting training."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_lines(path, offset):
    """Yield only complete records; retain a partial tail for the next poll."""
    if not path.exists():
        return
    with path.open('rb') as handle:
        handle.seek(offset)
        while True:
            raw = handle.readline()
            if not raw or not raw.endswith(b'\n'):
                break
            yield json.loads(raw), handle.tell()


def numbers(row, prefix):
    return {prefix + key: value for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}


def public_config(config, provenance):
    """Only publish model/training hyperparameters, not source snapshots or paths."""
    train = config['TRAIN']
    return dict(model='MVLM-memory-world', stage=train['WORLD']['STAGE'],
                parent_checkpoint_step=provenance.get('parent_checkpoint_step'),
                batch_execution=provenance.get('batch_execution', 'standard-actor'),
                trainable_parameters=provenance.get('trainable_parameters'),
                training_scope=provenance.get('training_scope'),
                precision=provenance.get('precision', 'fp32'),
                optimizer=train['OPTIMIZER'], learning_rate=train['LR'],
                encoder_learning_rate=train['LR'] * train.get('ENCODER_MULTIPLIER', .1),
                head_learning_rate=train['WORLD'].get('HEAD_LR'),
                cm_loss=train.get('CM_LOSS', False), cm_weight=train.get('CM_WEIGHT', 0.),
                freeze_encoder=train.get('FREEZE_ENCODER', True),
                weight_decay=train['WEIGHT_DECAY'], gradient_clip_norm=train['GRAD_CLIP_NORM'],
                per_gpu_batch_size=provenance['per_gpu_batch'],
                global_batch_size=provenance['global_batch_size'], world_size=provenance['world_size'],
                dataset_sampling_ratios=dict(zip(config['DATA']['TRAIN']['DATASETS_NAME'],
                                                 config['DATA']['TRAIN']['DATASETS_RATIO'])),
                temporal_training=dict(train['WORLD']), memory=dict(config['MODEL']['MEMORY']),
                world_model=dict(config['MODEL']['WORLD']), fusion=dict(config['MODEL']['FUSION']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--project', default='mvlm-reproduction')
    parser.add_argument('--entity', default='2896287267-hit')
    parser.add_argument('--name', default='memory-world-A-b512-2gpu-20261109')
    parser.add_argument('--retry-seconds', type=int, default=30)
    args = parser.parse_args()
    import wandb
    import yaml

    output, control = Path(args.output).resolve(), Path(args.control).resolve()
    metadata_path = output / 'wandb_run.json'
    state_path = output / 'wandb_bridge_status.json'
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if previous and (previous['project'] != args.project or previous['entity'] != args.entity):
        raise ValueError('Cannot change the W&B destination of an existing experiment')
    if previous and previous.get('logging_mode') != 'external_jsonl_bridge':
        raise ValueError('This experiment already has a different W&B logger')
    metadata = dict(id=previous.get('id') or wandb.util.generate_id(), project=args.project,
                    entity=args.entity, name=args.name, logging_mode='external_jsonl_bridge',
                    url=previous.get('url'), output=str(output))
    save(metadata_path, metadata)
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    state = dict(status='connecting', pid=os.getpid(), run_id=metadata['id'], updated_at=now(),
                 train_offset=old.get('train_offset', 0), gpu_offset=old.get('gpu_offset', 0),
                 validation_steps=old.get('validation_steps', []),
                 last_logged_train_step=old.get('last_logged_train_step', 0))
    save(state_path, state)
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    folder = output / 'wandb_bridge'
    folder.mkdir(exist_ok=True)
    config = yaml.safe_load((output / 'config.yaml').read_text())
    provenance = json.loads((output / 'run.json').read_text())
    run = None
    while run is None and not stopping[0]:
        try:
            run = wandb.init(project=args.project, entity=args.entity, name=args.name,
                id=metadata['id'], resume='allow', mode='online', dir=str(folder),
                group='memory-world', job_type='train-monitor',
                tags=['memory-world', 'stage-' + config['TRAIN']['WORLD']['STAGE'],
                      '{}-gpu'.format(provenance['world_size']),
                      'batch{}'.format(provenance['global_batch_size'])],
                config=public_config(config, provenance),
                settings=wandb.Settings(init_timeout=40, x_disable_stats=True,
                                        x_disable_meta=True, disable_git=True, save_code=False))
        except Exception as error:
            state.update(status='waiting_for_network', error=str(error), updated_at=now())
            save(state_path, state)
            print('W&B unavailable; training continues. Retrying in {}s: {}'.format(
                  args.retry_seconds, type(error).__name__), flush=True)
            for _ in range(args.retry_seconds):
                if stopping[0]:
                    break
                time.sleep(1)
    if run is None:
        state.update(status='stopped', updated_at=now())
        save(state_path, state)
        return
    metadata['url'] = run.url
    save(metadata_path, metadata)
    for axis, wildcard in [('train/step', 'train/*'), ('validation/step', 'validation/*'),
                           ('gpu/elapsed_seconds', 'gpu/*')]:
        run.define_metric(axis)
        run.define_metric(wildcard, step_metric=axis)
    print('W&B URL: ' + run.url, flush=True)
    launch_status = json.loads((control / 'supervisor_status.json').read_text())
    start = datetime.datetime.fromisoformat(launch_status['started_at']).timestamp()
    final_status = 'running'
    monitor_error = None
    try:
        while not stopping[0]:
            for row, offset in read_lines(output / 'metrics.jsonl', state['train_offset']):
                payload = numbers(row, 'train/')
                payload['train/clips_per_second_global'] = row['global_batch_size'] / max(row['seconds_per_step'], 1e-8)
                payload['train/active_prediction_ratio'] = row['active'] / max(row['predictions'], 1)
                run.log(payload)
                state.update(train_offset=offset, last_logged_train_step=row['step'])
            for path in sorted(output.glob('validation_*.json')):
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    # The trainer may still be writing a new validation report.
                    continue
                if row['step'] not in state['validation_steps']:
                    payload = numbers(row, 'validation/')
                    for key in ('giou_loss', 'l1_loss', 'focal_loss'):
                        usable = [clip for clip in row.get('rows', []) if key in clip and clip.get('supervised_frames', 0)]
                        count = sum(clip['supervised_frames'] for clip in usable)
                        if count:
                            payload['validation/' + key] = sum(clip[key] * clip['supervised_frames']
                                                               for clip in usable) / count
                    run.log(payload)
                    state['validation_steps'].append(row['step'])
            for row, offset in read_lines(control / 'gpu_resources.jsonl', state['gpu_offset']):
                seconds = datetime.datetime.fromisoformat(row['at']).timestamp() - start
                payload = {'gpu/elapsed_seconds': seconds}
                for gpu in row['gpus']:
                    index = int(gpu['index'])
                    payload.update(numbers({k:v for k,v in gpu.items() if k != 'index'}, 'gpu/{}/'.format(index)))
                run.log(payload)
                state['gpu_offset'] = offset
            training = json.loads((output / 'training_status.json').read_text())
            supervision = json.loads((control / 'supervisor_status.json').read_text())
            final_status = training['status']
            if supervision['status'] in ('failed', 'completed'):
                final_status = supervision['status']
            run.summary.update(dict(training_status=final_status,
                completed_optimizer_steps=training['completed_optimizer_steps'],
                last_synced_train_step=state['last_logged_train_step'],
                monitor_updated_at=now(), logging_mode='external_jsonl_bridge'))
            state.update(status='streaming', url=run.url, updated_at=now(), training_status=final_status)
            state.pop('error', None)
            save(state_path, state)
            if final_status in ('completed', 'failed'):
                break
            for _ in range(5):
                if stopping[0]:
                    break
                time.sleep(1)
    except BaseException as error:
        monitor_error = str(error)
        run.summary['monitor_error'] = type(error).__name__
        raise
    finally:
        monitor_status = 'failed' if monitor_error else ('stopped' if stopping[0] else final_status)
        run.summary['monitor_status'] = monitor_status
        run.finish(exit_code=1 if monitor_error or final_status == 'failed' else 0)
        state.update(status=monitor_status, updated_at=now())
        if monitor_error:
            state['error'] = monitor_error
        save(state_path, state)


if __name__ == '__main__':
    main()
