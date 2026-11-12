"""Stop one identified training run after its requested checkpoint is committed.

This controller never starts another training stage or changes training sources.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    folder = Path('/proc') / str(pid)
    try:
        stat = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=int(pid), parent=int(stat[1]), start=stat[19],
                    command=(folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(record):
    current = identity(record['pid'])
    return current is not None and current['start'] == record['start']


def descendants(parent):
    records = [record for folder in Path('/proc').iterdir() if folder.name.isdigit()
               for record in [identity(int(folder.name))] if record is not None]
    selected = {parent}
    while True:
        enlarged = selected | {r['pid'] for r in records if r['parent'] in selected}
        if enlarged == selected:
            return [r for r in records if r['pid'] in selected]
        selected = enlarged


def committed(output, log, step):
    required = [output / ('step_%06d.pth.tar' % step), output / 'last.pth.tar',
                output / ('validation_%06d.json' % step)]
    if not all(path.exists() and path.stat().st_size for path in required):
        return False
    # The trainer prints this record only AFTER step, last and best saves finish.
    with log.open('rb') as handle:
        handle.seek(max(0, log.stat().st_size - 65536))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(row, dict) and row.get('step') == step and row.get('split') == 'fixed_validation_clips':
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    output, control = Path(args.output).resolve(), Path(args.control).resolve()
    lock = (control / 'stop_watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = control / 'stop_watcher_status.json'
    state = dict(status='waiting_for_checkpoint', pid=os.getpid(), target_step=args.step,
                 output=str(output), created_at=now(), updated_at=now(), starts_next_stage=False)
    save(state_path, state)
    try:
        supervision = json.loads((control / 'supervisor_status.json').read_text())
        supervisor = identity(supervision['supervisor_pid'])
        trainer = identity(supervision['trainer_pid'])
        if supervisor is None or trainer is None:
            raise RuntimeError('Expected live supervisor and trainer')
        if ('supervise_memory_world.py' not in supervisor['command'] or str(control / 'launch.json') not in supervisor['command']
                or 'torch.distributed.run' not in trainer['command'] or str(output) not in trainer['command']):
            raise RuntimeError('Process identities do not match the requested experiment')
        while not committed(output, control / 'train.log', args.step):
            training = json.loads((output / 'training_status.json').read_text())
            if not same_process(trainer):
                raise RuntimeError('Training exited before the requested save completed')
            state.update(observed_step=training['completed_optimizer_steps'], updated_at=now())
            save(state_path, state)
            time.sleep(.25)
        # Verify deserialization and step metadata before terminating any process.
        import torch
        torch.set_num_threads(1)
        target = output / ('step_%06d.pth.tar' % args.step)
        checkpoint = torch.load(str(target), map_location='cpu')
        if checkpoint.get('step') != args.step or checkpoint.get('world_stage') != 'A':
            raise RuntimeError('Saved checkpoint is not the requested stage-A boundary')
        del checkpoint
        if not same_process(supervisor) or not same_process(trainer):
            raise RuntimeError('Training process identities changed before stop')
        owned = descendants(trainer['pid'])
        state.update(status='stopping', checkpoint_verified=True, stop_requested_at=now(), updated_at=now())
        save(state_path, state)
        os.kill(supervisor['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and (same_process(supervisor) or any(same_process(r) for r in owned)):
            time.sleep(.25)
        forced = []
        for record in owned + [supervisor]:
            if same_process(record):
                os.kill(record['pid'], signal.SIGKILL)
                forced.append(record['pid'])
        if forced:
            time.sleep(1)
        if any(same_process(record) for record in owned + [supervisor]):
            raise RuntimeError('An identified training process did not exit')
        original_training = json.loads((output / 'training_status.json').read_text())
        original_supervision = json.loads((control / 'supervisor_status.json').read_text())
        validations = []
        for path in output.glob('validation_*.json'):
            row = json.loads(path.read_text())
            if row['step'] <= args.step and row.get('loss') is not None:
                validations.append(row)
        best = min(validations, key=lambda row: row['loss'])
        source = output / ('step_%06d.pth.tar' % best['step'])
        selected = output / ('selected_for_stage_b_at_%06d.pth.tar' % args.step)
        temporary = selected.with_suffix(selected.suffix + '.tmp')
        shutil.copyfile(str(source), str(temporary))
        temporary.replace(selected)
        digest = hashlib.sha256()
        with selected.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        record = dict(reason='user_requested_early_stop', requested_step=args.step,
                      actual_completed_steps=original_training['completed_optimizer_steps'],
                      original_planned_steps=4000, stopped_at=now(), forced_pids=forced,
                      selected_checkpoint=str(selected), selected_checkpoint_step=best['step'],
                      selected_validation_loss=best['loss'], selected_sha256=digest.hexdigest(),
                      original_training_status=original_training,
                      original_supervisor_exit_status=original_supervision,
                      final_distributed_audit='not_run_due_to_external_user_stop',
                      next_stage_started=False)
        save(control / 'early_stop.json', record)
        original_training.update(status='early_stopped', stop_reason=record['reason'],
                                 requested_stop_step=args.step, updated_at=now())
        save(output / 'training_status.json', original_training)
        original_supervision.update(status='early_stopped', stop_reason=record['reason'],
                                    requested_stop_step=args.step, updated_at=now())
        save(control / 'supervisor_status.json', original_supervision)
        # Let the logger read the final training state, then request a clean finish.
        time.sleep(6)
        bridge_path = output / 'wandb_bridge_status.json'
        if bridge_path.exists():
            bridge = identity(json.loads(bridge_path.read_text()).get('pid', 0))
            if bridge and 'monitor_memory_world_wandb.py' in bridge['command'] and str(output) in bridge['command']:
                os.kill(bridge['pid'], signal.SIGTERM)
        state.update(status='stopped', actual_completed_steps=record['actual_completed_steps'],
                     selected_checkpoint=str(selected), selected_checkpoint_step=best['step'],
                     selected_validation_loss=best['loss'], updated_at=now())
        save(state_path, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error), updated_at=now())
        save(state_path, state)
        raise


if __name__ == '__main__':
    main()
