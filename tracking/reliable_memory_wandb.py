"""Optional W&B sidecar: local events first, no SDK/network in the trainer."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def monitoring_metrics(row, results, seconds, batch):
    """row has globally reduced losses/counts; per-frame diagnostics are rank zero."""
    payload = {'train/' + key: row[key] for key in
               ('loss', 'supervised', 'regression', 'outside', 'partial', 'background',
                'committed', 'pending', 'rejected', 'attempt')}
    payload.update({'train/lr': row['lr_used'], 'train/lr_next': row['lr'],
                    'train/gradient_norm': row['gradient_norm'],
                    'runtime/rank0/seconds_per_update': seconds,
                    'runtime/rank0/clips_per_second': batch/max(seconds, 1e-9)})
    writes = sum(row[key] for key in ('committed', 'pending', 'rejected'))
    payload['memory/commit_rate'] = row['committed']/max(1, writes)
    payload['memory/reject_rate'] = row['rejected']/max(1, writes)
    diagnostics = [result['diagnostics'] for result in results]
    for key in ('score', 'short_size', 'long_size', 'used_global_search'):
        values = [float(item[key]) for item in diagnostics if item.get(key) is not None]
        if values:
            payload['memory/rank0/' + key + '_mean'] = sum(values)/len(values)
    for key in ('raw_ratio', 'clipped_fraction', 'readout_ratio', 'gate', 'memory_fraction'):
        values = [item['fusion'][key] for item in diagnostics if key in item.get('fusion', {})]
        if values:
            payload['fusion/rank0/' + key + '_mean'] = sum(values)/len(values)
    reasons = Counter(reason for item in diagnostics for reason in item['write']['reasons'])
    for reason, count in reasons.items():
        payload['memory/rank0/reasons/' + reason] = count
    return payload


class WandbMonitor:
    """Rank-zero only. An SDK failure must never interrupt a training update."""
    def __init__(self, output, options, config, resume=False, rank=0):
        self.process = self.handle = self.log_handle = None
        self.folder = Path(output)/'wandb_monitor'
        self.last_step = None
        self.closed = False
        self.interval = int(options.get('LOG_INTERVAL', 20))
        if rank != 0 or not options.get('ENABLED') or options.get('MODE') == 'disabled':
            return
        try:
            self.folder.mkdir(exist_ok=True)
            identity_path = self.folder/'run.json'
            destination = {key.lower(): options.get(key) or None for key in ('PROJECT', 'ENTITY', 'MODE')}
            if resume and identity_path.exists():
                identity = json.loads(identity_path.read_text())
                if any(identity.get(key) != value for key, value in destination.items()):
                    raise ValueError('W&B destination changed on resume')
            else:
                identity = dict(destination, id=uuid.uuid4().hex[:8])
                write_json(identity_path, identity)
            session = self.folder/('session-' + uuid.uuid4().hex)
            session.mkdir()
            write_json(session/'request.json', dict(identity=identity, options=options, config=config,
                                                     name=options.get('NAME') or Path(output).name))
            self.session = session
            self.handle = (session/'events.jsonl').open('a', buffering=1)
            self.log_handle = (session/'worker.log').open('a')
            self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                             '--worker', str(session)], stdout=self.log_handle,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            print('W&B monitor: {} ({}); local events: {}'.format(
                identity['id'], identity['mode'], session), flush=True)
        except Exception as error:
            print('W&B monitor unavailable; training continues locally: {}'.format(error), flush=True)
            self.close('unavailable')

    def _append(self, event):
        if self.handle is not None and not self.closed:
            try:
                self.handle.write(json.dumps(event, allow_nan=False) + '\n')
            except Exception as error:
                print('W&B event write failed; training continues: {}'.format(error), flush=True)
                self.handle.close()
                self.handle = None

    def log(self, payload, step, force=False):
        if self.handle is None or (not force and step != 1 and step % self.interval):
            return
        if self.last_step == step:
            return
        # Reject nonfinite values from monitoring, without affecting optimization.
        clean = {key: float(value) for key, value in payload.items()
                 if isinstance(value, (int, float)) and math.isfinite(value)}
        self._append(dict(kind='metrics', step=step, metrics=clean))
        self.last_step = step

    def close(self, status, timeout=5):
        if self.closed:
            return
        self._append(dict(kind='finish', status=status))
        self.closed = True
        if self.handle is not None:
            self.handle.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Includes the SDK service spawned by the worker. Local files survive.
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
                print('W&B worker stopped after timeout; local events retained in {}'.format(self.folder), flush=True)
            if self.process.returncode:
                print('W&B worker did not finish normally; inspect {}'.format(self.session/'worker.log'), flush=True)
        if self.log_handle is not None:
            self.log_handle.close()


def worker(session):
    """SDK imports and network calls stay in this independent process."""
    request = json.loads((session/'request.json').read_text())
    identity, options = request['identity'], request['options']
    run = None
    state = dict(status='starting', id=identity['id'])
    try:
        import wandb
        run = wandb.init(project=identity['project'], entity=identity['entity'],
                         mode=identity['mode'], id=identity['id'], name=request['name'],
                         resume='allow' if identity['mode'] == 'online' else None,
                         dir=str(session), config=request['config'],
                         settings=wandb.Settings(init_timeout=20, save_code=False, disable_git=True))
        run.define_metric('global_step')
        run.define_metric('*', step_metric='global_step')
        state.update(status='running', url=run.url if identity['mode'] == 'online' else None)
        write_json(session/'status.json', state)
        with (session/'events.jsonl').open() as handle:
            while True:
                line = handle.readline()
                if not line:
                    time.sleep(.2)
                    continue
                event = json.loads(line)
                if event['kind'] == 'finish':
                    run.summary['training_status'] = event['status']
                    run.finish(exit_code=1 if event['status'] == 'failed' else 0)
                    state.update(status='finished', training_status=event['status'])
                    write_json(session/'status.json', state)
                    return
                if event['step'] >= run.step:
                    run.log(dict(event['metrics'], global_step=event['step']), step=event['step'])
                state['last_step'] = event['step']
    except Exception as error:
        state.update(status='unavailable', error=repr(error))
        write_json(session/'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, type=Path)
    worker(parser.parse_args().worker)
