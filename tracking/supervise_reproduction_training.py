"""Supervise one frozen-source MVLM training run on explicitly selected GPUs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--devices', required=True)
    args = parser.parse_args()
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    devices = args.devices.split(',')
    if not devices or any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices):
        raise ValueError('Supply distinct numeric physical GPU indices')
    config = yaml.safe_load((source / 'experiments/mvlm' / (args.config + '.yaml')).read_text())
    batch, epochs = config['TRAIN']['BATCH_SIZE'], config['TRAIN']['EPOCH']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = out / 'training_status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'completed':
        raise RuntimeError('Completed output; use a new directory for independent training')
    state = dict(status='starting', supervisor_pid=os.getpid(), source=str(source),
                 output=str(out), config=args.config, physical_gpus=devices,
                 world_size=len(devices), batch_size_per_rank=batch,
                 global_batch_size=batch * len(devices), target_epochs=epochs,
                 started_at=datetime.now(timezone.utc).isoformat())

    def write_status(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(status_path)

    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=args.devices,
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', WANDB_MODE='online', WANDB_CONSOLE='off',
               WANDB_INIT_TIMEOUT='180', WANDB_HTTP_TIMEOUT='60')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=' + str(len(devices)), '-m', 'lib.train.run_training',
               '--script', 'mvlm', '--config', args.config, '--save_dir', str(out)]
    write_status(command=command)
    with (out / 'train.log').open('a', buffering=1) as log:
        try:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as exc:
            write_status(status='failed', error=repr(exc))
            raise
        write_status(status='initializing', launcher_pid=process.pid)

        def stop(signum, frame):
            write_status(status='stopping', signal=signum)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        carry = ''
        with (out / 'train.log').open() as reader:
            while True:
                lines = (carry + reader.read()).split('\n')
                carry = lines.pop()
                changes = {}
                for line in lines:
                    match = re.search(r'\[train: (\d+), (\d+) / (\d+)\]', line)
                    if match:
                        epoch, batch_index, per_epoch = map(int, match.groups())
                        changes.update(status='training', epoch=epoch, batch=batch_index,
                                       batches_per_epoch=per_epoch,
                                       approximate_global_step=(epoch - 1) * per_epoch + batch_index)
                metadata = out / 'wandb_tracking/train/mvlm' / args.config / 'run.json'
                if metadata.exists():
                    changes['wandb'] = json.loads(metadata.read_text())
                code = process.poll()
                if code is not None:
                    stopped = state.get('status') == 'stopping'
                    final = 'stopped' if stopped else ('completed' if code == 0 else 'failed')
                    write_status(**dict(changes, status=final, exit_code=code))
                    return code
                write_status(**changes)
                time.sleep(5)


if __name__ == '__main__':
    sys.exit(main())
