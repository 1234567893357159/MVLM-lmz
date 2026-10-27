"""Launch one detached VastTrack run and persist its process/exit status."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='mvlm_TF_vasttrack')
    args = parser.parse_args()
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = out / 'training_status.json'
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if old.get('status') == 'completed':
            raise RuntimeError('This output already completed; use a new output directory')
    state = {'status': 'starting', 'supervisor_pid': os.getpid(), 'source': str(source),
             'output': str(out), 'config': args.config, 'world_size': 8,
             'batch_size_per_rank': 10, 'global_batch_size': 80, 'target_epochs': 60,
             'started_at': datetime.now(timezone.utc).isoformat()}

    def write_status(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(status_path)

    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               WANDB_MODE='online', WANDB_PROJECT='mvlm-reproduction', WANDB_CONSOLE='off',
               WANDB_INIT_TIMEOUT='180', WANDB_HTTP_TIMEOUT='60')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               'lib/train/run_training.py', '--script', 'mvlm', '--config', args.config,
               '--save_dir', str(out)]
    write_status(command=command)
    with (out / 'train.log').open('a', buffering=1) as log:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        write_status(status='initializing', launcher_pid=process.pid)

        def stop(signum, frame):
            write_status(status='stopping', signal=signum)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        position = 0
        carry = ''
        with (out / 'train.log').open() as reader:
            while True:
                reader.seek(position)
                new = reader.read()
                position = reader.tell()
                lines = (carry + new).split('\n')
                carry = lines.pop()
                changes = {}
                for line in lines:
                    match = re.search(r'\[train: (\d+), (\d+) / (\d+)\]', line)
                    if match:
                        epoch, batch, per_epoch = map(int, match.groups())
                        changes.update(status='training', epoch=epoch, batch=batch,
                                       batches_per_epoch=per_epoch,
                                       approximate_global_step=(epoch - 1) * per_epoch + batch)
                metadata = out / 'wandb_tracking/train/mvlm' / args.config / 'run.json'
                if metadata.exists():
                    changes['wandb'] = json.loads(metadata.read_text())
                code = process.poll()
                if code is not None:
                    stopped = state.get('status') == 'stopping'
                    write_status(**dict(changes, status='stopped' if stopped else ('completed' if code == 0 else 'failed'),
                                        exit_code=code))
                    return code
                write_status(**changes)
                time.sleep(10)


if __name__ == '__main__':
    sys.exit(main())
