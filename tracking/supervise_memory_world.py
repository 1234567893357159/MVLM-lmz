"""Run an explicitly configured GPU experiment and keep local status/resources."""
import argparse
import datetime
import json
import os
import resource
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-file', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.launch_file).read_text())
    gpus = spec['physical_gpus']
    if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ValueError('Assign distinct, nonnegative physical GPU indices explicitly')
    if spec['env'].get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, gpus)):
        raise ValueError('Physical GPU assignment must match CUDA_VISIBLE_DEVICES')
    control, output = Path(spec['control']), Path(spec['output'])
    control.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(spec['env'])
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 32768 if hard == resource.RLIM_INFINITY else min(32768, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    status = dict(status='starting', started_at=now(), supervisor_pid=os.getpid(),
                  physical_gpus=spec['physical_gpus'], command=spec['command'], output=str(output),
                  file_descriptor_soft_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    write_json(control / 'supervisor_status.json', status)
    with (control / 'train.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status.update(status='running', trainer_pid=process.pid)

        def terminate(signum, frame):
            status.update(status='stopping', signal=signum, updated_at=now())
            write_json(control / 'supervisor_status.json', status)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        while process.poll() is None:
            status['updated_at'] = now()
            training = output / 'training_status.json'
            if training.exists():
                status['training'] = json.loads(training.read_text())
            try:
                query = subprocess.run(['nvidia-smi', '-i', ','.join(map(str, spec['physical_gpus'])),
                    '--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
                if query.returncode == 0:
                    resources = [dict(zip(('index', 'memory_used_mib', 'memory_free_mib',
                        'gpu_utilization_percent', 'power_watts'),
                        [float(value.strip()) for value in row.split(',')]))
                        for row in query.stdout.strip().splitlines()]
                    status['gpu_resources'] = resources
                    with (control / 'gpu_resources.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(at=now(), gpus=resources)) + '\n')
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                # Resource sampling must never stop the authorized training job.
                status['resource_monitor_error'] = str(error)
            write_json(control / 'supervisor_status.json', status)
            time.sleep(10)
        status.update(status='completed' if process.returncode == 0 else 'failed',
                      exit_code=process.returncode, finished_at=now(), updated_at=now())
        training = output / 'training_status.json'
        if training.exists():
            status['training'] = json.loads(training.read_text())
        write_json(control / 'supervisor_status.json', status)
        return process.returncode


if __name__ == '__main__':
    sys.exit(main())
