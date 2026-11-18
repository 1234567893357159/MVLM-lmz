"""Validate immutable candidate checkpoints without blocking the training GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reliable_memory_runtime import atomic_json, sha256, source_hashes
from reliable_memory_wandb import WandbMonitor


def choose_best(rows):
    if not rows:
        return None
    return min(rows, key=lambda row: (-row['mean_AUC'], row['step']))


def link_best(checkpoint, destination):
    temporary = destination.with_name(destination.name + '.tmp')
    if temporary.exists():
        temporary.unlink()
    os.link(checkpoint, temporary)
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--group', default='M6')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--total-updates', type=int, default=4000)
    parser.add_argument('--candidate-every', type=int, default=1000)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--launcher-pid-file', type=Path,
                        help='Optional external PID file for the training launcher')
    parser.add_argument('--wandb-project', default='mvlm-reliable-memory')
    parser.add_argument('--wandb-name', default='M6-validation')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()
    if min(args.total_updates, args.candidate_every, args.poll_seconds) <= 0:
        parser.error('Invalid watcher budget')
    run, manifest = args.run_dir.resolve(), args.validation_manifest.resolve()
    validation = run/'validation'
    validation.mkdir(exist_ok=True)
    chosen = json.loads(manifest.read_text())
    expected_count = len(chosen['val'])
    expected_sources = {r['dataset'] for r in chosen['val']}
    monitor = WandbMonitor(validation, dict(ENABLED=not args.no_wandb, MODE='online',
                           PROJECT=args.wandb_project, ENTITY=os.environ.get('WANDB_ENTITY', ''),
                           NAME=args.wandb_name, LOG_INTERVAL=1),
                           dict(group=args.group, protocol='strict_language',
                                validation_manifest_sha256=sha256(manifest),
                                train_output=str(run), description='Full video validation at saved steps'),
                           resume=(validation/'wandb_monitor'/'run.json').exists())
    rows, evaluated = [], set()
    state = dict(status='waiting', validation_manifest_sha256=sha256(manifest), evaluated=[])
    try:
        while True:
            metadata_path = run/'run.json'
            status_path = run/'training_status.json'
            training = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'starting'}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata['source_hashes'] != source_hashes():
                    raise ValueError('Training/evaluation source changed after the run began')
                if metadata['group'] != args.group or metadata['candidate_every'] != args.candidate_every:
                    raise ValueError('Watcher does not match training configuration')
                if metadata['config']['TRAIN']['MAX_UPDATES'] != args.total_updates:
                    raise ValueError('Watcher total budget does not match training')
            for step in range(args.candidate_every, args.total_updates+1, args.candidate_every):
                candidate = run/'checkpoints'/('step_%06d.pth.tar' % step)
                if step in evaluated or not candidate.is_file() or not metadata_path.exists():
                    continue
                out = validation/('step_%06d' % step)
                command = [sys.executable, str(ROOT/'tracking/evaluate_reliable_memory.py'),
                           '--checkpoint', str(candidate), '--manifest', str(manifest), '--out', str(out),
                           '--device', args.device, '--group', args.group, '--protocol', 'strict_language']
                if (out/'manifest.json').exists():
                    command.append('--resume')
                state.update(status='evaluating', step=step, training_status=training['status'])
                atomic_json(validation/'watcher_status.json', state)
                with (validation/('step_%06d.log' % step)).open('a') as log:
                    result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError('Validation step {} failed; see {}'.format(step, log.name))
                summary = json.loads((out/'summary.json').read_text())
                if (summary['engineering_only'] or summary['sequences'] != expected_count
                        or set(summary['metrics']) != expected_sources):
                    raise ValueError('Incomplete or incompatible full-video validation')
                row = dict(step=step, mean_AUC=summary['mean_AUC'], metrics=summary['metrics'],
                           checkpoint_sha256=sha256(candidate), summary=str(out/'summary.json'))
                rows.append(row)
                evaluated.add(step)
                best = choose_best(rows)
                link_best(run/'checkpoints'/('step_%06d.pth.tar' % best['step']), run/'best.pth.tar')
                atomic_json(validation/'best.json', dict(best=best, candidates=rows,
                            rule='three-source mean AUC; ties choose earlier step'))
                payload = {'validation/mean_AUC': row['mean_AUC']}
                for dataset, metrics in row['metrics'].items():
                    payload.update({'validation/'+dataset+'/'+key: value for key, value in metrics.items()})
                monitor.log(payload, step, force=True)
                state.update(status='waiting', evaluated=sorted(evaluated), best_step=best['step'])
                atomic_json(validation/'watcher_status.json', state)
            if training['status'] in ('completed', 'failed'):
                expected = set(range(args.candidate_every, args.total_updates+1, args.candidate_every))
                if training['status'] == 'completed' and evaluated != expected:
                    raise RuntimeError('Training completed without all expected candidate validations')
                state.update(status='completed' if training['status']=='completed' else 'training_failed',
                             evaluated=sorted(evaluated), training_status=training['status'])
                atomic_json(validation/'watcher_status.json', state)
                break
            pid_path = args.launcher_pid_file or run/'launcher.pid'
            if pid_path.exists():
                pid = int(pid_path.read_text().strip())
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    state.update(status='training_process_missing', evaluated=sorted(evaluated))
                    atomic_json(validation/'watcher_status.json', state)
                    break
            time.sleep(args.poll_seconds)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        atomic_json(validation/'watcher_status.json', state)
        raise
    finally:
        monitor.close('completed' if state['status'] == 'completed' else 'failed')


if __name__ == '__main__':
    main()
