"""Watch training preparation without interrupting verification or repairs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from prepare_official_training import OUT, ROOT, saved_bytes, source_rows


def read_state():
    try:
        return json.loads((OUT / 'status.json').read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    try:
        return b'prepare_official_training.py' in Path('/proc/%d/cmdline' % pid).read_bytes()
    except (OSError, TypeError):
        return False


def snapshot():
    state = read_state()
    stages = {name: state.get(name, {}).get('status') for name in ('lasot', 'tnl2k')}
    count = sum(saved_bytes(source_rows(name)) for name in stages)
    return state, stages, count


def stop_preparation(pid):
    if not alive(pid):
        return
    children = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if int((p / 'stat').read_text().split()[3]) == pid:
                    children.append(int(p.name))
            except (OSError, ValueError, IndexError):
                pass
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not alive(pid):
            break
        time.sleep(0.5)
    if alive(pid):
        os.kill(pid, signal.SIGKILL)
    for child in children:
        try:
            if b'curl' in Path('/proc/%d/cmdline' % child).read_bytes().split(b'\0')[0]:
                os.kill(child, signal.SIGKILL)
        except OSError:
            pass


def main(dry_run):
    state, stages, before = snapshot()
    sample_secs = max(1, int(os.environ.get('SAMPLE_SECS', '90')))
    time.sleep(sample_secs)
    state, stages, after = snapshot()
    pid = state.get('_runtime', {}).get('pid')
    delta = max(0, after - before)
    speed = delta / sample_secs / 1_000_000
    processing = any(stage in ('repairing', 'verifying', 'extracting', 'validating')
                     for stage in stages.values())
    terminal = all(stage in ('ready', 'failed', 'paused') for stage in stages.values())
    old = {}
    try:
        old = json.loads((OUT / 'watchdog_state.json').read_text())
    except (OSError, ValueError):
        pass
    streak = old.get('zero_progress_streak', 0) + 1 if delta == 0 else 0
    status, action = 'OK', 'none'
    if terminal:
        status = 'COMPLETE' if all(s == 'ready' for s in stages.values()) else 'NEEDS_ATTENTION'
    elif not alive(pid):
        status, action = 'DEAD', 'restart_download'
    elif processing:
        status, streak = 'PROCESSING', 0
    elif delta == 0:
        status = 'WAITING'
        if streak >= 3:
            status, action = 'STALLED', 'restart_download'
    if action == 'restart_download' and not dry_run:
        if alive(pid):
            stop_preparation(pid)
        proxy = state.get('_runtime', {}).get('proxy') or 'http://127.0.0.1:17892'
        result = subprocess.run([
            '/root/miniconda3/envs/mvlm/bin/python', str(ROOT / 'downloads/start_training_downloads.py'),
            '--proxy', proxy, '--workers-per-dataset', '16', '--segment-mib', '4'],
            capture_output=True, text=True)
        if result.returncode:
            action = 'restart_failed'
        streak = 0
    report = dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                  status=status, action=action, speed_mbps=speed, delta_bytes=delta,
                  sample_secs=sample_secs, pid=pid, process_alive=int(alive(pid)),
                  downloaded_bytes=after, stages=stages, zero_progress_streak=streak)
    if not dry_run:
        temp = OUT / 'watchdog_state.tmp'
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(OUT / 'watchdog_state.json')
        with (OUT / 'watchdog.log').open('a') as log:
            log.write('%s status=%s speed=%.3fMB/s delta=%s pid=%s action=%s stages=%s\n' % (
                report['checked_at'], status, speed, delta, pid, action, stages))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with (OUT / 'watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.dry_run)
