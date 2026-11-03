#!/usr/bin/env python3
"""Launch only training-data preparation, with the VastTrack-style relay."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
from urllib.parse import urlparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy', default='http://127.0.0.1:17891')
    parser.add_argument('--workers-per-dataset', type=int, default=10)
    parser.add_argument('--segment-mib', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers_per_dataset <= 32:
        parser.error('workers-per-dataset must be between 1 and 32')
    if not 1 <= args.segment_mib <= 16:
        parser.error('--segment-mib must be between 1 and 16')
    proxy = urlparse(args.proxy)
    if proxy.scheme != 'http' or proxy.username or proxy.password or not proxy.hostname or not proxy.port:
        parser.error('proxy must be an HTTP proxy without embedded credentials')
    with socket.create_connection((proxy.hostname, proxy.port), timeout=5):
        pass
    root = Path(__file__).resolve().parents[1]
    folder = root / 'downloads/official_training'
    folder.mkdir(exist_ok=True)
    with (folder / 'launch.lock').open('a') as launcher:
        fcntl.flock(launcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder / 'prepare.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('数据准备任务已在运行，不重复启动。')
        env = dict(os.environ, MVLM_DOWNLOAD_WORKERS=str(args.workers_per_dataset),
                   MVLM_DOWNLOAD_SEGMENT_MIB=str(args.segment_mib), MVLM_DOWNLOAD_HTTP2='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for name in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
            env[name] = args.proxy
        with (folder / 'prepare.log').open('ab') as log:
            child = subprocess.Popen(['/root/miniconda3/envs/mvlm/bin/python', '-u',
                                      str(root / 'downloads/prepare_official_training.py')],
                                     cwd=str(root), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (folder / 'prepare.pid').write_text(str(child.pid) + '\n')
        print('已启动数据准备，PID %s；不会启动模型训练。' % child.pid)
