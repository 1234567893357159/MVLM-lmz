#!/bin/bash
# 自启入口：等网络就绪后拉起下载 + 守护
BASE=/root/MVLM/downloads
D=$BASE/official_training

# 等隧道端口出现（最多 10 分钟）
for i in $(seq 1 60); do
  if python3 -c "
import sys
for line in open('/proc/net/tcp'):
    f=line.split()
    if len(f)<4: continue
    if f[3]=='0A':
        ip,port=f[1].split(':')
        if int(port,16)==17892: sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
    echo "$(date -u '+%F %T UTC') tunnel 17892 ready after ${i}0s"
    break
  fi
  sleep 10
done

# 下载没跑就拉起来
if ! ps -eo cmd | grep -q '[p]repare_official_training.py'; then
  cd /root/MVLM || exit 1
  # Keep the flock inode stable; never remove prepare.lock.
  nohup env MVLM_HF_BASE=https://hf-mirror.com \
    VASTTRACK_PROXY=http://127.0.0.1:17892 \
    http_proxy=http://127.0.0.1:17892 https_proxy=http://127.0.0.1:17892 \
    HTTP_PROXY=http://127.0.0.1:17892 HTTPS_PROXY=http://127.0.0.1:17892 \
    ALL_PROXY=http://127.0.0.1:17892 all_proxy=http://127.0.0.1:17892 \
    MVLM_DOWNLOAD_WORKERS=16 MVLM_DOWNLOAD_SEGMENT_MIB=4 MVLM_DOWNLOAD_HTTP2=1 \
    /root/miniconda3/envs/mvlm/bin/python -u downloads/prepare_official_training.py \
    >> $D/prepare_new.log 2>&1 &
  echo $! > $D/prepare.pid
  echo "$(date -u '+%F %T UTC') download started pid=$!"
fi

# 守护起来
bash $BASE/watchdog_daemon.sh start
