#!/bin/bash
# 谨慎版 autotune：先测当前，只在明确更优时才重启
D=/root/MVLM/downloads/official_training
LOG=$D/autotune.log
URL="https://hf-mirror.com/datasets/l-lt/LaSOT/resolve/main/airplane.zip"
PROXY=17892

ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
log() { echo "$(ts) $*" >> "$LOG"; }

probe() {
  local N=$1
  local T1=$(date +%s%N)
  for i in $(seq 0 $((N-1))); do
    S=$((2300000000 + i*4194304)); E=$((S+4194303))
    ( curl -sL -o /dev/null --max-time 18 -x http://127.0.0.1:$PROXY \
        -H 'Accept-Encoding: identity' --range $S-$E \
        -w "%{size_download}\n" "$URL" 2>/dev/null > /tmp/at_$i.txt ) &
  done
  wait
  local T2=$(date +%s%N)
  local SUM=$(cat /tmp/at_*.txt 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s+0}')
  rm -f /tmp/at_*.txt
  local EL=$(( (T2-T1)/1000000000 )); [ "$EL" -lt 1 ] && EL=1
  awk -v s="$SUM" -v e="$EL" 'BEGIN{printf "%.3f", s/e/1000000}'
}

echo "=== autotune $(ts) ==="

# 1. 当前实际速度（不干扰）
A=$(du -sb $D 2>/dev/null | awk '{printf "%.0f", $1}')
sleep 60
B=$(du -sb $D 2>/dev/null | awk '{printf "%.0f", $1}')
CUR=$(awk -v d="$((B-A))" 'BEGIN{printf "%.3f", d/60/1000000}')
echo "  current: $CUR MB/s (16 workers)"

# 2. 暂停下载，纯压测各并发档
P=$(ps -eo pid,cmd | grep '[p]repare_official_training' | awk '{print $1}' | head -n1)
[ -n "$P" ] && kill -TERM "$P" 2>/dev/null
sleep 10
[ -n "$P" ] && ps -p "$P" >/dev/null 2>&1 && kill -9 "$P" 2>/dev/null
sleep 2

BEST_N=16
BEST_V=0
for N in 16 24 32; do
  V=$(probe $N)
  echo "  probe N=$N => $V MB/s"
  if awk -v v="$V" -v b="$BEST_V" 'BEGIN{exit !(v>b)}'; then
    BEST_V=$V
    BEST_N=$N
  fi
done

echo "  best probe: N=$BEST_N ($BEST_V MB/s)"

# 3. 只有明显优于当前才换并发，否则维持 16
USE_N=16
if awk -v b="$BEST_V" -v c="$CUR" 'BEGIN{exit !(b > c*1.15)}'; then
  USE_N=$BEST_N
  echo "  -> switching to N=$USE_N (probe %.1f%% faster)"
else
  echo "  -> keeping N=16 (probe not >15% faster than current)"
fi

# 4. 重启
cd /root/MVLM || exit 1
# Keep the flock inode stable; never remove prepare.lock.
nohup env MVLM_HF_BASE=https://hf-mirror.com \
  VASTTRACK_PROXY=http://127.0.0.1:$PROXY \
  http_proxy=http://127.0.0.1:$PROXY https_proxy=http://127.0.0.1:$PROXY \
  HTTP_PROXY=http://127.0.0.1:$PROXY HTTPS_PROXY=http://127.0.0.1:$PROXY \
  ALL_PROXY=http://127.0.0.1:$PROXY all_proxy=http://127.0.0.1:$PROXY \
  MVLM_DOWNLOAD_WORKERS=$USE_N MVLM_DOWNLOAD_SEGMENT_MIB=4 MVLM_DOWNLOAD_HTTP2=1 \
  /root/miniconda3/envs/mvlm/bin/python -u downloads/prepare_official_training.py \
  >> $D/prepare_new.log 2>&1 &
echo $! > $D/prepare.pid
echo "  restarted pid=$! with N=$USE_N"
log "autotune: cur=$CUR best_probe_n=$BEST_N best_probe_v=$BEST_V use_n=$USE_N"
