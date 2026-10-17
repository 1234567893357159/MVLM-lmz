#!/bin/bash
# 下载监控：输出两个数据集的累计字节数、速度、进程状态
D1=/root/MVLM/downloads/tnl2k.tar.gz.parts
D2=/root/MVLM/downloads/vasttrack/vasttrack.tar.gz.part.aa.parts

bytes() {
  local d=$1 tot=0
  for f in "$d"/*; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    case "$b" in *.headers) continue;; esac
    s=$(stat -c %s "$f" 2>/dev/null) || s=0
    tot=$((tot+s))
  done
  echo $tot
}

T1=$(bytes $D1); V1=$(bytes $D2)
sleep 30
T2=$(bytes $D1); V2=$(bytes $D2)

TS=$(awk -v a=$T1 -v b=$T2 'BEGIN{printf "%.3f", (b-a)/30/1048576}')
VS=$(awk -v a=$V1 -v b=$V2 'BEGIN{printf "%.3f", (b-a)/30/1048576}')

# 按进程名判断存活, 而不是硬编码 PID
P1=$(pgrep -f 'import prepare_test_datasets' >/dev/null 2>&1 && echo alive || echo dead)
P2=$(pgrep -f 'prepare_vasttrack.py' >/dev/null 2>&1 && echo alive || echo dead)

echo "TS=$TS"
echo "VS=$VS"
echo "TNL2K_BYTES=$T2"
echo "VAST_BYTES=$V2"
echo "TNL2K_PID=$P1"
echo "VAST_PID=$P2"
echo "TNL2K_MIB=$((T2/1048576))"
echo "VAST_MIB=$((V2/1048576))"
echo "TNL2K_FULLPARTS=$(find $D1 -maxdepth 1 -type f -regextype posix-extended -regex '.*/[0-9]{6}' -size 256M 2>/dev/null | wc -l)"
echo "VAST_FULLPARTS=$(find $D2 -maxdepth 1 -type f -regextype posix-extended -regex '.*/[0-9]{6}' -size 256M 2>/dev/null | wc -l)"
echo "NOW=$(date '+%Y-%m-%d %H:%M:%S')"
echo "DISK=$(df -h /root/MVLM | tail -1 | awk '{print $4}')"
echo "VAST_STATUS=$(grep -o '"status": "[a-z]*"' /root/MVLM/downloads/vasttrack/status.json | head -1)"