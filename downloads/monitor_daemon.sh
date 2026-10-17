#!/bin/bash
LOG=/root/MVLM/downloads/monitor_log.csv
BOOT=/root/MVLM/downloads/monitor_boot.log
[ -f "$LOG" ] || echo "time,tnl2k_mib,tnl2k_mbps,vast_mib,vast_mbps,tnl2k_pid,vast_pid,disk_avail" > "$LOG"

# 统一的进程存活判断：必须只有一个 TNL2K / VastTrack 脚本进程
tnl2k_alive() { [ "$(pgrep -fc 'prepare_test_datasets|tnl2k_download.pid' 2>/dev/null || echo 0)" -gt 0 ] && pgrep -f 'import prepare_test_datasets' >/dev/null 2>&1; }
vast_alive()  { pgrep -f 'prepare_vasttrack.py' >/dev/null 2>&1; }

restart_tnl2k() {
  # 幂等：已有进程就不再启动第二个
  if pgrep -f 'import prepare_test_datasets' >/dev/null 2>&1; then return; fi
  echo "[$(date '+%F %T')] RESTART tnl2k" >> "$BOOT"
  cd /root/MVLM || return
  # 关键：恢复代理环境变量, 否则 curl 会直连 HF 导致 15s 超时
  source /root/.bashrc >/dev/null 2>&1
  export http_proxy=http://127.0.0.1:17891 https_proxy=http://127.0.0.1:17891
  export HTTP_PROXY=http://127.0.0.1:17891 HTTPS_PROXY=http://127.0.0.1:17891
  setsid nohup /root/miniconda3/envs/mvlm/bin/python -u -c 'import sys; sys.path.insert(0,"downloads"); import prepare_test_datasets as d; d.STATUS=d.DOWNLOADS/"tnl2k_status.json"; d.state={"tnl2k":{}}; d.tnl2k()' >> /root/MVLM/downloads/tnl2k_restart.log 2>&1 < /dev/null &
  echo $! > /root/MVLM/downloads/tnl2k_download.pid
}

restart_vast() {
  if pgrep -f 'prepare_vasttrack.py' >/dev/null 2>&1; then return; fi
  echo "[$(date '+%F %T')] RESTART vasttrack" >> "$BOOT"
  cd /root/MVLM || return
  source /root/.bashrc >/dev/null 2>&1
  export http_proxy=http://127.0.0.1:17891 https_proxy=http://127.0.0.1:17891
  export HTTP_PROXY=http://127.0.0.1:17891 HTTPS_PROXY=http://127.0.0.1:17891
  setsid nohup /root/miniconda3/envs/mvlm/bin/python -u /root/MVLM/downloads/prepare_vasttrack.py >> /root/MVLM/downloads/vasttrack/download.log 2>&1 < /dev/null &
  echo $! > /root/MVLM/downloads/vasttrack/download.pid
}

while true; do
  out=$(/root/MVLM/downloads/monitor_downloads.sh 2>/dev/null)
  TS=$(echo "$out" | grep '^TS=' | cut -d= -f2)
  VS=$(echo "$out" | grep '^VS=' | cut -d= -f2)
  TB=$(echo "$out" | grep '^TNL2K_MIB=' | cut -d= -f2)
  VB=$(echo "$out" | grep '^VAST_MIB=' | cut -d= -f2)
  P1=$(echo "$out" | grep '^TNL2K_PID=' | cut -d= -f2)
  P2=$(echo "$out" | grep '^VAST_PID=' | cut -d= -f2)
  DSK=$(echo "$out" | grep '^DISK=' | cut -d= -f2)
  echo "$(date '+%F %T'),$TB,$TS,$VB,$VS,$P1,$P2,$DSK" >> "$LOG"
  if [ "$P1" = "dead" ]; then restart_tnl2k; fi
  if [ "$P2" = "dead" ]; then restart_vast; fi
  sleep 600
done