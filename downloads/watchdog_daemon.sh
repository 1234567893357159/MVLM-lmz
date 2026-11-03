#!/bin/bash
# 常驻守护：每 30 分钟调用一次 watchdog_download.sh
# 用法: watchdog_daemon.sh start|stop|status
BASE=/root/MVLM/downloads
WD=$BASE/watchdog_download.sh
PIDF=$BASE/watchdog_daemon.pid
LOGF=$BASE/official_training/watchdog_daemon.log
INTERVAL=${INTERVAL:-1800}

case "$1" in
  start)
    if [ -f "$PIDF" ] && ps -p "$(cat $PIDF)" > /dev/null 2>&1; then
      echo "already running pid=$(cat $PIDF)"; exit 0
    fi
    /root/miniconda3/envs/mvlm/bin/python - "$LOGF" "$PIDF" "$INTERVAL" "$WD" <<'DAEMON_PY'
import os
from pathlib import Path
import subprocess
import sys
import time
logfile, pidfile, interval, watchdog = sys.argv[1:]
env = dict(os.environ, MVLM_WATCHDOG_INTERVAL=interval, MVLM_WATCHDOG_SCRIPT=watchdog)
command = 'while true; do sleep "$MVLM_WATCHDOG_INTERVAL"; bash "$MVLM_WATCHDOG_SCRIPT"; done'
with open(logfile, 'ab', buffering=0) as log:
    log.write((time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + ' daemon started interval=' + interval + 's\n').encode())
    child = subprocess.Popen(['/bin/bash', '-c', command], env=env,
                             stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
Path(pidfile).write_text(str(child.pid) + '\n')
DAEMON_PY
    echo "started pid=$(cat $PIDF) interval=${INTERVAL}s"
    ;;
  stop)
    if [ -f "$PIDF" ]; then
      PID=$(cat "$PIDF")
      pkill -P "$PID" 2>/dev/null
      kill -TERM "$PID" 2>/dev/null
      rm -f "$PIDF"
      echo "stopped pid=$PID"
    else
      echo "not running"
    fi
    ;;
  status)
    if [ -f "$PIDF" ] && ps -p "$(cat $PIDF)" > /dev/null 2>&1; then
      echo "running pid=$(cat $PIDF)"
    else
      echo "not running"
    fi
    echo "--- last 10 watchdog log lines ---"
    tail -n 10 "$BASE/official_training/watchdog.log" 2>/dev/null
    ;;
  *)
    echo "usage: $0 start|stop|status"; exit 1
    ;;
esac
