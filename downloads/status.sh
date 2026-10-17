#!/bin/bash
# ===== MVLM 下载状态一键查看 =====
CSV=/root/MVLM/downloads/monitor_log.csv
BOOT=/root/MVLM/downloads/monitor_boot.log
D1=/root/MVLM/downloads/tnl2k.tar.gz.parts
D2=/root/MVLM/downloads/vasttrack/vasttrack.tar.gz.part.aa.parts
TNL2K_TOTAL=91092953063      # tnl2k.tar.gz 真实大小
VAST_TOTAL=736604489019      # vasttrack 14 个分卷合计

G="\033[32m"; R="\033[31m"; Y="\033[33m"; C="\033[36m"; B="\033[1m"; N="\033[0m"

bytes() { local t=0; for f in "$1"/*; do [ -e "$f" ] || continue; case "$(basename $f)" in *.headers) continue;; esac; s=$(stat -c %s "$f" 2>/dev/null)||s=0; t=$((t+s)); done; echo $t; }
gib() { awk -v x=$1 'BEGIN{printf "%.2f", x/1073741824}'; }
pct() { awk -v x=$1 -v t=$2 'BEGIN{printf "%.2f", x/t*100}'; }
pbar() { awk -v x=$1 -v t=$2 'BEGIN{p=int(x/t*20); s=""; for(i=0;i<20;i++) s=s (i<p?"#":"-"); printf "[%s]", s}'; }

echo -e "${C}====== MVLM 下载监控 ======${N}"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo

if pgrep -f 'import prepare_test_datasets' >/dev/null 2>&1; then
  echo -e "TNL2K 进程:     ${G}运行中${N} (PID $(pgrep -f 'import prepare_test_datasets' | tr '\n' ' '))"
else
  echo -e "TNL2K 进程:     ${R}已停止${N}"
fi
if pgrep -f 'prepare_vasttrack.py' >/dev/null 2>&1; then
  echo -e "VastTrack 进程: ${G}运行中${N} (PID $(pgrep -f 'prepare_vasttrack.py'))"
else
  echo -e "VastTrack 进程: ${R}已停止${N}"
fi
if pgrep -f monitor_daemon.sh >/dev/null 2>&1; then
  echo -e "监控守护进程:   ${G}运行中${N}"
else
  echo -e "监控守护进程:   ${R}已停止${N}"
fi
echo

T1=$(bytes $D1); V1=$(bytes $D2)
echo -e "${Y}正在采样 10 秒测实时速度...${N}"
sleep 10
T2=$(bytes $D1); V2=$(bytes $D2)

TS=$(awk -v a=$T1 -v b=$T2 'BEGIN{printf "%.2f", (b-a)/10/1048576}')
VS=$(awk -v a=$V1 -v b=$V2 'BEGIN{printf "%.2f", (b-a)/10/1048576}')
TR=$(awk -v a=$TS -v b=$VS 'BEGIN{printf "%.2f", a+b}')

echo -e "${Y}-- 当前进度 --${N}"
printf "TNL2K:     %s %8s / %6s GiB (%5s%%)\n" "$(pbar $T2 $TNL2K_TOTAL)" "$(gib $T2)" "$(gib $TNL2K_TOTAL)" "$(pct $T2 $TNL2K_TOTAL)"
printf "VastTrack: %s %8s / %6s GiB (%5s%%)\n" "$(pbar $V2 $VAST_TOTAL)" "$(gib $V2)" "$(gib $VAST_TOTAL)" "$(pct $V2 $VAST_TOTAL)"
printf "${B}合  计:       %8s / %6s GiB (%5s%%)${N}\n" "$(gib $((T2+V2)))" "$(gib $((TNL2K_TOTAL+VAST_TOTAL)))" "$(pct $((T2+V2)) $((TNL2K_TOTAL+VAST_TOTAL)))"
echo

echo -e "${Y}-- 实时速度(10秒采样) --${N}"
printf "TNL2K:     %6s MB/s\n" "$TS"
printf "VastTrack: %6s MB/s\n" "$VS"
printf "${B}合  计:    %6s MB/s${N}\n" "$TR"

# 预计剩余时间
awk -v t=$TR -v r=$((TNL2K_TOTAL+VAST_TOTAL-T2-V2)) 'BEGIN{
  if (t<=0) {print "\n预计: 速度过低, 无法估算"; exit}
  h=r/1048576/t/3600
  if (h>72) printf "\n按当前总速度预计还需: %.1f 天\n", h/24
  else printf "\n按当前总速度预计还需: %.1f 小时\n", h
}'
echo

echo -e "${Y}-- 最近记录 --${N}"
printf "%-20s %10s %9s %10s %9s\n" "时间" "TNL2K(MiB)" "速率MB/s" "Vast(MiB)" "速率MB/s"
tail -n +2 "$CSV" | tail -5 | awk -F, '{printf "%-20s %10s %9s %10s %9s\n",$1,($2==""?"-":$2),$3,($4==""?"-":$4),$5}'
echo

echo -e "${Y}-- 重启记录 --${N}"
if [ -s "$BOOT" ]; then cat "$BOOT"; else echo "(无)"; fi
echo
echo -e "${C}提示: 完整历史 tail -f $CSV${N}"