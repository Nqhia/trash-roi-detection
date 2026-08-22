#!/usr/bin/env bash
# Bat lan chay 24h tren camera EcoVision, tach hoan toan khoi phien terminal.
#
#   bash tools/run24h.sh
#
# `setsid` la phan quan trong: khong co no thi tien trinh chet theo terminal ma
# nguoi bat no. Hai lan chay truoc deu chet giua chung.
#
# Nguon RTSP lay tu bien moi truong TRASH_RTSP de khong ghi mat khau vao repo.
set -u
cd "$(dirname "$0")/.." || exit 1
SRC=${TRASH_RTSP:?dat TRASH_RTSP='rtsp://user:pass@ip:554/...' truoc khi chay}
OUT=${OUT:-runs/shadow24}
mkdir -p "$OUT"
setsid nohup bash tools/keepalive.sh "$SRC" config/zone_live.json \
       config/day_cfg.yaml "$OUT" > "$OUT/keepalive.out" 2>&1 < /dev/null &
sleep 3
echo "da bat, PID $!  ->  $OUT/run.log"
