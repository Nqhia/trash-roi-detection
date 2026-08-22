#!/usr/bin/env bash
# Chay shadow lien tuc, tu bat lai neu chet.
#
#   bash tools/keepalive.sh 'rtsp://...' config/zone_live.json config/day_cfg.yaml runs/shadow
#
# Vi sao can: hai lan chay truoc deu chet giua chung — mot lan may crash, mot
# lan may ngu / phien WSL dong — va khong lan nao co dong "mat ket noi han"
# trong log, tuc tien trinh bi giet tu ben ngoai chu khong phai loi RTSP.
# `run_video.py` da co backoff cho RTSP, nhung no khong tu song lai duoc.
#
# State (nen + mat na nhieu) nam trong <out>/state va duoc nap lai khi khoi
# dong, nen bat lai KHONG mat 5 gio hoc mat na. scans.csv thi bi GHI DE moi lan
# -> doi ten file cu truoc khi chay lai.
set -u
P=${PYTHON:-/home/nqhia/miniconda3/envs/cv-base/bin/python}
SRC=${1:?thieu nguon rtsp}
ZONE=${2:?thieu zone json}
CFG=${3:?thieu config yaml}
OUT=${4:-runs/shadow}
mkdir -p "$OUT"

n=0
while true; do
  n=$((n + 1))
  if [ -f "$OUT/scans.csv" ]; then
    mv "$OUT/scans.csv" "$OUT/scans_$(date +%Y%m%d_%H%M%S).csv"
  fi
  echo "=== lan chay thu $n · $(date +'%Y-%m-%d %H:%M:%S') ===" >> "$OUT/run.log"
  "$P" -u tools/run_video.py --source "$SRC" --zone "$ZONE" --config "$CFG" \
      --out "$OUT" >> "$OUT/run.log" 2>&1
  echo "=== thoat luc $(date +'%H:%M:%S'), bat lai sau 30s ===" >> "$OUT/run.log"
  sleep 30
done
