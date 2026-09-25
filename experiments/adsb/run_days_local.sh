#!/usr/bin/env bash
# Local driver: for each DATE argument, download -> stage 1 (points) -> stage 2 (events) -> delete tar parts.
# Memory-capped and niced for the laptop. Keeps only cache/adsb_lol/{points,events}/<date>.parquet.
#   bash experiments/adsb/run_days_local.sh 2025-07-14 2025-07-15 ...
# Set KEEP_TAR=1 to keep the downloaded parts.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/home/thinklex/prc2026-venv/bin/python
RAW="$REPO/cache/adsb_lol/raw"
LOG="$REPO/cache/adsb_lol/meta/run_days_local.log"
mkdir -p "$RAW" "$(dirname "$LOG")"
for D in "$@"; do
  if [[ -f "$REPO/cache/adsb_lol/events/$D.parquet" ]]; then echo "$D done already" | tee -a "$LOG"; continue; fi
  t0=$(date +%s)
  mapfile -t PARTS < <($PY "$REPO/experiments/adsb/adsb_download.py" "$D" --dest "$RAW" 2>>"$LOG")
  t1=$(date +%s)
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 nice -n 5 \
    $PY "$REPO/experiments/adsb/adsb_extract_day.py" "$D" --parts "${PARTS[@]}" --workers 6 >>"$LOG" 2>&1
  t2=$(date +%s)
  sz=$(du -cb "${PARTS[@]}" | tail -1 | cut -f1)
  echo "$D download_s=$((t1-t0)) extract_s=$((t2-t1)) bytes=$sz" | tee -a "$LOG"
  if [[ "${KEEP_TAR:-0}" != "1" ]]; then rm -f "${PARTS[@]}"; fi
done
