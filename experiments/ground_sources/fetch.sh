#!/usr/bin/env bash
# fetch.sh URL OUTFILE [extra curl args...]
# Downloads with curl, retries on network failure, appends bytes to tally.tsv.
set -u
url="$1"; out="$2"; shift 2
tally=/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/ground_sources/tally.tsv
for i in 1 2 3 4; do
  n=$(curl -sS -L --max-time 300 -A "Mozilla/5.0 (research; prc2026)" -o "$out" -w '%{size_download}' "$@" "$url") && break
  echo "retry $i for $url" >&2; sleep 10
done
n=${n:-0}
printf '%s\t%s\t%s\n' "$(date -Is)" "$n" "$url" >> "$tally"
echo "$n bytes -> $out"
