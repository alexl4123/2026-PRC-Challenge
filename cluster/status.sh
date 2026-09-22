#!/usr/bin/env bash
# One-shot status for every PRC campaign on the cluster. Read-only.
#   bash cluster/status.sh
cd "$(dirname "$0")/.." || exit 1

echo "===== JOBS ====="
squeue -u "$USER" -o "%.14i %.12j %.8T %.10M %.6C %R" | grep -E "JOBID|prc-" || echo "  no prc-* jobs in the queue"

echo
echo "===== PROGRESS (files written / expected) ====="
printf '  %-12s %6s / %-6s\n' tailspec "$(ls cache/cv/tailspec/*.npy 2>/dev/null | wc -l)" 240
printf '  %-12s %6s / %-6s  (216 cpu + 36 gpu)\n' study "$(ls cache/cv/study/*.npy 2>/dev/null | wc -l)" 252

echo
echo "===== PER-ARM BREAKDOWN of the study campaign ====="
if compgen -G "cache/cv/study/*.npy" > /dev/null; then
    ls cache/cv/study/*.npy | sed 's#.*/##; s#__seed.*##' | sort | uniq -c | sed 's/^/  /'
else
    echo "  (nothing yet)"
fi

echo
echo "===== FAILURES / TIMEOUTS (array tasks only) ====="
for j in $(squeue -u "$USER" -h -o "%A" | sort -u) $(sacct -u "$USER" -S today -n -X -o JobID 2>/dev/null | cut -d_ -f1 | sort -u); do :; done
sacct -u "$USER" -S today -n -X --name=prc-tailspec,prc-study,prc-nn \
      -o JobName%14,State%12 2>/dev/null | awk '{print $1, $2}' | sort | uniq -c | sed 's/^/  /' \
      || echo "  (sacct returned nothing)"

echo
echo "===== MOST RECENT TASK OUTPUT (one line per campaign) ====="
for pat in prc-tailspec prc-study prc-nn; do
    f=$(ls -t logs/${pat}_*.out 2>/dev/null | head -1)
    if [ -n "$f" ]; then printf '  %-14s %s\n' "$pat" "$(tail -1 "$f")"; else printf '  %-14s (no logs yet)\n' "$pat"; fi
done

echo
echo "===== DISK ====="
du -sh cache 2>/dev/null | sed 's/^/  cache      /'
du -sh logs  2>/dev/null | sed 's/^/  logs       /'
df -h . | tail -1 | awk '{print "  filesystem  "$4" free of "$2" ("$5" used)"}'
