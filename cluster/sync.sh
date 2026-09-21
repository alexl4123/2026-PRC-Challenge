#!/usr/bin/env bash
# Push code and the cached parquet to the cluster. Read-only on the remote side
# apart from ~/prc2026. Never syncs outputs/ or any credential.
set -euo pipefail
REMOTE="${1:-copperhead}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "syncing code -> $REMOTE:~/prc2026"
rsync -az --delete \
    --exclude 'cache/' --exclude 'outputs/' --exclude '__pycache__/' \
    --exclude '.git/' --exclude '*.pyc' \
    "$HERE/" "$REMOTE:~/prc2026/"

echo "syncing cached parquet (~264 MB, only if changed)"
rsync -az --info=progress2 \
    "$HERE/cache/training.parquet" "$HERE/cache/ranking.parquet" \
    "$REMOTE:~/prc2026/cache/"

echo "syncing external data (METAR, ATFM)"
rsync -az "$HERE/cache/external/" "$REMOTE:~/prc2026/cache/external/" \
    --exclude 'atfm.xlsx'          # 97 MB source; the extracted CSV is enough

echo "done."
