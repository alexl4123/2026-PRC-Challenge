#!/usr/bin/env bash
# Push code and the cached parquet to the cluster. Read-only on the remote side
# apart from the remote repo path. Never syncs outputs/ or any credential.
set -euo pipefail
REMOTE="${1:-vivaldi}"        # see cluster/README.md for the ~/.ssh/config stanza
RPATH="${2:-03_2026_PRC_Challenge/2026-PRC-Challenge}"   # path on the remote, relative to $HOME
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "syncing code -> $REMOTE:~/$RPATH"
rsync -az --delete \
    --exclude 'cache/' --exclude 'outputs/' --exclude '__pycache__/' \
    --exclude '.git/' --exclude '*.pyc' \
    "$HERE/" "$REMOTE:~/$RPATH/"

echo "syncing cached parquet (~264 MB, only if changed)"
rsync -az --info=progress2 \
    "$HERE/cache/training.parquet" "$HERE/cache/ranking.parquet" \
    "$REMOTE:~/$RPATH/cache/"

echo "syncing external data (METAR, ATFM)"
rsync -az "$HERE/cache/external/" "$REMOTE:~/$RPATH/cache/external/" \
    --exclude 'atfm.xlsx'          # 97 MB source; the extracted CSV is enough

echo "done."
