#!/usr/bin/env bash
# Push the cached parquet and external data to the cluster. Read-only on the remote side
# apart from the remote repo path. Never syncs outputs/ or any credential.
set -euo pipefail
REMOTE="${1:-vivaldi}"        # see cluster/README.md for the ~/.ssh/config stanza
RPATH="${2:-03_2026_PRC_Challenge/2026-PRC-Challenge}"   # path on the remote, relative to $HOME
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# CODE IS NOT SYNCED HERE. It reaches the cluster through git (commit locally,
# push, then `git pull` on the cluster) -- see cluster/README.md. This script used
# to rsync the code with --delete as well, which would have removed everything on
# the remote that does not exist locally -- including the cluster-only logs/
# directory that running array jobs are writing into. Data only, no --delete.

echo "syncing cached parquet (~264 MB, only if changed)"
rsync -az --info=progress2 \
    "$HERE/cache/training.parquet" "$HERE/cache/ranking.parquet" \
    "$REMOTE:~/$RPATH/cache/"

echo "syncing external data (METAR, ATFM)"
rsync -az "$HERE/cache/external/" "$REMOTE:~/$RPATH/cache/external/" \
    --exclude 'atfm.xlsx'          # 97 MB source; the extracted CSV is enough

echo "done."
