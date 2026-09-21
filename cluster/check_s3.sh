#!/usr/bin/env bash
# Verify S3 credentials and list the team bucket. Read-only: lists, never writes.
# Works on the laptop and on the cluster -- picks whichever interpreter is there
# rather than hardcoding one.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ -x /home/thinklex/prc2026-venv/bin/python ]]; then
    PY=/home/thinklex/prc2026-venv/bin/python           # laptop
elif command -v python >/dev/null; then
    PY=$(command -v python)                             # cluster, inside the conda env
else
    echo "no python found; activate the conda env first" >&2; exit 1
fi

"$PY" -c "
import sys; sys.path.insert(0,'.')
from prc.s3 import ls, BUCKET, ENDPOINT
print(f'endpoint {ENDPOINT}')
print(f'bucket   {BUCKET}')
print('contents:')
ls()
"
