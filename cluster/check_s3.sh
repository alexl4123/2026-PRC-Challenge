#!/usr/bin/env bash
# Verify S3 credentials and list the team bucket. Safe: read-only.
cd "$(dirname "$0")/.." || exit 1
/home/thinklex/prc2026-venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from prc.s3 import ls, BUCKET, ENDPOINT
print(f'endpoint {ENDPOINT}\nbucket   {BUCKET}\ncontents:')
ls()
"
