"""Uploading submissions to the team's OpenSky bucket.

CREDENTIALS ARE NEVER STORED IN THIS REPO OR ANYWHERE UNDER DROPBOX.
The repo is destined to be public under GPLv3, and the whole 00_src tree is a
synced Dropbox folder, so either location would leak the key. They live in
    ~/.config/prc2026/s3.env      (mode 600, outside both)

Create that file with, substituting your own values:

    install -m 600 /dev/null ~/.config/prc2026/s3.env
    cat > ~/.config/prc2026/s3.env <<'EOF'
    PRC_S3_ACCESS_KEY=your_access_key
    PRC_S3_SECRET_KEY=your_secret_key
    EOF

Environment variables of the same names override the file, which is how a
cluster job should receive them.
"""
import os
from pathlib import Path

ENDPOINT = 'https://s3.opensky-network.org'
BUCKET = 'prc-2026-understanding-accordion'
CRED_FILE = Path.home() / '.config' / 'prc2026' / 's3.env'


def _credentials():
    key, secret = os.environ.get('PRC_S3_ACCESS_KEY'), os.environ.get('PRC_S3_SECRET_KEY')
    if key and secret:
        return key, secret
    if not CRED_FILE.exists():
        raise FileNotFoundError(
            f'no credentials: set PRC_S3_ACCESS_KEY / PRC_S3_SECRET_KEY, or create '
            f'{CRED_FILE} (see the module docstring)')
    mode = CRED_FILE.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f'{CRED_FILE} is mode {mode:o}; run: chmod 600 {CRED_FILE}')
    vals = {}
    for line in CRED_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            vals[k.strip()] = v.strip().strip('"\'')
    try:
        return vals['PRC_S3_ACCESS_KEY'], vals['PRC_S3_SECRET_KEY']
    except KeyError as e:
        raise KeyError(f'{CRED_FILE} is missing {e}') from None


def client():
    import boto3
    from botocore.config import Config
    key, secret = _credentials()
    return boto3.client('s3', endpoint_url=ENDPOINT, aws_access_key_id=key,
                        aws_secret_access_key=secret,
                        config=Config(signature_version='s3v4', retries={'max_attempts': 3}))


def ls(bucket=BUCKET):
    """List the team bucket. Result files from the scorer appear here too."""
    c = client()
    resp = c.list_objects_v2(Bucket=bucket)
    out = [(o['Key'], o['Size'], o['LastModified']) for o in resp.get('Contents', [])]
    for k, sz, ts in sorted(out, key=lambda r: r[2]):
        print(f'  {ts:%Y-%m-%d %H:%M}  {sz:>10,}  {k}')
    if not out:
        print('  (bucket empty)')
    return out


def upload(path, bucket=BUCKET, dry_run=False):
    """Upload one submission. Validates the file again before sending."""
    from .submit import validate
    path = Path(path)
    validate(path)                    # never upload something that has not just passed
    if dry_run:
        print(f'  DRY RUN: would upload {path.name} -> s3://{bucket}/{path.name}')
        return None
    c = client()
    c.upload_file(str(path), bucket, path.name)
    print(f'  uploaded {path.name} -> s3://{bucket}/{path.name}')
    return f's3://{bucket}/{path.name}'
