"""Download ONE day of the adsb.lol globe_history archive (split tar parts).

    python experiments/adsb/adsb_download.py 2025-07-16 [--dest DIR] [--variant auto|prod|staging]

Source: https://github.com/adsblol/globe_history_<YEAR>/releases (ODbL 1.0).
Each day is a release tagged  vYYYY.MM.DD-planes-readsb-<variant>  whose assets are
the parts  <tag>.tar.aa, <tag>.tar.ab, ...  of one tar (cat them, then untar).

Variant choice (--variant auto, the default): the repo's own PREFERRED_RELEASES.txt,
which names for each day the larger of prod / staging (and the '0tmp' releases that
fill 2025-05-23..06-12). That file is fetched once and cached in <dest>/../meta/.
Days of December 2025 are in both the 2025 and 2026 repos; the year repo is tried
first, then the next year's.

Idempotent: a finished day has <dest>/<tag>.ok listing each part and its size.
Partial parts are resumed with an HTTP Range request. Sizes are checked against the
server's Content-Length, so no GitHub API call (60/h unauthenticated) is needed.
Prints the local part paths, one per line, on stdout (for piping into the extractor).
Exits non-zero, loudly, on any failure.
"""
import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / 'cache' / 'adsb_lol' / 'raw'
UA = {'User-Agent': 'prc2026-adsb-research (python urllib)'}
PREF_URL = 'https://raw.githubusercontent.com/adsblol/globe_history_{y}/main/PREFERRED_RELEASES.txt'
REL_URL = 'https://github.com/adsblol/globe_history_{y}/releases/download/{tag}/{tag}.tar.{part}'


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def preferred_urls(date: str, meta: Path, variant: str) -> list[str]:
    """URLs of all parts for `date` (YYYY-MM-DD)."""
    y = int(date[:4])
    dot = date.replace('-', '.')
    for repo_year in (y, y + 1):
        if variant != 'auto':
            # Probe parts aa, ab, ... until a 404.
            tag = f'v{dot}-planes-readsb-{variant}-0'
            urls = []
            for part in ('aa', 'ab', 'ac', 'ad', 'ae', 'af'):
                u = REL_URL.format(y=repo_year, tag=tag, part=part)
                try:
                    _get(u, timeout=60).close()
                    urls.append(u)
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        break
                    raise
            if urls:
                return urls
            continue
        cache = meta / f'PREFERRED_RELEASES_{repo_year}.txt'
        for attempt in (0, 1):
            if attempt == 1 or not cache.exists() or time.time() - cache.stat().st_mtime > 86400:
                try:
                    with _get(PREF_URL.format(y=repo_year)) as r:
                        tmp = cache.with_suffix(f'.{os.getpid()}.tmp')
                        tmp.write_bytes(r.read())
                        os.replace(tmp, cache)       # atomic: parallel tasks never see a half file
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        break
                    raise
            for line in cache.read_text().splitlines():
                urls = [u.strip() for u in line.split(',') if u.strip()]
                if urls and re.search(rf'/v{re.escape(dot)}-planes-readsb-', urls[0]):
                    return urls
    raise SystemExit(f'ERROR: no adsb.lol release found for {date} (variant={variant})')


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get('Content-Length')
    if n is None:
        raise SystemExit(f'ERROR: no Content-Length for {url}')
    return int(n)


def fetch(url: str, out: Path, want: int, retries: int = 8):
    for attempt in range(retries):
        have = out.stat().st_size if out.exists() else 0
        if have == want:
            return
        if have > want:
            log(f'  {out.name}: local {have} > remote {want}; restarting')
            out.unlink()
            have = 0
        try:
            hdr = {'Range': f'bytes={have}-'} if have else {}
            with _get(url, headers=hdr, timeout=120) as r, open(out, 'ab' if have else 'wb') as f:
                if have and r.status != 206:
                    raise IOError(f'server ignored Range (status {r.status})')
                while True:
                    buf = r.read(1 << 22)
                    if not buf:
                        break
                    f.write(buf)
        except Exception as e:  # noqa: BLE001 - retry any transport error
            wait = min(300, 10 * 2 ** attempt)
            log(f'  {out.name}: attempt {attempt + 1} failed ({e!r}); retry in {wait}s')
            time.sleep(wait)
    if not out.exists() or out.stat().st_size != want:
        raise SystemExit(f'ERROR: {out} incomplete after {retries} attempts')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date')
    ap.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    ap.add_argument('--variant', default='auto', choices=['auto', 'prod', 'staging'])
    a = ap.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', a.date):
        raise SystemExit('ERROR: date must be YYYY-MM-DD')
    a.dest.mkdir(parents=True, exist_ok=True)
    meta = a.dest.parent / 'meta'
    meta.mkdir(parents=True, exist_ok=True)

    urls = preferred_urls(a.date, meta, a.variant)
    tag = urls[0].rsplit('/', 2)[-2]
    ok = a.dest / f'{tag}.ok'
    paths = [a.dest / u.rsplit('/', 1)[-1] for u in urls]
    if ok.exists() and all(p.exists() for p in paths):
        sizes = dict(line.split() for line in ok.read_text().splitlines())
        if all(p.stat().st_size == int(sizes.get(p.name, -1)) for p in paths):
            log(f'{tag}: already complete')
            print('\n'.join(map(str, paths)))
            return
    t0 = time.time()
    total = 0
    lines = []
    for u, p in zip(urls, paths):
        want = remote_size(u)
        fetch(u, p, want)
        total += want
        lines.append(f'{p.name} {want}')
    ok.write_text('\n'.join(lines) + '\n')
    dt = time.time() - t0
    log(f'{tag}: {len(paths)} parts, {total / 1e9:.2f} GB in {dt:.0f} s '
        f'({total / 1e6 / max(dt, 1e-9):.1f} MB/s)')
    print('\n'.join(map(str, paths)))


if __name__ == '__main__':
    main()
