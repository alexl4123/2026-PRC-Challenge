"""Stage 1 + 2 for ONE day of adsb.lol globe_history: tar parts -> airport points -> departure events.

    python experiments/adsb/adsb_extract_day.py 2025-07-16 [--parts P1 P2 ...] [--workers 6]
           [--raw-dir DIR] [--out-dir DIR] [--force] [--keep-points]

Stage 1 streams the split tar WITHOUT unpacking it to disk (python tarfile in stream
mode over the concatenated parts), parses every traces/*/trace_full_<hex>.json.gz in a
process pool, and keeps only points within 12 km of one of the ten airports and below
6,000 ft above the field (or reported on the ground). Output:
    <out-dir>/points/<date>.parquet   (one row per kept trace point)
Stage 2 (adsb_events.py) turns those points into one row per detected take-off with
ground-movement features:
    <out-dir>/events/<date>.parquet

Idempotent: if the outputs exist and --force is not given, the day is skipped.
Outputs are written to a temp name and renamed, so a killed job leaves no half file.
Fails loudly (non-zero exit) if the parts are missing, the tar is truncated, or fewer
than 1,000 traces were read (a sign of a bad archive).

Trace point layout (readsb README-json.md, 'trace jsons'):
  [dt, lat, lon, alt_ft|'ground'|null, gs_kt, track|true_heading_on_ground, flags,
   vert_rate_fpm, details{flight,...}|null, source_type, geom_alt, geom_rate, ias, roll]
  flags: 1 stale, 2 new leg, 4 geometric vrate, 8 geometric altitude.
"""
import argparse
import gzip
import io
import json
import os
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from airports import CODES, LAT, LON, ELEV, R_KEEP_M, H_KEEP_FT, EARTH_R  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / 'cache' / 'adsb_lol'

# Coarse box (degrees) around every ARP for a cheap first cut.
_DLAT = R_KEEP_M / EARTH_R * 180 / np.pi
_DLON = _DLAT / np.cos(np.radians(LAT))


class ConcatReader(io.RawIOBase):
    """Read several files OR http(s) URLs back to back as one stream (the split tar parts).

    For URLs a dropped connection is resumed with an HTTP Range request at the current
    offset (up to 8 retries per part), so --stream needs no local disk at all.
    """

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        self.i = 0
        self.pos = 0            # offset within the current part
        self.nread = 0
        self.f = self._open(self.paths[0], 0)

    @staticmethod
    def _open(p, offset):
        if p.startswith('http'):
            import urllib.request
            hdr = {'User-Agent': 'prc2026-adsb-research (python urllib)'}
            if offset:
                hdr['Range'] = f'bytes={offset}-'
            r = urllib.request.urlopen(urllib.request.Request(p, headers=hdr), timeout=120)
            if offset and r.status != 206:
                raise IOError(f'server ignored Range for {p}')
            cl = r.headers.get('Content-Length')
            r.expected_end = offset + int(cl) if cl is not None else None
            return r
        f = open(p, 'rb')
        f.seek(offset)
        return f

    def readable(self):
        return True

    def readinto(self, b):
        retries = 0
        while True:
            try:
                n = self.f.readinto(b)
            except Exception as e:  # noqa: BLE001 - network hiccup: resume at offset
                if not self.paths[self.i].startswith('http') or retries >= 8:
                    raise
                retries += 1
                print(f'  resume {self.paths[self.i]} at {self.pos} after {e!r}', file=sys.stderr, flush=True)
                time.sleep(min(120, 5 * 2 ** retries))
                try:
                    self.f.close()
                except Exception:  # noqa: BLE001
                    pass
                self.f = self._open(self.paths[self.i], self.pos)
                continue
            if n:
                self.pos += n
                self.nread += n
                return n
            end = getattr(self.f, 'expected_end', None)
            if end is not None and self.pos < end:      # clean close before the end: resume
                if retries >= 8:
                    raise IOError(f'{self.paths[self.i]} truncated at {self.pos} of {end}')
                retries += 1
                self.f.close()
                self.f = self._open(self.paths[self.i], self.pos)
                continue
            self.f.close()
            self.i += 1
            self.pos = 0
            if self.i >= len(self.paths):
                return 0
            self.f = self._open(self.paths[self.i], 0)


def _parse(blob: bytes):
    """One trace file -> dict of arrays for points near an airport, or None."""
    try:
        raw = gzip.decompress(blob)
    except OSError:
        raw = blob
    d = json.loads(raw)
    tr = d.get('trace') or []
    if not tr:
        return None
    n = len(tr)
    lat = np.fromiter((p[1] for p in tr), float, n)
    lon = np.fromiter((p[2] for p in tr), float, n)
    # cheap box test against all ten airports
    near = (np.abs(lat[:, None] - LAT[None, :]) < _DLAT) & (np.abs(lon[:, None] - LON[None, :]) < _DLON[None, :])
    if not near.any():
        return None
    ts0 = float(d['timestamp'])
    t = ts0 + np.fromiter((p[0] for p in tr), float, n)
    altraw = [p[3] for p in tr]
    ground = np.fromiter((a == 'ground' for a in altraw), bool, n)
    alt = np.fromiter((a if isinstance(a, (int, float)) else np.nan for a in altraw), float, n)
    # callsign: only present in the details object; propagate both directions,
    # restarting at readsb's 'new leg' flag so a turnaround does not inherit the inbound callsign
    flags = np.fromiter((p[6] if isinstance(p[6], int) else 0 for p in tr), np.int16, n)
    cs = [None] * n
    for i, p in enumerate(tr):
        det = p[8] if len(p) > 8 else None
        if det and det.get('flight'):
            cs[i] = det['flight'].strip() or None
    cs_s = pd.Series(cs, dtype=object)
    leg = np.cumsum((flags & 2) > 0)
    cs_ff = cs_s.groupby(leg).ffill()
    cs_bf = cs_s.groupby(leg).bfill()
    squawk = [None] * n
    for i, p in enumerate(tr):
        det = p[8] if len(p) > 8 else None
        if det and det.get('squawk'):
            squawk[i] = det['squawk']
    sq_ff = pd.Series(squawk, dtype=object).groupby(leg).ffill()

    out = []
    for k in np.flatnonzero(near.any(axis=0)):
        m = near[:, k]
        dx = np.radians(lon[m] - LON[k]) * EARTH_R * np.cos(np.radians(LAT[k]))
        dy = np.radians(lat[m] - LAT[k]) * EARTH_R
        r = np.hypot(dx, dy)
        h = alt[m] - ELEV[k]
        keep = (r < R_KEEP_M) & (ground[m] | np.isnan(h) | (h < H_KEEP_FT))
        if not keep.any():
            continue
        idx = np.flatnonzero(m)[keep]
        sel = [tr[i] for i in idx]
        out.append(dict(
            apt=CODES[k], idx=idx, x=dx[keep], y=dy[keep], t=t[idx],
            alt=alt[idx], ground=ground[idx],
            gs=np.array([p[4] if isinstance(p[4], (int, float)) else np.nan for p in sel], float),
            trk=np.array([p[5] if isinstance(p[5], (int, float)) else np.nan for p in sel], float),
            flags=flags[idx],
            vr=np.array([p[7] if isinstance(p[7], (int, float)) else np.nan for p in sel], float),
            src=[(p[9] if len(p) > 9 else None) for p in sel],
            cs_ff=cs_ff.values[idx], cs_bf=cs_bf.values[idx], squawk=sq_ff.values[idx],
        ))
    if not out:
        return None
    return dict(icao=d.get('icao'), reg=d.get('r'), typ=d.get('t'), dbflags=d.get('dbFlags'), parts=out)


def _parse_safe(item):
    name, blob = item
    try:
        return name, _parse(blob), None
    except Exception as e:  # noqa: BLE001 - count, report, carry on
        return name, None, repr(e)


def _members(tf):
    for m in tf:
        if m.isfile() and '/traces/' in m.name and 'trace_full_' in m.name:
            yield m.name, tf.extractfile(m).read()


def stage1(date: str, parts, out_path: Path, workers: int):
    t0 = time.time()
    rd = io.BufferedReader(ConcatReader(parts), buffer_size=1 << 22)
    # ignore_zeros: a stream-mode TarFile ends SILENTLY at an invalid header after offset 0; on 2026-10-10 that cut
    # three June 2026 staging archives after their first 2 GB part. Skip such blocks like GNU tar --ignore-zeros.
    tf = tarfile.open(fileobj=rd, mode='r|', ignore_zeros=True)
    cols = {k: [] for k in ('icao', 'reg', 'typ', 'apt', 't', 'x', 'y', 'alt', 'ground', 'gs', 'trk',
                            'flags', 'vr', 'src', 'cs_ff', 'cs_bf', 'squawk')}
    n_files = n_hit = n_err = 0
    errs = []
    with Pool(workers) as pool:
        for name, res, err in pool.imap_unordered(_parse_safe, _members(tf), chunksize=32):
            n_files += 1
            if err:
                n_err += 1
                errs.append((name, err))
                continue
            if res is None:
                continue
            n_hit += 1
            for p in res['parts']:
                k = len(p['t'])
                cols['icao'].extend([res['icao']] * k)
                cols['reg'].extend([res['reg']] * k)
                cols['typ'].extend([res['typ']] * k)
                cols['apt'].extend([p['apt']] * k)
                for c in ('t', 'x', 'y', 'alt', 'ground', 'gs', 'trk', 'flags', 'vr'):
                    cols[c].append(p[c])
                for c in ('src', 'cs_ff', 'cs_bf', 'squawk'):
                    cols[c].extend(list(p[c]))
    tf.close()
    left = 0                                        # completeness guard: nothing of the parts may stay unread
    while True:
        chunk = rd.read(1 << 22)
        if not chunk:
            break
        left += len(chunk.rstrip(b'\0'))
    if left > (1 << 20):
        raise SystemExit(f'ERROR: tar stream ended with {left:,} non-zero bytes of {parts} unread - truncated read')
    if n_files < 1000:
        raise SystemExit(f'ERROR: only {n_files} trace files read from {parts} - bad or truncated archive')
    if n_err > 0.01 * n_files:
        raise SystemExit(f'ERROR: {n_err} of {n_files} traces failed to parse, e.g. {errs[:3]}')
    arr = {}
    for c in ('t', 'x', 'y', 'alt', 'ground', 'gs', 'trk', 'flags', 'vr'):
        arr[c] = np.concatenate(cols[c]) if cols[c] else np.array([])
    df = pd.DataFrame({
        'icao': pd.Categorical(cols['icao']), 'reg': pd.Categorical(cols['reg']),
        'typ': pd.Categorical(cols['typ']), 'apt': pd.Categorical(cols['apt'], categories=CODES),
        't': arr['t'], 'x': arr['x'].astype(np.float32), 'y': arr['y'].astype(np.float32),
        'alt': arr['alt'].astype(np.float32), 'ground': arr['ground'].astype(bool),
        'gs': arr['gs'].astype(np.float32), 'trk': arr['trk'].astype(np.float32),
        'flags': arr['flags'].astype(np.int8), 'vr': arr['vr'].astype(np.float32),
        'src': pd.Categorical(cols['src']), 'cs_ff': pd.Categorical(cols['cs_ff']),
        'cs_bf': pd.Categorical(cols['cs_bf']), 'squawk': pd.Categorical(cols['squawk']),
    })
    df = df.sort_values(['apt', 'icao', 't'], kind='stable').reset_index(drop=True)
    tmp = out_path.with_suffix('.tmp.parquet')
    df.to_parquet(tmp, compression='zstd', index=False)
    os.replace(tmp, out_path)
    stats = dict(date=date, n_trace_files=n_files, n_traces_near=n_hit, n_parse_errors=n_err,
                 n_points=len(df), n_icao=df.icao.nunique(), seconds=round(time.time() - t0, 1),
                 bytes_read=rd.raw.nread, out_mb=round(out_path.stat().st_size / 1e6, 2))
    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date')
    ap.add_argument('--parts', nargs='*', type=Path)
    ap.add_argument('--raw-dir', type=Path, default=BASE / 'raw')
    ap.add_argument('--out-dir', type=Path, default=BASE)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip-events', action='store_true')
    ap.add_argument('--stream', action='store_true',
                    help='read the tar parts straight from GitHub (no local disk); resumes on errors')
    a = ap.parse_args()

    dot = a.date.replace('-', '.')
    if a.stream:
        from adsb_download import preferred_urls
        meta = a.out_dir / 'meta'
        meta.mkdir(parents=True, exist_ok=True)
        parts = preferred_urls(a.date, meta, 'auto')
        tags = {parts[0].rsplit('/', 2)[-2]}
    else:
        parts = a.parts or sorted(a.raw_dir.glob(f'v{dot}-planes-readsb-*.tar.a?'))
        if not parts:
            raise SystemExit(f'ERROR: no tar parts for {a.date} in {a.raw_dir}; run adsb_download.py or use --stream')
        tags = {p.name.rsplit('.tar.', 1)[0] for p in parts}
        if len(tags) != 1:
            raise SystemExit(f'ERROR: parts of more than one release for {a.date}: {sorted(tags)}; pass --parts')
        for p in parts:
            if not p.exists() or p.stat().st_size == 0:
                raise SystemExit(f'ERROR: missing/empty part {p}')
    if f'v{dot}-' not in next(iter(tags)):
        raise SystemExit(f'ERROR: release {tags} does not belong to {a.date}')

    (a.out_dir / 'points').mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'events').mkdir(parents=True, exist_ok=True)
    pts_path = a.out_dir / 'points' / f'{a.date}.parquet'
    ev_path = a.out_dir / 'events' / f'{a.date}.parquet'

    if pts_path.exists() and not a.force:
        print(f'{a.date}: points exist, stage 1 skipped', file=sys.stderr)
        df = None
    else:
        df, st = stage1(a.date, parts, pts_path, a.workers)
        st['release'] = tags.pop()
        print(json.dumps(st), flush=True)
        with open(a.out_dir / 'points' / 'stage1_stats.jsonl', 'a') as f:
            f.write(json.dumps(st) + '\n')

    if a.skip_events:
        return
    if ev_path.exists() and not a.force:
        print(f'{a.date}: events exist, stage 2 skipped', file=sys.stderr)
        return
    from adsb_events import detect_events
    t0 = time.time()
    if df is None:
        df = pd.read_parquet(pts_path)
    ev = detect_events(df)
    ev.insert(0, 'date', a.date)
    tmp = ev_path.with_suffix('.tmp.parquet')
    ev.to_parquet(tmp, compression='zstd', index=False)
    os.replace(tmp, ev_path)
    print(json.dumps(dict(date=a.date, n_events=len(ev), seconds=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    main()
