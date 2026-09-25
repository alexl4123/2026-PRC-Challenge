"""Stage 2: airport points (stage 1 output) -> one row per detected take-off.

detect_events(points) is called by adsb_extract_day.py; it can also be run alone:
    python experiments/adsb/adsb_events.py 2025-07-16 [--prev-hours 4]
which re-reads cache/adsb_lol/points/<date>.parquet (plus the last --prev-hours of the
previous day's points file if it exists, so early-morning departures keep their
pre-midnight ground segment) and rewrites cache/adsb_lol/events/<date>.parquet.
Only take-offs between 00:00 and 24:00 UTC of <date> (+-30 min) are written.

Definitions (times are unix seconds UTC; x/y metres from the ARP):
  h           baro altitude - field elevation - baro offset (ft); NaN on 'ground' points. The offset is the
              pressure-altitude reading on the runway (per airport-hour, see baro_offsets): without it a
              low-QNH day (EHAM 2026-01-21: +400 ft) makes every take-off roll look airborne.
  air         not flagged 'ground' AND h > 200 ft ("airborne proper": readsb's ground
              flag often drops on the take-off roll while the aircraft is still rolling).
  climb-out   first 'air' point with h < 5,500 ft after a non-air point or a > 10 min gap,
              followed by climbing airborne points (5,500 not 3,000: see CLIMB_START_MAX_FT).
  t_to        take-off estimate = t_air - h_air / climb rate, clipped to
              [t_last_ground, t_air] if the transition was observed (<= 180 s apart,
              to_src='obs'); else extrapolated (to_src='extrap').
  segment     ground-ish points (ground flag, or h < 150 ft and gs < 80 kt) before the
              climb-out, after the previous landing, cut at the last coverage gap > 20 min,
              not earlier than t_to - 3 h, and started at the last rest of >= 10 min
              (the stand dwell when the transponder stayed on).
  rest        maximal run of points within 15 m of the run's first point.
  t_first     first segment point.   gs_first = its reported ground speed.
  t_stat_end  last point of the first rest; t_move = next point (first >= 15 m away);
              stat_s = t_stat_end - t_first; move_gap_s = t_move - t_stat_end.
  parking map per airport: 20 m cells where any aircraft rested >= 15 min on the file's
              day(s); at_park = first rest within 40 m of such a cell.
  push        reverse movement: step course within 45 deg of (heading + 180), step speed
              0.3-7 kt, reported gs <= 6 kt, step <= 40 s, bout total >= 10 m. The last
              such bout starting >= 2 min before t_to. t_push_prev = the point before it.
  t_best      off-block estimate: t_push_prev if a push was found; else t_stat_end if the
              first rest lasted >= 10 s, move_gap_s <= 60 s and (at_park or stat_s >= 600);
              else NaN (start of movement not observed = censored).
  t_roll      first segment point with gs >= 40 kt (take-off roll).
  hold_s      t_to - end of the last rest >= 30 s after t_move (runway queue / hold).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from airports import AIRPORTS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / 'cache' / 'adsb_lol'

GAP_SPLIT_S = 20 * 60
STAND_DWELL_S = 10 * 60
PARK_S = 15 * 60
WINDOW_S = 3 * 3600
REST_M = 15.0
CELL_M = 20.0
# first airborne point of a climb-out may be this high: on some days (e.g. EHAM 2026-01-21 afternoon) low-level
# airborne coverage is missing and the first airborne point is at 3-5,000 ft although the stand/pushback was seen.
CLIMB_START_MAX_FT = 5500.0


def _mode(vals):
    vals = [v for v in vals if isinstance(v, str) and v]
    if not vals:
        return None
    u, c = np.unique(vals, return_counts=True)
    return u[np.argmax(c)]


def _rests(x, y, radius=REST_M):
    """Maximal runs where every point stays within `radius` of the run's first point."""
    out, i, n = [], 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and (x[j + 1] - x[i]) ** 2 + (y[j + 1] - y[i]) ** 2 < radius * radius:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def parking_cells(p: pd.DataFrame) -> dict:
    """{apt: set of (cx, cy)} of 20 m cells where an aircraft rested >= 15 min on the ground."""
    cells = {}
    g = p[p.ground]
    for (apt, _), q in g.groupby(['apt', 'icao'], observed=True, sort=False):
        if len(q) < 3:
            continue
        q = q.sort_values('t')
        t, x, y = q.t.to_numpy(), q.x.to_numpy(float), q.y.to_numpy(float)
        # split at gaps > 20 min: a rest must be continuously observed
        for a, b in _rests(x, y):
            if t[b] - t[a] >= PARK_S and np.diff(t[a:b + 1]).max(initial=0) < GAP_SPLIT_S:
                cells.setdefault(apt, set()).add((int(np.floor(x[a] / CELL_M)), int(np.floor(y[a] / CELL_M))))
    return cells


def _near_park(cells, x, y, r=2):
    if not cells:
        return False
    cx, cy = int(np.floor(x / CELL_M)), int(np.floor(y / CELL_M))
    return any((cx + i, cy + j) in cells for i in range(-r, r + 1) for j in range(-r, r + 1))


def _push_bouts(ts, xs, ys, gss, hs):
    """Reverse-movement bouts: list of (k_start, k_end) point indices (k_start = last point before)."""
    if len(ts) < 2:
        return []
    dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(ts)
    step = np.hypot(dx, dy)
    course = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    ang = np.abs(((course - hs[1:]) + 180) % 360 - 180)
    spd = step / np.maximum(dt, 0.1) * 1.94384
    gmax = np.fmax(np.nan_to_num(gss[1:], nan=0), np.nan_to_num(gss[:-1], nan=0))
    rev = (ang > 135) & (step >= 1.5) & (dt <= 40) & (spd >= 0.3) & (spd <= 7) & (gmax <= 6) & np.isfinite(hs[1:])
    bouts, k = [], 0
    while k < len(rev):
        if rev[k]:
            j = k
            while j + 1 < len(rev) and rev[j + 1]:
                j += 1
            if step[k:j + 1].sum() >= 10:
                bouts.append((k, j + 1))
            k = j + 1
        else:
            k += 1
    return bouts


def baro_offsets(p: pd.DataFrame) -> pd.Series:
    """Pressure-altitude reading on the runway minus field elevation (ft), per (apt, UTC hour).

    alt_baro is PRESSURE altitude (1013.25 hPa datum), so on the ground it reads about
    (1013.25 - QNH) * 27 ft above the field: +400 ft at EHAM on 2026-01-21 (low pressure), which
    made every take-off roll look 'airborne' once readsb dropped the ground flag (~45 kt) and broke
    the detector. Estimated from the data: at each ground-flag flip of one aircraft (<= 30 s apart,
    non-ground side at 30-160 kt, i.e. still on the runway), the non-ground point's altitude.
    Hour median with >= 3 flips, else the airport-day median, else 0.
    """
    q = p[['apt', 'icao', 't', 'ground', 'alt', 'gs']].sort_values(['apt', 'icao', 't'])
    same = (q.icao.to_numpy()[1:] == q.icao.to_numpy()[:-1]) & (q.apt.to_numpy()[1:] == q.apt.to_numpy()[:-1])
    g = q.ground.to_numpy()
    flip = same & (g[1:] != g[:-1]) & (np.diff(q.t.to_numpy()) <= 30)
    alt, gs = q.alt.to_numpy(float), q.gs.to_numpy(float)
    ng_alt = np.where(g[1:], alt[:-1], alt[1:])
    ng_gs = np.where(g[1:], gs[:-1], gs[1:])
    ok = flip & np.isfinite(ng_alt) & (ng_gs >= 30) & (ng_gs <= 160)
    apt = q.apt.astype(str).to_numpy()[1:][ok]
    elev = np.array([AIRPORTS[a][2] for a in apt], float)
    d = pd.DataFrame({'apt': apt, 'hr': (q.t.to_numpy()[1:][ok] // 3600).astype(np.int64),
                      'off': ng_alt[ok] - elev})
    hr = d.groupby(['apt', 'hr']).off.agg(['median', 'size'])
    day = d.groupby('apt').off.median()
    return hr[hr['size'] >= 3]['median'], day


def _offsets_for(g: pd.DataFrame, apt: str, off_hr, off_day) -> np.ndarray:
    hrs = (g.t.to_numpy() // 3600).astype(np.int64)
    base = float(off_day.get(apt, 0.0)) if off_day is not None else 0.0
    if off_hr is None:
        return np.full(len(hrs), base)
    lut = {h: v for (a, h), v in off_hr.items() if a == apt}
    return np.array([lut.get(h, base) for h in hrs], float)


def _events_one(g: pd.DataFrame, apt: str, cells: set, boff=None):
    elev = AIRPORTS[apt][2]
    t = g.t.to_numpy()
    x = g.x.to_numpy(float)
    y = g.y.to_numpy(float)
    gs = g.gs.to_numpy(float)
    hdg = g.trk.to_numpy(float)
    gnd = g.ground.to_numpy()
    h = g.alt.to_numpy(float) - elev - (boff if boff is not None else 0.0)
    vr = g.vr.to_numpy(float)
    src = g.src.astype(object).to_numpy()
    cs = g.cs_bf.astype(object).where(g.cs_bf.notna(), g.cs_ff.astype(object)).to_numpy()
    sq = g.squawk.astype(object).to_numpy()
    n = len(t)
    air = (~gnd) & (h > 200)
    groundish = gnd | ((h < 150) & (np.nan_to_num(gs, nan=0) < 80))
    out = []
    i = 0
    while i < n:
        if not (air[i] and h[i] < CLIMB_START_MAX_FT):
            i += 1
            continue
        if not ((i == 0) or (not air[i - 1]) or (t[i] - t[i - 1] > 600)):
            i += 1
            continue
        fw = np.flatnonzero(air[i:] & (t[i:] < t[i] + 300))[:12] + i
        if len(fw) < 2 or np.nanmax(h[fw]) < h[i] + 300:
            i += 1
            continue
        # ---------------- take-off time
        vri = vr[i] if np.isfinite(vr[i]) and vr[i] > 300 else np.nan
        k_prev = i - 1
        obs = k_prev >= 0 and (not air[k_prev]) and t[i] - t[k_prev] <= 180   # not air = ground or h <= 200
        if obs:
            rate = vri if np.isfinite(vri) else 1500.0
            t_to = min(max(t[i] - max(h[i], 0) / rate * 60, t[k_prev]), t[i])
        else:
            rate = vri if np.isfinite(vri) else 2000.0
            t_to = t[i] - max(h[i], 0) / max(rate, 800) * 60
        # ---------------- ground segment
        prev_air = np.flatnonzero(air[:i])
        lo = prev_air[-1] + 1 if len(prev_air) else 0
        seg = np.arange(lo, i)
        seg = seg[groundish[seg] & (t[seg] >= t_to - WINDOW_S)]
        if len(seg):
            cut = np.flatnonzero(np.diff(t[seg]) > GAP_SPLIT_S)
            if len(cut):
                seg = seg[cut[-1] + 1:]
        cs_ev = _mode(cs[i:fw[-1] + 1])
        if cs_ev is None and len(seg):
            cs_ev = _mode(cs[seg])
        ev = dict(apt=apt, icao=g.icao.iat[0], reg=g.reg.iat[0], typ=g.typ.iat[0], cs=cs_ev,
                  squawk=_mode(sq[i:fw[-1] + 1]), t_to=t_to, to_src='obs' if obs else 'extrap',
                  t_air=t[i], h_air=h[i], t_last_ground=t[k_prev] if obs else np.nan,
                  x_air=x[i], y_air=y[i], n_seg=len(seg))
        if len(seg):
            ts, xs, ys, gss, hs = t[seg], x[seg], y[seg], gs[seg], hdg[seg]
            rests = _rests(xs, ys)
            long_r = [(a, b) for a, b in rests if ts[b] - ts[a] >= STAND_DWELL_S]
            dwell = 0.0
            if long_r:
                a, b = long_r[-1]
                dwell = ts[b] - ts[a]
                ts, xs, ys, gss, hs = ts[a:], xs[a:], ys[a:], gss[a:], hs[a:]
                seg = seg[a:]
                rests = _rests(xs, ys)
            a0, b0 = rests[0]
            t_first, t_stat_end = ts[0], ts[b0]
            t_move = ts[b0 + 1] if b0 + 1 < len(ts) else np.nan
            stat_s = t_stat_end - t_first
            at_park = _near_park(cells, xs[0], ys[0])
            bouts = [(ka, kb) for ka, kb in _push_bouts(ts, xs, ys, gss, hs) if ts[ka] <= t_to - 120]
            n_push = len(bouts)
            t_push_prev = t_push = push_m = np.nan
            if bouts:
                ka, kb = bouts[-1]
                t_push_prev, t_push = ts[ka], ts[ka + 1]
                push_m = float(np.hypot(np.diff(xs[ka:kb + 1]), np.diff(ys[ka:kb + 1])).sum())
                at_park = at_park or _near_park(cells, xs[ka], ys[ka])
            move_gap = t_move - t_stat_end if np.isfinite(t_move) else np.nan
            if np.isfinite(t_push_prev):
                t_best, best_src = t_push_prev, 'push'
            elif stat_s >= 10 and np.isfinite(move_gap) and move_gap <= 60 and (at_park or stat_s >= 600):
                t_best, best_src = t_stat_end, 'rest'
            else:
                t_best, best_src = np.nan, 'none'
            dt = np.diff(ts)
            step = np.hypot(np.diff(xs), np.diff(ys))
            t0m = t_move if np.isfinite(t_move) else ts[0]
            path = float(step[ts[1:] > t0m].sum()) if len(step) else 0.0
            stops = [(ts[a], ts[b]) for a, b in rests if ts[b] - ts[a] >= 30 and ts[a] >= t0m]
            roll = np.flatnonzero(gss >= 40)
            ev.update(
                t_first=t_first, gs_first=gss[0], x_first=xs[0], y_first=ys[0], hdg_first=hs[0],
                t_stat_end=t_stat_end, stat_s=stat_s, t_move=t_move, move_gap_s=move_gap,
                dwell_s=dwell, at_park=bool(at_park), n_push=n_push,
                t_push_prev=t_push_prev, t_push=t_push, push_m=push_m,
                t_best=t_best, best_src=best_src,
                t_last=ts[-1], n_pts=len(ts), max_gap_s=float(dt.max()) if len(dt) else np.nan,
                cov_s_per_pt=float((ts[-1] - ts[0]) / max(len(ts) - 1, 1)),
                gap_last_to_air=t[i] - ts[-1],
                path_m=path, n_stops=len(stops), stop_s=float(sum(b - a for a, b in stops)),
                t_last_stop_end=stops[-1][1] if stops else np.nan,
                t_roll=ts[roll[0]] if len(roll) else np.nan,
                gs_med_moving=float(np.nanmedian(gss[gss > 3])) if (gss > 3).any() else np.nan,
                mlat_frac=float(np.mean(src[seg] == 'mlat')),
            )
        out.append(ev)
        i = fw[-1] + 1
    return out


def detect_events(p: pd.DataFrame) -> pd.DataFrame:
    cells = parking_cells(p)
    off_hr, off_day = baro_offsets(p)
    rows = []
    for (apt, icao), g in p.groupby(['apt', 'icao'], observed=True, sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values('t')
        rows.extend(_events_one(g, apt, cells.get(apt, set()), _offsets_for(g, apt, off_hr, off_day)))
    ev = pd.DataFrame(rows)
    if len(ev):
        ev['baro_off_ft'] = [float(off_day.get(a, 0.0)) for a in ev.apt]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date')
    ap.add_argument('--base', type=Path, default=BASE)
    ap.add_argument('--prev-hours', type=float, default=4.0)
    a = ap.parse_args()
    t0 = time.time()
    p = pd.read_parquet(a.base / 'points' / f'{a.date}.parquet')
    day0 = pd.Timestamp(a.date, tz='UTC').timestamp()
    prev = a.base / 'points' / f'{(pd.Timestamp(a.date) - pd.Timedelta(days=1)).date()}.parquet'
    if prev.exists() and a.prev_hours > 0:
        q = pd.read_parquet(prev, filters=[('t', '>=', day0 - a.prev_hours * 3600)])
        p = pd.concat([q.astype({c: object for c in q.select_dtypes('category')}),
                       p.astype({c: object for c in p.select_dtypes('category')})], ignore_index=True)
        for c in ('icao', 'reg', 'typ', 'apt', 'src', 'cs_ff', 'cs_bf', 'squawk'):
            p[c] = p[c].astype('category')
    ev = detect_events(p)
    ev = ev[(ev.t_to >= day0 - 1800) & (ev.t_to < day0 + 86400 + 1800)]
    ev.insert(0, 'date', a.date)
    (a.base / 'events').mkdir(parents=True, exist_ok=True)
    out = a.base / 'events' / f'{a.date}.parquet'
    tmp = out.with_suffix('.tmp.parquet')
    ev.to_parquet(tmp, compression='zstd', index=False)
    os.replace(tmp, out)
    print(f'{a.date}: {len(ev)} take-off events in {time.time() - t0:.1f} s -> {out}')


if __name__ == '__main__':
    main()
