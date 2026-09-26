"""ADS-B v2, per-day stage: departure ground runs, reappearance after coverage gaps, calibration samples, visibility grid.

    python experiments/adsb/adsb_events_v2.py 2025-07-16 [--prev-hours 8]

Reads (never writes) the production files of adsb_events.py:
    cache/adsb_lol/points/<date>.parquet  (+ the last --prev-hours of the previous day's points, if present)
    cache/adsb_lol/events/<date>.parquet  (take-off events; the v2 rows are keyed like them by apt, icao, t_air)
and writes three NEW files (atomic rename; re-running overwrites them):
    cache/adsb_lol/events_v2/<date>.parquet  one row per take-off event: the v2 departure-run description
    cache/adsb_lol/calib_v2/<date>.parquet   simulated-censoring samples from departures whose off-block WAS observed
    cache/adsb_lol/grid_v2/<date>.parquet    per airport 50 m cells: distinct aircraft seen on the ground that day
    cache/adsb_lol/cong_v2/<date>.parquet    per airport and UTC minute: aircraft seen on the ground, moving (3-50 kt),
                                             and stopped away from parking cells (queues / holds) -- surface congestion
Nothing here uses labels (BLOCK / TAXITIME); the files are safe for the ranking months.

Why (measured 2026-09-26, reports/2026-09-26_adsb_v2.md): where the stand is not seen, adsb_events.py's ground segment can
start with the ARRIVAL of the same airframe (LFPG: one arrival roll-out point glued to a departure that has no ground
point at all -> 'first point' 66 min before off-block), and the first departure point is typically hundreds of metres
from the stand, already taxiing.

Definitions (unix seconds; x/y metres from the ARP; h, air, groundish exactly as in adsb_events.py, incl. baro offset):
  landing      last airborne point before the take-off climb-out (the airframe's previous arrival), if within 8 h.
  G            groundish points of the airframe after the landing (or after t_air - 8 h) and before the climb-out.
  final run    the last part of G whose consecutive gaps are all <= RUN_GAP_S (300 s); then, like adsb_events.py, started
               at its last rest >= 10 min (stand dwell) if one exists.
  connected    the final run ends <= CONNECT_S (900 s) before the first airborne point. If not, the 'run' belongs to the
               arrival side (the departure had no ground point) -> no departure run (the LFPG failure mode above).
  R            first point of a connected final run = reappearance point (t_R, x_R, y_R, gs_R, hdg_R).
  gap_R        t_R - previous groundish point of the airframe (or landing time); NaN if none.
  A            last groundish point before R (end of what was seen of the arrival side): t_A, x_A, y_A, gs_A;
               A_rest_s = seconds at rest at the end of that side (>= 60 s at a parking cell ~ in-block seen).
  rest_R_s     seconds at rest at the start of the run (first rest, 15 m radius).
  roll         first run point with gs >= 40 kt (take-off roll start); d_R_roll = |R - roll|.
Calibration samples (label-free): for events whose off-block WAS observed by adsb_events.py (best_src push, or a stand-rest
end), every run point after t_best (thinned to >= 15 s apart) gives (d = distance from the off-block position,
e = t_k - t_best, gs_k, d_roll_k). adsb_build_features_v2.py fits e = f(d, moving) per airport on the training year and
applies it to the reappearance point of censored flights (back-calculation).
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
from adsb_events import _rests, _offsets_for, baro_offsets, parking_cells, _near_park  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / 'cache' / 'adsb_lol'

LOOKBACK_S = 8 * 3600
RUN_GAP_S = 300.0
CONNECT_S = 900.0
STAND_DWELL_S = 600.0
CALIB_STEP_S = 15.0
GRID_M = 50.0


def _v2_one(g, apt, evs, cells, boff):
    """All take-off events of one airframe at one airport -> (list of v2 rows, list of calibration sample frames)."""
    elev = AIRPORTS[apt][2]
    t = g.t.to_numpy()
    x = g.x.to_numpy(float)
    y = g.y.to_numpy(float)
    gs = g.gs.to_numpy(float)
    hdg = g.trk.to_numpy(float)
    gnd = g.ground.to_numpy()
    h = g.alt.to_numpy(float) - elev - boff
    air = (~gnd) & (h > 200)
    groundish = gnd | ((h < 150) & (np.nan_to_num(gs, nan=0) < 80))
    rows, calib = [], []
    for e in evs.itertuples(index=False):
        i = int(np.searchsorted(t, e.t_air))
        r = dict(apt=apt, icao=e.icao, t_air=e.t_air)
        prev_air = np.flatnonzero(air[:i])
        lo = prev_air[-1] + 1 if len(prev_air) else 0
        t_land = t[prev_air[-1]] if len(prev_air) and t[prev_air[-1]] >= e.t_air - LOOKBACK_S else np.nan
        r['t_land'] = t_land
        G = np.arange(lo, i)
        G = G[groundish[G] & (t[G] >= e.t_air - LOOKBACK_S)]
        r['n_g'] = len(G)
        if len(G) == 0:
            r['connected'] = 0
            rows.append(r)
            continue
        tg = t[G]
        cuts = np.flatnonzero(np.diff(tg) > RUN_GAP_S)
        r0 = cuts[-1] + 1 if len(cuts) else 0
        run = G[r0:]
        stand_dwell = 0.0
        rr = _rests(x[run], y[run])
        long_r = [(a, b) for a, b in rr if t[run[b]] - t[run[a]] >= STAND_DWELL_S]
        if long_r:
            a, b = long_r[-1]
            stand_dwell = t[run[b]] - t[run[a]]
            r0 = r0 + a
            run = run[a:]
        gap_air = e.t_air - t[run[-1]]
        r['gap_run_air'] = gap_air
        r['connected'] = int(gap_air <= CONNECT_S)
        before = G[:r0]
        if not r['connected']:
            # the whole of G is the arrival side (or stray points): describe it, no departure run
            r['n_arr_side'] = len(G)
            r['t_A'] = t[G[-1]]
            r['x_A'], r['y_A'], r['gs_A'] = x[G[-1]], y[G[-1]], gs[G[-1]]
            rows.append(r)
            continue
        R = run[0]
        r.update(t_R=t[R], x_R=x[R], y_R=y[R], gs_R=gs[R], hdg_R=hdg[R], n_run=len(run), stand_dwell_s=stand_dwell,
                 R_at_park=bool(_near_park(cells, x[R], y[R])))
        rr = _rests(x[run], y[run])
        a0, b0 = rr[0]
        r['rest_R_s'] = t[run[b0]] - t[run[a0]]
        r['t_move_R'] = t[run[b0 + 1]] if b0 + 1 < len(run) else np.nan
        roll = run[np.flatnonzero(gs[run] >= 40)]
        if len(roll):
            r.update(t_roll2=t[roll[0]], x_roll=x[roll[0]], y_roll=y[roll[0]],
                     d_R_roll=float(np.hypot(x[R] - x[roll[0]], y[R] - y[roll[0]])))
        if len(run) > 1:
            st = np.hypot(np.diff(x[run]), np.diff(y[run]))
            r['path_run_m'] = float(st.sum())
            r['max_gap_run_s'] = float(np.diff(t[run]).max())
        if len(before):
            A = before[-1]
            r.update(gap_R=t[R] - t[A], t_A=t[A], x_A=x[A], y_A=y[A], gs_A=gs[A], n_arr_side=len(before),
                     d_R_A=float(np.hypot(x[R] - x[A], y[R] - y[A])))
            rb = _rests(x[before], y[before])
            a1, b1 = rb[-1]
            r['A_rest_s'] = t[before[b1]] - t[before[a1]]
            r['A_at_park'] = bool(_near_park(cells, x[A], y[A]))
        elif np.isfinite(t_land):
            r['gap_R'] = t[R] - t_land
        # ---------------- calibration samples from observed off-blocks (label-free)
        tb = getattr(e, 't_best', np.nan)
        if np.isfinite(tb) and getattr(e, 'best_src', 'none') in ('push', 'rest'):
            k0 = int(np.searchsorted(t, tb))
            if k0 < len(t) and t[k0] == tb:
                ks = run[(t[run] >= tb) & (t[run] <= e.t_air)]
                if len(ks) >= 3:
                    keep, last = [], -1e18
                    for k in ks:
                        if t[k] - last >= CALIB_STEP_S:
                            keep.append(k)
                            last = t[k]
                    ks = np.array(keep)
                    xr, yr = (x[roll[0]], y[roll[0]]) if len(roll) else (np.nan, np.nan)
                    calib.append(pd.DataFrame({
                        'apt': apt, 'icao': e.icao, 't_air': e.t_air, 'src': e.best_src,
                        'e_s': t[ks] - tb, 'd_m': np.hypot(x[ks] - x[k0], y[ks] - y[k0]),
                        'gs': gs[ks], 'd_roll_m': np.hypot(x[ks] - xr, y[ks] - yr),
                        'x': x[ks], 'y': y[ks], 'x0': x[k0], 'y0': y[k0]}))
        rows.append(r)
    return rows, calib


def visibility_grid(p: pd.DataFrame) -> pd.DataFrame:
    """Per airport 50 m cells: distinct aircraft and points seen on the ground (label-free)."""
    g = p[p.ground]
    cx = np.floor(g.x.to_numpy(float) / GRID_M).astype(np.int32)
    cy = np.floor(g.y.to_numpy(float) / GRID_M).astype(np.int32)
    d = pd.DataFrame({'apt': g.apt.astype(str).to_numpy(), 'cx': cx, 'cy': cy, 'icao': g.icao.astype(str).to_numpy()})
    out = d.groupby(['apt', 'cx', 'cy']).agg(n_ac=('icao', 'nunique'), n_pts=('icao', 'size')).reset_index()
    out['n_ac'] = out.n_ac.astype(np.int32)
    out['n_pts'] = out.n_pts.astype(np.int32)
    return out


def congestion(p: pd.DataFrame, cells: dict, t_min: float) -> pd.DataFrame:
    """Per (apt, UTC minute): distinct aircraft on the ground, moving 3-50 kt, and stopped (< 1 kt) away from parking
    cells. Measures the real surface traffic (other flights' actual off-blocks), label-free."""
    g = p[p.ground & (p.t >= t_min)]
    apt = g.apt.astype(str).to_numpy()
    x, y, gs = g.x.to_numpy(float), g.y.to_numpy(float), g.gs.to_numpy(float)
    mv = (gs >= 3) & (gs < 50)
    stop = np.nan_to_num(gs, nan=99) < 1
    away = np.zeros(len(g), bool)
    for a in np.unique(apt):
        ix = np.flatnonzero((apt == a) & stop)
        cs = cells.get(a, set())
        away[ix] = [not _near_park(cs, xx, yy) for xx, yy in zip(x[ix], y[ix])]
    d = pd.DataFrame({'apt': apt, 'minute': (g.t.to_numpy() // 60).astype(np.int64), 'icao': g.icao.astype(str).to_numpy(),
                      'mv': mv, 'hold': stop & away})
    out = d.groupby(['apt', 'minute']).agg(n_gnd=('icao', 'nunique')).reset_index()
    for c in ('mv', 'hold'):
        k = d[d[c]].groupby(['apt', 'minute']).icao.nunique().rename(f'n_{c}')
        out = out.merge(k, on=['apt', 'minute'], how='left')
    for c in ('n_gnd', 'n_mv', 'n_hold'):
        out[c] = out[c].fillna(0).astype(np.int16)
    return out


def run_day(date: str, base: Path = BASE, prev_hours: float = 8.0):
    p = pd.read_parquet(base / 'points' / f'{date}.parquet')
    day0 = pd.Timestamp(date, tz='UTC').timestamp()
    prev = base / 'points' / f'{(pd.Timestamp(date) - pd.Timedelta(days=1)).date()}.parquet'
    if prev.exists() and prev_hours > 0:
        q = pd.read_parquet(prev, filters=[('t', '>=', day0 - prev_hours * 3600)])
        p = pd.concat([q.astype({c: object for c in q.select_dtypes('category')}),
                       p.astype({c: object for c in p.select_dtypes('category')})], ignore_index=True)
        for c in ('icao', 'reg', 'typ', 'apt', 'src', 'cs_ff', 'cs_bf', 'squawk'):
            if c in p:
                p[c] = p[c].astype('category')
    evf = base / 'events' / f'{date}.parquet'
    if not evf.exists():
        raise SystemExit(f'ERROR: {evf} missing (run adsb_events.py {date} first)')
    ev = pd.read_parquet(evf)
    ev['apt'] = ev.apt.astype(str)
    ev['icao'] = ev.icao.astype(str)
    cells = parking_cells(p)
    off_hr, off_day = baro_offsets(p)
    p['apt_s'] = p.apt.astype(str)
    p['icao_s'] = p.icao.astype(str)
    evg = {k: v for k, v in ev.groupby(['apt', 'icao'], sort=False)}
    rows, calib = [], []
    for (apt, icao), g in p.groupby(['apt_s', 'icao_s'], sort=False):
        evs = evg.get((apt, icao))
        if evs is None or len(g) < 2:
            continue
        g = g.sort_values('t', kind='stable')
        rr, cc = _v2_one(g, apt, evs, cells.get(apt, set()), _offsets_for(g, apt, off_hr, off_day))
        rows.extend(rr)
        calib.extend(cc)
    v2 = pd.DataFrame(rows)
    cal = pd.concat(calib, ignore_index=True) if calib else pd.DataFrame(
        columns=['apt', 'icao', 't_air', 'src', 'e_s', 'd_m', 'gs', 'd_roll_m', 'x', 'y', 'x0', 'y0'])
    grid = visibility_grid(p[p.t >= day0])
    cong = congestion(p, cells, day0 - 3600)
    for sub, df in (('events_v2', v2), ('calib_v2', cal), ('grid_v2', grid), ('cong_v2', cong)):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
        out = d / f'{date}.parquet'
        tmp = out.with_suffix('.tmp.parquet')
        df.to_parquet(tmp, compression='zstd', index=False)
        os.replace(tmp, out)
    return len(ev), len(v2), len(cal), len(grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dates', nargs='+')
    ap.add_argument('--base', type=Path, default=BASE)
    ap.add_argument('--prev-hours', type=float, default=8.0)
    a = ap.parse_args()
    for d in a.dates:
        t0 = time.time()
        n_ev, n_v2, n_cal, n_grid = run_day(d, a.base, a.prev_hours)
        if n_v2 != n_ev:
            print(f'WARNING {d}: {n_ev} events but {n_v2} v2 rows', file=sys.stderr)
        print(f'{d}: events {n_ev}, v2 rows {n_v2}, calib samples {n_cal}, grid cells {n_grid} in {time.time() - t0:.1f} s',
              flush=True)


if __name__ == '__main__':
    main()
