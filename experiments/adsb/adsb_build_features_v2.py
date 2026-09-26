"""ADS-B v2 feature table: extra columns for flights whose off-block is not observed (stage-2 style, label-free).

    python experiments/adsb/adsb_build_features_v2.py [--workers 8] [--redo] [--which training ranking]
           [--days D1 D2 ...]      # local test: only these days (default: every day with points, as in production)

Needs the production outputs of adsb_extract_day.py / adsb_events.py (points/, events/) and stands.parquet
(adsb_build_features.py). Never modifies them. Steps:
 1. adsb_events_v2.py for every day with points+events (parallel; skipped when events_v2/<date> exists, unless --redo)
 2. back-calculation curves (label-free): from calib_v2 samples of the TRAINING year (flights whose off-block was
    observed by ADS-B), per airport the median elapsed time since off-block e as a function of the straight-line
    distance d from the off-block position, separately for moving (gs > 3 kt) and stopped points; and the same as a
    function of the distance to the take-off roll start. Written to cache/adsb_lol/v2_calib_curves.parquet.
 3. per month (as adsb_build_features.py): match events to departures with the production matcher, join the v2 rows,
    extended stand positions, the day's stand visibility, the back-calculated off-block estimates.
 4. writes cache/adsb_lol/features_v2_<which>.parquet keyed by MVT_ID_mvt (unique, asserted), all departures of the
    processed months, adsb2_* columns (NaN where not applicable).

Columns ('off_X' = MVT_TIME - X as in production):
  adsb2_run              1 if a departure ground run connected to the take-off exists (0: matched but no departure
                         ground point, e.g. only the arrival roll-out was seen; NaN: not matched / day missing)
  adsb2_off_R            MVT - reappearance point (first point of the connected final run; = production off_first
                         unless the production segment started on the arrival side)
  adsb2_contam           1 if the production first point (adsb_off_first) lies > 600 s before t_R (arrival-side start)
  adsb2_gap_R            seconds unseen before R (since the previous ground point of the airframe, or the landing)
  adsb2_arr_seen         1 if the airframe's landing at this airport within 8 h before is seen
  adsb2_off_land         MVT - that landing time (turnaround span)
  adsb2_A_gs, adsb2_A_rest_s, adsb2_d_R_A   arrival-side end: speed, rest at the end, distance to R
  adsb2_gs_R, adsb2_rest_R_s, adsb2_R_at_park, adsb2_x_R, adsb2_y_R, adsb2_hdg_R   reappearance point description
  adsb2_d_R_roll         distance from R to the take-off roll start; adsb2_path_run_m, adsb2_max_gap_run_s
  adsb2_stand_src        position of the departure stand: 1 production table, 2 local table (>= 1 push-from-rest),
                         3 interpolated between numbered neighbours, 4 nearest numbered neighbour, 0 unknown
  adsb2_d_R_stand        distance from R to that stand position
  adsb2_vis_stand        distinct aircraft seen on the ground within ~75 m of the stand position that day
  adsb2_vis_R            distinct aircraft seen that day in R's 50 m cell (is this a well-covered place?)
  adsb2_apt_day_vis      share of the airport's known stands with >= 2 aircraft seen that day
  adsb2_bc_e_stand       back-calculated elapsed time off-block -> R from d(stand, R)   (estimator a)
  adsb2_off_bc_stand     MVT - (t_R - adsb2_bc_e_stand)
  adsb2_bc_e_roll        elapsed time off-block -> R from d(R, roll start) (no stand needed)
  adsb2_off_bc_roll      MVT - (t_R - adsb2_bc_e_roll)
  adsb2_off_best2        production adsb_off_best if present, else off_bc_stand, else off_bc_roll
  adsb2_best2_src        0 none, 1 production t_best, 2 back-calc from stand, 3 back-calc from roll
  adsb2_cong_mv15        surface congestion (cong_v2): mean number of aircraft seen moving on the ground (3-50 kt) per
                         minute over [MVT-15 min, MVT); adsb2_cong_mv30_15 the same over [MVT-30, MVT-15)
  adsb2_cong_hold10      mean number stopped away from parking cells (queues, holds) over [MVT-10, MVT)
  adsb2_cong_gnd15       mean number seen on the ground at all over [MVT-15, MVT)
  adsb2_cong_mv_day      airport-day mean of the per-minute moving count (coverage context for the counts above)
Leakage: no label and no BLOCK is read; stands / curves come from ADS-B positions of the training year only.
"""
import argparse
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adsb_days import days as all_days  # noqa: E402
from adsb_match_measure import load_deps, match  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / 'cache' / 'adsb_lol'
PY = sys.executable
D_EDGES = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1300, 1700, 2200, 3000, 4000, 5500, 8000, 20000.])
MIN_BIN = 25
V2COLS = ['t_land', 'n_g', 'connected', 'gap_run_air', 't_R', 'x_R', 'y_R', 'gs_R', 'hdg_R', 'n_run', 'stand_dwell_s',
          'R_at_park', 'rest_R_s', 't_move_R', 't_roll2', 'x_roll', 'y_roll', 'd_R_roll', 'path_run_m', 'max_gap_run_s',
          'gap_R', 't_A', 'x_A', 'y_A', 'gs_A', 'n_arr_side', 'd_R_A', 'A_rest_s', 'A_at_park']


def _v2_day(day):
    import subprocess
    out = BASE / 'events_v2' / f'{day}.parquet'
    r = subprocess.run([PY, str(Path(__file__).resolve().parent / 'adsb_events_v2.py'), day],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        return day, f'FAILED: {r.stderr[-400:]}'
    return day, r.stdout.strip()


# ----------------------------------------------------------------------------------------------- calibration curves
def fit_curves(days) -> pd.DataFrame:
    """Median elapsed time since the ADS-B off-block per (apt, kind, moving, distance bin); label-free."""
    parts = []
    for d in days:
        f = BASE / 'calib_v2' / f'{d}.parquet'
        if f.exists():
            parts.append(pd.read_parquet(f, columns=['apt', 'e_s', 'd_m', 'gs', 'd_roll_m']))
    if not parts:
        raise SystemExit('ERROR: no calib_v2 samples for the training days')
    c = pd.concat(parts, ignore_index=True)
    c['apt'] = c.apt.astype(str)
    c['mov'] = (c.gs.fillna(0) > 3).astype(np.int8)
    rows = []
    for kind, col in (('stand', 'd_m'), ('roll', 'd_roll_m')):
        q = c[c[col].notna()].copy()
        q['bin'] = np.clip(np.searchsorted(D_EDGES, q[col].to_numpy(), side='right') - 1, 0, len(D_EDGES) - 2)
        for apt_key, qq in list(q.groupby('apt')) + [('ALL', q)]:
            s = qq.groupby(['mov', 'bin']).agg(e_med=('e_s', 'median'), d_med=(col, 'median'), n=('e_s', 'size'))
            s = s.reset_index()
            s['apt'] = apt_key
            s['kind'] = kind
            rows.append(s)
    cur = pd.concat(rows, ignore_index=True)
    return cur


def apply_curve(cur, kind, apt, mov, d):
    """Piecewise-linear interpolation of the median elapsed time over distance; airport curve if it has >= MIN_BIN
    samples in the bins used, else the pooled ('ALL') curve. Monotone non-decreasing in d (cumulative max)."""
    out = np.full(len(d), np.nan)
    apt = np.asarray(apt).astype(str)
    mov = np.asarray(mov)
    d = np.asarray(d, float)
    fin = np.isfinite(d)
    for a in np.unique(apt[fin]):
        for mv in (0, 1):
            ix = np.flatnonzero(fin & (apt == a) & (mov == mv))
            if not len(ix):
                continue
            s = cur[(cur.kind == kind) & (cur.apt == a) & (cur.mov == mv) & (cur.n >= MIN_BIN)].sort_values('bin')
            if len(s) < 4:
                s = cur[(cur.kind == kind) & (cur.apt == 'ALL') & (cur.mov == mv) & (cur.n >= MIN_BIN)].sort_values('bin')
            if len(s) == 0:
                continue
            out[ix] = np.interp(d[ix], s.d_med.to_numpy(), np.maximum.accumulate(s.e_med.to_numpy()))
    return out


# ----------------------------------------------------------------------------------------------- stand positions
_NAME = re.compile(r'^([A-Za-z]*)(\d+)([A-Za-z]*)$')


def extend_stands(prod: pd.DataFrame, local: pd.DataFrame, wanted: pd.DataFrame) -> pd.DataFrame:
    """Positions for (APT, STAND): 1 production table, 2 local table (n >= 1), 3 interpolated between the nearest
    lower/upper numbered stands of the same letter prefix (numbers <= 6 apart), 4 nearest such neighbour (<= 3 apart)."""
    known = {}
    for src, tab in ((2, local), (1, prod)):          # production overrides local
        for r in tab.itertuples(index=False):
            known[(r.apt, r.stand)] = (r.sx, r.sy, src)
    by_prefix = {}
    for (a, s), (sx, sy, _) in known.items():
        m = _NAME.match(s)
        if m:
            by_prefix.setdefault((a, m.group(1).upper()), []).append((int(m.group(2)), sx, sy))
    rows = []
    for a, s in wanted[['apt', 'stand']].drop_duplicates().itertuples(index=False):
        if (a, s) in known:
            sx, sy, src = known[(a, s)]
            rows.append((a, s, sx, sy, src))
            continue
        m = _NAME.match(s) if isinstance(s, str) else None
        if not m:
            continue
        n = int(m.group(2))
        cand = by_prefix.get((a, m.group(1).upper()), [])
        lo = [c for c in cand if n - 6 <= c[0] < n]
        hi = [c for c in cand if n < c[0] <= n + 6]
        if lo and hi:
            l, h = max(lo), min(hi)
            w = (n - l[0]) / (h[0] - l[0])
            rows.append((a, s, l[1] + w * (h[1] - l[1]), l[2] + w * (h[2] - l[2]), 3))
        else:
            near = sorted([c for c in cand if abs(c[0] - n) <= 3], key=lambda c: abs(c[0] - n))
            if near:
                rows.append((a, s, near[0][1], near[0][2], 4))
    return pd.DataFrame(rows, columns=['apt', 'stand', 'sx', 'sy', 'stand_src'])


# ----------------------------------------------------------------------------------------------- per month
def month_table(dep: pd.DataFrame, evd: list) -> pd.DataFrame:
    ev = pd.concat([pd.read_parquet(BASE / 'events' / f'{d}.parquet') for d in evd], ignore_index=True)
    ev = ev.drop_duplicates(['apt', 'icao', 't_air']).reset_index(drop=True)
    ev['ev_id'] = np.arange(len(ev))
    v2 = []
    for d in evd:
        f = BASE / 'events_v2' / f'{d}.parquet'
        if f.exists():
            v2.append(pd.read_parquet(f))
    v2 = pd.concat(v2, ignore_index=True).drop_duplicates(['apt', 'icao', 't_air'])
    m = match(dep, ev)
    m['_icao'] = m.ev_icao.astype(str)
    m['_apt'] = m.APT.astype(str)
    v2 = v2.rename(columns={'apt': '_apt', 'icao': '_icao', 't_air': 'ev_t_air'})
    v2['_apt'] = v2._apt.astype(str)
    v2['_icao'] = v2._icao.astype(str)
    m = m.merge(v2, on=['_apt', '_icao', 'ev_t_air'], how='left')
    if len(m) != len(dep):
        raise SystemExit('ERROR: v2 join changed the row count')
    return m


def build(which: str, month_days: dict, cur: pd.DataFrame, prod_st: pd.DataFrame, local_st: pd.DataFrame,
          ok_days: set) -> pd.DataFrame:
    parts = []
    for mo, ds in month_days.items():
        dep = load_deps(ds, which)
        if len(dep) == 0:
            print(f'{which} {mo}: no departures, skipped', flush=True)
            continue
        evd = [d for d in sorted(set(ds) | {str((pd.Timestamp(ds[0]) - pd.Timedelta(days=1)).date()),
                                            str((pd.Timestamp(ds[-1]) + pd.Timedelta(days=1)).date())})
               if (BASE / 'events' / f'{d}.parquet').exists()]
        f = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt.to_numpy()})
        if not evd:
            parts.append(f)
            print(f'{which} {mo}: {len(dep):,} deps, NO events', flush=True)
            continue
        m = month_table(dep, evd)
        mv = m.mvt_s.to_numpy()
        day = pd.to_datetime(m.mvt_s, unit='s').dt.strftime('%Y-%m-%d').to_numpy()
        matched = (m.match != 'none').to_numpy()
        run = np.where(matched, (m.connected.fillna(0) == 1).astype(float), np.nan)
        f['adsb2_run'] = run
        f['adsb2_off_R'] = mv - m.t_R.to_numpy()
        f['adsb2_contam'] = np.where(np.isfinite(m.ev_t_first.to_numpy()),
                                     ((m.t_R - m.ev_t_first).fillna(1e9) > 600).astype(float), np.nan)
        f['adsb2_gap_R'] = m.gap_R.to_numpy()
        f['adsb2_arr_seen'] = np.where(matched, m.t_land.notna().astype(float), np.nan)
        f['adsb2_off_land'] = mv - m.t_land.to_numpy()
        f['adsb2_A_gs'] = m.gs_A.to_numpy()
        f['adsb2_A_rest_s'] = m.A_rest_s.to_numpy()
        f['adsb2_d_R_A'] = m.d_R_A.to_numpy()
        for c in ('gs_R', 'rest_R_s', 'x_R', 'y_R', 'hdg_R', 'd_R_roll', 'path_run_m', 'max_gap_run_s'):
            f[f'adsb2_{c}'] = m[c].to_numpy(float)
        f['adsb2_R_at_park'] = m.R_at_park.astype(float).to_numpy()
        # stand position
        w = pd.DataFrame({'apt': m._apt.to_numpy(), 'stand': m.STAND_mvt.astype(str).to_numpy()})
        st = extend_stands(prod_st, local_st, w)
        j = w.merge(st, on=['apt', 'stand'], how='left')
        f['adsb2_stand_src'] = j.stand_src.fillna(0).to_numpy()
        f['adsb2_d_R_stand'] = np.hypot(m.x_R.to_numpy() - j.sx.to_numpy(), m.y_R.to_numpy() - j.sy.to_numpy())
        # visibility that day
        vis_st = np.full(len(m), np.nan)
        vis_r = np.full(len(m), np.nan)
        apt_day_vis = np.full(len(m), np.nan)
        for dd in np.unique(day):
            gf = BASE / 'grid_v2' / f'{dd}.parquet'
            if not gf.exists():
                continue
            gr = pd.read_parquet(gf)
            lut = {(a, cx, cy): n for a, cx, cy, n in zip(gr.apt.astype(str), gr.cx, gr.cy, gr.n_ac)}
            ix = np.flatnonzero(day == dd)
            for k in ix:
                a = m._apt.iat[k]
                sx, sy = j.sx.iat[k], j.sy.iat[k]
                if np.isfinite(sx):
                    cx, cy = int(np.floor(sx / 50)), int(np.floor(sy / 50))
                    vis_st[k] = sum(lut.get((a, cx + u, cy + v), 0) for u in (-1, 0, 1) for v in (-1, 0, 1))
                xr, yr = m.x_R.iat[k], m.y_R.iat[k]
                if np.isfinite(xr):
                    vis_r[k] = lut.get((a, int(np.floor(xr / 50)), int(np.floor(yr / 50))), 0)
            # share of the airport's production-table stands seen that day
            for a in np.unique(m._apt.to_numpy()[ix]):
                ps = prod_st[prod_st.apt == a]
                if len(ps) == 0:
                    continue
                cnt = [sum(lut.get((a, int(np.floor(sx / 50)) + u, int(np.floor(sy / 50)) + v), 0)
                           for u in (-1, 0, 1) for v in (-1, 0, 1)) for sx, sy in zip(ps.sx, ps.sy)]
                apt_day_vis[ix[m._apt.to_numpy()[ix] == a]] = float(np.mean(np.array(cnt) >= 2))
        f['adsb2_vis_stand'] = vis_st
        f['adsb2_vis_R'] = vis_r
        f['adsb2_apt_day_vis'] = apt_day_vis
        # back-calculation (estimator a) and the no-stand variant
        apts = m._apt.to_numpy()
        mov = (m.gs_R.fillna(0).to_numpy() > 3).astype(np.int8)
        e_st = apply_curve(cur, 'stand', apts, mov, f.adsb2_d_R_stand.to_numpy())
        e_ro = apply_curve(cur, 'roll', apts, mov, f.adsb2_d_R_roll.to_numpy())
        f['adsb2_bc_e_stand'] = e_st
        f['adsb2_off_bc_stand'] = f.adsb2_off_R.to_numpy() + e_st
        f['adsb2_bc_e_roll'] = e_ro
        f['adsb2_off_bc_roll'] = f.adsb2_off_R.to_numpy() + e_ro
        best = mv - m.ev_t_best.to_numpy()
        b2 = np.where(np.isfinite(best), best, np.where(np.isfinite(f.adsb2_off_bc_stand), f.adsb2_off_bc_stand,
                                                        f.adsb2_off_bc_roll))
        src = np.where(np.isfinite(best), 1, np.where(np.isfinite(f.adsb2_off_bc_stand), 2,
                                                      np.where(np.isfinite(f.adsb2_off_bc_roll), 3, 0)))
        f['adsb2_off_best2'] = b2
        f['adsb2_best2_src'] = src.astype(float)
        # surface congestion around the departure's own take-off (all departures of a covered airport-day)
        for c in ('cong_mv15', 'cong_mv30_15', 'cong_hold10', 'cong_gnd15', 'cong_mv_day'):
            f[f'adsb2_{c}'] = np.nan
        mvm = (mv // 60).astype(np.int64)
        for dd in np.unique(day):
            cf = BASE / 'cong_v2' / f'{dd}.parquet'
            if not cf.exists():
                continue
            cg = pd.read_parquet(cf)
            cg['apt'] = cg.apt.astype(str)
            d0 = int(pd.Timestamp(dd, tz='UTC').timestamp() // 60)
            for a, ca in cg.groupby('apt'):
                ix = np.flatnonzero((day == dd) & (m._apt.to_numpy() == a))
                if not len(ix):
                    continue
                lo = d0 - 60
                n = 1440 + 120
                arr = {}
                for col in ('n_mv', 'n_hold', 'n_gnd'):
                    v = np.zeros(n)
                    k = ca.minute.to_numpy() - lo
                    ok = (k >= 0) & (k < n)
                    v[k[ok]] = ca[col].to_numpy()[ok]
                    arr[col] = np.r_[0, np.cumsum(v)]
                mm = np.clip(mvm[ix] - lo, 30, n)

                def win(col, a0, a1):
                    return (arr[col][mm - a1] - arr[col][mm - a0]) / (a0 - a1)
                f.loc[ix, 'adsb2_cong_mv15'] = win('n_mv', 15, 0)
                f.loc[ix, 'adsb2_cong_mv30_15'] = win('n_mv', 30, 15)
                f.loc[ix, 'adsb2_cong_hold10'] = win('n_hold', 10, 0)
                f.loc[ix, 'adsb2_cong_gnd15'] = win('n_gnd', 15, 0)
                f.loc[ix, 'adsb2_cong_mv_day'] = arr['n_mv'][60 + 1440] / 1440 - arr['n_mv'][60] / 1440
        dok = np.isin(day, list(ok_days))
        for c in [c for c in f.columns if c.startswith('adsb2_')]:
            f.loc[~dok, c] = np.nan
        parts.append(f)
        print(f'{which} {mo}: {len(dep):,} deps; run {np.nanmean(run):.1%} of matched, contam '
              f'{np.nanmean(f.adsb2_contam):.1%}, best2 {np.mean(src > 0):.1%} (prod best {np.mean(src == 1):.1%})',
              flush=True)
    out = pd.concat(parts, ignore_index=True)
    for c in out.columns:
        if c.startswith('adsb2_'):
            out[c] = out[c].astype(np.float32)
    return out


def local_stand_table(month_days: dict) -> pd.DataFrame:
    """Median first position of pushbacks-from-rest per (APT, STAND) over the given TRAINING days, n >= 1."""
    rows = []
    for mo, ds in month_days.items():
        dep = load_deps(ds, 'training')
        evd = [d for d in ds if (BASE / 'events' / f'{d}.parquet').exists()]
        if not evd:
            continue
        ev = pd.concat([pd.read_parquet(BASE / 'events' / f'{d}.parquet') for d in evd], ignore_index=True)
        ev = ev.drop_duplicates(['apt', 'icao', 't_air']).reset_index(drop=True)
        ev['ev_id'] = np.arange(len(ev))
        m = match(dep, ev)
        p = m[(m.ev_best_src == 'push') & ((m.ev_t_push_prev - m.ev_t_first) < 600) & m.ev_x_first.notna()]
        rows.append(pd.DataFrame({'apt': p.APT.astype(str), 'stand': p.STAND_mvt.astype(str),
                                  'x': p.ev_x_first, 'y': p.ev_y_first}))
    if not rows:
        return pd.DataFrame(columns=['apt', 'stand', 'sx', 'sy'])
    r = pd.concat(rows, ignore_index=True)
    return r.groupby(['apt', 'stand']).agg(sx=('x', 'median'), sy=('y', 'median')).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--which', nargs='+', default=['training', 'ranking'])
    ap.add_argument('--days', nargs='*', help='local test: restrict to these days')
    ap.add_argument('--suffix', default='', help='output name suffix (local tests)')
    a = ap.parse_args()
    t0 = time.time()
    universe = a.days if a.days else all_days()
    have = [d for d in universe if (BASE / 'points' / f'{d}.parquet').exists() and (BASE / 'events' / f'{d}.parquet').exists()]
    todo = [d for d in have if a.redo or not (BASE / 'events_v2' / f'{d}.parquet').exists()]
    print(f'days with points+events: {len(have)}; v2 stage to run: {len(todo)}', flush=True)
    fails = []
    with Pool(a.workers) as pool:
        for day, msg in pool.imap_unordered(_v2_day, todo):
            if msg.startswith('FAILED'):
                fails.append(day)
            print(day, msg[-200:], flush=True)
    if fails:
        raise SystemExit(f'ERROR: v2 stage failed for {len(fails)} days: {fails[:10]}')
    train_days = [d for d in have if d.startswith('2025')]
    cur = fit_curves(train_days)
    cur.to_parquet(BASE / f'v2_calib_curves{a.suffix}.parquet', index=False)
    print(f'calibration curves from {len(train_days)} training days: {len(cur)} bins', flush=True)
    prod_st = pd.read_parquet(BASE / 'stands.parquet').reset_index().rename(columns={'_apt': 'apt', '_stand': 'stand'})
    prod_st = prod_st.astype({'apt': str, 'stand': str})
    ok_days = set(have)

    def months(which):
        ds = [d for d in have if (d.startswith('2025') if which == 'training' else d.startswith('2026'))]
        out = {}
        for d in ds:
            out.setdefault(d[:7], []).append(d)
        return out
    local_st = local_stand_table(months('training'))
    local_st.to_parquet(BASE / f'v2_stands_local{a.suffix}.parquet', index=False)
    print(f'local stand table (n >= 1): {len(local_st)} stands; production: {len(prod_st)}', flush=True)
    for which in a.which:
        md = months(which)
        if a.days:           # local test: only the given days' departures
            pass
        else:                # production: the same months as adsb_build_features.py, every day of the month
            mos = ['2025-%02d' % k for k in range(1, 13)] if which == 'training' else ['2026-01', '2026-07']
            md = {mo: [d for d in all_days() if d.startswith(mo)] for mo in mos}
        out = build(which, md, cur, prod_st, local_st, ok_days)
        if not out.MVT_ID_mvt.is_unique:
            raise SystemExit(f'ERROR: duplicate MVT_ID_mvt in features_v2_{which}')
        dst = BASE / f'features_v2_{which}{a.suffix}.parquet'
        out.to_parquet(dst.with_suffix('.tmp.parquet'), index=False)
        dst.with_suffix('.tmp.parquet').replace(dst)
        print(f'-> {dst} ({len(out):,} rows, {dst.stat().st_size / 1e6:.1f} MB)', flush=True)
    print(f'total {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
