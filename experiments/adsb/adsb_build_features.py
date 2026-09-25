"""Stage 2 for all days + per-departure feature table (training and ranking).

    python experiments/adsb/adsb_build_features.py [--workers 8] [--redo-events] [--which training ranking]

1. (re)builds events/<date>.parquet for every day whose points exist (parallel, each day with the
   last 4 h of the previous day's points), skipping existing ones unless --redo-events;
2. matches events to departures month by month (adsb_match_measure.match) and writes
       cache/adsb_lol/features_<which>.parquet     keyed by MVT_ID_mvt, one row per departure
   Every departure of the month appears; adsb_* columns are NaN where nothing matched.
   Days whose points file is missing are reported and their departures get adsb_day_ok = 0,
   so a model can tell "no archive" from "no coverage".

Feature columns (seconds unless stated; 'off_X' = MVT_TIME - X, i.e. the taxi time implied by X):
  adsb_day_ok        1 if the archive day was processed
  adsb_matched       1 if a take-off event was matched   adsb_match_kind 1 callsign, 2 flight, 3 type+time
  adsb_to_dev        t_to - MVT (take-off agreement)     adsb_to_obs 1 if ground->air transition observed
  adsb_has_ground    1 if ground points precede the take-off
  adsb_off_first     MVT - first ground point            adsb_gs_first  speed at first point (kt)
  adsb_off_move      MVT - first movement point          adsb_off_statend MVT - last at-rest point
  adsb_off_push      MVT - pushback start                adsb_off_best  MVT - t_best (push/rest)
  adsb_best_src      0 none, 1 rest, 2 push              adsb_stat_s    seen at rest before moving
  adsb_move_gap_s    gap between last rest and first move points   adsb_at_park 1 near a parking cell
  adsb_dwell_s       stand dwell seen (>= 10 min rest)   adsb_n_push, adsb_push_m
  adsb_n_pts, adsb_max_gap_s, adsb_cov_s_per_pt, adsb_gap_last_to_air, adsb_mlat_frac  (coverage quality)
  adsb_path_m        taxi path observed after first move (m)
  adsb_n_stops, adsb_stop_s   stops >= 30 s after first move      adsb_hold_s  t_to - end of last stop
  adsb_off_roll      MVT - first point >= 40 kt          adsb_off_last  MVT - last ground point
  adsb_gs_med_moving median taxi speed (kt)
  adsb_apt_day_ground  share of the airport-day's departures with ground points (coverage context)
  adsb_apt_day_best    share with an off-block estimate (t_best)
  adsb_d_stand_first   distance (m) of the first ground point from the stand's typical pushback position
                       (stands.parquet: median over 2025 pushbacks-from-rest per (APT, STAND_mvt), n >= 3;
                       positions only, no labels). Within 30 m the first sighting is ~at off-block.
  adsb_stand_known     1 if the stand is in that table
"""
import argparse
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


def _events_day(day):
    import subprocess
    out = BASE / 'events' / f'{day}.parquet'
    r = subprocess.run([PY, str(Path(__file__).resolve().parent / 'adsb_events.py'), day],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        return day, f'FAILED: {r.stderr[-400:]}'
    return day, r.stdout.strip()


def features_from_match(m: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({'MVT_ID_mvt': m.MVT_ID_mvt})
    mv = m.mvt_s
    f['adsb_matched'] = (m.match != 'none').astype(np.int8)
    f['adsb_match_kind'] = m.match.map({'none': 0, 'callsign': 1, 'flight': 2, 'type_time': 3}).astype(np.int8)
    f['adsb_to_dev'] = m.ev_t_to - mv
    f['adsb_to_obs'] = (m.ev_to_src == 'obs').astype(float).where(m.match != 'none')
    f['adsb_has_ground'] = (m.ev_n_seg > 0).astype(float).where(m.match != 'none')
    for c, n in (('t_first', 'first'), ('t_move', 'move'), ('t_stat_end', 'statend'), ('t_push_prev', 'push'),
                 ('t_best', 'best'), ('t_roll', 'roll'), ('t_last', 'last')):
        f[f'adsb_off_{n}'] = mv - m[f'ev_{c}']
    f['adsb_best_src'] = m.ev_best_src.map({'none': 0, 'rest': 1, 'push': 2})
    for c in ('gs_first', 'stat_s', 'move_gap_s', 'dwell_s', 'n_push', 'push_m', 'n_pts', 'max_gap_s',
              'cov_s_per_pt', 'gap_last_to_air', 'mlat_frac', 'path_m', 'n_stops', 'stop_s', 'gs_med_moving'):
        f[f'adsb_{c}'] = m[f'ev_{c}'].astype(float)
    f['adsb_at_park'] = m.ev_at_park.astype(float)
    f['adsb_hold_s'] = m.ev_t_to - m.ev_t_last_stop_end
    # helpers for the stand table (dropped before writing)
    f['_apt'] = m.APT.astype(str).to_numpy()
    f['_stand'] = m.STAND_mvt.astype(str).to_numpy()
    f['_x'] = m.ev_x_first.astype(float)
    f['_y'] = m.ev_y_first.astype(float)
    f['_push_from_rest'] = ((m.ev_best_src == 'push') & ((m.ev_t_push_prev - m.ev_t_first) < 600)).to_numpy()
    for c in f.columns:                      # NEVER the key: float32 cannot hold MVT_ID_mvt (~1.9e8) exactly
        if c.startswith('adsb_') and f[c].dtype == np.float64:
            f[c] = f[c].astype(np.float32)
    return f


def stand_table(f: pd.DataFrame, min_n=3) -> pd.DataFrame:
    """Median first position of departures that pushed back from rest, per (APT, STAND_mvt).
    Built from ADS-B positions only (no labels), so it may be applied to the ranking set."""
    p = f[f._push_from_rest & f._x.notna()]
    t = p.groupby(['_apt', '_stand']).agg(sx=('_x', 'median'), sy=('_y', 'median'), sn=('_x', 'size'))
    return t[t.sn >= min_n]


def add_stand_features(f: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    f = f.join(st, on=['_apt', '_stand'])
    f['adsb_d_stand_first'] = np.hypot(f._x - f.sx, f._y - f.sy).astype(np.float32)
    f['adsb_stand_known'] = f.sx.notna().astype(np.int8)
    return f.drop(columns=['sx', 'sy', 'sn', '_apt', '_stand', '_x', '_y', '_push_from_rest'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--redo-events', action='store_true')
    ap.add_argument('--which', nargs='+', default=['training', 'ranking'])
    a = ap.parse_args()
    t0 = time.time()
    have = [d for d in all_days() if (BASE / 'points' / f'{d}.parquet').exists()]
    missing = [d for d in all_days() if d not in have]
    todo = [d for d in have if a.redo_events or not (BASE / 'events' / f'{d}.parquet').exists()]
    print(f'points for {len(have)} days, missing {len(missing)}; events to build: {len(todo)}', flush=True)
    if missing:
        print('  missing days:', ' '.join(missing[:40]) + (' ...' if len(missing) > 40 else ''), flush=True)
    fails = []
    with Pool(a.workers) as pool:
        for day, msg in pool.imap_unordered(_events_day, todo):
            if msg.startswith('FAILED'):
                fails.append(day)
                print(day, msg, flush=True)
    if fails:
        raise SystemExit(f'ERROR: stage 2 failed for {len(fails)} days: {fails[:10]}')
    print(f'events done in {time.time() - t0:.0f} s', flush=True)

    ok_days = set(have)
    results = {}
    for which in a.which:
        months = ['2025-%02d' % k for k in range(1, 13)] if which == 'training' else ['2026-01', '2026-07']
        parts = []
        for mo in months:
            ds = [d for d in all_days() if d.startswith(mo)]
            dep = load_deps(ds, which)
            evd = [d for d in sorted(set(ds) | {str((pd.Timestamp(ds[0]) - pd.Timedelta(days=1)).date()),
                                                str((pd.Timestamp(ds[-1]) + pd.Timedelta(days=1)).date())})
                   if (BASE / 'events' / f'{d}.parquet').exists()]
            if not evd:
                fm = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt})
                fm['adsb_day_ok'] = np.int8(0)
                parts.append(fm)
                print(f'{which} {mo}: {len(dep):,} deps, NO events', flush=True)
                continue
            ev = pd.concat([pd.read_parquet(BASE / 'events' / f'{d}.parquet') for d in evd], ignore_index=True)
            ev = ev.drop_duplicates(['apt', 'icao', 't_air']).reset_index(drop=True)
            ev['ev_id'] = np.arange(len(ev))
            m = match(dep, ev)
            fm = features_from_match(m)
            day = pd.to_datetime(m.mvt_s, unit='s').dt.strftime('%Y-%m-%d')
            fm['adsb_day_ok'] = day.isin(ok_days).astype(np.int8).to_numpy()
            key = m.APT.astype(str) + '_' + day
            fm['adsb_apt_day_ground'] = fm.adsb_has_ground.fillna(0).groupby(key.to_numpy()).transform('mean').astype(np.float32)
            fm['adsb_apt_day_best'] = fm.adsb_off_best.notna().astype(float).groupby(key.to_numpy()).transform('mean').astype(np.float32)
            for c in [c for c in fm.columns if c.startswith('adsb_') and c not in ('adsb_day_ok',)]:
                fm.loc[fm.adsb_day_ok == 0, c] = np.nan
            parts.append(fm)
            ok = fm.adsb_day_ok == 1
            print(f'{which} {mo}: {len(dep):,} deps, {ok.sum():,} on processed days: matched '
                  f'{fm.adsb_matched[ok].mean():.1%}, ground {fm.adsb_has_ground[ok].fillna(0).mean():.1%}, '
                  f'best {fm.adsb_off_best[ok].notna().mean():.1%}', flush=True)
        results[which] = pd.concat(parts, ignore_index=True)
    # stand coordinates from the TRAINING year's ADS-B (positions only); reuse a saved table otherwise
    st_path = BASE / 'stands.parquet'
    if 'training' in results:
        st = stand_table(results['training'])
        st.to_parquet(st_path)
        print(f'stand table: {len(st):,} (APT, STAND) with >= 3 pushbacks from rest -> {st_path}', flush=True)
    elif st_path.exists():
        st = pd.read_parquet(st_path)
    else:
        raise SystemExit('ERROR: no stand table; run with --which training (ranking) first')
    for which, out in results.items():
        if '_apt' not in out:
            out = out.assign(_apt='', _stand='', _x=np.nan, _y=np.nan, _push_from_rest=False)
        out['_push_from_rest'] = out['_push_from_rest'].fillna(False).astype(bool)
        out = add_stand_features(out, st)
        if not out.MVT_ID_mvt.is_unique:
            raise SystemExit(f'ERROR: duplicate MVT_ID_mvt in features_{which}')
        dst = BASE / f'features_{which}.parquet'
        out.to_parquet(dst.with_suffix('.tmp.parquet'), index=False)
        dst.with_suffix('.tmp.parquet').replace(dst)
        print(f'-> {dst} ({len(out):,} rows, {dst.stat().st_size / 1e6:.1f} MB)', flush=True)
    print(f'total {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
