"""Match adsb.lol take-off events to the challenge departures of one or more days and measure.

    python experiments/adsb/adsb_match_measure.py 2025-07-16 [2025-07-17 ...] [--which training]
           [--out cache/adsb_lol/match/<first>_<last>.parquet]

Matching: same airport (APT), callsign equal to CALLSIGN_flt (fallback FLIGHT_mvt),
|t_to - MVT_TIME_UTC| <= 15 min, nearest wins, one event per departure and one
departure per event. A second pass matches remaining departures to remaining events
by aircraft type + |t_to - MVT| <= 120 s when that pair is unique ('type_time').

Off-block estimators (all compared to BLOCK_TIME_UTC_mvt, the label's off-block):
  first   = t_first     first ground point of the departure segment
  move    = t_move      first point >= 15 m from the first position
  statend = t_stat_end  last point still at rest (only when stat_s >= 10 s)
  push    = t_push_prev last point before the last reverse (pushback) bout
  best    = t_best from adsb_events.py (push, else a stand rest with an observed end)
Reference proxy: AOBT_3_flt (NM actual off-block, present on ~98% of rows).
Only departures with BLOCK < MVT and a label <= 7200 s enter the accuracy rows,
so the stamped / day-shifted tail is reported separately.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / 'cache' / 'adsb_lol'
sys.path.insert(0, str(REPO_ROOT))
from prc.config import CACHE as _CACHE      # cache_final/ when PRC_RANKING=final (its ranking.parquet is the final file)
COLS = ['MVT_ID_mvt', 'APT', 'PHASE_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt',
        'STAND_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
        'AOBT_3_flt', 'TAXITIME_SEC_mvt']


def load_deps(days, which='training'):
    """Departures whose MVT_TIME falls on exactly the given UTC days (contiguous runs read in one go)."""
    cols = COLS if which == 'training' else [c for c in COLS if c not in ('BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt')]
    ds = sorted(pd.Timestamp(x, tz='UTC') for x in set(days))
    runs, start, prev = [], ds[0], ds[0]
    for x in ds[1:]:
        if x - prev != pd.Timedelta(days=1):
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    tabs = [pq.read_table(_CACHE / f'{which}.parquet', columns=cols,
                          filters=[('PHASE_mvt', '=', 'DEP'), ('MVT_TIME_UTC_mvt', '>=', a),
                                   ('MVT_TIME_UTC_mvt', '<', b + pd.Timedelta(days=1))]).to_pandas()
            for a, b in runs]
    d = pd.concat(tabs, ignore_index=True)
    secs = {'MVT_TIME_UTC_mvt': 'mvt_s', 'BLOCK_TIME_UTC_mvt': 'block_s',
            'SCHED_TIME_UTC_mvt': 'sched_s', 'AOBT_3_flt': 'aobt3_s'}
    for c, s in secs.items():
        if c in d:
            v = d[c]
            d[s] = (v - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    return d


def load_events(days):
    ev = []
    for day in days:
        f = BASE / 'events' / f'{day}.parquet'
        if not f.exists():
            raise SystemExit(f'ERROR: {f} missing; run adsb_extract_day.py {day}')
        ev.append(pd.read_parquet(f))
    ev = pd.concat(ev, ignore_index=True)
    ev = ev.drop_duplicates(['apt', 'icao', 't_air']).reset_index(drop=True)
    ev['ev_id'] = np.arange(len(ev))
    return ev


def match(d, ev, win=900):
    d = d.copy()
    d['ev_id'] = -1
    d['match'] = 'none'
    evk = ev[['ev_id', 'apt', 'cs', 'typ', 't_to']].copy()
    evk['apt'] = evk.apt.astype(str)
    evk['cs'] = evk.cs.astype(str)
    cand = []
    for key in ('CALLSIGN_flt', 'FLIGHT_mvt'):
        m = d[['MVT_ID_mvt', 'APT', key, 'mvt_s']].dropna().astype({'APT': str, key: str}).merge(
            evk, left_on=['APT', key], right_on=['apt', 'cs'])
        m['dt'] = m.t_to - m.mvt_s
        m = m[m.dt.abs() <= win]
        m['pri'] = 0 if key == 'CALLSIGN_flt' else 1
        cand.append(m[['MVT_ID_mvt', 'ev_id', 'dt', 'pri']])
    cand = pd.concat(cand).sort_values(['pri', 'dt'], key=lambda s: s.abs() if s.name == 'dt' else s)
    used_d, used_e, pairs = set(), set(), []
    for r in cand.itertuples(index=False):
        if r.MVT_ID_mvt in used_d or r.ev_id in used_e:
            continue
        used_d.add(r.MVT_ID_mvt)
        used_e.add(r.ev_id)
        pairs.append((r.MVT_ID_mvt, r.ev_id, 'callsign' if r.pri == 0 else 'flight'))
    # fallback: aircraft type + time, unique pair
    rest_d = d[~d.MVT_ID_mvt.isin(used_d)][['MVT_ID_mvt', 'APT', 'AIRCRAFT_TYPE_mvt', 'mvt_s']].astype(
        {'APT': str, 'AIRCRAFT_TYPE_mvt': str})
    rest_e = evk[~evk.ev_id.isin(used_e)].astype({'typ': str})
    m = rest_d.merge(rest_e, left_on=['APT', 'AIRCRAFT_TYPE_mvt'], right_on=['apt', 'typ'])
    m = m[(m.t_to - m.mvt_s).abs() <= 120]
    m = m[m.groupby('MVT_ID_mvt').ev_id.transform('size').eq(1) & m.groupby('ev_id').MVT_ID_mvt.transform('size').eq(1)]
    pairs += [(a, b, 'type_time') for a, b in zip(m.MVT_ID_mvt, m.ev_id)]
    pm = pd.DataFrame(pairs, columns=['MVT_ID_mvt', 'ev_id', 'match'])
    d = d.drop(columns=['ev_id', 'match']).merge(pm, on='MVT_ID_mvt', how='left')
    d['match'] = d.match.fillna('none')
    d['ev_id'] = d.ev_id.fillna(-1).astype(int)
    evc = ev.drop(columns=['apt']).add_prefix('ev_').rename(columns={'ev_ev_id': 'ev_id'})
    return d.merge(evc, on='ev_id', how='left')


def estimators(m):
    m = m.copy()
    m['est_first'] = m.ev_t_first
    m['est_move'] = m.ev_t_move
    m['est_statend'] = m.ev_t_stat_end.where(m.ev_stat_s >= 10)
    m['est_push'] = m.ev_t_push_prev
    m['est_best'] = m.ev_t_best
    return m


def q(s, p):
    s = s.dropna()
    return np.nan if s.empty else float(np.quantile(s, p))


def rmse(a):
    a = a.dropna()
    return np.nan if a.empty else float(np.sqrt(np.mean(a ** 2)))


def summarise(m):
    rows = []
    for apt, g in list(m.groupby('APT', observed=True)) + [('ALL', m)]:
        n = len(g)
        mt = g[g.match != 'none']
        ok = mt[(mt.block_s < mt.mvt_s) & (mt.TAXITIME_SEC_mvt <= 7200)] if 'block_s' in mt else mt
        r = dict(APT=apt, n_dep=n,
                 match_pct=100 * len(mt) / n,
                 match_cs_pct=100 * (g.match == 'callsign').mean(),
                 to_obs_pct=100 * (mt.ev_to_src == 'obs').sum() / n,
                 ground_pct=100 * (mt.ev_n_seg > 0).sum() / n,
                 to_err_med=q((mt.ev_t_to - mt.mvt_s).abs(), .5),
                 to_err_p90=q((mt.ev_t_to - mt.mvt_s).abs(), .9),
                 to_bias_med=q(mt.ev_t_to - mt.mvt_s, .5),
                 to_obs_err_med=q((mt.ev_t_to - mt.mvt_s)[mt.ev_to_src == 'obs'].abs(), .5),
                 )
        if 'block_s' in g:
            r.update(
                seen_before_block_pct=100 * (ok.ev_t_first <= ok.block_s + 60).sum() / n,
                first_minus_block_med=q(ok.ev_t_first - ok.block_s, .5),
                statobs_pct=100 * (ok.ev_stat_s >= 10).sum() / n,
                push_pct=100 * (ok.ev_best_src == 'push').sum() / n,
                rest_pct=100 * (ok.ev_best_src == 'rest').sum() / n,
                best_pct=100 * ok.est_best.notna().sum() / n,
            )
            b = ok[ok.est_best.notna() & ok.aobt3_s.notna()]
            r.update(
                n_best=len(b),
                best_abs_med=q((b.est_best - b.block_s).abs(), .5),
                best_abs_p90=q((b.est_best - b.block_s).abs(), .9),
                best_bias_med=q(b.est_best - b.block_s, .5),
                aobt3_abs_med_same=q((b.aobt3_s - b.block_s).abs(), .5),
                aobt3_abs_p90_same=q((b.aobt3_s - b.block_s).abs(), .9),
                best_closer_pct=100 * ((b.est_best - b.block_s).abs() < (b.aobt3_s - b.block_s).abs()).mean() if len(b) else np.nan,
                rmse_taxi_best_mvt=rmse(b.mvt_s - b.est_best - b.TAXITIME_SEC_mvt),
                rmse_taxi_best_adsb=rmse(b.ev_t_to - b.est_best - b.TAXITIME_SEC_mvt),
                rmse_taxi_aobt3_same=rmse(b.mvt_s - b.aobt3_s - b.TAXITIME_SEC_mvt),
            )
            f = ok[ok.ev_t_first.notna() & ok.aobt3_s.notna()]
            r.update(
                n_first=len(f),
                first_abs_med=q((f.est_first - f.block_s).abs(), .5),
                first_abs_p90=q((f.est_first - f.block_s).abs(), .9),
                aobt3_abs_med_first=q((f.aobt3_s - f.block_s).abs(), .5),
                rmse_taxi_first_mvt=rmse(f.mvt_s - f.est_first - f.TAXITIME_SEC_mvt),
                rmse_taxi_aobt3_first=rmse(f.mvt_s - f.aobt3_s - f.TAXITIME_SEC_mvt),
            )
        rows.append(r)
    return pd.DataFrame(rows).set_index('APT')


def coverage(m):
    """Label-free coverage table (works for the ranking set too)."""
    rows = []
    for apt, g in list(m.groupby('APT', observed=True)) + [('ALL', m)]:
        n = len(g)
        mt = g[g.match != 'none']
        rows.append(dict(APT=apt, n_dep=n, match_pct=100 * len(mt) / n,
                         to_obs_pct=100 * (mt.ev_to_src == 'obs').sum() / n,
                         ground_pct=100 * (mt.ev_n_seg > 0).sum() / n,
                         push_pct=100 * (mt.ev_best_src == 'push').sum() / n,
                         best_pct=100 * mt.ev_t_best.notna().sum() / n,
                         first_before_aobt3_pct=100 * (mt.ev_t_first <= mt.aobt3_s + 60).sum() / n,
                         to_err_med=q((mt.ev_t_to - mt.mvt_s).abs(), .5)))
    return pd.DataFrame(rows).set_index('APT')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('days', nargs='+')
    ap.add_argument('--which', default='training', choices=['training', 'ranking'])
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    d = load_deps(a.days, a.which)
    ev = load_events(sorted(set(a.days) | {str((pd.Timestamp(x) + pd.Timedelta(days=k)).date())
                                           for x in a.days for k in (-1, 1)
                                           if (BASE / 'events' / f'{(pd.Timestamp(x) + pd.Timedelta(days=k)).date()}.parquet').exists()}))
    m = estimators(match(d, ev))
    out = a.out or BASE / 'match' / f'{a.which}_{min(a.days)}_{max(a.days)}.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    c = coverage(m)
    print(c.round(1).T.to_string())
    c.to_csv(out.with_suffix('.coverage.csv'))
    if a.which == 'training':
        s = summarise(m)
        print(s.round(1).T.to_string())
        s.to_csv(out.with_suffix('.summary.csv'))
    print(f'-> {out}', file=sys.stderr)


if __name__ == '__main__':
    main()
