"""De-icing / disruption indicator, step 1: label-free airport-hour and airport-day aggregates of the adsb.lol
"remote rest" pattern, for the 2025 training departures and the Jan+Jul 2026 ranking departures.

Per-flight flags (from the FULL-SCALE per-flight columns in cache/adsb_lol/features_{which}.parquet, which cover
every 2025 day and Jan/Jul 2026; no events/points files are needed):
  cov   adsb_has_ground == 1                       the flight has ADS-B ground points (the denominator)
  rr    remote rest (the definition of drivers_0926/d08): adsb_best_src == 1 (off-block from a rest, no push
        seen) AND adsb_stat_s > 600 (more than 10 min at rest before moving) AND adsb_d_stand_first > 300 m
        (first seen more than 300 m from the stand's typical pushback position) -- de-icing pad / remote hold
  ls    long stop: adsb_stop_s >= 600 (>= 10 min in stops of >= 30 s AFTER the first movement: pad de-icing
        after pushback, or a long runway hold)
  taxi  adsb_off_best clipped to [0, 7200] where present (the ADS-B taxi of that flight; EXPLORATORY arm only)

Aggregates over OTHER departures of the same airport (the flight itself is always excluded), keyed by the
take-off time MVT_TIME (present for every departure in both files, so label-free):
  whole (look-ahead allowed):  hour = take-off within +-60 min of ours;  day = same local calendar date
  past  (conservative):        hour = take-off in [ours - 60 min, ours);  day = same local date, before ours
Columns (prefix dz_ = whole, dzp_ = past):
  _ncov_h/_d     covered other departures in the window
  _rr_sh_h/_d    share of remote-rest flights among them (NaN when none covered)
  _rr_n_h/_d     count of remote-rest flights
  _ls_sh_h/_d    share of long-stop flights among the covered
  _rest_h        mean adsb_stat_s of the remote-rest flights in the hour (NaN when none)
  _taxi_h, _ntaxi_h   mean ADS-B taxi of others with an off-block estimate, and their count (EXPLORATORY)

A flight's own label never enters: no y, BLOCK or AOBT is read. Other flights' ADS-B is open data (adsb.lol, ODbL,
approved 09-25) and other rows / look-ahead within the provided file are allowed by the organisers.

Outputs: cache/deice_0926/ind_{training,ranking}.parquet, row-aligned with cache/major/dep_{which}.parquet
(= load(which, phase='DEP') order), with MVT_ID_mvt for assertion.

    python experiments/deice_0926/e01_build_indicators.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from prc.config import CACHE
from prc.io import peak_rss_gb

OUT = CACHE / 'deice_0926'; OUT.mkdir(exist_ok=True)
ACOLS = ['adsb_has_ground', 'adsb_best_src', 'adsb_stat_s', 'adsb_d_stand_first', 'adsb_stop_s', 'adsb_off_best']
H = 3600


def flags(which):
    d = pd.read_parquet(CACHE / 'major' / f'dep_{which}.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt'])
    r = pd.read_parquet(CACHE / 'drivers_0926' / f'rows_{which}.parquet', columns=['apt', 'date_loc'])
    assert len(r) == len(d) and (r.apt.astype(str).to_numpy() == d.APT.astype(str).to_numpy()).all()
    a = pd.read_parquet(CACHE / 'adsb_lol' / f'features_{which}.parquet', columns=['MVT_ID_mvt'] + ACOLS)
    a = a.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(d.MVT_ID_mvt.to_numpy())
    assert len(a) == len(d)
    t = pd.to_datetime(d.MVT_TIME_UTC_mvt)
    if t.dt.tz is not None:
        t = t.dt.tz_convert('UTC').dt.tz_localize(None)
    f = pd.DataFrame({'MVT_ID_mvt': d.MVT_ID_mvt.to_numpy(), 'apt': d.APT.astype(str).to_numpy(),
                      't': (t - pd.Timestamp('1970-01-01')).dt.total_seconds().to_numpy(),
                      'date': r.date_loc.to_numpy()})
    assert np.isfinite(f.t).all(), 'missing take-off time'
    f['cov'] = (a.adsb_has_ground.to_numpy() == 1).astype(float)
    f['rr'] = ((a.adsb_best_src.to_numpy() == 1) & (a.adsb_stat_s.to_numpy() > 600)
               & (a.adsb_d_stand_first.to_numpy() > 300)).astype(float)
    f['ls'] = (a.adsb_stop_s.to_numpy() >= 600).astype(float)
    f['rest'] = np.where(f.rr == 1, a.adsb_stat_s.to_numpy(float), 0.0)
    tx = a.adsb_off_best.to_numpy(float)
    f['htx'] = np.isfinite(tx).astype(float)
    f['tx'] = np.where(np.isfinite(tx), np.clip(tx, 0, 7200), 0.0)
    return f


def window_sums(ts, v, lo_off, hi_off, hi_side, self_in):
    """Sum of v over flights with take-off in [t + lo_off, t + hi_off] (hi_side='right') or [.., t + hi_off)
    (hi_side='left'); ts sorted. Subtracts the flight itself when it lies inside the window."""
    C = np.r_[0.0, np.cumsum(v)]
    lo = np.searchsorted(ts, ts + lo_off, side='left')
    hi = np.searchsorted(ts, ts + hi_off, side=hi_side)
    return C[hi] - C[lo] - (v if self_in else 0.0)


def build(which):
    f = flags(which)
    n = len(f)
    V = ['cov', 'rr', 'ls', 'rest', 'htx', 'tx']
    out = {k: np.full(n, np.nan) for k in
           [f'{p}_{s}' for p in ('dz', 'dzp') for s in ('ncov_h', 'rr_n_h', 'ls_n_h', 'rest_sum_h', 'ntaxi_h', 'taxi_sum_h',
                                                         'ncov_d', 'rr_n_d', 'ls_n_d')]}
    for apt, idx in f.groupby('apt').indices.items():
        g = f.iloc[idx]
        o = np.argsort(g.t.to_numpy(), kind='stable')
        ii = idx[o]; ts = g.t.to_numpy()[o]; dt = g.date.to_numpy()[o]
        assert (np.diff(dt) >= 0).all(), f'{apt}: local date not monotone in take-off time'
        vals = {k: g[k].to_numpy()[o] for k in V}
        hmap = {'cov': 'ncov_h', 'rr': 'rr_n_h', 'ls': 'ls_n_h', 'rest': 'rest_sum_h', 'htx': 'ntaxi_h', 'tx': 'taxi_sum_h'}
        for k, name in hmap.items():
            out[f'dz_{name}'][ii] = window_sums(ts, vals[k], -H, H, 'right', True)       # +-60 min, self removed
            out[f'dzp_{name}'][ii] = window_sums(ts, vals[k], -H, 0, 'left', False)     # [t-60 min, t)
        # day: same local date. whole = day total - self; past = day-so-far strictly before our take-off
        u, first = np.unique(dt, return_index=True)
        start = first[np.searchsorted(u, dt)]
        before = np.searchsorted(ts, ts, side='left')                                  # flights with t' < t
        for k, name in (('cov', 'ncov_d'), ('rr', 'rr_n_d'), ('ls', 'ls_n_d')):
            C = np.r_[0.0, np.cumsum(vals[k])]
            end = np.r_[first[1:], len(ts)][np.searchsorted(u, dt)]
            out[f'dz_{name}'][ii] = C[end] - C[start] - vals[k]
            out[f'dzp_{name}'][ii] = C[np.maximum(before, start)] - C[start]
    X = pd.DataFrame({'MVT_ID_mvt': f.MVT_ID_mvt.to_numpy()})
    with np.errstate(invalid='ignore', divide='ignore'):
        for p in ('dz', 'dzp'):
            nc_h, nc_d = out[f'{p}_ncov_h'], out[f'{p}_ncov_d']
            X[f'{p}_ncov_h'] = nc_h
            X[f'{p}_rr_sh_h'] = np.where(nc_h > 0, out[f'{p}_rr_n_h'] / nc_h, np.nan)
            X[f'{p}_rr_n_h'] = out[f'{p}_rr_n_h']
            X[f'{p}_ls_sh_h'] = np.where(nc_h > 0, out[f'{p}_ls_n_h'] / nc_h, np.nan)
            X[f'{p}_rest_h'] = np.where(out[f'{p}_rr_n_h'] > 0, out[f'{p}_rest_sum_h'] / out[f'{p}_rr_n_h'], np.nan)
            X[f'{p}_ncov_d'] = nc_d
            X[f'{p}_rr_sh_d'] = np.where(nc_d > 0, out[f'{p}_rr_n_d'] / nc_d, np.nan)
            X[f'{p}_rr_n_d'] = out[f'{p}_rr_n_d']
            X[f'{p}_ls_sh_d'] = np.where(nc_d > 0, out[f'{p}_ls_n_d'] / nc_d, np.nan)
            X[f'{p}_ntaxi_h'] = out[f'{p}_ntaxi_h']
            X[f'{p}_taxi_h'] = np.where(out[f'{p}_ntaxi_h'] > 0, out[f'{p}_taxi_sum_h'] / out[f'{p}_ntaxi_h'], np.nan)
    for c in X.columns[1:]:
        X[c] = X[c].astype('float32')
    # own flags kept for the shift check / report only (never a candidate column)
    X['own_cov'] = f['cov'].astype('int8').to_numpy(); X['own_rr'] = f['rr'].astype('int8').to_numpy()
    X['own_ls'] = f['ls'].astype('int8').to_numpy()
    return X


def main():
    t0 = time.time()
    for which in ('training', 'ranking'):
        X = build(which)
        X.to_parquet(OUT / f'ind_{which}.parquet', index=False)
        print(f'{which}: {len(X):,} rows, own rr {int(X.own_rr.sum()):,}, own ls {int(X.own_ls.sum()):,}; '
              f'rows with dz_rr_sh_h > 0: {(X.dz_rr_sh_h > 0).mean():.4f}, > 0.15: {(X.dz_rr_sh_h > 0.15).mean():.4f}',
              flush=True)
        print(X.drop(columns=['MVT_ID_mvt']).describe(percentiles=[.5, .9, .99]).T.round(3).to_string(), flush=True)
    print(f'peak RSS {peak_rss_gb():.2f} GB | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
