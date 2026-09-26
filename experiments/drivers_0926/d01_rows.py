"""Drivers 09-26, step 1: one compact per-row table for the 2025 OOF (with v17-structure error) and the 2026
ranking set (label-free columns only), so every later decomposition reads the same cells.

Inputs (read only):
  cache/w2/v17s_oof.npy          v17-structure 12-fold LOMO OOF (experiments/w2/v17_anatomy.py), DEP order
  cache/v11/base.npz             y, month, P, D in the same order
  cache/major/dep_{training,ranking}.parquet   clocks + off_* differences, same order as load(.., 'DEP')
  cache/adsb_lol/features_{training,ranking}.parquet   adsb.lol columns (joined BY MVT_ID: the training file
                                                       is NOT in DEP order -- checked 09-26)
Outputs:
  cache/drivers_0926/rows_training.parquet, rows_ranking.parquet

    python experiments/drivers_0926/d01_rows.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, TAIL_CAP
from prc.io import peak_rss_gb
from prc.models.routed import route_of

OUT = CACHE / 'drivers_0926'; OUT.mkdir(exist_ok=True)
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam',
      'LEBL': 'Europe/Madrid', 'LEMD': 'Europe/Madrid', 'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome',
      'LSZH': 'Europe/Zurich', 'LTFM': 'Europe/Istanbul'}
ADSB = ['adsb_matched', 'adsb_has_ground', 'adsb_off_best', 'adsb_best_src', 'adsb_day_ok', 'adsb_off_first']
DEP = ['MVT_ID_mvt', 'APT', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_OPERATOR_flt', 'MVT_TIME_UTC_mvt',
       'SCHED_TIME_UTC_mvt', 'off_sched', 'off_aobt3', 'off_eobt1', 'off_iobt', 'month']


def local_parts(apt, t_utc):
    """Local hour and local calendar date (as int yyyymmdd) per airport."""
    hour = np.full(len(apt), -1, np.int16); date = np.zeros(len(apt), np.int32)
    t = pd.Series(pd.to_datetime(t_utc))
    if t.dt.tz is None:
        t = t.dt.tz_localize('UTC')
    for a, tz in TZ.items():
        m = (apt == a)
        if not m.any():
            continue
        lt = t[m].dt.tz_convert(tz)
        hour[m] = lt.dt.hour.to_numpy(); date[m] = (lt.dt.year * 10000 + lt.dt.month * 100 + lt.dt.day).to_numpy()
    return hour, date


def build(which):
    d = pd.read_parquet(CACHE / 'major' / f'dep_{which}.parquet', columns=DEP + (['y'] if which == 'training' else []))
    a = pd.read_parquet(CACHE / 'adsb_lol' / f'features_{which}.parquet', columns=['MVT_ID_mvt'] + ADSB)
    a = a.set_index('MVT_ID_mvt').reindex(d.MVT_ID_mvt.to_numpy())
    assert a.index.is_unique and len(a) == len(d)
    apt = d.APT.astype(str).to_numpy()
    r = pd.DataFrame({'apt': pd.Categorical(apt), 'month': d.month.to_numpy().astype(np.int8)})
    r['off_sched'] = d.off_sched.to_numpy(float); r['off_aobt3'] = d.off_aobt3.to_numpy(float)
    r['off_eobt1'] = d.off_eobt1.to_numpy(float); r['off_iobt'] = d.off_iobt.to_numpy(float)
    rt = route_of(r.off_aobt3.to_numpy())
    so = r.off_sched.to_numpy()
    P = (rt == 'D') & np.isfinite(so) & (so > TAIL_CAP)
    r['route'] = pd.Categorical(np.where(P, 'P', np.where(rt == 'D', 'D', 'ABC')))
    r['route4'] = pd.Categorical(rt)
    h_mvt, dt_mvt = local_parts(apt, d.MVT_TIME_UTC_mvt)
    h_sch, dt_sch = local_parts(apt, d.SCHED_TIME_UTC_mvt)
    r['hour_loc'] = h_mvt; r['date_loc'] = dt_mvt; r['sched_hour_loc'] = h_sch
    r['date_change'] = (dt_mvt > dt_sch)                        # takes off on a later local date than scheduled
    r['utc_date'] = pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.strftime('%Y%m%d').astype(int).to_numpy()
    r['stand_na'] = d.STAND_mvt.isna().to_numpy()
    off_best = a.adsb_off_best.to_numpy(float)
    r['adsb_off'] = np.isfinite(off_best)                        # an ADS-B off-block estimate exists
    r['adsb_ground'] = a.adsb_has_ground.fillna(0).to_numpy().astype(bool)
    r['adsb_matched'] = a.adsb_matched.fillna(0).to_numpy().astype(bool)
    r['adsb_taxi'] = off_best                                    # adsb_off_best is already MVT - off-block (s)
    r['adsb_src'] = a.adsb_best_src.astype(str).to_numpy()
    if which == 'training':
        b = np.load(CACHE / 'v11' / 'base.npz')
        f = np.load(CACHE / 'w2' / 'v17s_oof.npy').astype(float)
        assert np.allclose(b['y'], d.y.to_numpy()) and (b['P'] == P).all() and (b['D'] == (rt == 'D')).all()
        r['y'] = b['y']; r['pred'] = f
    return r


def main():
    t0 = time.time()
    for which in ('training', 'ranking'):
        r = build(which)
        r.to_parquet(OUT / f'rows_{which}.parquet', index=False)
        print(f'{which}: {len(r):,} rows, adsb_off {r.adsb_off.mean():.3f}, ground {r.adsb_ground.mean():.3f}, '
              f'routes {r.route.value_counts().to_dict()}')
        if which == 'training':
            print('  median adsb_taxi - y on rows with an estimate:',
                  float(np.nanmedian(r.adsb_taxi[r.adsb_off] - r.y[r.adsb_off])))
    print(f'peak RSS {peak_rss_gb():.2f} GB | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
