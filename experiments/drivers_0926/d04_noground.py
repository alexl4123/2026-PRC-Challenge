"""Drivers 09-26, step 4: is the 'no ADS-B ground data' state at well-covered airports a shift risk for 2026?

In 2025 OOF, v17 is WORSE than v15 on ordinary rows WITHOUT ground data at EDDF/EDDM/EHAM/LSZH/LEBL (d03 (e)).
In 2026 the no-ground share rises at EDDF (2-7% -> 14%) and falls at EDDM July (59% -> 5%). This checks
(1) 2025: mean residual y - v17 and mean shift v17 - v15 on no-ground vs ground rows, per airport;
(2) 2026: mean shift v17 - v16 (the shipped files) on the same status groups;
(3) whether no-ground rows are day-clustered (feeder outage) or spread (stands out of view), both years.
No labels of 2026 are used (there are none).

    python experiments/drivers_0926/d04_noground.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
from d02_decompose import adsb_status

OUT = CACHE / 'drivers_0926'
pd.set_option('display.width', 220)


def main():
    r = pd.read_parquet(OUT / 'rows_training.parquet'); k = pd.read_parquet(OUT / 'rows_ranking.parquet')
    r['adsb'] = adsb_status(r); k['adsb'] = adsb_status(k)
    v15 = np.load(CACHE / 'w2' / 'v15s_oof.npy').astype(float)
    ok = (r.route == 'ABC').to_numpy() & (r.y.to_numpy() <= 7200)
    r['res'] = r.y - r.pred; r['shift'] = r.pred - v15
    ids = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    s16 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v16.parquet'); s17 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v17.parquet')
    idc = [c for c in s17.columns if 'ID' in c.upper()][0]; pc = [c for c in s17.columns if c != idc][0]
    p16 = s16.set_index(idc)[pc].reindex(ids).to_numpy(float); p17 = s17.set_index(idc)[pc].reindex(ids).to_numpy(float)
    assert np.isfinite(p16).all() and np.isfinite(p17).all()
    k['shift'] = p17 - p16
    kk = (k.route == 'ABC').to_numpy()
    apts = ['EDDF', 'EDDM', 'EHAM', 'LSZH', 'LEBL', 'LEMD', 'EGLL', 'LIRF', 'LFPG']
    t = (r[ok & r.apt.isin(apts).to_numpy()].groupby(['apt', 'adsb'], observed=True)
         .agg(rows25=('res', 'size'), resid_mean25=('res', 'mean'), shift_v17_v15_25=('shift', 'mean')))
    t2 = k[kk & k.apt.isin(apts).to_numpy()].groupby(['apt', 'adsb'], observed=True).agg(rows26=('shift', 'size'),
                                                                                        shift_v17_v16_26=('shift', 'mean'))
    t = t.join(t2); t.to_csv(OUT / 't14_noground.csv')
    print('ordinary A/B/C rows (2025: y <= 2 h): mean residual and mean shift by ADS-B status (s)'); print(t.round(1).to_string())
    for name, x in (('2025', r[ok]), ('2026', k[kk])):
        g = x[x.apt.isin(['EDDF', 'EHAM', 'LSZH', 'EDDM']).to_numpy()].assign(ng=lambda v: v.adsb == 'no ground')
        dd = g.groupby([g.apt.astype(str), g.date_loc]).ng.mean()
        q = dd.groupby(level=0).agg(days=('size'), med=('median'), p90=(lambda v: v.quantile(0.9)), days_gt50=(lambda v: int((v > 0.5).sum())))
        print(f'\n{name}: per-day share of no-ground rows (ordinary rows)'); print(q.round(3).to_string())


if __name__ == '__main__':
    main()
