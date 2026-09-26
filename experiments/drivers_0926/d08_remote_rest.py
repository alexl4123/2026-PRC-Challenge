"""Drivers 09-26, step 8: is the ADS-B 'remote rest' pattern (de-icing / remote hold) a label-free disruption signal?

remote rest = adsb_best_src == 1 (rest, no push seen), adsb_stat_s > 600, adsb_d_stand_first > 300 m.
Reports (1) v17 OOF residual on those flights per airport, (2) per airport-hour (>= 10 ordinary A/B/C flights):
the share of remote-rest flights vs the hour's ordinary-flight MSE relative to the airport median.

    python experiments/drivers_0926/d08_remote_rest.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE

OUT = CACHE / 'drivers_0926'


def main():
    r = pd.read_parquet(OUT / 'rows_training.parquet')
    ids = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    a = pd.read_parquet(CACHE / 'adsb_lol' / 'features_training.parquet',
                        columns=['MVT_ID_mvt', 'adsb_best_src', 'adsb_stat_s', 'adsb_d_stand_first']).set_index('MVT_ID_mvt').reindex(ids)
    r['rr'] = (a.adsb_best_src.to_numpy() == 1) & (a.adsb_stat_s.to_numpy() > 600) & (a.adsb_d_stand_first.to_numpy() > 300)
    r['res'] = r.y - r.pred; r['e2'] = r.res ** 2
    ok = r.y <= 7200
    print(r[ok].groupby(['apt', 'rr'], observed=True).agg(n=('res', 'size'), mean_res=('res', 'mean'),
                                                        rmse=('e2', lambda v: np.sqrt(v.mean()))).round(1).to_string())
    r['hkey'] = r.apt.astype(str) + '_' + r.date_loc.astype(str) + '_' + r.hour_loc.astype(str)
    h = r[ok & (r.route == 'ABC')].groupby('hkey').agg(n=('res', 'size'), rr_share=('rr', 'mean'), mse=('e2', 'mean'), apt=('apt', 'first'))
    h = h[h.n >= 10]; h['ratio'] = h.mse / h.groupby('apt').mse.transform('median')
    print('airport-hours', len(h), 'Spearman(rr_share, MSE ratio)', round(h[['rr_share', 'ratio']].corr('spearman').iloc[0, 1], 3))
    print(h.groupby(pd.cut(h.rr_share, [-.01, 0, .05, .15, 1])).ratio.agg(['size', 'median', 'mean']).round(2).to_string())


if __name__ == '__main__':
    main()
