"""m03b -- what orders FLIGHT_ID_mvt / MVT_ID_mvt? Within airport-day Spearman with each clock."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from prc.config import CACHE
M = CACHE / 'major'
for which in ['training', 'ranking']:
    d = pd.read_parquet(M / f'dep_{which}.parquet')
    print(which, 'rows', len(d), 'FLIGHT_ID non-null', d.FLIGHT_ID_mvt.notna().mean().round(4),
          'unique', d.FLIGHT_ID_mvt.nunique(), 'MVT_ID non-null', d.MVT_ID_mvt.notna().mean().round(4),
          'unique', d.MVT_ID_mvt.nunique())
    print('  FLIGHT_ID min/max', d.FLIGHT_ID_mvt.min(), d.FLIGHT_ID_mvt.max(), ' MVT_ID min/max', d.MVT_ID_mvt.min(), d.MVT_ID_mvt.max())
d = pd.read_parquet(M / 'dep_training.parquet')
d = d[d.month.isin([3, 7])].copy()
for c in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt']:
    d[c + '_s'] = d[c].astype('int64') / 1e6 if d[c].notna().all() else d[c].map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
d['day'] = d.MVT_TIME_UTC_mvt.dt.floor('D')
res = {}
for idc in ['FLIGHT_ID_mvt', 'MVT_ID_mvt']:
    rows = []
    for (a, day), g in d.groupby(['APT', 'day'], observed=True):
        if len(g) < 50: continue
        r = {'apt': a}
        for c in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt']:
            r[c.split('_')[0]] = spearmanr(g[idc], g[c + '_s'], nan_policy='omit')[0]
        rows.append(r)
    t = pd.DataFrame(rows).groupby('apt').median().round(4)
    print('\nwithin airport-day median spearman of', idc, 'with each clock'); print(t.to_string())
