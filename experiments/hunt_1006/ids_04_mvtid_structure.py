"""ids_04 -- how are MVT_IDs allocated? Batches per airport / month / phase; order key inside a batch."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from prc.io import load

H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
C = ['MVT_ID_mvt', 'APT', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'FLIGHT_mvt',
     'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt']
fr = []
for which in ['training', 'ranking']:
    d = load(which, columns=C); d['src'] = which[0]; fr.append(d)
d = pd.concat(fr, ignore_index=True); del fr
for c in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']:
    v = d[c].values.astype('datetime64[s]').astype('float64')
    v[d[c].isna().values] = np.nan
    d[c[:5]] = v
d['mvt_month'] = d.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m')
d['sch_month'] = d.SCHED_TIME_UTC_mvt.dt.strftime('%Y-%m')
d = d.sort_values('MVT_ID_mvt').reset_index(drop=True)
# runs of APT along MVT_ID
key = d.APT.astype(str) + '|' + d.PHASE_mvt.astype(str)
brk = (key != key.shift()).values
print('rows', len(d), 'runs of APT|PHASE along MVT_ID:', brk.sum(), ' runs of APT:', (d.APT != d.APT.shift()).sum())
rl = pd.Series(np.diff(np.r_[np.flatnonzero(brk), len(d)]))
print('APT|PHASE run length quantiles', rl.quantile([.5, .9, .99, 1]).to_dict())
# per airport-month span
g = d.groupby(['APT', 'mvt_month'], observed=True).MVT_ID_mvt.agg(['min', 'max', 'count'])
g['span'] = g['max'] - g['min']
print(g.head(30).to_string())
# id gap distribution between consecutive rows (all airports)
gap = np.diff(d.MVT_ID_mvt.values)
print('gap quantiles', np.quantile(gap, [.1, .5, .9, .99, .999]))
# ID vs month: per month, min/median/max ID, all rows
m = d.groupby('mvt_month').MVT_ID_mvt.agg(['min', 'median', 'max', 'count'])
print(m.to_string())
# Overlap: share of rows whose ID lies below the max ID of the previous mvt_month
# Within airport-month: spearman of ID rank with several keys
rows = []
for (a, mm), g in d[d.mvt_month.isin(['2025-03', '2025-07', '2026-01', '2026-07'])].groupby(['APT', 'mvt_month'], observed=True):
    r = {'apt': a, 'm': mm, 'n': len(g)}
    for k in ['MVT_T', 'SCHED', 'BLOCK']:
        r[k] = spearmanr(g.MVT_ID_mvt, g[k], nan_policy='omit')[0]
    r['flight'] = spearmanr(g.MVT_ID_mvt, g.FLIGHT_mvt.astype(str).rank(method='average'))[0]
    r['phase'] = spearmanr(g.MVT_ID_mvt, g.PHASE_mvt.eq('DEP'))[0]
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string())
d[['MVT_ID_mvt', 'APT', 'PHASE_mvt', 'src', 'MVT_T', 'SCHED', 'BLOCK', 'FLIGHT_mvt', 'mvt_month']].to_parquet(H + 'cache/hunt/ids_04_sorted.parquet')
