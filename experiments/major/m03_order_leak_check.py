"""m03 -- do row order or the id columns encode BLOCK time beyond what MVT/SCHED give?

A classic data-generation leak. If ids/rows were assigned in BLOCK order, residual
ordering information would predict y. Checked on the raw monthly training files
(original row order) and on the ids. Reported, not exploited: using such a leak
would be against the spirit of the rules and is flagged as such in the report.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from prc.config import DATA

f = sorted(glob.glob(str(DATA / 'training_*.parquet')))[6]   # one month, raw order
d = pd.read_parquet(f, columns=['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt',
                                'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                                'TAXITIME_SEC_mvt'])
print(f, len(d))
d['row'] = np.arange(len(d))
dep = d[d.PHASE_mvt == 'DEP'].copy()
for c in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']:
    dep[c] = dep[c].astype('int64') / 1e6
# is the file sorted by anything?
for c in ['row', 'MVT_ID_mvt', 'FLIGHT_ID_mvt']:
    for t in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']:
        print(f'  spearman({c}, {t}) = {spearmanr(dep[c], dep[t], nan_policy="omit")[0]:.4f}')
# residual test: within airport-day, after ordering by MVT, does the id rank predict y?
dep['day'] = (dep.MVT_TIME_UTC_mvt // 86400).astype(int)
out = []
for c in ['row', 'MVT_ID_mvt', 'FLIGHT_ID_mvt']:
    rs = []
    for (_, _), g in dep.groupby(['ADEP_mvt', 'day']):
        if len(g) < 30 or g[c].isna().all():
            continue
        # rank of id minus rank of MVT: positive = id says "earlier event than takeoff order implies"
        dr = g[c].rank() - g.MVT_TIME_UTC_mvt.rank()
        rs.append(spearmanr(dr, g.TAXITIME_SEC_mvt, nan_policy='omit')[0])
    out.append((c, np.nanmedian(rs), np.nanmean(rs)))
print('within airport-day: spearman(rank(id)-rank(MVT), y): median / mean')
for c, a, b in out:
    print(f'  {c:14s} {a:+.4f} {b:+.4f}')
