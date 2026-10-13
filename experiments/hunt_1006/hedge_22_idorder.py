"""LIRF departures: is MVT_ID order = take-off (MVT) order?  If so, ID position carries no BLOCK (label) information for
the stamp / day-shift question.  Rank agreement per month, and the day-shift / stamp rows' ID rank vs MVT rank vs BLOCK rank."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'month', 'off_sched', 'y'])
d = d[d.APT == 'LIRF'].copy()
for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'):
    d[c[:4]] = d[c].astype('int64')
out = []
for m, g in d.groupby('month'):
    r = g.MVT_ID_mvt.rank()
    out.append((m, len(g), round(r.corr(g.MVT_.rank()), 5), round(r.corr(g.SCHE.rank()), 5), round(r.corr(g.BLOC.rank()), 5),
                round(float((np.diff(g.sort_values('MVT_ID_mvt').MVT_.to_numpy()) < 0).mean()), 4)))
print('month, n, spearman(ID, MVT), (ID, SCHED), (ID, BLOCK), share of MVT inversions along ID order'); [print(o) for o in out]
cls = np.where(np.abs(d.y - d.off_sched) <= 120, 'stamp', np.where(d.y > 43200, 'day', 'normal'))
for c in ('day', 'stamp'):
    g = d[(cls == c) & (d.off_sched > 43200)]
    print(c, 'rows', len(g))
