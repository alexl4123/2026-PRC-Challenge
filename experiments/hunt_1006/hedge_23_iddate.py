"""LIRF: the MVT_ID is ordered by DAY (spearman .9, ~45% within-day inversions).  Which day -- SCHED's, MVT's or BLOCK's?
For rows whose SCHED and MVT fall on different UTC days, locate the ID inside the per-day ID ranges of ordinary rows."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250)
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'off_sched', 'y'])
d['P'] = V['P']
res = []
for apt in ('LIRF', 'LFPG', 'EGLL', 'EHAM'):
    g = d[d.APT == apt].copy()
    for c, n in (('MVT_TIME_UTC_mvt', 'dm'), ('SCHED_TIME_UTC_mvt', 'ds'), ('BLOCK_TIME_UTC_mvt', 'db')):
        g[n] = g[c].dt.tz_convert('Europe/Rome' if apt == 'LIRF' else 'UTC').dt.tz_localize(None).dt.normalize()
    ordi = g[(g.dm == g.ds) & (g.ds == g.db)]
    rng = ordi.groupby('dm').MVT_ID_mvt.agg(['min', 'max', 'median'])
    x = g[g.dm != g.ds].copy()
    def where(row):
        out = []
        for k in ('ds', 'dm', 'db'):
            if row[k] in rng.index:
                lo, hi = rng.loc[row[k], 'min'], rng.loc[row[k], 'max']
                out.append(lo <= row.MVT_ID_mvt <= hi)
            else:
                out.append(np.nan)
        return pd.Series(out, index=['in_sched_day', 'in_mvt_day', 'in_block_day'])
    w = x.apply(where, axis=1)
    x = pd.concat([x, w], axis=1)
    x['cls'] = np.where(np.abs(x.y - x.off_sched) <= 120, 'stamp', np.where(x.y > 43200, 'day', 'normal'))
    print(apt, '(local days for LIRF)' if apt == 'LIRF' else '(UTC days)', 'rows with SCHED day != MVT day:', len(x))
    print(pd.crosstab([x.cls, x.db == x.ds], [x.in_sched_day, x.in_mvt_day]))
