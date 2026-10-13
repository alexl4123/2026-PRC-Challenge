"""Does the raw row order (or MVT_ID order) carry BLOCK order beyond MVT/SCHED?"""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT', 'BLOCK', 'SCHED', 'LOBT'])
d['APT'] = d.APT.astype(str)
def mono(x, key):
    g = x.groupby('APT', sort=False)
    out = {}
    for c in ['MVT', 'BLOCK', 'SCHED', 'MVT_ID_mvt']:
        dif = g[c].diff()
        out[c] = float((dif >= 0).mean())
    return out
print('file order, consecutive rows same APT: share non-decreasing', mono(d, None))
s = d.sort_values('MVT_ID_mvt')
print('MVT_ID order:', mono(s, None))
# Within MVT_ID order: is BLOCK order better explained than by SCHED? Spearman of rank residuals in windows
from scipy.stats import spearmanr
for a in ['LIRF', 'EDDF', 'LFPG']:
    x = s[s.APT == a].head(200000)
    print(a, 'spearman(MVT_ID, BLOCK)', round(spearmanr(x.MVT_ID_mvt, x.BLOCK)[0], 6), ' (MVT_ID, SCHED)', round(spearmanr(x.MVT_ID_mvt, x.SCHED)[0], 6), ' (MVT_ID, MVT)', round(spearmanr(x.MVT_ID_mvt, x.MVT)[0], 6))
    # partial: within equal SCHED, does MVT_ID order follow BLOCK or MVT?
    x = x.copy(); x['r_id'] = x.groupby('SCHED').MVT_ID_mvt.rank(); x['r_b'] = x.groupby('SCHED').BLOCK.rank(); x['r_m'] = x.groupby('SCHED').MVT.rank()
    x['n'] = x.groupby('SCHED').MVT.transform('size'); z = x[x.n > 1]
    print('   within equal SCHED groups (n>1 rows', len(z), '): corr(rank id, rank BLOCK)', round(np.corrcoef(z.r_id, z.r_b)[0, 1], 4), ' corr(rank id, rank MVT)', round(np.corrcoef(z.r_id, z.r_m)[0, 1], 4))
