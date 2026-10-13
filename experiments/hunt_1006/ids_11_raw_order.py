"""ids_11 -- raw row order of the delivered files (training monthly files, ranking, submitting template):
what is it sorted by, and does the position carry BLOCK information beyond MVT/SCHED?"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')
D = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
H = D + 'prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
     'SCHED_TIME_UTC_mvt']
for f in ['training_2025-07-01_2025-08-01.parquet', 'ranking.parquet']:
    d = pd.read_parquet(D + f, columns=C)
    d['row'] = np.arange(len(d))
    print('=====', f, len(d))
    print(d.head(12).to_string())
    for k in ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']:
        v = d[k]
        if v.notna().sum() == 0:
            continue
        vv = v.astype('int64') if 'TIME' in k else v
        vv = vv.where(v.notna())
        print(f'  spearman(row, {k}) = {spearmanr(d.row, vv, nan_policy="omit")[0]:.4f};  adjacent nondecreasing {np.mean(np.diff(vv.dropna().values) >= 0):.4f}')
    ph = d.PHASE_mvt.astype(str)
    print('  phase runs', (ph != ph.shift()).sum(), ' ADEP runs', (d.ADEP_mvt.astype(str) != d.ADEP_mvt.astype(str).shift()).sum())
    # is it sorted by something like a hash? check rows sorted by MVT_ID within phase-airport? compute fraction of
    # adjacent rows with the same airport
    apt = np.where(ph == 'DEP', d.ADEP_mvt.astype(str), d.ADES_mvt.astype(str))
    print('  adjacent same airport share', np.mean(apt[1:] == apt[:-1]), ' adjacent same phase', np.mean(ph.values[1:] == ph.values[:-1]))
    if f.startswith('training'):
        dep = d[ph.eq('DEP')].copy()
        dep['apt'] = dep.ADEP_mvt.astype(str)
        dep['y'] = (dep.MVT_TIME_UTC_mvt - dep.BLOCK_TIME_UTC_mvt).dt.total_seconds()
        dep['day'] = dep.MVT_TIME_UTC_mvt.dt.floor('D')
        rr = []
        for (a, dd), g in dep.groupby(['apt', 'day']):
            if len(g) < 50:
                continue
            r_row = g.row.rank(); r_m = g.MVT_TIME_UTC_mvt.rank(); r_b = g.BLOCK_TIME_UTC_mvt.rank(); r_s = g.SCHED_TIME_UTC_mvt.rank()
            rr.append({'apt': a, 'rho_row_mvt': spearmanr(r_row, r_m)[0], 'rho(row-mvt, y)': spearmanr(r_row - r_m, g.y)[0],
                       'rho(row-sched, y)': spearmanr(r_row - r_s, g.y)[0], 'rho(row-mvt, block-mvt rank)': spearmanr(r_row - r_m, r_b - r_m)[0]})
        print(pd.DataFrame(rr).groupby('apt').median().round(4).to_string())
s = pd.read_parquet(D + 'submitting.parquet')
print('===== submitting.parquet', s.shape, list(s.columns))
print(s.head().to_string())
rk = pd.read_parquet(D + 'ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt'])
rk = rk[rk.PHASE_mvt.eq('DEP')]
print('submitting MVT_ID order == ranking DEP order:', len(s) == len(rk) and np.array_equal(s.MVT_ID_mvt.values, rk.MVT_ID_mvt.values))
print('submitting sorted by MVT_ID:', np.all(np.diff(s.MVT_ID_mvt.values) > 0))
