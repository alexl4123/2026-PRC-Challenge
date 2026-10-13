"""ids_05 -- inside each (airport, month, phase) MVT_ID batch: order key, gaps (excluded source rows), tie-breaks."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
d = pd.read_parquet(H + 'cache/hunt/ids_04_sorted.parquet')
d['APT'] = d.APT.astype(str); d['PHASE_mvt'] = d.PHASE_mvt.astype(str)
key = d.APT + '|' + d.PHASE_mvt + '|' + d.mvt_month
d['batch'] = (key != key.shift()).cumsum()
print('batches', d.batch.nunique(), ' distinct keys', key.nunique())
# a batch may contain rows whose mvt_month differs (month-boundary) -> check by APT|PHASE runs
k2 = d.APT + '|' + d.PHASE_mvt
d['run'] = (k2 != k2.shift()).cumsum()
r = d.groupby('run').agg(apt=('APT', 'first'), ph=('PHASE_mvt', 'first'), n=('MVT_ID_mvt', 'size'),
                         idmin=('MVT_ID_mvt', 'min'), idmax=('MVT_ID_mvt', 'max'),
                         m_mode=('mvt_month', lambda s: s.mode().iloc[0]), m_n=('mvt_month', 'nunique'),
                         smin=('SCHED', 'min'), smax=('SCHED', 'max'), mmin=('MVT_T', 'min'), mmax=('MVT_T', 'max'))
r['span'] = r.idmax - r.idmin + 1
r['missing'] = r.span - r.n
for c in ['smin', 'smax', 'mmin', 'mmax']:
    r[c] = pd.to_datetime(r[c], unit='s')
print(r[r.apt.isin(['EDDF', 'LIRF', 'EHAM'])].to_string())
print('missing share by apt/phase:\n', r.groupby(['apt', 'ph'])[['n', 'missing']].sum().assign(sh=lambda x: x.missing / (x.n + x.missing)).round(4).to_string())

# per run (= batch), spearman of within-batch id order with SCHED / MVT / BLOCK, by phase
rows = []
for run, g in d.groupby('run'):
    rr = {'apt': g.APT.iat[0], 'ph': g.PHASE_mvt.iat[0], 'm': g.mvt_month.mode().iat[0]}
    for k in ['SCHED', 'MVT_T', 'BLOCK']:
        if g[k].notna().sum() > 100:
            rr[k] = spearmanr(g.MVT_ID_mvt, g[k], nan_policy='omit')[0]
    # share of adjacent pairs that are SCHED-nondecreasing
    s = g.SCHED.values
    rr['adj_sched_up'] = np.mean(np.diff(s) >= 0)
    m = g.MVT_T.values
    rr['adj_mvt_up'] = np.mean(np.diff(m) >= 0)
    rows.append(rr)
t = pd.DataFrame(rows)
print(t.groupby(['apt', 'ph'])[['SCHED', 'MVT_T', 'BLOCK', 'adj_sched_up', 'adj_mvt_up']].median().round(4).to_string())
# print a sample stretch of a few batches
for apt in ['EDDF', 'EGLL', 'LEBL', 'EHAM', 'LTFM']:
    g = d[(d.APT == apt) & (d.PHASE_mvt == 'DEP') & (d.mvt_month == '2025-07')].head(40)
    print('\n', apt, 'DEP 2025-07 first 40 rows in MVT_ID order')
    print(g.assign(SCHED=pd.to_datetime(g.SCHED, unit='s'), MVT=pd.to_datetime(g.MVT_T, unit='s'),
                   BLOCK=pd.to_datetime(g.BLOCK, unit='s'))[['MVT_ID_mvt', 'FLIGHT_mvt', 'SCHED', 'MVT', 'BLOCK']].to_string())
d[['MVT_ID_mvt', 'run']].to_parquet(H + 'cache/hunt/ids_05_runs.parquet')
r.to_parquet(H + 'cache/hunt/ids_05_runtable.parquet')
