"""Where do the larger January-2026 corrections sit? Airport, route, day, and the feature values behind them."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.io import load
from prc.config import CACHE, ID
OUT = CACHE / 'v11'; FIN = CACHE / 'final'
b = np.load(OUT / 'base.npz'); P, month, y, base = b['P'], b['month'], b['y'], b['base']
Ftr = pd.read_parquet(OUT / 'feat_training.parquet'); Frk = pd.read_parquet(OUT / 'feat_ranking.parquet')
atr = load('training', phase='DEP', columns=['APT', 'MVT_TIME_UTC_mvt']); ark = load('ranking', phase='DEP', columns=['APT', 'MVT_TIME_UTC_mvt'])
z = np.load(FIN / 'stack_c_ranking.npz'); c_rk = z['corr']; c_oof = np.load(OUT / 'oof_c.npy').astype(float)
rows = []
for nm, F, A, cc, keep in (('2025', Ftr, atr, c_oof, ~P), ('2026', Frk, ark, c_rk, ~z['in_P'])):
    for m in (1, 7):
        k = keep & (F.month.to_numpy() == m)
        df = pd.DataFrame(dict(apt=A.APT.astype(str).to_numpy()[k], c=cc[k], big=np.abs(cc[k]) > 300, D=F.aobt_missing.to_numpy()[k] == 1))
        g = df.groupby('apt').agg(n=('c', 'size'), mean=('c', 'mean'), sd=('c', 'std'), big=('big', 'sum'))
        g.columns = pd.MultiIndex.from_product([[f'{nm}-{m:02d}'], g.columns]); rows.append(g)
print('stacker (c) correction by airport, outside P'); print(pd.concat(rows, axis=1).round(1).to_string())
# January 2026 big corrections: by day, and what their features look like
k = ~z['in_P'] & (Frk.month.to_numpy() == 1)
day = pd.to_datetime(ark.MVT_TIME_UTC_mvt.to_numpy()[k]).day
big = np.abs(c_rk[k]) > 300
print('\nJan 2026 |corr|>300 by day (top 8):', pd.Series(day[big]).value_counts().head(8).to_dict(), ' total', int(big.sum()))
cols = ['nm_delay', 'off_aobt3', 'off_sched', 'sl_to', 'sn_minus_to', 'si_before_sched', 'pa_inb_m_sibt']
f = Frk.loc[k, cols]
print('Jan 2026 rows with |corr|>300: feature medians vs all Jan 2026 rows vs 2025 Jan rows')
kt = ~P & (month == 1)
print(pd.DataFrame({'big 2026': f[big].median(), 'all 2026 Jan': f.median(), '2025 Jan': Ftr.loc[kt, cols].median()}).round(0).to_string())
# is the Jan-2025 OOF also larger on the equivalent (large nm_delay) rows? residual check on 2025 Jan rows with |corr|>300
kb = kt & (np.abs(c_oof) > 300)
print(f'\n2025 Jan OOF rows with |corr|>300: n {kb.sum()}, mean corr {c_oof[kb].mean():+.0f}, mean residual y-base {np.mean((y-base)[kb]):+.0f}, '
      f'RMSE base {np.sqrt(np.mean((y-base)[kb]**2)):.0f} -> with corr {np.sqrt(np.mean((y-base-c_oof)[kb]**2)):.0f}')
