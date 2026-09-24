"""m11c -- where did the state booster gain (trail vs null)? airport x month, day, route."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'y', 'off_aobt3', 'off_sched'])
f = pd.read_parquet(M / 'm11_state_training.parquet', columns=['hpT', 'dshT'])
n0 = np.load(M / 'm11_oof_null.npy'); t1 = np.load(M / 'm11_oof_trail.npy'); v9 = np.load(M / 'v9s_oof.npy')
y = d.y.to_numpy()
d['dsse'] = (y - t1) ** 2 - (y - n0) ** 2
d['dsse_v9'] = (y - t1) ** 2 - (y - v9) ** 2
d['D'] = d.off_aobt3.isna(); d['day'] = d.MVT_TIME_UTC_mvt.dt.day
g = d.groupby(['APT', 'month'], observed=True).dsse.sum().sort_values()
print('most improved airport-months (trail - null, SSE):'); print((g.head(8) / 1e8).round(2).to_string())
print('most worsened:'); print((g.tail(4) / 1e8).round(2).to_string())
top = g.index[0]
h = d[(d.APT == top[0]) & (d.month == top[1])]
dd = h.groupby('day').agg(n=('y', 'size'), D=('D', 'sum'), dsse=('dsse', 'sum'), ymean=('y', 'mean'))
dd['hpT_med'] = f.loc[h.index].groupby(h.day).hpT.median()
dd['dsh_med'] = f.loc[h.index].groupby(h.day).dshT.median()
print(f'\n{top}: by day (dsse in 1e8):'); dd['dsse'] /= 1e8; print(dd.sort_values('dsse').head(6).round(2).to_string())
print('\nSSE change by route x y band (trail - null, 1e8), 12 months:')
band = pd.cut(d.y, [0, 1800, 3600, 7200, 1e9])
print((d.groupby([d.D, band], observed=True).dsse.sum() / 1e8).round(2).to_string())
