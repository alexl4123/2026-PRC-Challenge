"""Value of EUROCONTROL's monthly 'Taxi-Out Additional Time' file (mean taxi-out per airport-month) as a calibration of
our predictions -- measured on the 2025 analogue: the 2025 file against the v28-structure 2025 out-of-fold predictions.
User decision 2026-10-09 (organiser approved twice). Published mean m_pub(a,mo) = (TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN)
/ TOTAL_REF_NB_FL * 60 s over flights EUROCONTROL keeps (it drops flights de-iced after off-block and taxi-out > 120 min).
Variants (shift applied to rows with prediction <= 7200 s, additive, per airport-month):
  raw     delta = m_pub - mean(pred | pred <= 7200)
  gap     delta = m_pub - gap(a) - mean(pred | pred <= 7200), gap(a) = mean over the OTHER 11 months of (m_pub - mean(y | y<=7200))
  x 0.5   the same, halved.
    python experiments/eurocontrol/ec01_value.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.validation import block_bootstrap
Z = np.load(CACHE / 'w2' / 'v28s_oof.npz'); y, month, f = Z['y'], Z['month'], Z['f']
dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT']); apt = dd.APT.astype(str).to_numpy()
e = pd.read_csv(CACHE / 'external' / 'eurocontrol' / 'taxi_out_additional_time_2025.csv')
e['m_pub'] = (e.TOTAL_REF_TIME_MIN + e.TOTAL_ADD_TIME_MIN) / e.TOTAL_REF_NB_FL * 60
pub = e.set_index(['APT_ICAO', 'MONTH_NUM']).m_pub
P = pd.Series([pub.get((a, m), np.nan) for a, m in zip(apt, month)]).to_numpy()
t = pd.DataFrame({'a': apt, 'm': month, 'y': y, 'f': f, 'P': P})
my = t[t.y <= 7200].groupby(['a', 'm']).y.mean(); mf = t[t.f <= 7200].groupby(['a', 'm']).f.mean()
G = (pub.rename_axis(['a', 'm']).reindex(my.index) - my)
print('published minus actual mean (y <= 7200), s: by airport mean / sd over months')
print(G.groupby(level=0).agg(['mean', 'std']).round(1).T.to_string())
print('our OOF mean error (pred - y, y<=7200 rows) by airport, s:', t[t.y <= 7200].assign(e=lambda q: q.f - q.y).groupby('a').e.mean().round(1).to_dict())
rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc'])
block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
held, S = ~np.isin(month, [3, 6, 9, 12]), np.isin(month, [1, 7])
r = lambda g, k: float(np.sqrt(np.mean((y[k] - g[k]) ** 2)))
for name, use_gap in (('raw', False), ('gap', True)):
    for alpha in (1.0, 0.5):
        g = f.copy()
        for (a, m), mfv in mf.items():
            gap = G.xs(a, level=0).drop(m, errors='ignore').mean() if use_gap else 0.0
            d = alpha * (pub.get((a, m), np.nan) - gap - mfv)
            k = (apt == a) & (month == m) & (f <= 7200)
            g[k] = np.maximum(f[k] + d, 0)
        res = {sc: block_bootstrap(y[k], g[k], f[k], block[k], n_boot=500) for sc, k in (('12m', np.ones(len(y), bool)), ('held', held), ('scored', S))}
        mb = sum(r(g, month == m) < r(f, month == m) for m in range(1, 13))
        print(f'{name:4s} x{alpha}: ' + '  '.join(f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items())
              + f'  Jan {r(g, month==1)-r(f, month==1):+.3f} Jul {r(g, month==7)-r(f, month==7):+.3f}  months better {mb}/12')
