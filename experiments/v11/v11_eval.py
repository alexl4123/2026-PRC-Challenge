"""v11 step 4 -- evaluate the stacker arms against the v10 family-blend base, OUTSIDE P.
Paired bootstrap (500, seed 0) on 12 months and on scored months 1+7; per-month sign counts;
pairwise (c)-(b), (b)-(a), diagnostics (a)-(a_pb), null-base; per-airport contributions.
    python experiments/v11/v11_eval.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.io import load, peak_rss_gb
from prc.config import CACHE
from prc.validation import paired_bootstrap

t0 = time.time()
OUT = CACHE / 'v11'
b = np.load(OUT / 'base.npz'); base, y, month, P = b['base'], b['y'], b['month'], b['P']
apt = load('training', phase='DEP', columns=['APT']).APT.astype(str).to_numpy()
o = ~P
pred = {'base': base}
for a in ['null', 'a_pb', 'a', 'b', 'c']:
    c = np.load(OUT / f'oof_{a}.npy').astype(float)
    assert np.isfinite(c[o]).all() and np.isnan(c[P]).all()
    pred[a] = np.where(P, base, base + np.nan_to_num(c))
rm = lambda p, k: np.sqrt(np.mean((y[k] - p[k]) ** 2))
sc = np.isin(month, [1, 7])
def line(a, ref):
    out = []
    for scope, k in (('12 months', o), ('scored 1+7', o & sc)):
        bb = paired_bootstrap(y[k], pred[a][k], pred[ref][k], n_boot=500)
        out.append(f'{scope}: {rm(pred[ref], k):.3f} -> {rm(pred[a], k):.3f}  {bb["diff"]:+.3f} +- {bb["sd"]:.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
    dm = [rm(pred[a], o & (month == m)) - rm(pred[ref], o & (month == m)) for m in range(1, 13)]
    neg = sum(v < 0 for v in dm)
    return ' | '.join(out) + f' | better in {neg}/12 months', dm
print('ARMS vs BASE (v10 family blend), rows OUTSIDE P, negative = better')
res = {}
for a in ['a', 'b', 'c', 'null', 'a_pb']:
    s, dm = line(a, 'base'); res[a] = dm
    print(f'  ({a:4s}) {s}')
print('\nPAIRWISE')
for a, ref, lab in [('c', 'b', '(c)-(b)  look-ahead on top of backward'), ('b', 'a', '(b)-(a)  backward turnover on top of core'),
                    ('c', 'a', '(c)-(a)  full turnover on top of core'), ('a', 'a_pb', '(a)-(a_pb) cost of NOT using the base prediction as input')]:
    s, dm = line(a, ref); res[f'{a}-{ref}'] = dm
    print(f'  {lab}\n      {s}')
print('\nPER MONTH (RMSE difference, outside P)')
print(pd.DataFrame({k: np.round(v, 2) for k, v in res.items()}, index=[f'm{m:02d}' for m in range(1, 13)]).T.to_string())
print('\nPER AIRPORT (contribution to the 12-month outside-P RMSE change; sums to the total)')
N = o.sum(); rows = {}
for a, ref in [('a', 'base'), ('b', 'base'), ('c', 'base'), ('c', 'b')]:
    e0 = (y - pred[ref]) ** 2; e1 = (y - pred[a]) ** 2; tot0 = e0[o].sum()
    rows[f'{a}-{ref}'] = {ap: round(float(np.sqrt((tot0 + (e1 - e0)[o & (apt == ap)].sum()) / N) - np.sqrt(tot0 / N)), 3) for ap in sorted(set(apt))}
print(pd.DataFrame(rows).T.to_string())
# context: route D rows and y bands
print('\nBY ROUTE (outside P): contribution to the 12-month change')
D = b['D']
for a, ref in [('c', 'base'), ('c', 'b')]:
    e0 = (y - pred[ref]) ** 2; e1 = (y - pred[a]) ** 2; tot0 = e0[o].sum()
    print(f'  {a}-{ref}: ' + '  '.join(f'{nm} {np.sqrt((tot0 + (e1 - e0)[o & msk].sum()) / N) - np.sqrt(tot0 / N):+.3f}' for nm, msk in [('route A/B/C', ~D), ('route D', D)]))
print(f'\npeak RSS {peak_rss_gb():.2f} GB | {time.time()-t0:.0f}s')
