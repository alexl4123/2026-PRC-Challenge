"""s14 -- how much would more seeds buy?  Measured on the cluster study OOFs (3 seeds per family).

For the network (nn) and LightGBM mixture (mix), and for a 0.35/0.65 mix+nn blend, compute the
RMSE outside P as a function of the number of seeds averaged (every subset of size k), fit
MSE(k) = A + B/k, and extrapolate to 5 and 10 seeds.  Outside P only (P is served by the specialist).
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE

M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['month', 'y', 'off_sched', 'off_aobt3'])
y = d.y.to_numpy(); mon = d.month.to_numpy(); P = ~np.isfinite(d.off_aobt3.to_numpy()) & (d.off_sched.to_numpy() > 7200)
o = ~P; S = np.isin(mon, [1, 7])
ST = CACHE / 'cv' / 'study'


def fam(f, s):
    out = np.full(len(y), np.nan)
    for m in range(1, 13):
        out[mon == m] = np.load(ST / f'{f}__seed{s:02d}__m{m:02d}.npy')
    return out


A = {f: [fam(f, s) for s in range(3)] for f in ['nn', 'mix']}
rm = lambda p, k: np.sqrt(np.mean((y[k] - p[k]) ** 2))
for scope, k in [('outside P, 12m', o), ('outside P, scored', o & S), ('outside P bulk y<=7200, 12m', o & (y <= 7200))]:
    print(f'\n=== {scope}')
    for nm, getp in [('nn', lambda sub: np.mean([A['nn'][i] for i in sub], 0)),
                     ('mix', lambda sub: np.mean([A['mix'][i] for i in sub], 0)),
                     ('0.35 mix(3) + 0.65 nn(k)', lambda sub: 0.35 * np.mean(A['mix'], 0) + 0.65 * np.mean([A['nn'][i] for i in sub], 0))]:
        ks, mses = [], []
        for kk in [1, 2, 3]:
            v = [rm(getp(sub), k) ** 2 for sub in itertools.combinations(range(3), kk)]
            ks.append(kk); mses.append(np.mean(v))
        Bc, Ac = np.polyfit(1 / np.array(ks), mses, 1)
        ext = {kk: np.sqrt(Ac + Bc / kk) for kk in [5, 10, 1000]}
        print(f'  {nm:26s} k=1 {np.sqrt(mses[0]):.2f}  k=2 {np.sqrt(mses[1]):.2f}  k=3 {np.sqrt(mses[2]):.2f}  | fit: k=5 {ext[5]:.2f}  k=10 {ext[10]:.2f}  '
              f'k=inf {ext[1000]:.2f}  (3->10: {ext[10]-np.sqrt(mses[2]):+.2f})')
