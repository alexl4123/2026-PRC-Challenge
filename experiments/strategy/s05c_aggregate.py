"""s05c -- pool the per-fold season-weight predictions (s05_m*.npz) into 12-month and scored-month
paired bootstraps (rows outside P, all y)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from prc.config import CACHE
from prc.validation import paired_bootstrap
y = np.load(CACHE / 'altfam_y.npy').astype(float); mon = np.load(CACHE / 'altfam_month.npy').astype(int)
te, pr = [], {'none': [], 's1.5': [], 's3': []}
for m in range(1, 13):
    z = np.load(CACHE / 'strategy' / f's05_m{m}.npz'); te.append(z['te'])
    for k in pr: pr[k].append(z[k])
te = np.concatenate(te); pr = {k: np.concatenate(v) for k, v in pr.items()}; yt = y[te]; mt = mon[te]
for arm in ['s1.5', 's3']:
    for scope, k in [('12 months', np.ones(len(te), bool)), ('scored 1+7', np.isin(mt, [1, 7])), ('winter 12+1+2', np.isin(mt, [12, 1, 2]))]:
        b = paired_bootstrap(yt[k], pr[arm][k], pr['none'][k])
        print(f'{arm:5s} {scope:14s} d {b["diff"]:+.3f} [{b["lo"]:+.3f},{b["hi"]:+.3f}]  p_worse {b["p_a_worse"]:.2f}')
    print('      months better:', sum(np.sqrt(np.mean((yt[mt==m]-pr[arm][mt==m])**2)) < np.sqrt(np.mean((yt[mt==m]-pr['none'][mt==m])**2)) for m in range(1,13)), '/ 12')
