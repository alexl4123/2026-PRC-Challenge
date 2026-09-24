"""v11 step 1 -- the BASE: twelve-fold OOF of the v10 family blend, outside P.

Computed exactly as experiments/exp66_family_blend.py computes `blend` (seed-averaged
study families, per-route NNLS weights from cache/exp66_weights_perroute_all.json),
and the production structure `prod` (LGB + network, v9 weights 0.65 / 0.5).

SANITY (check 4): blend - prod on the 8 held-out months outside P must reproduce
exp66's -1.556 (paired bootstrap, 500 resamples, seed 0).

Writes cache/v11/base.npz: base (= blend), prod, y, month, P, D  (DEP order of
load('training', phase='DEP')).
    python experiments/v11/v11_base.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from prc.io import load, peak_rss_gb
from prc.config import TARGET, TAIL_CAP, CACHE
from prc.validation import paired_bootstrap
from prc.models.routed import route_of
from exp66_family_blend import family_oof, SCREEN, SCORED

t0 = time.time()
OUT = CACHE / 'v11'; OUT.mkdir(exist_ok=True)
d = load('training', phase='DEP', columns=[TARGET, 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'])
y = d[TARGET].astype(float).to_numpy()
month = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
D = route_of((d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds().to_numpy()) == 'D'
so = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
P = D & np.isfinite(so) & (so > TAIL_CAP)
n = len(y)
W = json.load(open(CACHE / 'exp66_weights_perroute_all.json'))
names = W['members']
X = []
for fam in names:
    o, k = family_oof(fam, month, n)
    if o is None or k != 3:
        sys.exit(f'FATAL: family {fam} has {k} complete seeds (need 3) in cache/cv/study')
    X.append(o); print(f'  {fam:<8} seeds {k}')
X = np.column_stack(X)
blend = np.empty(n)
for gname, g in (('route D', D), ('routes A/B/C', ~D)):
    w = np.array([W['weights'][gname][nm] for nm in names])
    blend[g] = X[g] @ w
    print(f'  weights {gname}: ' + '  '.join(f'{nm}={wt:.4f}' for nm, wt in zip(names, w)))
wv = np.where(D, 0.5, 0.65)
prod = (1 - wv) * X[:, names.index('mix')] + wv * X[:, names.index('nn')]
print('\n  CHECK 4 -- base (family blend) minus production (LGB+network, v9 weights), outside P:')
for scope, k in (('8 held-out months', ~np.isin(month, SCREEN) & ~P), ('scored months 1+7', np.isin(month, SCORED) & ~P)):
    b = paired_bootstrap(y[k], blend[k], prod[k], n_boot=500)
    r = lambda p: np.sqrt(np.mean((y[k] - p[k]) ** 2))
    print(f'    {scope:<19} {r(prod):7.2f} -> {r(blend):7.2f}   {b["diff"]:+.3f} +- {b["sd"]:.3f}  CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]')
np.savez(OUT / 'base.npz', base=blend, prod=prod, y=y, month=month, P=P, D=D)
print(f'\n  saved cache/v11/base.npz | outside-P RMSE base {np.sqrt(np.mean((y[~P]-blend[~P])**2)):.3f} | '
      f'peak RSS {peak_rss_gb():.2f} GB | {time.time()-t0:.0f}s')
