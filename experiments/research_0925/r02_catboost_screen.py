"""r02 -- screen: does CatBoost add to the six-family blend? (old, pre-W2 features; read-only)

The cluster study left 22 of 36 CatBoost fold files (months 1-11 covered by 1-3 seeds, month 12 by none),
so CatBoost was never evaluated as a blend member. A competitor (kind-mango, public repo) reports
-2.87 s live from a CatBoost member blended 50/50 with its LightGBM base. This asks the cheap question on
OUR members: next to mix, nn, perapt, lintree, routed, xgboost (study OOFs, seeds 0-2), does a seventh
CatBoost member earn weight and cut held-out error?

PROTOCOL (mirrors exp66 --per-route, adapted to the missing December):
  months 1-11 only; rows OUTSIDE P; weights = NNLS per route (A/B/C, D) fitted on months 3, 6, 9;
  judged on the 8 other months (1, 2, 4, 5, 7, 8, 10, 11) and on the scored months (1, 7).
  CatBoost = mean over the seeds available for that month (1-3; conservative vs 3 for the others).
PRE-REGISTERED READING (written before running): CatBoost is a PROMISING W2-era member if
  blend7 - blend6 on held-out A/B/C rows is <= -0.3 s with the paired-bootstrap CI below 0 AND the scored
  months also improve. Otherwise it is not worth a W2-feature CatBoost array. This is a screen on old
  features, not a shipping decision.

    systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 nice -n 5 \
        python experiments/research_0925/r02_catboost_screen.py
"""
import os, sys, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import paired_bootstrap

STUDY = CACHE / 'cv' / 'study'
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost']
MONTHS = list(range(1, 12)); FIT = [3, 6, 9]; HELD = [m for m in MONTHS if m not in FIT]; SCORED = [1, 7]

b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'].astype(float), b['month']
n = len(y)


def fam_oof(fam, seeds):
    o = np.full(n, np.nan); used = {}
    for m in MONTHS:
        arrs = []
        for s in seeds:
            f = STUDY / f'{fam}__seed{s:02d}__m{m:02d}.npy'
            if f.exists():
                a = np.load(f); assert len(a) == (month == m).sum(), (f.name, len(a))
                arrs.append(a.astype(float))
        if arrs:
            o[month == m] = np.mean(arrs, axis=0); used[m] = len(arrs)
    return o, used


X, names = [], []
for fam in FAMS + ['catboost']:
    o, used = fam_oof(fam, [0, 1, 2])
    print(f'{fam:9s} seeds per month {used}')
    X.append(o); names.append(fam)
X = np.column_stack(X)
inm = np.isin(month, MONTHS)
assert np.isfinite(X[inm]).all()
r = lambda p, k: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))

out = ~P & inm
for nm in names:
    k = out & ~D & (y <= 7200)
    print(f'  {nm:9s} RMSE A/B/C outside P, months 1-11: all {r(X[:, names.index(nm)], out & ~D):7.2f}  '
          f'bulk(y<=7200) {r(X[:, names.index(nm)], k):7.2f}')

blends = {}
for tag, cols in (('blend6', list(range(6))), ('blend7', list(range(7)))):
    bl = np.full(n, np.nan)
    for gname, g in (('A/B/C', ~D), ('D', D)):
        fit = np.isin(month, FIT) & ~P & g
        w, _ = nnls(X[fit][:, cols], y[fit])
        bl[g] = X[g][:, cols] @ w
        print(f'  {tag} weights {gname}: ' + ' '.join(f'{names[c]}={wt:.3f}' for c, wt in zip(cols, w)))
    blends[tag] = bl

res6 = y - blends['blend6']; cb_res = y - X[:, names.index('catboost')]
k = out & ~D & (y <= 7200)
print(f'  residual corr (A/B/C bulk rows, months 1-11): catboost vs blend6 {np.corrcoef(res6[k], cb_res[k])[0, 1]:.4f}; '
      f'catboost vs mix {np.corrcoef(y[k] - X[k, 0], cb_res[k])[0, 1]:.4f}; perapt vs mix '
      f'{np.corrcoef(y[k] - X[k, 0], y[k] - X[k, 2])[0, 1]:.4f}')

print('\nblend7 - blend6 (negative = CatBoost helps), rows outside P:')
for scope, mm in (('held-out 8 months', HELD), ('scored 1+7', SCORED)):
    for gname, g in (('A/B/C', ~D), ('D', D), ('all', np.ones(n, bool))):
        kk = ~P & np.isin(month, mm) & g
        bb = paired_bootstrap(y[kk], blends['blend7'][kk], blends['blend6'][kk], n_boot=500)
        print(f'  {scope:18s} {gname:5s} {r(blends["blend6"], kk):8.2f} -> {r(blends["blend7"], kk):8.2f}  '
              f'{bb["diff"]:+.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
per = [r(blends['blend7'], ~P & ~D & (month == m)) - r(blends['blend6'], ~P & ~D & (month == m)) for m in HELD]
print(f'  A/B/C held-out months better: {sum(v < 0 for v in per)}/{len(per)}  ' +
      ' '.join(f'm{m}:{v:+.2f}' for m, v in zip(HELD, per)))
print('peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
