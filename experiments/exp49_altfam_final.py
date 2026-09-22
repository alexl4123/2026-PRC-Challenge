"""EXP49 -- the final table, on whatever fold set each member actually has.

exp46 compares families across all three screening months. CatBoost cannot be
compared there, because it was OOM-killed on month 1 at the laptop's memory
ceiling and so exists for a SUBSET of the folds. Silently pooling a two-fold
model with three-fold models would be wrong, and dropping CatBoost entirely
would throw away the most interesting ensemble result in the study.

So this script does the four-way comparison on exactly the months where every
member is present, states that month set explicitly in its output, and rotates
the blend weights within it. Nothing is pooled across differing fold sets.

Read alongside exp46, which is the three-fold table for the families that have
all three.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import nnls

from prc.config import TAIL_CAP
from prc.models.altfam import CACHE, SCREEN_MONTHS, fold_path, mixture, report
from prc.validation import paired_bootstrap, rmse

y_all = np.load(CACHE / 'altfam' / 'y_screen.npy')
mo_all = np.load(CACHE / 'altfam' / 'month_screen.npy')
os_all = np.load(CACHE / 'altfam' / 'off_sched_screen.npy').astype(np.float64)

NAMES = {'lgb350': 'lgb_bulk', 'lgb900': 'lgb900_bulk_s0', 'cat': 'catboost_bulk_s0',
         'xgb': 'xgboost_bulk_s0', 'nn': 'nn_bulk_s0'}
avail = {k: [m for m in SCREEN_MONTHS if fold_path(v, m).exists()]
         for k, v in NAMES.items()}
for k, v in avail.items():
    print(f'  {k:8s} months {v}')
months = sorted(set.intersection(*(set(v) for v in avail.values())))
if not months:
    sys.exit('no month has every member; nothing to compare')
print(f'\nCOMMON MONTH SET: {months}   '
      f'(CatBoost is missing month 1: OOM-killed at the 8 GB cap, 1,930,786 '
      'training rows)')

sel = np.isin(mo_all, months)
y, mo, osv = y_all[sel], mo_all[sel], os_all[sel]
b = y <= TAIL_CAP


def stitch(nm):
    out = np.empty(int(sel.sum()))
    for m in months:
        out[mo == m] = np.load(fold_path(nm, m))
    return out


P = {k: stitch(v) for k, v in NAMES.items()}
ptail = stitch('ptail')
tailm = stitch('tailmean')
print(f'n={len(y):,}  bulk={int(b.sum()):,}  tail={int((~b).sum())}\n')

print('=' * 100)
print('BULK REGRESSOR ALONE, bulk rows, common month set')
print('=' * 100)
for k in sorted(P, key=lambda k: rmse(y[b], P[k][b])):
    r = paired_bootstrap(y[b], P[k][b], P['lgb900'][b])
    print(f'  {k:8s} {rmse(y[b], P[k][b]):7.2f}   corr vs lgb900 '
          f'{np.corrcoef(y[b] - P[k][b], y[b] - P["lgb900"][b])[0, 1]:6.3f}   '
          f'paired vs lgb900 {r["diff"]:+6.2f} +- {r["sd"]:4.2f} '
          f'CI [{r["lo"]:+6.2f},{r["hi"]:+6.2f}] P(worse) {r["p_a_worse"]:.3f}')

print('\n' + '=' * 100)
print('BLENDS, weights rotated over the common month set (fit on the others, '
      'applied to the held-out one)')
print('=' * 100)
MEM = ['lgb900', 'cat', 'nn', 'xgb']


def rotated(keys):
    A = np.column_stack([P[k] for k in keys])
    out = np.empty(len(y))
    for m in months:
        fit = b & (mo != m)
        w, _ = nnls(A[fit], y[fit])
        out[mo == m] = A[mo == m] @ w
    w_all, _ = nnls(A[b], y[b])
    return out, w_all


best = None
for r in range(1, len(MEM) + 1):
    for combo in itertools.combinations(MEM, r):
        pred, w = rotated(list(combo))
        v = rmse(y[b], pred[b])
        print(f'  {"+".join(combo):24s} bulk {v:7.2f}   w={np.round(w, 3)}')
        if best is None or v < best[0]:
            best = (v, combo, pred)

print(f'\n  BEST SUBSET: {"+".join(best[1])} at {best[0]:.2f}')
r = paired_bootstrap(y[b], best[2][b], P['lgb900'][b])
print(f'  vs lgb900 alone: {r["diff"]:+.2f} +- {r["sd"]:.2f} '
      f'CI [{r["lo"]:+.2f},{r["hi"]:+.2f}] P(worse) {r["p_a_worse"]:.3f}')
r = paired_bootstrap(y[b], best[2][b], P['lgb350'][b])
print(f'  vs production lgb350: {r["diff"]:+.2f} +- {r["sd"]:.2f} '
      f'CI [{r["lo"]:+.2f},{r["hi"]:+.2f}] P(worse) {r["p_a_worse"]:.3f}')

print('\n' + '=' * 100)
print('FULL MIXTURE (classifier and tail estimate identical throughout)')
print('=' * 100)
MIX = {k: mixture(ptail, v, osv, tailm) for k, v in P.items()}
MIX['blend'] = mixture(ptail, best[2], osv, tailm)
for k in MIX:
    report(y, MIX[k], k, width=10)
print('\n  paired vs the production LightGBM mixture')
for k in MIX:
    if k == 'lgb350':
        continue
    r = paired_bootstrap(y, MIX[k], MIX['lgb350'])
    print(f'    {k:10s} {r["diff"]:+7.2f} +- {r["sd"]:5.2f} '
          f'CI [{r["lo"]:+7.2f},{r["hi"]:+7.2f}] P(worse) {r["p_a_worse"]:.3f}')
