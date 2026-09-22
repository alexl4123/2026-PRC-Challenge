"""EXP46 -- score every family, measure residual decorrelation, and blend.

Reads whatever exp41/exp45 have written into cache/altfam/ and reports, on the
three screening folds (months 1, 7, 10):

  1. each family's BULK regressor RMSE over bulk rows -- the quantity this whole
     study is aimed at, with production LightGBM at 240.02 as the mark, and the
     same LightGBM at 900 rounds as the anti-confound control;
  2. the correlation of each family's residuals with LightGBM's, on bulk rows.
     This is the number that decides whether a blend can help at all: two models
     at equal RMSE with residual correlation r blend to sqrt((1+r)/2) of it, so
     r = 0.94 caps the gain at 1.5% and r = 0.80 would allow 5%;
  3. blends, three ways.

ON BLEND WEIGHTS AND HONESTY. Weights chosen on the same rows they are scored on
are optimistic, and with 530k rows and four models the optimism is small but it
is not zero. So three numbers are reported: equal weights (no fitting at all),
weights fitted in-sample (the CEILING, explicitly labelled as such), and weights
fitted on two months and applied to the third, rotated -- which is the honest
one and the one any recommendation rests on.

Every difference gets a paired bootstrap. The whole point of the local protocol
is that unpaired absolute RMSE differences below ~100s here are noise.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import nnls

from prc.config import TAIL_CAP
from prc.models.altfam import CACHE, SCREEN_MONTHS, load_folds, mixture, report
from prc.validation import decompose, paired_bootstrap, per_month, rmse

y = np.load(CACHE / 'altfam' / 'y_screen.npy')
month = np.load(CACHE / 'altfam' / 'month_screen.npy')
osv = np.load(CACHE / 'altfam' / 'off_sched_screen.npy').astype(np.float64)
ptail = load_folds('ptail', month)
tailm = load_folds('tailmean', month)
bulk = y <= TAIL_CAP
print(f'screening set: months {SCREEN_MONTHS}, n={len(y):,}, '
      f'{int((~bulk).sum())} tail rows\n')

# 'lightgbm' is the PRODUCTION regressor at 350 rounds -- the thing a new family
# has to beat to be worth anything. 'lgb900' is the same model given the round
# count exp43's learning curve says it should have at 1.9M rows; it is here to
# stop a 'the NN wins' claim that is really 'LightGBM was under-boosted'.
CANDIDATES = [('lightgbm', ['lgb_bulk']),
              ('lgb900', ['lgb900_bulk_s0']),
              ('catboost', ['catboost_bulk_s0']),
              ('xgboost', ['xgboost_bulk_s0']),
              ('nn', ['nn_bulk_s0', 'nn_bulk_s1', 'nn_bulk_s2']),
              ('nn_s0', ['nn_bulk_s0'])]

P = {}
for label, names in CANDIDATES:
    parts = [load_folds(n, month) for n in names]
    have = [p for p in parts if p is not None]
    if not have:
        print(f'  {label}: MISSING ({names}) -- skipped')
        continue
    if len(have) < len(parts):
        print(f'  {label}: only {len(have)}/{len(parts)} seeds present, averaging those')
    P[label] = np.mean(have, axis=0)

print('\n' + '=' * 104)
print('1. BULK REGRESSOR ALONE, bulk rows only  (LightGBM reference = the mark)')
print('=' * 104)
res = {k: y[bulk] - v[bulk] for k, v in P.items()}
base = 'lightgbm'
order = sorted(P, key=lambda k: rmse(y[bulk], P[k][bulk]))
for k in order:
    v = P[k]
    r = rmse(y[bulk], v[bulk])
    c = np.corrcoef(res[k], res[base])[0, 1]
    pb = paired_bootstrap(y[bulk], v[bulk], P[base][bulk])
    print(f'  {k:10s} bulk RMSE {r:7.2f}   corr-resid vs lgb {c:6.3f}   '
          f'paired vs lgb350 {pb["diff"]:+7.2f} +- {pb["sd"]:5.2f}  '
          f'CI [{pb["lo"]:+7.2f},{pb["hi"]:+7.2f}]  P(worse) {pb["p_a_worse"]:.3f}')

# LightGBM at 350 rounds is the production model, but the honest question for a
# NEW family is whether it beats the best tree we can actually build. That is
# lgb900, so every family is also paired against it.
if 'lgb900' in P:
    print('\n  paired against the BEST tree available (lgb900), bulk rows')
    for k in order:
        if k == 'lgb900':
            continue
        pb = paired_bootstrap(y[bulk], P[k][bulk], P['lgb900'][bulk])
        print(f'    {k:10s} {pb["diff"]:+7.2f} +- {pb["sd"]:5.2f}  '
              f'CI [{pb["lo"]:+7.2f},{pb["hi"]:+7.2f}]  P(worse) {pb["p_a_worse"]:.3f}')

print('\n  residual correlation matrix (bulk rows)')
ks = list(P)
print('             ' + ' '.join(f'{k:>10s}' for k in ks))
for a in ks:
    print(f'  {a:10s} ' + ' '.join(f'{np.corrcoef(res[a], res[b])[0, 1]:10.3f}' for b in ks))

# ---------------------------------------------------------------------------
print('\n' + '=' * 104)
print('2. BLENDS OF THE BULK REGRESSOR  (bulk rows)')
print('=' * 104)
MEMBERS = [k for k in ('lgb900', 'catboost', 'xgboost', 'nn') if k in P]
A = np.column_stack([P[k][bulk] for k in MEMBERS])
yb = y[bulk]


def fit_w(mask):
    w, _ = nnls(A[mask], yb[mask])
    return w


for r in range(2, len(MEMBERS) + 1):
    for combo in itertools.combinations(range(len(MEMBERS)), r):
        names = [MEMBERS[i] for i in combo]
        sub = A[:, combo]
        eq = sub.mean(axis=1)
        w_all, _ = nnls(sub, yb)
        insample = sub @ w_all
        # rotated: weights from the other two months, applied to the held-out one
        rot = np.empty(len(yb))
        mb = month[bulk]
        for m in SCREEN_MONTHS:
            hold = mb == m
            w, _ = nnls(sub[~hold], yb[~hold])
            rot[hold] = sub[hold] @ w
        print(f'  {"+".join(names):34s} equal {rmse(yb, eq):7.2f} | '
              f'in-sample {rmse(yb, insample):7.2f} (ceiling) | '
              f'rotated {rmse(yb, rot):7.2f} | w={np.round(w_all, 3)}')
        if r == len(MEMBERS):
            # Rotated over ALL screening rows, not just the bulk ones. The tail
            # rows need a bulk-regressor value too (the mixture feeds them
            # through (1-p)*pred_bulk), and giving those 192 rows in-sample
            # weights while the other 529,901 get rotated ones would quietly
            # put a thumb on the total-RMSE number. Weights are still FITTED on
            # bulk rows only -- that is what the bulk regressor is trained for.
            Afull = np.column_stack([P[k] for k in MEMBERS])
            P['blend_rot'] = np.empty(len(y))
            for mm in SCREEN_MONTHS:
                fit_on = bulk & (month != mm)
                w, _ = nnls(Afull[fit_on], y[fit_on])
                P['blend_rot'][month == mm] = Afull[month == mm] @ w
            wfull, _ = nnls(A, yb)
            BLEND_W = dict(zip(MEMBERS, np.round(wfull, 4)))

# ---------------------------------------------------------------------------
print('\n' + '=' * 104)
print('3. FULL MIXTURE  (bulk regressor swapped, classifier and tail estimate held fixed)')
print('=' * 104)
MIX = {}
for k in list(P):
    MIX[k] = mixture(ptail, P[k], osv, tailm)
    report(y, MIX[k], k, width=12)

print('\n  paired bootstrap of the FULL mixture, vs the LightGBM mixture')
for k in MIX:
    if k == base:
        continue
    pb = paired_bootstrap(y, MIX[k], MIX[base])
    print(f'    {k:12s} {pb["diff"]:+8.2f} +- {pb["sd"]:6.2f}  '
          f'CI [{pb["lo"]:+8.2f},{pb["hi"]:+8.2f}]  P(worse) {pb["p_a_worse"]:.3f}')

print('\n  per-month, full mixture')
for k in MIX:
    pm = per_month(y, MIX[k], month)
    print(f'    {k:12s} ' + '  '.join(
        f'm{int(r.month):02d} bulk {r.rmse_bulk:6.1f}' for r in pm.itertuples()))

if 'BLEND_W' in dir():
    print(f'\n  blend weights (all months): {BLEND_W}')
d = decompose(y, MIX[base])
print(f'\n  reference: LightGBM mixture RMSE {d["rmse"]:.2f} bulk {d["rmse_bulk"]:.2f} '
      f'on these three folds (the published 287.45 / 236.60 is 10 seeds over 12 months '
      'and is NOT this quantity)')
