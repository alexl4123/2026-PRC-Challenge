"""EXP41 -- the LightGBM reference, on the exp40+ screening folds.

This is the yardstick every alternative family is measured against. It fits the
current best arm (regressor on the full 72 features, classifier on the lean 18
-- r-full__c-base from the cluster grid) on months 1, 7 and 10 and stores three
things per fold:

    lgb_bulk   E[y|~T,x] from the LightGBM regressor  (what alternatives replace)
    ptail      P(T|x) from the LightGBM classifier    (held fixed everywhere)
    tailmean   mean of the training tail rows         (held fixed everywhere)

Holding the classifier and the tail estimate fixed is deliberate. It makes the
comparison between model families a comparison of BULK REGRESSORS and nothing
else, and it means a CatBoost or NN run costs one fit instead of two.

CAVEAT ON THE HEADLINE NUMBER. The published baseline (bulk 236.60, total
287.45) is a 10-seed average over all twelve months. This is one seed over three
months, so it is NOT the same quantity and will not reproduce 236.60. Months 1
and 7 are the two hardest; month 10 is typical. Every comparison in exp42-46 is
against the number produced HERE, on these folds, with this seed.

Runtime ~12 min, peak ~4 GB.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np

from prc.config import TAIL_CAP
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.altfam import (BASE, FULL, SCREEN_MONTHS, config_cols, design,
                               have_fold, load_folds, mixture, report, save_fold, screen)
from prc.validation import per_month, rmse

THREADS = 4
SEED = 0

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=SEED,
           bagging_seed=SEED + 1, feature_fraction_seed=SEED + 2,
           deterministic=True, force_row_wise=True)
CLF = dict(REG, objective='binary', num_leaves=63, min_data_in_leaf=50)
CLF['metric'] = 'auc'

t0 = time.time()
X, y, month, groups = design()
reg_cols = config_cols(groups, FULL)
clf_cols = config_cols(groups, BASE)
reg_cats = [c for c in CATS if c in reg_cols]
clf_cats = [c for c in CATS if c in clf_cols]
print(f'design {X.shape[0]:,} x {X.shape[1]}  reg {len(reg_cols)} cols  '
      f'clf {len(clf_cols)} cols   ({time.time() - t0:.0f}s)', flush=True)

for m in SCREEN_MONTHS:
    if have_fold('lgb_bulk', m) and have_fold('ptail', m) and have_fold('tailmean', m):
        print(f'  month {m:02d}: cached, skipping')
        continue
    ts = time.time()
    va = month == m
    tr = ~va
    ytr = y[tr]
    k = ytr <= TAIL_CAP

    mb = lgb.train(REG, lgb.Dataset(X.loc[tr, reg_cols][k], ytr[k],
                                    categorical_feature=reg_cats), num_boost_round=350)
    pb = np.clip(mb.predict(X.loc[va, reg_cols]), 0, None)
    del mb

    mc = lgb.train(CLF, lgb.Dataset(X.loc[tr, clf_cols], (ytr > TAIL_CAP).astype(int),
                                    categorical_feature=clf_cats), num_boost_round=300)
    pt = mc.predict(X.loc[va, clf_cols])
    del mc

    save_fold('lgb_bulk', m, pb)
    save_fold('ptail', m, pt)
    save_fold('tailmean', m, np.full(int(va.sum()), ytr[ytr > TAIL_CAP].mean()))
    print(f'  month {m:02d}: n={int(va.sum()):,}  bulk-only RMSE '
          f'{rmse(y[va][y[va] <= TAIL_CAP], pb[y[va] <= TAIL_CAP]):.2f}  '
          f'({time.time() - ts:.0f}s)', flush=True)

sel = screen(month)
ys = y[sel]
pb = load_folds('lgb_bulk', month)
pt = load_folds('ptail', month)
tm = load_folds('tailmean', month)
osv = X.loc[sel, 'off_sched'].to_numpy(dtype=np.float64, na_value=np.nan)
np.save('cache/altfam/off_sched_screen.npy', osv.astype(np.float32))
np.save('cache/altfam/y_screen.npy', ys)
np.save('cache/altfam/month_screen.npy', month[sel])

pred = mixture(pt, pb, osv, tm)
bulk_rows = ys <= TAIL_CAP

print('\n' + '=' * 100)
print(f'LIGHTGBM REFERENCE -- months {SCREEN_MONTHS}, seed {SEED}, n={int(sel.sum()):,}, '
      f'{int((~bulk_rows).sum())} tail rows')
print('=' * 100)
print(f'  regressor alone, bulk rows only   RMSE {rmse(ys[bulk_rows], pb[bulk_rows]):8.2f}'
      '     <- the quantity exp42+ tries to beat')
report(ys, pred, 'full mixture')
print('\nper month')
print(per_month(ys, pred, month[sel]).round(1).to_string(index=False))

print(f'\nclassifier: mean p {pt.mean():.6f} vs actual rate {(~bulk_rows).mean():.6f}  '
      f'ratio {pt.mean() / max((~bulk_rows).mean(), 1e-12):.2f}')
print(f'peak RSS {peak_rss_gb():.2f} GB   total {time.time() - t0:.0f}s')
