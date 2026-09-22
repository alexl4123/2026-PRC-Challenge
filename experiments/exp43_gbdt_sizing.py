"""EXP43 -- how many iterations do CatBoost and XGBoost need, and how fast?

LightGBM's 350 rounds at 255 leaves is a fixed, already-chosen budget. Giving a
new family an arbitrary budget would confound "different model family" with
"differently tuned", so this script measures the learning curve of each family
against a real held-out month and reads the plateau off it.

MONTH 4 IS USED, NOT A SCREENING MONTH. Choosing an iteration count on month 1,
7 or 10 and then reporting on those same months would be selection on the
evaluation set. Month 4 is a typical month (OOF 221.9, bulk 206.0) and is never
scored in exp41-46.

Subsample: 600k of the 1.89M bulk training rows, so the curve is measured in
~10 min per family rather than an hour. The plateau location is then scaled by
the row ratio, which is the standard (rough) correction -- more rows need more
trees at fixed learning rate.

Runtime ~25 min, peak ~4.5 GB.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from prc.config import TAIL_CAP
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.altfam import FULL, config_cols, design
from prc.validation import rmse

THREADS = 4
SEED = 0
N_SUB = 600_000
VAL_MONTH = 4

t0 = time.time()
X, y, month, groups = design()
cols = config_cols(groups, FULL)
cats = [c for c in CATS if c in cols]
num_cols = [c for c in cols if c not in cats]

va = month == VAL_MONTH
tr = (~va) & (y <= TAIL_CAP)
rng = np.random.default_rng(SEED)
idx_tr = rng.choice(np.flatnonzero(tr), N_SUB, replace=False)
idx_va = np.flatnonzero(va)
Xtr, Xva = X.iloc[idx_tr][cols], X.iloc[idx_va][cols]
ytr, yva = y[idx_tr], y[idx_va]
bulk = yva <= TAIL_CAP
yb, n_full = yva[bulk], int(tr.sum())
print(f'train {N_SUB:,} of {n_full:,}  val {len(idx_va):,} (month {VAL_MONTH}, '
      f'{int(bulk.sum()):,} bulk)   ({time.time() - t0:.0f}s)', flush=True)


def curve(name, preds_by_iter, elapsed):
    best_k, best_v = None, np.inf
    line = []
    for k, p in preds_by_iter:
        v = rmse(yb, np.clip(p, 0, None)[bulk])
        line.append(f'{k}:{v:.2f}')
        if v < best_v:
            best_k, best_v = k, v
    print(f'{name:10s} {elapsed:5.0f}s  ' + '  '.join(line), flush=True)
    print(f'{"":10s}        best {best_v:.2f} at {best_k} iters -> scaled to '
          f'{n_full:,} rows ~ {int(best_k * (n_full / N_SUB) ** 0.5):,} iters', flush=True)
    return best_k, best_v


# --- LightGBM, for reference on the same subsample --------------------------
ts = time.time()
m = lgb.train(dict(objective='regression', metric='rmse', learning_rate=0.05,
                   num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   num_threads=THREADS, verbose=-1, seed=SEED, deterministic=True,
                   force_row_wise=True),
              lgb.Dataset(Xtr, ytr, categorical_feature=cats), num_boost_round=1200)
curve('LightGBM', [(k, m.predict(Xva, num_iteration=k))
                   for k in (200, 350, 500, 800, 1200)], time.time() - ts)
del m

# --- CatBoost ---------------------------------------------------------------
def to_pool(Xf, yf=None):
    d = {c: Xf[c].cat.codes.to_numpy().astype(np.int32) + 1 for c in cats}
    for c in num_cols:
        d[c] = Xf[c].to_numpy(dtype=np.float32, na_value=np.nan)
    return Pool(pd.DataFrame(d, columns=cats + num_cols), yf,
                cat_features=list(range(len(cats))))


ts = time.time()
ptr, pva = to_pool(Xtr, ytr), to_pool(Xva)
cb = CatBoostRegressor(loss_function='RMSE', learning_rate=0.05, depth=8,
                       iterations=3000, max_ctr_complexity=2, one_hot_max_size=16,
                       border_count=128, bootstrap_type='Bernoulli', subsample=0.8,
                       rsm=0.8, l2_leaf_reg=3.0, random_seed=SEED,
                       thread_count=THREADS, verbose=0)
cb.fit(ptr)
curve('CatBoost', [(k, cb.predict(pva, ntree_end=k))
                   for k in (500, 1000, 1500, 2000, 2500, 3000)], time.time() - ts)
del cb, ptr, pva

# --- XGBoost ----------------------------------------------------------------
# grow_policy='lossguide' with max_leaves=255 and max_depth=0 makes XGBoost's
# tree shape match LightGBM's, so the comparison is about the split-finding and
# the categorical handling rather than about depthwise-vs-leafwise growth.
ts = time.time()
dtr = xgb.QuantileDMatrix(Xtr, ytr, enable_categorical=True, max_bin=256)
dva = xgb.DMatrix(Xva, enable_categorical=True)
bst = xgb.train(dict(objective='reg:squarederror', eval_metric='rmse',
                     tree_method='hist', max_bin=256, grow_policy='lossguide',
                     max_leaves=255, max_depth=0, learning_rate=0.05,
                     min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=1.0, max_cat_to_onehot=8, max_cat_threshold=128,
                     nthread=THREADS, seed=SEED),
                dtr, num_boost_round=1200)
curve('XGBoost', [(k, bst.predict(dva, iteration_range=(0, k)))
                  for k in (200, 350, 500, 800, 1200)], time.time() - ts)

print(f'\npeak RSS {peak_rss_gb():.2f} GB   total {time.time() - t0:.0f}s')
