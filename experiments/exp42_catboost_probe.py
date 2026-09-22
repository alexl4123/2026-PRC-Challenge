"""EXP42 -- CatBoost probe on a subsample: is it worth a full fold?

CatBoost is the most plausible of the untried families for THIS dataset, because
its ordered target statistics are a genuinely different treatment of the
high-cardinality categoricals (STAND_mvt 1,899 levels, AIRCRAFT_OPERATOR_flt
676) than LightGBM's gradient-sorted partitioning. If it is going to help
anywhere it is there.

This script does not try to settle anything. It measures, on a 300k subsample
trained against a real held-out month, (a) how long CatBoost takes per 100
iterations at this width, so the full run can be sized, and (b) whether the
bulk RMSE is in the right neighbourhood at all. Only then is a full fold worth
1,900,000 rows of CPU time.

Target to beat on the full folds: LightGBM bulk-rows RMSE 240.02 (exp41).
Subsample numbers are NOT comparable to that -- a sixth of the training rows
costs real accuracy -- so LightGBM is refit on the SAME subsample here.

Runtime ~15 min, peak ~4 GB.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from prc.config import TAIL_CAP
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.altfam import FULL, config_cols, design
from prc.validation import rmse

THREADS = 4
SEED = 0
N_SUB = 300_000
VAL_MONTH = 7          # the hardest month; if a family cannot cope it shows here

t0 = time.time()
X, y, month, groups = design()
cols = config_cols(groups, FULL)
cats = [c for c in CATS if c in cols]
num_cols = [c for c in cols if c not in cats]

va = month == VAL_MONTH
tr = (~va) & (y <= TAIL_CAP)          # bulk regressor: trained on bulk rows only
rng = np.random.default_rng(SEED)
idx_tr = rng.choice(np.flatnonzero(tr), N_SUB, replace=False)
idx_va = np.flatnonzero(va)
print(f'train {len(idx_tr):,} (subsample of {int(tr.sum()):,})  '
      f'val {len(idx_va):,} month {VAL_MONTH}   ({time.time() - t0:.0f}s)', flush=True)

Xtr = X.iloc[idx_tr][cols]
Xva = X.iloc[idx_va][cols]
ytr = y[idx_tr]
yva = y[idx_va]
bulk_va = yva <= TAIL_CAP

# ---------------------------------------------------------------------------
# LightGBM on the same subsample, so the comparison is like for like.
# ---------------------------------------------------------------------------
REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=SEED,
           deterministic=True, force_row_wise=True)
ts = time.time()
m = lgb.train(REG, lgb.Dataset(Xtr, ytr, categorical_feature=cats), num_boost_round=350)
p_lgb = np.clip(m.predict(Xva), 0, None)
t_lgb = time.time() - ts
print(f'\nLightGBM  350 rounds  {t_lgb:5.0f}s   bulk RMSE '
      f'{rmse(yva[bulk_va], p_lgb[bulk_va]):7.2f}', flush=True)
del m

# ---------------------------------------------------------------------------
# CatBoost. Categoricals go in as integer codes (code + 1, 0 = missing), which
# CatBoost accepts directly and which avoids materialising 300k python strings
# per column. NaN is not allowed in a CatBoost categorical, hence the +1.
# ---------------------------------------------------------------------------
def to_pool(Xf, yf=None):
    d = {c: Xf[c].cat.codes.to_numpy().astype(np.int32) + 1 for c in cats}
    for c in num_cols:
        d[c] = Xf[c].to_numpy(dtype=np.float32, na_value=np.nan)
    df = pd.DataFrame(d, columns=cats + num_cols)
    return Pool(df, yf, cat_features=list(range(len(cats))))


ts = time.time()
ptr, pva = to_pool(Xtr, ytr), to_pool(Xva)
print(f'pools built ({time.time() - ts:.0f}s)', flush=True)

# max_ctr_complexity controls how many categoricals get COMBINED into a single
# target statistic. The default of 4 over eight categoricals with 1,899 levels
# is a combinatorial memory hazard on a 29 GB laptop; 1-2 is the safe range and
# is also where most of the benefit sits.
GRID = [
    dict(depth=6, iterations=1000, max_ctr_complexity=1, tag='d6 n1000 ctr1'),
    dict(depth=8, iterations=1000, max_ctr_complexity=1, tag='d8 n1000 ctr1'),
    dict(depth=8, iterations=1000, max_ctr_complexity=2, tag='d8 n1000 ctr2'),
]
for g in GRID:
    tag = g.pop('tag')
    ts = time.time()
    cb = CatBoostRegressor(loss_function='RMSE', learning_rate=0.05,
                           random_seed=SEED, thread_count=THREADS, verbose=0,
                           one_hot_max_size=16, border_count=128,
                           bootstrap_type='Bernoulli', subsample=0.8,
                           rsm=0.8, l2_leaf_reg=3.0, **g)
    cb.fit(ptr)
    p_cb = np.clip(cb.predict(pva), 0, None)
    el = time.time() - ts
    print(f'CatBoost  {tag:16s} {el:5.0f}s   bulk RMSE '
          f'{rmse(yva[bulk_va], p_cb[bulk_va]):7.2f}   '
          f'corr(resid, lgb) {np.corrcoef(yva[bulk_va] - p_cb[bulk_va], yva[bulk_va] - p_lgb[bulk_va])[0, 1]:.4f}',
          flush=True)
    del cb, p_cb

print(f'\npeak RSS {peak_rss_gb():.2f} GB   total {time.time() - t0:.0f}s')
print(f'SIZING: a full fold has {int(tr.sum()):,} rows, {int(tr.sum()) / N_SUB:.1f}x this '
      'subsample; CatBoost scales roughly linearly in rows.')
