"""EXP45 -- fit one alternative family on the full screening folds.

    python experiments/exp45_altfam_folds.py --family catboost
    python experiments/exp45_altfam_folds.py --family xgboost
    python experiments/exp45_altfam_folds.py --family nn --seeds 0 1 2

Each (family, seed, month) writes one .npy into cache/altfam/ and is SKIPPED if
that file already exists, so a run interrupted after two folds resumes where it
stopped rather than starting over. Same reasoning as cluster/cv_task.py: these
are long, they are run unattended, and a partial result must not be silently
half-written.

Only the BULK REGRESSOR is fitted here. The tail classifier P(T|x) and the tail
estimate come from exp41 and are identical for every family, so any difference
between the families is attributable to the regressor and nothing else.

Hyperparameters are the ones exp43 sized against month 4, which is deliberately
NOT one of the three evaluation months.

Runtime per family: CatBoost ~110 min, XGBoost ~25 min, NN ~30 min for 3 seeds.
Peak ~5 GB.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from prc.config import TAIL_CAP
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.altfam import (FULL, SCREEN_MONTHS, config_cols, design, fold_path,
                               save_fold)
from prc.validation import rmse

ap = argparse.ArgumentParser()
ap.add_argument('--family', required=True,
                choices=['catboost', 'xgboost', 'nn', 'lightgbm'])
ap.add_argument('--tag', default=None, help='override the fold-store name')
ap.add_argument('--lgb-rounds', type=int, default=350)
ap.add_argument('--seeds', type=int, nargs='+', default=[0])
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--months', type=int, nargs='+', default=list(SCREEN_MONTHS))
ap.add_argument('--cb-iters', type=int, default=3000)
ap.add_argument('--cb-lr', type=float, default=0.08)
# used_ram_limit alone did NOT keep CatBoost inside a 6 GB cgroup on 1.9M rows:
# two OOM kills, journalctl --user 'killed by the OOM killer', no traceback.
# ctr_leaf_count_limit is the parameter that actually bounds the ordered-target-
# statistic hash tables, which is where the memory goes at max_ctr_complexity=2
# over eight categoricals whose pairwise products reach 1899 x 676 levels.
ap.add_argument('--cb-ctr-limit', type=int, default=1_000_000)
# one_hot_max_size is the OTHER memory lever, and the principled one. A
# categorical below this size is one-hot encoded and takes no part in the CTR
# combinatorics at all. At 16 all eight categoricals get target statistics, so
# max_ctr_complexity=2 builds 8 + C(8,2) = 36 projections. At 64, APT (10),
# RUNWAY_mvt (53), MARKET_SEGMENT_flt (8) and WK_TBL_CAT_flt (4) become one-hot
# and only PIER / STAND_mvt / AIRCRAFT_TYPE_mvt / AIRCRAFT_OPERATOR_flt keep
# CTRs -- 4 + C(4,2) = 10 projections, roughly a 3.5x cut in CTR memory. Those
# four are exactly the high-cardinality columns where ordered target statistics
# are supposed to earn their keep, so this trims the part that was never the
# point rather than weakening the mechanism under test.
ap.add_argument('--cb-onehot', type=int, default=16)
ap.add_argument('--xgb-rounds', type=int, default=900)
ap.add_argument('--nn-epochs', type=int, default=18)
ap.add_argument('--nn-missing', default='median', choices=['median', 'sentinel'],
                help="how a NaN numeric is filled; see altfam.numeric_design "
                     "and exp47 for why 'median' is measurably the wrong one")
a = ap.parse_args()

t0 = time.time()
X, y, month, groups = design()
cols = config_cols(groups, FULL)
cats = [c for c in CATS if c in cols]
num_cols = [c for c in cols if c not in cats]
print(f'design {X.shape[0]:,} x {len(cols)}  family={a.family} seeds={a.seeds} '
      f'months={a.months}   ({time.time() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------------------
def fit_catboost(tr_mask, va_mask, seed):
    """MEMORY NOTE, learned by being OOM-killed inside the 6 GB scope.

    The first version built both Pools while the 594 MB design frame was still
    alive, and CatBoost's CTR tables on top of that blew straight through the
    cap. journalctl --user showed 'killed by the OOM killer' with no python
    traceback at all, which is what a cgroup kill looks like -- do not mistake
    it for a silent logic bug.

    Three fixes, all needed:
      * free each intermediate frame the moment its Pool exists;
      * drop the whole design matrix before fit(), since the Pools hold their
        own quantized copy (the caller reloads it for the next month);
      * used_ram_limit, which is the ONE CatBoost parameter that bounds CTR
        memory. Without it the ordered target statistics over eight
        categoricals at max_ctr_complexity=2 are unbounded.
    """
    import gc

    import pandas as pd
    from catboost import CatBoostRegressor, Pool

    def pool(mask, with_y):
        Xf = X.loc[mask, cols]
        d = {c: Xf[c].cat.codes.to_numpy().astype(np.int32) + 1 for c in cats}
        for c in num_cols:
            d[c] = Xf[c].to_numpy(dtype=np.float32, na_value=np.nan)
        del Xf
        df = pd.DataFrame(d, columns=cats + num_cols)
        d.clear()
        p = Pool(df, y[mask] if with_y else None, cat_features=list(range(len(cats))))
        del df
        gc.collect()
        return p

    ptr = pool(tr_mask, True)
    pva = pool(va_mask, False)
    globals()['X'] = None
    gc.collect()

    m = CatBoostRegressor(loss_function='RMSE', learning_rate=a.cb_lr, depth=8,
                          iterations=a.cb_iters, max_ctr_complexity=2,
                          one_hot_max_size=a.cb_onehot, border_count=96,
                          bootstrap_type='Bernoulli', subsample=0.8, rsm=0.8,
                          l2_leaf_reg=3.0, random_seed=seed, thread_count=a.threads,
                          used_ram_limit='3gb', ctr_leaf_count_limit=a.cb_ctr_limit,
                          allow_writing_files=False, verbose=0)
    m.fit(ptr)
    del ptr
    gc.collect()
    return np.clip(m.predict(pva), 0, None)


def fit_xgboost(tr_mask, va_mask, seed):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(X.loc[tr_mask, cols], y[tr_mask],
                              enable_categorical=True, max_bin=256)
    bst = xgb.train(dict(objective='reg:squarederror', eval_metric='rmse',
                         tree_method='hist', max_bin=256, grow_policy='lossguide',
                         max_leaves=255, max_depth=0, learning_rate=0.05,
                         min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, max_cat_to_onehot=8, max_cat_threshold=128,
                         nthread=a.threads, seed=seed),
                    dtr, num_boost_round=a.xgb_rounds)
    del dtr
    dva = xgb.DMatrix(X.loc[va_mask, cols], enable_categorical=True)
    return np.clip(bst.predict(dva), 0, None)


_NUMCACHE = {}


def fit_nn(tr_mask, va_mask, seed):
    from prc.models.altfam import numeric_design
    from prc.models.embed_nn import fit_predict
    key = int(np.flatnonzero(va_mask)[0])
    if key not in _NUMCACHE:
        _NUMCACHE.clear()          # one fold's design at a time: 534 MB each
        _NUMCACHE[key] = numeric_design(X[cols], cols, fit_mask=tr_mask,
                                        missing=a.nn_missing)[:3]
    codes, cards, Z = _NUMCACHE[key]
    return fit_predict(codes, Z, y, tr_mask, va_mask, epochs=a.nn_epochs,
                       batch=4096, lr=3e-3, hidden=(512, 256, 128), dropout=0.10,
                       seed=seed, threads=a.threads,
                       log=lambda s: print(s, flush=True))


def fit_lightgbm(tr_mask, va_mask, seed):
    """LightGBM at an arbitrary round count.

    THIS IS A FAIRNESS CHECK, NOT A NEW FAMILY. exp43 sized LightGBM's plateau
    at 500 rounds on 600k rows, which scales to ~900 on the full 1.9M. If the
    production 350 is simply under-boosted at full scale, then part of any
    'the NN beats LightGBM' result is really 'LightGBM was not run long enough',
    and the comparison is worthless. Run it before believing anything else here.
    """
    import lightgbm as lgb
    m = lgb.train(dict(objective='regression', metric='rmse', learning_rate=0.05,
                       num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                       num_threads=a.threads, verbose=-1, seed=seed,
                       bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
                       deterministic=True, force_row_wise=True),
                  lgb.Dataset(X.loc[tr_mask, cols], y[tr_mask],
                              categorical_feature=cats),
                  num_boost_round=a.lgb_rounds)
    return np.clip(m.predict(X.loc[va_mask, cols]), 0, None)


FIT = {'catboost': fit_catboost, 'xgboost': fit_xgboost, 'nn': fit_nn,
       'lightgbm': fit_lightgbm}[a.family]

for m in a.months:
    if X is None:                     # fit_catboost drops it to survive the cap
        X, y, month, groups = design()
    va = month == m
    tr = (~va) & (y <= TAIL_CAP)      # bulk regressor: bulk training rows only
    yv = y[va]
    bulk = yv <= TAIL_CAP
    for seed in a.seeds:
        name = f'{a.tag or a.family}_bulk_s{seed}'
        if fold_path(name, m).exists():
            print(f'  {name} m{m:02d}: cached, skipping', flush=True)
            continue
        ts = time.time()
        if X is None:                 # second seed after a CatBoost fold
            X, y, month, groups = design()
        p = FIT(tr, va, seed)
        save_fold(name, m, p)
        print(f'  {name} m{m:02d}: n_tr={int(tr.sum()):,} n_va={int(va.sum()):,}  '
              f'bulk RMSE {rmse(yv[bulk], p[bulk]):7.2f}  ({time.time() - ts:.0f}s)  '
              f'RSS {peak_rss_gb():.2f} GB', flush=True)

print(f'\ndone. peak RSS {peak_rss_gb():.2f} GB   total {(time.time() - t0) / 60:.0f} min')
