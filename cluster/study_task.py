"""One unit of the model-family study: a single (family, seed, month) fit.

Writes cache/cv/<campaign>/<family>__seed<NN>__m<MM>.npy, so cluster/collect.py
scores every family as an arm and paired-bootstraps them against each other.

All families share the SAME tail treatment unless the family itself changes it,
so a measured difference is attributable to the thing being varied:

  mix       the shipped v5 mixture (baseline). Bulk regressor on 72 features
            trained on y<=7200, classifier on the 18 base features, off_sched
            as the tail anchor.
  routed    prc/models/routed.py -- four disjoint routes by the NM proxy. This
            replaces the whole architecture, not just the regressor.
  lintree   mix, but the bulk regressor uses linear_tree=True. Linear leaves
            extrapolate beyond the training range, which is the one thing a
            constant-leaf regressor trained on y<=7200 structurally cannot do.
  perapt    mix, but the bulk regressor is ten per-airport models. A LIRF-only
            bulk specialist measured -7.7 +- 1.7 on its own rows (exp51).
  catboost  mix with a CatBoost bulk regressor. Ordered target statistics are a
            genuinely different treatment of STAND_mvt (1,899 levels).
  xgboost   mix with an XGBoost bulk regressor. Mostly ensemble diversity.

  python cluster/study_task.py --campaign study --family routed --seed 0 --month 7
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np
import pandas as pd

from cluster.cv_task import assemble, CONFIGS
from prc.config import TAIL_CAP, CACHE
from prc.features import CATS

FAMILIES = ['mix', 'routed', 'lintree', 'perapt', 'catboost', 'xgboost']


def reg_params(seed, threads, **kw):
    p = dict(objective='regression', metric='rmse', learning_rate=0.025,
             num_leaves=511, min_data_in_leaf=100, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
             num_threads=threads, verbose=-1, seed=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2, deterministic=True,
             force_row_wise=True)
    p.update(kw)
    return p


def clf_params(seed, threads):
    return dict(objective='binary', metric='auc', learning_rate=0.05,
                num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=threads, verbose=-1, seed=seed, bagging_seed=seed + 1,
                feature_fraction_seed=seed + 2, deterministic=True,
                force_row_wise=True)


def fit_bulk(family, X, y, trn, cols, cats, seed, threads, apt=None, va=None):
    """Return bulk predictions for the rows in `va`."""
    k = trn & (y <= TAIL_CAP)
    if family in ('mix', 'routed'):
        m = lgb.train(reg_params(seed, threads),
                      lgb.Dataset(X.loc[k, cols], y[k], categorical_feature=cats),
                      num_boost_round=1400)
        return np.clip(m.predict(X.loc[va, cols]), 0, None)

    if family == 'lintree':
        # linear_tree needs the categoricals as plain ints and no NaN in the
        # linear part, so it gets a numeric view of the same columns.
        m = lgb.train(reg_params(seed, threads, linear_tree=True, linear_lambda=1.0,
                                 num_leaves=255),
                      lgb.Dataset(X.loc[k, cols], y[k], categorical_feature=cats),
                      num_boost_round=700)
        # Linear leaves extrapolate without bound; clip to something physical.
        return np.clip(m.predict(X.loc[va, cols]), 0, 4 * TAIL_CAP)

    if family == 'perapt':
        out = np.full(int(va.sum()), np.nan)
        va_apt = apt[va]
        for a in np.unique(apt):
            ka = k & (apt == a)
            sel = va_apt == a
            if not sel.any():
                continue
            m = lgb.train(reg_params(seed, threads, num_leaves=255),
                          lgb.Dataset(X.loc[ka, cols], y[ka], categorical_feature=cats),
                          num_boost_round=700)
            out[sel] = m.predict(X.loc[va, cols][sel])
        return np.clip(out, 0, None)

    if family == 'catboost':
        from catboost import CatBoostRegressor, Pool
        Xa = X.loc[k, cols].copy()
        Xb = X.loc[va, cols].copy()
        for c in cats:
            Xa[c] = Xa[c].astype(str)
            Xb[c] = Xb[c].astype(str)
        m = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8,
                              loss_function='RMSE', random_seed=seed,
                              thread_count=threads, verbose=0)
        m.fit(Pool(Xa, y[k], cat_features=cats))
        return np.clip(m.predict(Pool(Xb, cat_features=cats)), 0, None)

    if family == 'xgboost':
        import xgboost as xgb
        Xa = X.loc[k, cols].copy()
        Xb = X.loc[va, cols].copy()
        for c in cats:
            Xa[c] = Xa[c].cat.codes.astype('int32')
            Xb[c] = Xb[c].cat.codes.astype('int32')
        m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.05, max_depth=10,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             tree_method='hist', random_state=seed,
                             n_jobs=threads, verbosity=0)
        m.fit(Xa, y[k])
        return np.clip(m.predict(Xb), 0, None)

    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--family', required=True, choices=FAMILIES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--month', type=int, required=True)
    ap.add_argument('--threads', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    a = ap.parse_args()

    outdir = CACHE / 'cv' / a.campaign
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'{a.family}__seed{a.seed:02d}__m{a.month:02d}.npy'
    if out.exists():
        print(f'{out.name} exists, skipping')
        return

    t0 = time.time()
    X, y, month, groups = assemble()
    cols = [c for g in CONFIGS['full'] for c in groups[g]]
    clf_cols = [c for g in CONFIGS['base'] for c in groups[g]]
    cats = [c for c in CATS if c in cols]
    ccats = [c for c in CATS if c in clf_cols]
    va, trn = month == a.month, month != a.month
    apt = X['APT'].astype(str).to_numpy()
    sched_off = X['off_sched'].to_numpy()

    if a.family == 'routed':
        from prc.models.routed import RoutedModel
        proxy = X['off_aobt3'].to_numpy()
        rm = RoutedModel(cols, clf_cols, CATS, seed=a.seed, threads=a.threads)
        rm.fit(X.loc[trn], y[trn], proxy[trn], sched_off[trn])
        pred = rm.predict(X.loc[va], proxy[va], sched_off[va])
    else:
        pb = fit_bulk(a.family, X, y, trn, cols, cats, a.seed, a.threads,
                      apt=apt, va=va)
        yt = y[trn]
        mc = lgb.train(clf_params(a.seed, a.threads),
                       lgb.Dataset(X.loc[trn, clf_cols], (yt > TAIL_CAP).astype(int),
                                   categorical_feature=ccats), num_boost_round=300)
        pt = mc.predict(X.loc[va, clf_cols])
        so = sched_off[va]
        et = np.where(np.isfinite(so) & (so > TAIL_CAP), so, yt[yt > TAIL_CAP].mean())
        pred = np.clip(pt * et + (1 - pt) * pb, 0, None)

    np.save(out, pred.astype(np.float32))
    rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
    print(f'{out.name}: n={va.sum():,} RMSE {rmse:.2f}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
