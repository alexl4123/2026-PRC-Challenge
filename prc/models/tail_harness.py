"""Shared harness for tail-model experiments (exp30+).

The mixture factorises as

    pred = p * e_tail + (1 - p) * r

where `r` (the bulk regressor) is by far the most expensive component and is
IDENTICAL across every tail variant we want to compare. So each component gets
its own out-of-fold pass, cached to cache/tail/<name>__m<MM>.npy, and the
variants are assembled afterwards from cached arrays in milliseconds.

That makes a tail experiment cost one classifier pass (~1/3 of a regressor pass)
instead of a full model rebuild, and it guarantees that two variants compared
with a paired bootstrap differ ONLY in the component under test.

Everything here is fold-safe: a component fitted for month m never sees month m.
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, CACHE
from prc.features import build as build_base, align_categories, CATS, NEEDED
from prc.io import load

TAILDIR = CACHE / 'tail'
TAILDIR.mkdir(parents=True, exist_ok=True)

# The artifact: the airport stamped BLOCK_TIME at SCHED_TIME, so y == off_sched.
# 60s is the shoulder of the delta distribution -- see exp30: on tail rows the
# modal |delta| values are 0-6s and 432/584 fall inside 60s, while the all-row
# median |delta| is 481s.
ARTIFACT_TOL = 60.0

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, deterministic=True,
           force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, deterministic=True,
           force_row_wise=True)


def params(base, seed, threads):
    return dict(base, seed=seed, bagging_seed=seed + 1,
                feature_fraction_seed=seed + 2, num_threads=threads)


def assemble():
    """Feature matrix (the `base` config of cluster/cv_task.py) plus the label
    pieces the tail work needs: y, month, off_sched, delta and the airport."""
    cols = sorted(set(NEEDED + [TARGET, 'BLOCK_TIME_UTC_mvt']))
    tr = load('training', phase='DEP', columns=cols)
    X = build_base(tr)
    X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    X, = align_categories(X)
    X = X.reset_index(drop=True)

    meta = pd.DataFrame({
        'y': tr[TARGET].astype(float).to_numpy(),
        'month': tr.MVT_TIME_UTC_mvt.dt.month.to_numpy(),
        'off_sched': X.off_sched.to_numpy(),
        'delta': (tr.BLOCK_TIME_UTC_mvt - tr.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy(),
        'apt': tr.APT.astype(str).to_numpy(),
        'aobt_missing': X.aobt_missing.to_numpy().astype(bool),
    })
    return X, meta


def artifact_label(meta, tol=ARTIFACT_TOL):
    return (np.abs(meta['delta'].to_numpy()) <= tol).astype(int)


# ---------------------------------------------------------------------------
# Component runners. Each writes cache/tail/<name>__m<MM>.npy and skips work
# that is already on disk, so a run can be interrupted and resumed.
# ---------------------------------------------------------------------------

def run_component(name, X, meta, fit_predict, months=None, seed=0, threads=4,
                  force=False, verbose=True):
    """Out-of-fold values for one component. Returns an array aligned to meta."""
    month = meta['month'].to_numpy()
    out = np.full(len(month), np.nan)
    todo = sorted(set(month.tolist())) if months is None else list(months)
    for m in todo:
        f = TAILDIR / f'{name}__seed{seed:02d}__m{m:02d}.npy'
        va = month == m
        if f.exists() and not force:
            v = np.load(f)
            if len(v) != va.sum():
                raise RuntimeError(f'{f.name}: {len(v)} values for {va.sum()} rows '
                                   '-- stale cache, delete cache/tail and rerun')
            out[va] = v
            if verbose:
                print(f'    {name} m{m:02d}: cached', flush=True)
            continue
        t0 = time.time()
        v = np.asarray(fit_predict(X, meta, ~va, va, seed, threads), float)
        np.save(f, v.astype(np.float32))
        out[va] = v
        if verbose:
            print(f'    {name} m{m:02d}: fitted {va.sum():,} rows ({time.time()-t0:.0f}s)',
                  flush=True)
    return out


def _cats(cols):
    return [c for c in CATS if c in cols]


def fp_bulk_regressor(X, meta, tr, va, seed, threads):
    """E[y | y <= TAIL_CAP] -- the existing bulk model, unchanged."""
    y = meta['y'].to_numpy()
    k = tr & (y <= TAIL_CAP)
    m = lgb.train(params(REG, seed, threads),
                  lgb.Dataset(X[k], y[k], categorical_feature=_cats(X.columns)),
                  num_boost_round=350)
    return np.clip(m.predict(X[va]), 0, None)


def fp_notA_regressor(X, meta, tr, va, seed, threads):
    """E[y | not A] -- the correct second component of the artifact mixture."""
    y = meta['y'].to_numpy()
    A = artifact_label(meta).astype(bool)
    k = tr & ~A
    m = lgb.train(params(REG, seed, threads),
                  lgb.Dataset(X[k], y[k], categorical_feature=_cats(X.columns)),
                  num_boost_round=350)
    return np.clip(m.predict(X[va]), 0, None)


def fp_tail_classifier(X, meta, tr, va, seed, threads):
    """P(y > TAIL_CAP | x) -- the baseline classifier."""
    y = meta['y'].to_numpy()
    m = lgb.train(params(CLF, seed, threads),
                  lgb.Dataset(X[tr], (y[tr] > TAIL_CAP).astype(int),
                              categorical_feature=_cats(X.columns)),
                  num_boost_round=300)
    return m.predict(X[va])


def fp_artifact_classifier(X, meta, tr, va, seed, threads):
    """P(|BLOCK - SCHED| <= 60s | x) -- 199,634 positives instead of 584."""
    a = artifact_label(meta)
    m = lgb.train(params(CLF, seed, threads),
                  lgb.Dataset(X[tr], a[tr], categorical_feature=_cats(X.columns)),
                  num_boost_round=300)
    return m.predict(X[va])


def e_tail_classic(meta):
    """The baseline tail estimate: off_sched where it is usable, else the mean
    of the training tail. Mirrors cluster/cv_task.py exactly."""
    y = meta['y'].to_numpy()
    os_ = meta['off_sched'].to_numpy()
    tm = y[y > TAIL_CAP].mean()
    return np.where(np.isfinite(os_) & (os_ > TAIL_CAP), os_, tm)


def mixture(p, e, r):
    return np.clip(p * e + (1.0 - p) * r, 0, None)
