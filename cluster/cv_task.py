"""One unit of cluster work: a single (config, seed, fold) fit.

Writes the fold's out-of-fold predictions to
    cache/cv/<campaign>/<config>__seed<NN>__m<MM>.npy
so `collect.py` can assemble them without the tasks talking to each other.

Designed to be small and timeout-bounded, per the local admin guidance against
long-running job scripts.

  python cluster/cv_task.py --campaign seeds --config full --seed 3 --month 7
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, CACHE
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.disruption import attach as attach_atfm
from prc.io import load
from prc.weather import download, prepare, attach as attach_wx


def assemble():
    """Full feature matrix plus target and month. Identical to the local path --
    cluster and laptop must not drift apart."""
    full = load('training')
    cong = build_cong(full)
    tr = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    del full
    X = build_base(tr)
    X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    base = list(X.columns)
    wx = attach_wx(tr, prepare(download()))
    atfm = attach_atfm(tr)
    groups = {'base': base, 'cong': list(cong.columns),
              'wx': list(wx.columns), 'atfm': list(atfm.columns)}
    X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
                   atfm.set_axis(X.index)], axis=1)
    y = tr[TARGET].astype(float).to_numpy()
    month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    X, = align_categories(X)
    return X, y, month, groups


CONFIGS = {
    'base':  ['base'],
    'cong':  ['base', 'cong'],
    'wx':    ['base', 'cong', 'wx'],
    'full':  ['base', 'cong', 'wx', 'atfm'],
}

# EXP15 found the rich feature set helps the BULK regressor (July bulk -8.0s)
# but hurts the TAIL classifier, which learns from only 435 positives in 1.74M
# rows and gets noisier with 28 extra mostly-irrelevant columns. So the two
# components are configured independently.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--config', default='full', choices=sorted(CONFIGS),
                    help='feature set for the BULK REGRESSOR')
    ap.add_argument('--clf-config', default=None, choices=sorted(CONFIGS),
                    help='feature set for the TAIL CLASSIFIER; defaults to --config')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--month', type=int, required=True)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    a = ap.parse_args()

    outdir = CACHE / 'cv' / a.campaign
    outdir.mkdir(parents=True, exist_ok=True)
    clf_cfg = a.clf_config or a.config
    out = outdir / f'r-{a.config}__c-{clf_cfg}__seed{a.seed:02d}__m{a.month:02d}.npy'
    if out.exists():
        print(f'{out.name} exists, skipping')
        return

    t0 = time.time()
    X, y, month, groups = assemble()
    cols = [c for g in CONFIGS[a.config] for c in groups[g]]
    clf_cols = [c for g in CONFIGS[clf_cfg] for c in groups[g]]
    cats = [c for c in CATS if c in cols]
    clf_cats = [c for c in CATS if c in clf_cols]
    va = month == a.month
    tr_ = ~va

    reg = dict(objective='regression', metric='rmse', learning_rate=0.05,
               num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               num_threads=a.threads, verbose=-1, seed=a.seed,
               bagging_seed=a.seed + 1, feature_fraction_seed=a.seed + 2,
               deterministic=True, force_row_wise=True)
    clf = dict(reg, objective='binary', metric='auc', num_leaves=63, min_data_in_leaf=50)
    clf.pop('metric', None)
    clf['metric'] = 'auc'

    yt = y[tr_]
    k = yt <= TAIL_CAP
    mb = lgb.train(reg, lgb.Dataset(X.loc[tr_, cols][k], yt[k], categorical_feature=cats),
                   num_boost_round=350)
    mc = lgb.train(clf, lgb.Dataset(X.loc[tr_, clf_cols], (yt > TAIL_CAP).astype(int),
                                    categorical_feature=clf_cats), num_boost_round=300)
    Xv = X.loc[va, cols]
    pb = np.clip(mb.predict(Xv), 0, None)
    pt = mc.predict(X.loc[va, clf_cols])
    osv = Xv['off_sched'].to_numpy()
    et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
    pred = np.clip(pt * et + (1 - pt) * pb, 0, None)

    np.save(out, pred.astype(np.float32))
    rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
    print(f'{out.name}: n={va.sum():,} RMSE {rmse:.2f}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
