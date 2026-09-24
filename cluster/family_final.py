"""Final fit of one model FAMILY, one seed: all of 2025 -> the ranking set.

WHY
---
exp66 found that a blend of several families beats the production LightGBM +
network blend on rows outside P (fitted on months 3/6/9/12, judged on the rest):

    all six families, separate weights for route D   held-out -1.56   scored -2.17
    all six families, one set of weights             held-out -1.12   scored -1.99
    network + per-airport alone                      held-out +0.04   (nothing)

The gain comes from DIVERSITY -- lintree, xgboost and routed each err
differently. To ship it, each family needs predictions for the ranking set.

This script produces them with EXACTLY the code that made the validated
out-of-fold predictions: `fit_bulk` from cluster/study_task.py for the bulk
families, and `RoutedModel` for routed, same hyper-parameters, same seed. The
only difference from a study fold is the training set: all twelve months
instead of eleven.

Output (cache/family_final/, one file per family and seed, ranking DEP order):
    perapt / lintree / xgboost / mix  -> the BULK prediction pb (seconds). The
        submission assembly applies the production tail branch (global
        classifier + specialist) on top, exactly as for v9.
    routed -> its full prediction (it carries its own route-D handling).
    ranking_ids.npy -> the matching MVT_ID order.
A file that exists is skipped, so reruns are safe.

    sbatch --array=0-11 cluster/family_final.sbatch                 # 4 families x 3 seeds
    python cluster/family_final.py --family perapt --seed 0 --smoke # plumbing test, minutes
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, ID, CACHE
from prc.features import CATS, align_categories
from prc.io import peak_rss_gb
from study_task import fit_bulk                     # the validated family code

FAMILIES = ('perapt', 'lintree', 'xgboost', 'routed', 'mix')
N_BASE = 18          # the base group (classifier features) = the first 18 columns of the build


def assemble(which, extra_cols):
    """The v5/v6 feature build, identical to experiments/exp54_submit_v6.py. Copied rather
    than imported from nn_final.py, which pulls in torch (absent from the CPU env)."""
    from prc.features import build as build_base
    from prc.congestion import build as build_cong
    from prc.weather import download, prepare, attach as attach_wx
    from prc.disruption import attach as attach_atfm
    from prc.io import load
    full = load(which)
    cong = build_cong(full)
    d = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    del full
    X = build_base(d)
    X['aobt_missing'] = d.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = d.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    if X.shape[1] != N_BASE:
        sys.exit(f'FATAL: the base group has {X.shape[1]} columns, expected {N_BASE}')
    wx = attach_wx(d, prepare(download()))
    atfm = attach_atfm(d)
    X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
                   atfm.set_axis(X.index)], axis=1)
    return d[extra_cols], X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=FAMILIES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--smoke', action='store_true',
                    help='40k training rows, write to family_final_smoke/ -- plumbing only')
    a = ap.parse_args()

    outdir = CACHE / ('family_final_smoke' if a.smoke else 'family_final')
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'{a.family}__seed{a.seed:02d}.npy'
    if out.exists():
        print(f'{out} exists, skipping')
        return

    t0 = time.time()
    tr_meta, Xtr = assemble('training', [TARGET])
    rk_meta, Xrk = assemble('ranking', [ID])
    Xtr, Xrk = align_categories(Xtr, Xrk)
    cols = list(Xtr.columns)
    if len(cols) != 72 or list(Xrk.columns) != cols:
        sys.exit(f'FATAL: expected the 72-column v6 feature set on both sides, got {len(cols)}')
    y_tr = tr_meta[TARGET].astype(float).to_numpy()
    n_tr, n_rk = len(Xtr), len(Xrk)
    if a.smoke:
        keep = np.sort(np.random.default_rng(0).choice(n_tr, 40_000, replace=False))
        Xtr, y_tr, n_tr = Xtr.iloc[keep].reset_index(drop=True), y_tr[keep], 40_000
    X = pd.concat([Xtr, Xrk], ignore_index=True)
    bad = [c for c in CATS if c in cols and str(X[c].dtype) != 'category']
    if bad:
        sys.exit(f'FATAL: categories diverged between training and ranking for {bad}')
    y = np.r_[y_tr, np.full(n_rk, np.nan)]          # ranking labels do not exist
    trn = np.r_[np.ones(n_tr, bool), np.zeros(n_rk, bool)]
    va = ~trn
    cats = [c for c in CATS if c in cols]
    apt = X['APT'].astype(str).to_numpy()
    print(f'{a.family} seed {a.seed}: train {n_tr:,}  ranking {n_rk:,} | {time.time()-t0:.0f}s | '
          f'RSS {peak_rss_gb():.2f} GB', flush=True)

    if a.family == 'routed':
        from prc.models.routed import RoutedModel
        # study_task: RoutedModel(cols, clf_cols=base group, CATS); assemble() asserts that
        # the base group is exactly the first N_BASE columns.
        base_cols = cols[:N_BASE]
        proxy = X['off_aobt3'].to_numpy(); so = X['off_sched'].to_numpy()
        rm = RoutedModel(cols, base_cols, CATS, seed=a.seed, threads=a.threads)
        rm.fit(X.loc[trn], y[trn], proxy[trn], so[trn])
        pred = rm.predict(X.loc[va], proxy[va], so[va])
    else:
        pred = fit_bulk(a.family, X, y, trn, cols, cats, a.seed, a.threads, apt=apt, va=va)
    pred = np.asarray(pred, dtype=np.float64)
    if len(pred) != n_rk or not np.isfinite(pred).all():
        sys.exit(f'FATAL: {len(pred)} predictions for {n_rk} rows, finite={bool(np.isfinite(pred).all())}')

    ids = rk_meta[ID].to_numpy()
    idf = outdir / 'ranking_ids.npy'
    if idf.exists():
        if not np.array_equal(np.load(idf, allow_pickle=True), ids):
            sys.exit('FATAL: ranking row order differs from an earlier run')
    else:
        np.save(idf, ids)
    tmp = outdir / f'.{a.family}__seed{a.seed:02d}.tmp.npy'
    np.save(tmp, pred.astype(np.float32))
    os.replace(tmp, out)
    print(f'{out.name}: {n_rk:,} predictions, mean {pred.mean():.1f}s, median {np.median(pred):.1f}s, '
          f'max {pred.max():.0f}s | {(time.time()-t0)/60:.1f} min | peak RSS {peak_rss_gb():.2f} GB',
          flush=True)


if __name__ == '__main__':
    main()
