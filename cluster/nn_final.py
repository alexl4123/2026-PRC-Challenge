"""Final entity-embedding network fit: all of 2025 -> the ranking set, one seed.

WHY
---
The network is not in v5 or v6, which are pure LightGBM. Blending it 50/50 into
the bulk branch was measured over twelve leave-one-month-out folds against the
v5/v6 LightGBM configuration (study campaign, LGB seed 2 vs network seeds 0-2):

    LightGBM alone                 286.94   bulk 234.80
    50/50 LGB + 3-seed network     283.06   bulk 230.52
    paired  -3.94 +- 0.41   CI [-4.72, -3.09]   P(worse) 0.0%
    outside P  -5.02 +- 0.26 | scored months 1+7  -4.78 +- 0.83

This script produces the network half of that blend for the submission. It is
the SAME configuration the 36 cluster runs used -- `fit_predict` defaults, 12
epochs, batch 8192, embedding cap 32, trained on bulk rows (y <= 7200) only,
features = cv_task's `full` set -- so the submission ships exactly what was
validated. (cap16 was -0.59 better on the sweep's screening folds but was never
run on all twelve; it is deliberately NOT used here.)

The ranking rows are encoded TOGETHER with the training rows: categories come
from align_categories, and the numeric scaling / imputation medians are fitted
on the training bulk rows only (fit_mask), exactly as in each CV fold.

Runs on the GPU node (minutes) or the laptop CPU (~45 min per seed):

  sbatch --array=0-2 cluster/nn_final.sbatch                  # cluster, seeds 0-2
  python cluster/nn_final.py --seed 0 --device cpu --threads 6 # laptop
  python cluster/nn_final.py --smoke --device cpu              # plumbing test, ~2 min

Writes cache/nn_final/nn_final_seedNN.npy (float32, bulk seconds, one value per
ranking departure) and cache/nn_final/ranking_ids.npy (the matching MVT_ID
order). A seed whose output exists is skipped, so reruns are safe.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, ID, CACHE
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.models.altfam import numeric_design
from prc.models.embed_nn import fit_predict, resolve_device


def assemble(which, extra_cols):
    """The v5/v6 feature build (identical to experiments/exp54_submit_v6.py)."""
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
    wx = attach_wx(d, prepare(download()))
    atfm = attach_atfm(d)
    X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
                   atfm.set_axis(X.index)], axis=1)
    return d[extra_cols], X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--threads', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)))
    ap.add_argument('--smoke', action='store_true',
                    help='50k training rows, 1 epoch, write to nn_final_smoke/ -- plumbing only')
    a = ap.parse_args()

    outdir = CACHE / ('nn_final_smoke' if a.smoke else 'nn_final')
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'nn_final_seed{a.seed:02d}.npy'
    if out.exists():
        print(f'{out} exists, skipping')
        return

    # Resolve the device FIRST, so a GPU job submitted without --gres fails in
    # seconds rather than after the ~1 minute feature build.
    dev = resolve_device(None if a.device == 'auto' else a.device)
    print(f'device: {dev} | seed {a.seed} | epochs {a.epochs} | batch {a.batch}', flush=True)

    t0 = time.time()
    tr_meta, Xtr = assemble('training', [TARGET])
    rk_meta, Xrk = assemble('ranking', [ID])
    Xtr, Xrk = align_categories(Xtr, Xrk)
    cols = list(Xtr.columns)
    if len(cols) != 72 or list(Xrk.columns) != cols:
        sys.exit(f'FATAL: expected the 72-column v6 feature set on both sides, got '
                 f'{len(cols)} / {len(Xrk.columns)}')
    y = tr_meta[TARGET].astype(float).to_numpy()
    n_tr, n_rk = len(Xtr), len(Xrk)
    print(f'train {Xtr.shape}  ranking {Xrk.shape} | {time.time()-t0:.0f}s | '
          f'RSS {peak_rss_gb():.2f} GB', flush=True)

    X = pd.concat([Xtr, Xrk], ignore_index=True)
    del Xtr, Xrk
    bad = [c for c in CATS if c in cols and str(X[c].dtype) != 'category']
    if bad:
        sys.exit(f'FATAL: categories diverged between training and ranking for {bad}; '
                 'the stacked frame lost its categorical dtype')

    fit = np.r_[y <= TAIL_CAP, np.zeros(n_rk, bool)]
    pred = np.r_[np.zeros(n_tr, bool), np.ones(n_rk, bool)]
    epochs = a.epochs
    if a.smoke:
        keep = np.flatnonzero(fit)
        keep = np.random.default_rng(0).choice(keep, 50_000, replace=False)
        fit = np.zeros(len(X), bool); fit[keep] = True
        epochs = 1
    y_ext = np.r_[y, np.zeros(n_rk)]

    codes, cards, Z = numeric_design(X[cols], cols, fit_mask=fit)[:3]
    del X
    print(f'design: {codes.shape[1]} categorical, {Z.shape[1]} numeric | '
          f'fit rows {int(fit.sum()):,} | predict rows {int(pred.sum()):,} | '
          f'RSS {peak_rss_gb():.2f} GB', flush=True)

    p = fit_predict(codes, Z, y_ext, fit, pred, epochs=epochs, batch=a.batch,
                    seed=a.seed, threads=a.threads, device=str(dev),
                    log=lambda m: print(m, flush=True))
    p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
    if len(p) != n_rk or not np.isfinite(p).all():
        sys.exit(f'FATAL: got {len(p)} predictions for {n_rk} ranking rows, '
                 f'finite={bool(np.isfinite(p).all())}')

    ids = rk_meta[ID].to_numpy()
    idf = outdir / 'ranking_ids.npy'
    if idf.exists():
        if not np.array_equal(np.load(idf, allow_pickle=True), ids):
            sys.exit('FATAL: ranking row order differs from an earlier seed')
    else:
        np.save(idf, ids)
    tmp = outdir / f'.nn_final_seed{a.seed:02d}.tmp.npy'
    np.save(tmp, p.astype(np.float32))
    os.replace(tmp, out)
    print(f'{out.name}: {n_rk:,} predictions, mean {p.mean():.1f}s, median '
          f'{np.median(p):.1f}s, max {p.max():.0f}s | {(time.time()-t0)/60:.1f} min | '
          f'peak RSS {peak_rss_gb():.2f} GB', flush=True)


if __name__ == '__main__':
    main()
