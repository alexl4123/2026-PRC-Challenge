"""W2: one (family, seed, month) fold fit with the v11 signals INSIDE the member.

Identical to the study campaign (cluster/study_task.py, cluster/nn_task.py) except for the
bulk regressor's columns (cluster/w2_features.py: +23 numeric v11 columns, +ADES_mvt) and, for
the network, the embedding cap 16 the 2026-09-23 sweep preferred (-0.59 s on its screening
folds). The tail classifier, its columns, its parameters and the tail anchor are unchanged, so
old-vs-new differences are the bulk regressors'.

Writes, per task:
  cache/cv/<campaign>/<family>__seed<NN>__m<MM>.npy        full mixture (exp66 format)
  cache/cv/<campaign>/<family>__seed<NN>__m<MM>.bulk.npy   the bulk regressor alone (not for
                                                           routed), for P-branch evaluation
Idempotent: a task whose mixture file exists is skipped.

  python cluster/w2_task.py --family perapt --seed 0 --month 7
  python cluster/w2_task.py --family nn --seed 0 --month 7 --device cuda
  python cluster/w2_task.py --family mix --seed 0 --month 7 --smoke     # laptop plumbing test
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np

from cluster.study_task import fit_bulk, clf_params
from cluster.w2_features import assemble_w2, columns, W2_CATS
from prc.config import TAIL_CAP, CACHE
from prc.features import CATS
from prc.io import peak_rss_gb

FAMILIES = ['mix', 'perapt', 'lintree', 'xgboost', 'routed', 'nn']


def save(path, arr):
    tmp = path.with_name('.' + path.name + '.tmp.npy')
    np.save(tmp, np.asarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='w2')
    ap.add_argument('--family', required=True, choices=FAMILIES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--month', type=int, required=True)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='nn only')
    ap.add_argument('--emb-cap', type=int, default=16, help='nn only')
    ap.add_argument('--epochs', type=int, default=12, help='nn only')
    ap.add_argument('--batch', type=int, default=8192, help='nn only')
    ap.add_argument('--smoke', action='store_true',
                    help='5%% of the training rows, 1 nn epoch, campaign w2_smoke -- plumbing only')
    a = ap.parse_args()

    campaign = 'w2_smoke' if a.smoke else a.campaign
    outdir = CACHE / 'cv' / campaign
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f'{a.family}__seed{a.seed:02d}__m{a.month:02d}'
    out, out_bulk = outdir / f'{stem}.npy', outdir / f'{stem}.bulk.npy'
    if out.exists():
        print(f'{out.name} exists, skipping')
        return
    if a.family == 'nn':
        from prc.models.embed_nn import resolve_device
        dev = resolve_device(None if a.device == 'auto' else a.device)   # fail fast without --gres
        print(f'device: {dev}', flush=True)

    t0 = time.time()
    X, y, month, groups = assemble_w2()
    cols, clf_cols = columns(groups)
    cats = [c for c in W2_CATS if c in cols]
    ccats = [c for c in CATS if c in clf_cols]
    va, trn = month == a.month, month != a.month
    if a.smoke:
        keep = np.random.default_rng(0).random(len(y)) < 0.05
        trn = trn & keep
    print(f'assembled {X.shape} ({len(cols)} regressor / {len(clf_cols)} classifier columns) '
          f'{time.time() - t0:.0f}s | peak RSS {peak_rss_gb():.2f} GB', flush=True)
    apt = X['APT'].astype(str).to_numpy()
    so_all = X['off_sched'].to_numpy()
    yt = y[trn]
    pb = None

    if a.family == 'routed':
        from prc.models.routed import RoutedModel
        proxy = X['off_aobt3'].to_numpy()
        rm = RoutedModel(cols, clf_cols, W2_CATS, seed=a.seed, threads=a.threads)
        rm.fit(X.loc[trn], y[trn], proxy[trn], so_all[trn])
        pred = rm.predict(X.loc[va], proxy[va], so_all[va])
    elif a.family == 'nn':
        from prc.models.altfam import numeric_design, mixture
        from prc.models.embed_nn import fit_predict
        bulk_trn = trn & (y <= TAIL_CAP)
        codes, cards, Z = numeric_design(X[cols], cols, fit_mask=bulk_trn, cats=W2_CATS)[:3]
        pb = np.clip(fit_predict(codes, Z, y, bulk_trn, va, epochs=1 if a.smoke else a.epochs,
                                 batch=a.batch, seed=a.seed, threads=a.threads, device=str(dev),
                                 emb_cap=a.emb_cap, log=lambda m: print(m, flush=True)), 0, None)
        # tail classifier exactly as cluster/nn_task.py
        mc = lgb.train(dict(objective='binary', metric='auc', learning_rate=0.05,
                            num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                            num_threads=a.threads, verbose=-1, seed=a.seed,
                            deterministic=True, force_row_wise=True),
                       lgb.Dataset(X.loc[trn, clf_cols], (yt > TAIL_CAP).astype(int),
                                   categorical_feature=ccats), num_boost_round=300)
        pt = mc.predict(X.loc[va, clf_cols])
        pred = mixture(pt, pb, so_all[va], yt[yt > TAIL_CAP].mean())
    else:
        # bulk regressor + tail classifier exactly as cluster/study_task.py
        pb = fit_bulk(a.family, X, y, trn, cols, cats, a.seed, a.threads, apt=apt, va=va)
        mc = lgb.train(clf_params(a.seed, a.threads),
                       lgb.Dataset(X.loc[trn, clf_cols], (yt > TAIL_CAP).astype(int),
                                   categorical_feature=ccats), num_boost_round=300)
        pt = mc.predict(X.loc[va, clf_cols])
        so = so_all[va]
        et = np.where(np.isfinite(so) & (so > TAIL_CAP), so, yt[yt > TAIL_CAP].mean())
        pred = np.clip(pt * et + (1 - pt) * pb, 0, None)

    if len(pred) != int(va.sum()) or not np.all(np.isfinite(pred)):
        sys.exit(f'FATAL: {len(pred)} predictions for {int(va.sum())} rows, finite={bool(np.all(np.isfinite(pred)))}')
    if pb is not None:
        save(out_bulk, pb)
    save(out, pred)                       # written last: its presence marks the task complete
    rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
    print(f'{out.name}: n={int(va.sum()):,} RMSE {rmse:.2f} ({time.time() - t0:.0f}s) | '
          f'peak RSS {peak_rss_gb():.2f} GB', flush=True)


if __name__ == '__main__':
    main()
