"""W2 final fit of one member, one seed: all of 2025 -> the ranking set, with the v11 columns.

The same code path as the W2 fold fits (cluster/w2_task.py) and as the shipped final fits
(cluster/family_final.py, cluster/nn_final.py); the only difference from a W2 fold is the
training set: all twelve months instead of eleven. Training and ranking are assembled together
(cluster/w2_features.assemble_w2_pair) so every categorical, ADES_mvt included, shares one
vocabulary, and the network's scaling is fitted on the training bulk rows only.

Output (cache/w2_final/, ranking DEP order, float32):
    mix / perapt / lintree / xgboost / nn / w3 -> the BULK prediction pb (seconds); the submission
        builder applies the production tail branch on top, as for v12
    routed -> its full prediction (it carries its own tail handling)
    ranking_ids.npy -> the matching MVT_ID order
A file that exists is skipped, so reruns are safe.

    sbatch cluster/w2_final.sbatch          # CPU members, 5 families x 3 seeds
    sbatch cluster/w2_final_nn.sbatch       # network, 3 seeds, GPU
    python cluster/w2_final.py --family perapt --seed 0 --smoke   # plumbing only (laptop)

C1 (2026-09-26): --adsb --extra --deice --nn-missing sentinel --tag c1 -> cache/c1_final/ (a _config.json in the
folder stops a later run with other flags from mixing into it).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from cluster.w2_features import assemble_w2_pair, columns, W2_CATS
from prc.config import TARGET, TAIL_CAP, ID, CACHE
from prc.io import peak_rss_gb

FAMILIES = ('mix', 'perapt', 'lintree', 'xgboost', 'routed', 'nn', 'w3')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=FAMILIES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='nn only')
    ap.add_argument('--emb-cap', type=int, default=16, help='nn only')
    ap.add_argument('--epochs', type=int, default=12, help='nn only')
    ap.add_argument('--batch', type=int, default=8192, help='nn only')
    ap.add_argument('--smoke', action='store_true', help='40k training rows, 1 nn epoch, w2_final_smoke/')
    ap.add_argument('--adsb', action='store_true', help='add the adsb.lol columns; writes cache/adsb_final/')
    ap.add_argument('--extra', action='store_true', help='A3 arm 2 (+ wake / backlog); with --adsb writes cache/adsb2_final/')
    ap.add_argument('--deice', action='store_true', help='C1: + de-icing / disruption shares; needs --tag')
    ap.add_argument('--nn-missing', default='median', choices=['median', 'sentinel'], help='nn only; non-default needs --tag')
    ap.add_argument('--tag', default=None, help='output folder cache/<tag>_final/ (default from the flags: w2 / adsb / adsb2)')
    a = ap.parse_args()

    if (a.deice or a.nn_missing != 'median') and a.tag is None:
        sys.exit('FATAL: --deice / --nn-missing need an explicit --tag (e.g. --tag c1)')
    tag = a.tag or (('adsb' if a.adsb else 'w2') + ('2' if a.extra else ''))
    outdir = CACHE / (tag + ('_final_smoke' if a.smoke else '_final'))
    outdir.mkdir(parents=True, exist_ok=True)
    if a.tag is not None:
        conf = dict(adsb=a.adsb, extra=a.extra, deice=a.deice, nn_missing=a.nn_missing, emb_cap=a.emb_cap,
                    epochs=a.epochs, batch=a.batch)
        cf = outdir / '_config.json'
        if cf.exists():
            old = json.load(open(cf))
            diff = {k: (old.get(k), v) for k, v in conf.items() if old.get(k) != v}
            if diff:
                sys.exit(f'FATAL: {outdir.name} was started with other settings (old, new): {diff}')
        else:
            tmp = outdir / f'._config.{os.getpid()}.json'
            json.dump(conf, open(tmp, 'w'), indent=1)
            os.replace(tmp, cf)
    out = outdir / f'{a.family}__seed{a.seed:02d}.npy'
    if out.exists():
        print(f'{out} exists, skipping')
        return
    if a.family in ('nn', 'w3'):
        from prc.models.embed_nn import resolve_device
        dev = resolve_device(None if a.device == 'auto' else a.device)
        print(f'device: {dev}', flush=True)

    t0 = time.time()
    Xtr, mtr, Xrk, mrk, groups = assemble_w2_pair(adsb=a.adsb, extra=a.extra, deice=a.deice)
    cols, clf_cols = columns(groups)
    y_tr = mtr[TARGET].to_numpy()
    n_tr, n_rk = len(Xtr), len(Xrk)
    if a.smoke:
        keep = np.sort(np.random.default_rng(0).choice(n_tr, 40_000, replace=False))
        Xtr, y_tr, n_tr = Xtr.iloc[keep].reset_index(drop=True), y_tr[keep], 40_000
    X = pd.concat([Xtr, Xrk], ignore_index=True)
    del Xtr, Xrk
    bad = [c for c in W2_CATS if c in cols and str(X[c].dtype) != 'category']
    if bad:
        sys.exit(f'FATAL: categories diverged between training and ranking for {bad}')
    y = np.r_[y_tr, np.full(n_rk, np.nan)]
    trn = np.r_[np.ones(n_tr, bool), np.zeros(n_rk, bool)]
    va = ~trn
    cats = [c for c in W2_CATS if c in cols]
    print(f'{a.family} seed {a.seed}: train {n_tr:,} ranking {n_rk:,}, {len(cols)} columns | '
          f'{time.time() - t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)

    if a.family == 'routed':
        from prc.models.routed import RoutedModel
        proxy = X['off_aobt3'].to_numpy(); so = X['off_sched'].to_numpy()
        rm = RoutedModel(cols, clf_cols, W2_CATS, seed=a.seed, threads=a.threads)
        rm.fit(X.loc[trn], y[trn], proxy[trn], so[trn])
        pred = rm.predict(X.loc[va], proxy[va], so[va])
    elif a.family == 'nn':
        from prc.models.altfam import numeric_design
        from prc.models.embed_nn import fit_predict
        fit = trn & (np.nan_to_num(y, nan=np.inf) <= TAIL_CAP)
        codes, cards, Z = numeric_design(X[cols], cols, fit_mask=fit, missing=a.nn_missing, cats=W2_CATS)[:3]
        del X
        y_ext = np.nan_to_num(y, nan=0.0)
        pred = fit_predict(codes, Z, y_ext, fit, va, epochs=1 if a.smoke else a.epochs, batch=a.batch,
                           seed=a.seed, threads=a.threads, device=str(dev), emb_cap=a.emb_cap,
                           log=lambda m: print(m, flush=True))
        pred = np.clip(pred, 0, None)
    elif a.family == 'w3':
        from prc.models import w3_net
        fit = trn & (np.nan_to_num(y, nan=np.inf) <= TAIL_CAP)
        codes, num, _ = w3_net.design(X, cols, W2_CATS)
        heads = w3_net.head_index(X['APT'].astype(str).to_numpy())
        del X
        pred = w3_net.fit_predict(codes, num, heads, np.nan_to_num(y, nan=0.0), fit, va,
                                  epochs=1 if a.smoke else a.epochs, batch=a.batch, seed=a.seed,
                                  threads=a.threads, device=str(dev), emb_cap=a.emb_cap,
                                  log=lambda m: print(m, flush=True))
    else:
        from cluster.study_task import fit_bulk
        apt = X['APT'].astype(str).to_numpy()
        pred = fit_bulk(a.family, X, y, trn, cols, cats, a.seed, a.threads, apt=apt, va=va)
    pred = np.asarray(pred, dtype=np.float64)
    if len(pred) != n_rk or not np.isfinite(pred).all():
        sys.exit(f'FATAL: {len(pred)} predictions for {n_rk} rows, finite={bool(np.isfinite(pred).all())}')

    ids = mrk[ID].to_numpy()
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
          f'max {pred.max():.0f}s | {(time.time() - t0) / 60:.1f} min | peak RSS {peak_rss_gb():.2f} GB', flush=True)


if __name__ == '__main__':
    main()
