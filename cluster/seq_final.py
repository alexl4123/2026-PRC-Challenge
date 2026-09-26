"""Bet D: final fit of the 'seq' member, one seed: all of 2025 -> the ranking set (Jan + Jul 2026).

Same code path as the fold fits (cluster/seq_task.py); the differences are only the training set (all twelve
months) and the frames: training and ranking are assembled together (cluster/w2_features.assemble_w2_pair with
adsb=True, extra=True -- the adsb2 members' columns) so every categorical shares one vocabulary, and the movement
table is ONE table of training + ranking movements with the training token vocabulary. Contexts are built per
(airport, calendar month) group, so a 2026 departure only ever sees 2026 movements of its own month, exactly as a
2025 departure sees its own month in training.

Output (cache/seq_final/, ranking DEP order, float32):
    seq__seed<NN>.npy   the BULK prediction (seconds); the submission builder applies the production tail
                        (seqnc__seed<NN>.npy with --no-context)
                        branch on top, as for the other network members (cluster/w2_final.py)
    ranking_ids.npy     the matching MVT_ID order (checked against an earlier run)
A file that exists is skipped, so reruns are safe.

    python cluster/seq_final.py --seed 0 --device cuda       # GPU node (--gres=gpu:1)
    python cluster/seq_final.py --seed 0 --smoke             # laptop plumbing test -> cache/seq_final_smoke/
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from prc.config import TARGET, TAIL_CAP, ID, CACHE
from prc.io import peak_rss_gb
from cluster.seq_task import check_inputs, check_alignment

REQUIRED = ['cache/v11/feat_training.parquet', 'cache/v11/feat_ranking.parquet',
            'cache/adsb_lol/features_training.parquet', 'cache/adsb_lol/features_ranking.parquet']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--window', type=int, default=5400)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--no-context', action='store_true', help="the ablation twin 'seqnc' (no context branch)")
    ap.add_argument('--deice', action='store_true', help='C1 feature set (+ de-icing shares) -> cache/seq_c1_final/')
    ap.add_argument('--smoke', action='store_true',
                    help='laptop plumbing test: 2 airports, 3%% of their training rows, K=16, 1 epoch')
    ap.add_argument('--tag', default=None,
                    help='output folder cache/<tag>_final/ -- REQUIRED for any non-default setting, so a variant can '
                         'never be skipped into (or mixed with) the shipped cache/seq_final/')
    a = ap.parse_args()
    nondefault = (a.epochs, a.batch, a.lr, a.K, a.window, a.d, a.layers) != (10, 4096, 2e-3, 64, 5400, 64, 2)
    if nondefault and a.tag is None:
        sys.exit('FATAL: non-default settings need --tag (e.g. --epochs 15 --tag seq_ep15)')
    fam = 'seqnc' if a.no_context else 'seq'

    check_inputs(REQUIRED + (['cache/deice_0926/ind_training.parquet', 'cache/deice_0926/ind_ranking.parquet']
                             if a.deice else []))
    base = a.tag or ('seq_c1' if a.deice else 'seq')
    outdir = CACHE / (base + ('_final_smoke' if a.smoke else '_final'))
    outdir.mkdir(parents=True, exist_ok=True)
    if a.tag is not None:                       # one configuration per tagged folder (as the fold campaigns)
        import json
        conf = dict(deice=a.deice, K=a.K, window=a.window, d=a.d, layers=a.layers, epochs=a.epochs, batch=a.batch, lr=a.lr)
        cf = outdir / '_config.json'
        if cf.exists():
            diff = {k: (v0, conf[k]) for k, v0 in json.load(open(cf)).items() if k in conf and conf[k] != v0}
            if diff:
                sys.exit(f'FATAL: {outdir.name} was started with other settings (old, new): {diff}')
        else:
            json.dump(conf, open(cf, 'w'), indent=1)
    out = outdir / f'{fam}__seed{a.seed:02d}.npy'
    if out.exists():
        print(f'{out} exists, skipping')
        return
    import pandas as pd
    from cluster.w2_features import assemble_w2_pair, columns, W2_CATS
    from prc.io import load
    from prc.models import seq_net as S
    dev = S.resolve_device(None if a.device == 'auto' else a.device)
    print(f'device: {dev}', flush=True)
    K, epochs = (16, 1) if a.smoke else (a.K, a.epochs)

    t0 = time.time()
    Xtr, mtr, Xrk, mrk, groups = assemble_w2_pair(adsb=True, extra=True, **({'deice': True} if a.deice else {}))
    cols, _ = columns(groups)
    n_tr, n_rk = len(Xtr), len(Xrk)
    X = pd.concat([Xtr, Xrk], ignore_index=True)
    del Xtr, Xrk
    bad = [c for c in W2_CATS if c in cols and str(X[c].dtype) != 'category']
    if bad:
        sys.exit(f'FATAL: categories diverged between training and ranking for {bad}')
    y = np.r_[mtr[TARGET].to_numpy(dtype=float), np.full(n_rk, np.nan)]
    trn = np.r_[np.ones(n_tr, bool), np.zeros(n_rk, bool)]
    print(f'assembled train {n_tr:,} + ranking {n_rk:,}, {len(cols)} columns | {time.time() - t0:.0f}s | '
          f'RSS {peak_rss_gb():.2f} GB', flush=True)

    # one movement table: training then ranking movements, training vocabulary
    ftr = load('training', columns=S.MV_COLS)
    frk = load('ranking', columns=S.MV_COLS)
    vocab = S.vocabularies(ftr)
    full = pd.concat([ftr.astype({c: object for c in ftr.columns if str(ftr[c].dtype) == 'category'}),
                      frk.astype({c: object for c in frk.columns if str(frk[c].dtype) == 'category'})],
                     ignore_index=True)
    del ftr, frk
    adsb = pd.concat([pd.read_parquet(CACHE / 'adsb_lol' / f'features_{w}.parquet', columns=['MVT_ID_mvt'] + S.ADSB_TOK)
                      for w in ('training', 'ranking')], ignore_index=True)
    if adsb.MVT_ID_mvt.duplicated().any():
        sys.exit('FATAL: an MVT_ID occurs in both the training and the ranking adsb.lol features')
    mv, _ = S.movement_table(full, adsb, vocab=vocab)
    del full, adsb
    tgt_mv = check_alignment(mv, X['off_sched'].to_numpy(), 'training+ranking')
    ids_mv = mv['id'][tgt_mv]
    if not np.array_equal(ids_mv[n_tr:], mrk[ID].to_numpy()) or not np.array_equal(ids_mv[:n_tr], mtr[ID].to_numpy()):
        sys.exit('FATAL: movement-table departure ids differ from the assembled frames')
    nb = S.build_context(mv, K=K, window=a.window, log=lambda m: print(m, flush=True))
    if a.smoke:
        sub = np.isin(X['APT'].astype(str).to_numpy(), ['LSZH', 'EDDM'])
        keep_tr = sub & trn & (np.random.default_rng(0).random(len(y)) < 0.03)
        rows = np.flatnonzero(keep_tr | ~trn)                 # a few training rows, every ranking row
        X = X.iloc[rows].reset_index(drop=True)
        y, trn, tgt_mv, nb = y[rows], trn[rows], tgt_mv[rows], nb[rows]
    fit = trn & (np.nan_to_num(y, nan=np.inf) <= TAIL_CAP)
    va = ~trn
    codes, Z, cat_cols, _ = S.target_design(X, cols, W2_CATS, fit)
    del X
    ctx = S.Context(mv, dev)
    del mv
    print(f'design: {Z.shape[1]} numerics+flags, {ctx.n_tok} token numerics; fit {fit.sum():,}, predict {va.sum():,} '
          f'| {time.time() - t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)
    pred = S.fit_predict(codes, Z, np.nan_to_num(y, nan=0.0), fit, va, ctx=ctx, tgt_mv=tgt_mv, nb=nb,
                         use_context=not a.no_context, epochs=epochs, batch=a.batch, lr=a.lr, d=a.d, layers=a.layers,
                         seed=a.seed, threads=a.threads, device=str(dev), log=lambda m: print(m, flush=True))
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    if len(pred) != n_rk or not np.isfinite(pred).all():
        sys.exit(f'FATAL: {len(pred)} predictions for {n_rk} rows, finite={bool(np.isfinite(pred).all())}')

    ids = mrk[ID].to_numpy()
    idf = outdir / 'ranking_ids.npy'
    if idf.exists():
        if not np.array_equal(np.load(idf, allow_pickle=True), ids):
            sys.exit('FATAL: ranking row order differs from an earlier run')
    else:
        np.save(idf, ids)
    tmp = outdir / f'.{fam}__seed{a.seed:02d}.tmp.npy'
    np.save(tmp, pred.astype(np.float32))
    os.replace(tmp, out)
    print(f'{out.name}: {n_rk:,} predictions, mean {pred.mean():.1f}s, median {np.median(pred):.1f}s, '
          f'max {pred.max():.0f}s | {(time.time() - t0) / 60:.1f} min | peak RSS {peak_rss_gb():.2f} GB', flush=True)


if __name__ == '__main__':
    main()
