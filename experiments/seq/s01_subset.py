"""Bet D, step 1: laptop-sized signal test of the set model (prc/models/seq_net.py) on a held-out MONTH.

Subset: a few airports x a few months of 2025. Train on the train months' bulk rows (y <= 7200), predict every
row of the held-out month. Arms (same network, same rows, same budget):
  ctx        context of K neighbouring movements ON  (the bet)
  noctx      context OFF: the identical target branch alone = tabular MLP on the 143 member columns (ablation a)
  ctx_res    ctx, fitted on the residual y - v17s_oof  (diagnostic: information beyond the v17 structure)
  noctx_res  noctx on the same residual (the stacking control for ctx_res)
Predictions -> cache/seq/subset/<tag>/<arm>__s<seed>.npy (+ rows.npz with the frame row index / ids / y / month).
Evaluation is experiments/seq/s02_subset_eval.py (never here).

    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 nice -n 5 \
      /home/thinklex/prc2026-venv/bin/python experiments/seq/s01_subset.py --arms ctx noctx --seeds 0 1
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, TAIL_CAP
from prc.io import load, peak_rss_gb
from prc.features import CATS
from prc.models import seq_net as S

W2_CATS = CATS + ['ADES_mvt']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--airports', nargs='+', default=['LTFM', 'LFPG', 'EHAM'])
    ap.add_argument('--train-months', nargs='+', type=int, default=[5, 6, 8])
    ap.add_argument('--test-month', type=int, default=7)
    ap.add_argument('--arms', nargs='+', default=['ctx', 'noctx'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--window', type=int, default=5400)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--max-train', type=int, default=0, help='subsample train rows (timing tests)')
    ap.add_argument('--ablate', default='none', choices=['none', 'past', 'noarr', 'nodep', 'noadsb', 'nostand'],
                    help='context ablation: past = only neighbours with MVT <= ours; noarr / nodep = drop arrivals / '
                         'departures from the context; noadsb = blank the neighbours adsb.lol fields; nostand = drop '
                         'neighbours at our own stand (the stand-turnover channel)')
    a = ap.parse_args()
    tag = a.tag or f"{'-'.join(a.airports)}__tr{''.join(f'{m:02d}' for m in a.train_months)}__te{a.test_month:02d}__K{a.K}"
    if a.ablate != 'none':
        tag += f'__{a.ablate}'
    out = CACHE / 'seq' / 'subset' / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    months = sorted(set(a.train_months) | {a.test_month})

    meta = json.load(open(CACHE / 'seq' / 'w2x_groups.json'))
    cols = meta['cols']
    F = pd.read_parquet(CACHE / 'seq' / 'w2x_training.parquet')
    frame_row = np.flatnonzero(F.APT.astype(str).isin(a.airports).to_numpy() & np.isin(F.month.to_numpy(), months))
    F = F.iloc[frame_row].reset_index(drop=True)
    y, month = F.y.to_numpy(float), F.month.to_numpy()
    print(f'frame rows {len(F):,} ({time.time() - t0:.0f}s, RSS {peak_rss_gb():.2f} GB)', flush=True)

    full = load('training', columns=S.MV_COLS)
    keep = full.APT.astype(str).isin(a.airports).to_numpy() & np.isin(full.MVT_TIME_UTC_mvt.dt.month.to_numpy(), months)
    full = full[keep].reset_index(drop=True)
    adsb = pd.read_parquet(CACHE / 'adsb_lol' / 'features_training.parquet', columns=['MVT_ID_mvt'] + S.ADSB_TOK)
    mv, vocab = S.movement_table(full, adsb)
    del full, adsb
    tgt_mv = np.flatnonzero(mv['is_dep'])
    if not np.array_equal(mv['id'][tgt_mv], F.MVT_ID_mvt.to_numpy()):
        sys.exit('FATAL: movement-table departures are not row-aligned with the feature frame')
    if a.ablate == 'noadsb':
        for c in S.ADSB_TOK:
            mv['S'][:, mv['S_names'].index(c)] = np.nan
    nb = S.build_context(mv, K=a.K, window=a.window)
    if a.ablate in ('past', 'noarr', 'nodep', 'nostand'):
        j = np.clip(nb, 0, None)
        drop = {'past': mv['t_abs'][j] > mv['t_abs'][tgt_mv][:, None],
                'noarr': ~mv['is_dep'][j], 'nodep': mv['is_dep'][j],
                'nostand': mv['stand'][j] == mv['stand'][tgt_mv][:, None]}[a.ablate]
        nb = np.where(drop, -1, nb)
        print(f'ablation {a.ablate}: filled slots now {(nb >= 0).sum(1).mean():.1f}', flush=True)
    print(f'movements {len(mv["id"]):,}, token statics {mv["S"].shape[1]} ({time.time() - t0:.0f}s, RSS {peak_rss_gb():.2f} GB)',
          flush=True)

    trn = np.isin(month, a.train_months)
    va = month == a.test_month
    fit = trn & (y <= TAIL_CAP)
    if a.max_train:
        rng = np.random.default_rng(0)
        fit = fit & (rng.random(len(y)) < a.max_train / fit.sum())
    codes, Z, cat_cols, names = S.target_design(F, cols, W2_CATS, fit)
    v17 = np.load(CACHE / 'w2' / 'v17s_oof.npy').astype(float)[frame_row]
    np.savez(out / 'rows.npz', frame_row=frame_row, ids=F.MVT_ID_mvt.to_numpy(), y=y, month=month,
             apt=F.APT.astype(str).to_numpy(), va=va, v17=v17)
    del F
    import torch
    ctx = S.Context(mv, torch.device('cpu'))
    print(f'target design: {len(cat_cols)} cats, {Z.shape[1]} numerics (+flags); tokens {ctx.n_tok} numerics; '
          f'fit rows {fit.sum():,}, held-out rows {va.sum():,} | RSS {peak_rss_gb():.2f} GB', flush=True)

    bulk_va = va & (y <= TAIL_CAP)
    yv = y[va]

    def hook(ep, p):
        k = bulk_va[va]
        return f'held-out bulk RMSE {np.sqrt(np.mean((yv[k] - p[k]) ** 2)):.2f}'

    for seed in a.seeds:
        for arm in a.arms:
            f = out / f'{arm}__s{seed}.npy'
            if f.exists():
                print(f'{f.name} exists, skipping'); continue
            t1 = time.time()
            use_ctx = arm.startswith('ctx')
            offset = v17 if arm.endswith('_res') else None
            print(f'--- {arm} seed {seed}', flush=True)
            p = S.fit_predict(codes, Z, y, fit, va, ctx=ctx, tgt_mv=tgt_mv, nb=nb, offset=offset, use_context=use_ctx,
                              epochs=a.epochs, batch=a.batch, lr=a.lr, d=a.d, layers=a.layers, seed=seed,
                              threads=a.threads, device='cpu', log=lambda m: print(m, flush=True), eval_hook=hook)
            np.save(f, p.astype(np.float32))
            print(f'{arm} seed {seed}: {hook(0, p)} | {time.time() - t1:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)


if __name__ == '__main__':
    main()
