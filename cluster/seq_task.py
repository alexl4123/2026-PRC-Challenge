"""Bet D: one (seed, month) fold fit of the 'seq' member -- the set model over neighbouring movements
(prc/models/seq_net.py) -- in the same mixture format as cluster/w2_task.py's nn branch.

Inputs, identical to the adsb2 members (w2_task --adsb --extra):
  the target's own columns   cluster/w2_features.assemble_w2(adsb=True, extra=True)   (143 regressor columns)
  the neighbour tokens       load('training') movements (both phases; departure BLOCK_TIME blanked in
                             seq_net.movement_table) + cache/adsb_lol/features_training.parquet
Bulk regressor: seq_net.fit_predict on the fold's training rows with y <= TAIL_CAP (standardised target).
Tail: the classifier of cluster/w2_task.py's nn branch, unchanged (binary LightGBM on the 18 base columns, 300
rounds, same parameters and seed), mixture pred = clip(p * et + (1 - p) * bulk, 0) with et = off_sched when that
is > TAIL_CAP, else the training tail mean -- so seq is directly comparable with the seven members.

Writes (idempotent: a task whose mixture file exists is skipped; the mixture is written last):
  cache/cv/<campaign>/seq__seed<NN>__m<MM>.npy        full mixture (exp66 format)
  cache/cv/<campaign>/seq__seed<NN>__m<MM>.bulk.npy   the bulk regressor alone
  (with --no-context: seqnc__... -- the same network without the context branch; tail classifier identical, so
   seq - seqnc = (1 - p) * (bulk_seq - bulk_seqnc) isolates what the context adds)

    python cluster/seq_task.py --seed 0 --month 7 --device cuda          # GPU node (--gres=gpu:1)
    python cluster/seq_task.py --seed 0 --month 7 --no-context --device cuda   # the ablation twin seqnc
    python cluster/seq_task.py --seed 0 --month 7 --smoke                # laptop plumbing test -> cv/seq_smoke
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from prc.config import TAIL_CAP, CACHE
from prc.io import peak_rss_gb

REQUIRED = ['cache/v11/feat_training.parquet', 'cache/adsb_lol/features_training.parquet']


def save(path, arr):
    tmp = path.with_name('.' + path.name + '.tmp.npy')
    np.save(tmp, np.asarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def check_inputs(paths):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    miss = [p for p in paths if not os.path.exists(os.path.join(repo, p))]
    if miss:
        sys.exit(f'FATAL: missing inputs {miss} -- rsync them to the cluster first')


def movements(which, adsb_path, vocab=None, keep=None):
    """seq_net movement table for `which` (+ its adsb.lol departure features)."""
    import pandas as pd
    from prc.io import load
    from prc.models import seq_net as S
    full = load(which, columns=S.MV_COLS)
    if keep is not None:
        full = full[keep(full)].reset_index(drop=True)
    adsb = pd.read_parquet(adsb_path, columns=['MVT_ID_mvt'] + S.ADSB_TOK)
    return S.movement_table(full, adsb, vocab=vocab)


def check_alignment(mv, off_sched, label):
    """The movement table's departures must be the frame's rows in the same order: compare off_sched computed from
    the movement table (MVT - SCHED, float32 month-relative seconds) with the frame's own column."""
    from prc.models.seq_net import TIME_KEYS
    dep = np.flatnonzero(mv['is_dep'])
    if len(dep) != len(off_sched):
        sys.exit(f'FATAL [{label}]: {len(dep):,} departures in the movement table vs {len(off_sched):,} frame rows')
    T = mv['T'][dep]
    so = T[:, TIME_KEYS.index('t_mvt')].astype(float) - T[:, TIME_KEYS.index('t_sched')].astype(float)
    ok = np.isfinite(so) & np.isfinite(off_sched)
    bad = np.abs(so[ok] - off_sched[ok]) > 2.0
    if bad.any() or (np.isfinite(so) != np.isfinite(off_sched)).any():
        sys.exit(f'FATAL [{label}]: movement table not row-aligned with the feature frame '
                 f'({int(bad.sum()):,} off_sched mismatches)')
    return dep



def _read_conf(cf, tries=20):
    """Read a campaign _config.json; retry briefly in case a parallel task is replacing it (2026-09-27 race)."""
    import json, time
    for i in range(tries):
        try:
            return json.load(open(cf))
        except json.JSONDecodeError:
            if i == tries - 1:
                raise
            time.sleep(0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='seq')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--month', type=int, required=True)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--K', type=int, default=64, help='neighbours per departure')
    ap.add_argument('--window', type=int, default=5400, help='max |dt| to a neighbour (s)')
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--deice', action='store_true',
                    help='C1 feature set: + the de-icing share columns (w2_features deice group); use --campaign seq_c1 '
                         'and judge against the c1 members (seq_eval.py --members c1)')
    ap.add_argument('--no-context', action='store_true',
                    help="the ablation twin 'seqnc': identical network and rows with the context branch removed "
                         "(a tabular MLP on the same 143 columns); its difference to seq isolates the context")
    ap.add_argument('--smoke', action='store_true',
                    help='laptop plumbing test: 2 airports, 3%% of their training rows, K=16, 1 epoch -> cv/seq_smoke')
    a = ap.parse_args()
    fam = 'seqnc' if a.no_context else 'seq'

    check_inputs(REQUIRED + (['cache/deice_0926/ind_training.parquet'] if a.deice else []))
    if a.deice and a.campaign == 'seq':
        sys.exit('FATAL: --deice must not write into the seq campaign; pass --campaign seq_c1')
    campaign = 'seq_smoke' if a.smoke else a.campaign
    outdir = CACHE / 'cv' / campaign
    outdir.mkdir(parents=True, exist_ok=True)
    # one configuration per campaign folder (as cluster/w2_task.py): a task with other settings stops
    conf = dict(deice=a.deice, K=a.K, window=a.window, d=a.d, layers=a.layers, epochs=a.epochs, batch=a.batch, lr=a.lr)
    cf = outdir / '_config.json'
    if cf.exists():
        diff = {k: (v0, conf[k]) for k, v0 in _read_conf(cf).items() if k in conf and conf[k] != v0}
        if diff:
            sys.exit(f'FATAL: campaign {campaign} was started with other settings (old, new): {diff}')
    else:
        tmp = cf.with_name(f'._config.{os.getpid()}.json')   # atomic: parallel tasks never see a half-written file
        json.dump(conf, open(tmp, 'w'), indent=1)
        os.replace(tmp, cf)
    stem = f'{fam}__seed{a.seed:02d}__m{a.month:02d}'
    out, out_bulk = outdir / f'{stem}.npy', outdir / f'{stem}.bulk.npy'
    if out.exists():
        print(f'{out.name} exists, skipping')
        return
    import lightgbm as lgb
    import torch
    from cluster.w2_features import assemble_w2, columns, W2_CATS
    from prc.features import CATS
    from prc.models import seq_net as S
    dev = S.resolve_device(None if a.device == 'auto' else a.device)      # fail fast without --gres
    print(f'device: {dev}', flush=True)
    K, epochs = (16, 1) if a.smoke else (a.K, a.epochs)

    t0 = time.time()
    X, y, month, groups = assemble_w2(adsb=True, extra=True, **({'deice': True} if a.deice else {}))
    cols, clf_cols = columns(groups)
    ccats = [c for c in CATS if c in clf_cols]
    so_all = X['off_sched'].to_numpy()
    print(f'assembled {X.shape} ({len(cols)} regressor / {len(clf_cols)} classifier columns) '
          f'{time.time() - t0:.0f}s | peak RSS {peak_rss_gb():.2f} GB', flush=True)
    mv, _ = movements('training', CACHE / 'adsb_lol' / 'features_training.parquet')
    tgt_mv = check_alignment(mv, so_all, 'training')
    nb = S.build_context(mv, K=K, window=a.window, log=lambda m: print(m, flush=True))
    if a.smoke:              # 2 small airports only (contexts are per airport-month, so they are unchanged)
        sub = np.isin(X['APT'].astype(str).to_numpy(), ['LSZH', 'EDDM'])
        X = X[sub].reset_index(drop=True)
        y, month, so_all, tgt_mv, nb = y[sub], month[sub], so_all[sub], tgt_mv[sub], nb[sub]
    va, trn = month == a.month, month != a.month
    if a.smoke:
        trn &= np.random.default_rng(0).random(len(y)) < 0.03
    bulk_trn = trn & (y <= TAIL_CAP)
    codes, Z, cat_cols, names = S.target_design(X, cols, W2_CATS, bulk_trn)
    Xc = X[clf_cols].copy()
    del X
    ctx = S.Context(mv, dev)
    del mv
    print(f'design: {len(cat_cols)} cats, {Z.shape[1]} numerics+flags, {ctx.n_tok} token numerics; '
          f'train bulk {bulk_trn.sum():,}, predict {va.sum():,} | {time.time() - t0:.0f}s | '
          f'peak RSS {peak_rss_gb():.2f} GB', flush=True)

    pb = np.clip(S.fit_predict(codes, Z, y, bulk_trn, va, ctx=ctx, tgt_mv=tgt_mv, nb=nb, use_context=not a.no_context,
                               epochs=epochs, batch=a.batch, lr=a.lr, d=a.d, layers=a.layers, seed=a.seed,
                               threads=a.threads, device=str(dev), log=lambda m: print(m, flush=True)), 0, None)
    del ctx, Z, codes, nb
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    # tail classifier exactly as cluster/w2_task.py's nn / w3 branch
    yt = y[trn]
    mc = lgb.train(dict(objective='binary', metric='auc', learning_rate=0.05,
                        num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                        num_threads=a.threads, verbose=-1, seed=a.seed,
                        deterministic=True, force_row_wise=True),
                   lgb.Dataset(Xc.loc[trn, clf_cols], (yt > TAIL_CAP).astype(int),
                               categorical_feature=ccats), num_boost_round=300)
    pt = mc.predict(Xc.loc[va, clf_cols])
    so = so_all[va]
    tail_mean = yt[yt > TAIL_CAP].mean()
    et = np.where(np.isfinite(so) & (so > TAIL_CAP), so, tail_mean)
    pred = np.clip(pt * et + (1 - pt) * pb, 0, None)

    if len(pred) != int(va.sum()) or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(pb)):
        sys.exit(f'FATAL: {len(pred)} predictions for {int(va.sum())} rows, '
                 f'finite={bool(np.all(np.isfinite(pred)))}/{bool(np.all(np.isfinite(pb)))}')
    save(out_bulk, pb)
    save(out, pred)                       # written last: its presence marks the task complete
    yv = y[va]
    k = yv <= TAIL_CAP
    print(f'{out.name}: n={int(va.sum()):,} RMSE {np.sqrt(np.mean((yv - pred) ** 2)):.2f}, bulk rows '
          f'{np.sqrt(np.mean((yv[k] - pb[k]) ** 2)):.2f} ({time.time() - t0:.0f}s) | peak RSS {peak_rss_gb():.2f} GB',
          flush=True)


if __name__ == '__main__':
    main()
