"""One (seed, month) fit of the entity-embedding network, for the GPU node.

Writes cache/cv/<campaign>/nn__seed<NN>__m<MM>.npy alongside the CPU study's
arms, so `cluster/collect.py` scores the network as one more arm and
paired-bootstraps it against the shipped mixture.

Only the BULK REGRESSOR is a network. The tail classifier and the tail estimate
are held at the configuration the 480-task grid selected, so any difference
measured is attributable to the regressor alone -- the same discipline the CPU
study uses.

  python cluster/nn_task.py --campaign study --seed 0 --month 7 --device cuda
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np

from cluster.cv_task import assemble, CONFIGS
from prc.config import TAIL_CAP, CACHE
from prc.features import CATS
from prc.models.altfam import numeric_design, mixture
from prc.models.embed_nn import fit_predict, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--month', type=int, required=True)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--threads', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    a = ap.parse_args()

    outdir = CACHE / 'cv' / a.campaign
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'nn__seed{a.seed:02d}__m{a.month:02d}.npy'
    if out.exists():
        print(f'{out.name} exists, skipping')
        return

    # Fail loudly and immediately if the job was submitted without --gres, rather
    # than silently training on CPU for hours.
    dev = resolve_device(None if a.device == 'auto' else a.device)
    print(f'device: {dev}', flush=True)

    t0 = time.time()
    X, y, month, groups = assemble()
    cols = [c for g in CONFIGS['full'] for c in groups[g]]
    clf_cols = [c for g in CONFIGS['base'] for c in groups[g]]
    ccats = [c for c in CATS if c in clf_cols]
    va, trn = month == a.month, month != a.month

    # The network serves the bulk only, so it is fitted on bulk rows -- matching
    # how the LightGBM regressor is trained in every other arm.
    bulk_trn = trn & (y <= TAIL_CAP)
    # numeric_design returns SIX values (codes, cards, Z, cat_cols, num_cols,
    # stats) -- its own docstring says four, which is wrong. Take the first
    # three, and pass X[cols] rather than X, matching how exp45 calls it.
    codes, cards, Z = numeric_design(X[cols], cols, fit_mask=bulk_trn)[:3]
    pred_bulk_all = fit_predict(codes, Z, y, bulk_trn, va, epochs=a.epochs,
                                batch=a.batch, seed=a.seed, threads=a.threads,
                                device=str(dev), log=lambda m: print(m, flush=True))

    yt = y[trn]
    mc = lgb.train(dict(objective='binary', metric='auc', learning_rate=0.05,
                        num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                        num_threads=a.threads, verbose=-1, seed=a.seed,
                        deterministic=True, force_row_wise=True),
                   lgb.Dataset(X.loc[trn, clf_cols], (yt > TAIL_CAP).astype(int),
                               categorical_feature=ccats), num_boost_round=300)
    pt = mc.predict(X.loc[va, clf_cols])
    pred = mixture(pt, np.clip(pred_bulk_all, 0, None),
                   X.loc[va, 'off_sched'].to_numpy(), yt[yt > TAIL_CAP].mean())

    np.save(out, np.asarray(pred, dtype=np.float32))
    rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
    print(f'{out.name}: n={va.sum():,} RMSE {rmse:.2f}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
