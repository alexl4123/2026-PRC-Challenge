"""One unit of cluster work for the EXP20/EXP21 architectures.

Same contract as cv_task.py -- one (config, seed, fold) fit, written to its own
file so tasks never talk to each other and a failed array can be resubmitted
without redoing finished work -- with two differences that the distributional
model forces:

1. The array index is decoded HERE, not in bash. A campaign is a list of named
   configs and a seed range, and `--index` maps onto (config, seed, month).
   cv_array.sbatch does this arithmetic in the shell, which works when the grid
   is a clean rectangle of 4 configs; it stops being readable once configs have
   different shapes. `--size` prints the array bound so the sbatch line and the
   config list cannot drift apart.

2. The output is an .npz holding SEVERAL prediction vectors, not one. Fitting
   the softmax is the expensive part; turning a predicted distribution into a
   conditional mean is free, and there are nine ways to do it worth comparing
   (three tail representatives x three recalibration factors). Refitting the
   model nine times to compare nine integrations would be pure waste.

  python cluster/dist_task.py --campaign dist --size
  python cluster/dist_task.py --campaign dist --index 17
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np

from prc.config import TAIL_CAP, CACHE
from prc.features import CATS
from prc.models.anchored import fit_predict as anchored_fit_predict
from prc.models.dataset import assemble_base
from prc.models.distributional import DistributionalRegressor, make_edges

MONTHS = list(range(1, 13))

# Integration variants evaluated from a single fitted distribution.
TAIL_REPS = ('centroid', 'sched_global', 'sched_inbin')
TAIL_CALS = (1.0, 1.5, 2.0)
VARIANTS = [f'{r}@{c:g}' for r, c in itertools.product(TAIL_REPS, TAIL_CALS)]

CONFIGS = {
    # --- distributional: bins x capacity ---------------------------------
    # n_tail is the ablation that matters. 1 pools every tail row into one
    # class; 10 spends resolution on tail magnitude and splits 584 rows ten
    # ways. core_width sets the bulk discretisation floor (100 s -> 28.9 s).
    'dist_nt1_l63':    dict(arch='dist', n_tail=1, leaves=63, rounds=300, lr=0.1),
    'dist_nt1_l127':   dict(arch='dist', n_tail=1, leaves=127, rounds=300, lr=0.1),
    'dist_nt3_l63':    dict(arch='dist', n_tail=3, leaves=63, rounds=300, lr=0.1),
    'dist_nt10_l63':   dict(arch='dist', n_tail=10, leaves=63, rounds=300, lr=0.1),
    'dist_nt1_lr05':   dict(arch='dist', n_tail=1, leaves=63, rounds=700, lr=0.05),
    'dist_nt1_w50':    dict(arch='dist', n_tail=1, leaves=63, rounds=300, lr=0.1,
                            core_width=50.0),
    # --- reparameterised target ------------------------------------------
    'anchored0':       dict(arch='anchored0'),
    'anchored':        dict(arch='anchored'),
    # --- the incumbent, re-fitted so the cluster has its own baseline -----
    'mixture':         dict(arch='mixture'),
}

CAMPAIGNS = {
    'dist':  (['dist_nt1_l63', 'dist_nt1_l127', 'dist_nt3_l63', 'dist_nt10_l63',
               'dist_nt1_lr05', 'dist_nt1_w50'], [0]),
    'arch':  (['mixture', 'anchored0', 'anchored'], [0]),
    # Seed averaging was worth 3.2% to the 2024 winner. Only run this once a
    # config has earned it on the `dist` campaign.
    'seeds': (['dist_nt1_l63'], list(range(10))),
}


def decode(campaign, index):
    cfgs, seeds = CAMPAIGNS[campaign]
    n = len(cfgs) * len(seeds) * len(MONTHS)
    if not 0 <= index < n:
        raise SystemExit(f'index {index} outside 0..{n-1} for campaign {campaign}')
    m = MONTHS[index % len(MONTHS)]
    q = index // len(MONTHS)
    return cfgs[q // len(seeds)], seeds[q % len(seeds)], m


def run(cfg_name, seed, month_held, threads):
    spec = CONFIGS[cfg_name]
    X, y, month = assemble_base()
    va = month == month_held
    tr = ~va
    Xt, yt, Xv = X[tr], y[tr], X[va]

    reg = dict(objective='regression', metric='rmse', learning_rate=0.05,
               num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               num_threads=threads, verbose=-1, seed=seed, bagging_seed=seed + 1,
               feature_fraction_seed=seed + 2, deterministic=True, force_row_wise=True)
    clf = dict(reg, objective='binary', num_leaves=63, min_data_in_leaf=50)
    clf['metric'] = 'auc'

    osv = Xv['off_sched'].to_numpy(dtype=float)
    tail_anchor = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv,
                           yt[yt > TAIL_CAP].mean())
    arch = spec['arch']

    if arch == 'dist':
        edges = make_edges(core_width=spec.get('core_width', 100.0), core_hi=4000.0,
                           n_mid=6, n_tail=spec['n_tail'])
        m = DistributionalRegressor(
            edges, num_boost_round=spec['rounds'],
            params=dict(num_leaves=spec['leaves'], learning_rate=spec['lr'],
                        num_threads=threads, seed=seed, bagging_seed=seed + 1,
                        feature_fraction_seed=seed + 2))
        m.fit(Xt, yt, categorical_feature=CATS)
        P = m.predict_proba(Xv).astype(np.float64)
        tb = edges[:-1] >= TAIL_CAP
        tm = P[:, tb].sum(1)
        preds, names = [], []
        for rep, cal in itertools.product(TAIL_REPS, TAIL_CALS):
            Q = P.copy()
            if cal != 1.0:
                Q[:, tb] *= cal
                Q[:, ~tb] *= (np.maximum(1 - cal * tm, 0.0)
                              / np.maximum(1 - tm, 1e-12))[:, None]
            preds.append(m.expectation(Q.astype(np.float32), tail_rep=rep, off_sched=osv))
            names.append(f'{rep}@{cal:g}')
        extra = dict(tail_mass_mean=float(tm.mean()),
                     tail_rate=float((y[va] > TAIL_CAP).mean()))
        return np.asarray(preds, np.float32), names, extra

    if arch == 'anchored0':
        p = anchored_fit_predict(reg, Xt, yt, Xv, categorical_feature=CATS)
        return p[None].astype(np.float32), ['anchored0'], {}

    k = yt <= TAIL_CAP
    if arch == 'anchored':
        pb = anchored_fit_predict(reg, Xt, yt, Xv, categorical_feature=CATS, trim=TAIL_CAP)
    else:
        mb = lgb.train(reg, lgb.Dataset(Xt[k], yt[k], categorical_feature=CATS),
                       num_boost_round=350)
        pb = np.clip(mb.predict(Xv), 0, None)
    mc = lgb.train(clf, lgb.Dataset(Xt, (yt > TAIL_CAP).astype(int),
                                    categorical_feature=CATS), num_boost_round=300)
    pt = mc.predict(Xv)
    p = np.clip(pt * tail_anchor + (1 - pt) * pb, 0, None)
    return p[None].astype(np.float32), [arch], dict(tail_mass_mean=float(pt.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, choices=sorted(CAMPAIGNS))
    ap.add_argument('--index', type=int)
    ap.add_argument('--size', action='store_true', help='print the array bound and exit')
    ap.add_argument('--threads', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    a = ap.parse_args()

    cfgs, seeds = CAMPAIGNS[a.campaign]
    if a.size:
        n = len(cfgs) * len(seeds) * len(MONTHS)
        print(f'campaign {a.campaign}: {len(cfgs)} configs x {len(seeds)} seeds x '
              f'{len(MONTHS)} months = {n} tasks  ->  --array=0-{n-1}')
        return
    if a.index is None:
        raise SystemExit('--index is required unless --size is given')

    cfg, seed, m = decode(a.campaign, a.index)
    outdir = CACHE / 'cv' / a.campaign
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'{cfg}__seed{seed:02d}__m{m:02d}.npz'
    if out.exists():
        print(f'{out.name} exists, skipping')
        return

    t0 = time.time()
    preds, names, extra = run(cfg, seed, m, a.threads)
    np.savez_compressed(out, pred=preds, variants=np.array(names), month=m)
    # Scoring here needs the labels, which `run` already dropped; recompute the
    # one number a human reads in the array log without keeping y around.
    from prc.config import TARGET
    from prc.io import load
    y = load('training', phase='DEP', columns=[TARGET, 'MVT_TIME_UTC_mvt'])
    yv = y[TARGET].astype(float).to_numpy()[
        y.MVT_TIME_UTC_mvt.dt.month.to_numpy() == m]
    best = min(float(np.sqrt(np.mean((yv - p) ** 2))) for p in preds)
    print(f'{out.name}: n={len(yv):,} variants={len(names)} bestRMSE {best:.2f} '
          f'{extra} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
