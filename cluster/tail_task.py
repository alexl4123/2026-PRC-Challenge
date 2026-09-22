"""One (config, seed, fold) fit of the mixture WITH the tail specialist.

Identical to cluster/cv_task.py in every other respect -- same features, same
hyper-parameters, same output layout -- so that collect.py can score a campaign
that mixes the two and the paired bootstrap between them is clean. The ONLY
difference is that P(y > 7200 | x) is replaced, inside

    P = {AOBT_3_flt missing} AND {off_sched > 7200}   (0.24% of rows, 77% of the tail)

by prc.models.tail_specialist. See that module's docstring for why.

Writes cache/cv/<campaign>/spec-r-<cfg>__c-<clfcfg>__seedNN__mMM.npy -- note the
`spec-` prefix, so a specialist arm and a baseline arm can live in the same
campaign directory without colliding, and collect.py treats them as two arms.

  python cluster/tail_task.py --campaign tailspec --config full --clf-config base \
      --seed 0 --month 7
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np

from prc.config import TARGET, TAIL_CAP, CACHE
from prc.features import CATS
from prc.models.tail_specialist import TailSpecialist, population, apply_to_mixture

from cv_task import assemble, CONFIGS  # noqa: E402  (same directory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--config', default='full', choices=sorted(CONFIGS))
    ap.add_argument('--clf-config', default=None, choices=sorted(CONFIGS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--month', type=int, required=True)
    ap.add_argument('--threads', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--no-specialist', action='store_true',
                    help='write the plain baseline arm instead (for a matched pair)')
    a = ap.parse_args()

    outdir = CACHE / 'cv' / a.campaign
    outdir.mkdir(parents=True, exist_ok=True)
    clf_cfg = a.clf_config or a.config
    prefix = '' if a.no_specialist else 'spec-'
    out = outdir / f'{prefix}r-{a.config}__c-{clf_cfg}__seed{a.seed:02d}__m{a.month:02d}.npy'
    if out.exists():
        print(f'{out.name} exists, skipping')
        return

    t0 = time.time()
    X, y, month, groups = assemble()
    cols = [c for g in CONFIGS[a.config] for c in groups[g]]
    clf_cols = [c for g in CONFIGS[clf_cfg] for c in groups[g]]
    cats = [c for c in CATS if c in cols]
    clf_cats = [c for c in CATS if c in clf_cols]

    for need in ('off_sched', 'log_off_sched', 'aobt_missing', 'APT', 'PIER'):
        if need not in X.columns:
            sys.exit(f'FATAL: column {need!r} missing from the feature matrix; the '
                     'specialist cannot be built. Check prc/features.py.')

    va = month == a.month
    tr_ = ~va
    reg = dict(objective='regression', metric='rmse', learning_rate=0.05,
               num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               num_threads=a.threads, verbose=-1, seed=a.seed,
               bagging_seed=a.seed + 1, feature_fraction_seed=a.seed + 2,
               deterministic=True, force_row_wise=True)
    clf = dict(reg, objective='binary', num_leaves=63, min_data_in_leaf=50)
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

    os_ = X['off_sched'].to_numpy()
    miss = X['aobt_missing'].to_numpy().astype(bool)
    apt = X['APT'].astype(str).to_numpy()

    if not a.no_specialist:
        spec = TailSpecialist(threads=a.threads).fit(
            X.loc[tr_], y[tr_], apt[tr_], os_[tr_], miss[tr_])
        p_spec, in_P = spec.predict(X.loc[va], apt[va], os_[va], miss[va])
        n_before = int(in_P.sum())
        pt = apply_to_mixture(pt, p_spec, in_P)
        print(f'  specialist applied to {n_before:,} of {int(va.sum()):,} validation rows')

    osv = os_[va]
    et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
    pred = np.clip(pt * et + (1 - pt) * pb, 0, None)

    np.save(out, pred.astype(np.float32))
    rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
    print(f'{out.name}: n={va.sum():,} RMSE {rmse:.2f}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
