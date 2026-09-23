"""EXP66 -- does a blend of several model FAMILIES beat v8/v9's LightGBM + network?

WHY
---
The network blend was worth -3.94 s over twelve folds and -6.52 s live, so
combining model families pays here. The cluster study now holds twelve-fold OOF
predictions for up to seven families (mix = LightGBM, nn, perapt, lintree,
routed, catboost, xgboost), three seeds each. Some have different strengths:
lintree's bulk is worse but its tail better; perapt's bulk is slightly better.
This asks whether a weighted combination beats the production structure.

HOW
---
* Each family is seed-averaged over its complete seeds.
* Weights are non-negative least squares, fitted on the screening months
  (3, 6, 9, 12) ONLY, and only on rows OUTSIDE P -- inside P the production
  model uses the tail specialist, which none of these families has, so the
  blend is meant for the bulk branch.
* The comparison is against the production structure on the same rows: the
  LightGBM + network blend with v9's per-route weights (0.65 on rows with an NM
  record, 0.5 without).
* Judged on the 8 months not used for fitting, and on the scored months, with a
  paired bootstrap. Adopt only if the held-out CI is clear of zero.

Every family file is a full mixture (bulk regressor + the same LightGBM tail
classifier), so outside P they differ essentially in their bulk regressor.

    python experiments/exp66_family_blend.py            # uses every complete family found
    python experiments/exp66_family_blend.py --per-route  # separate weights for route D
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import nnls

from prc.io import load, peak_rss_gb
from prc.config import TARGET, TAIL_CAP, CACHE
from prc.validation import paired_bootstrap
from prc.models.routed import route_of

FAMILIES = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'catboost', 'xgboost']
SCREEN, SCORED = [3, 6, 9, 12], [1, 7]
STUDY = CACHE / 'cv' / 'study'


def family_oof(fam, month, n):
    """Seed-average of every seed that has all 12 months; None if no complete seed."""
    seeds = []
    for s in range(10):
        files = [STUDY / f'{fam}__seed{s:02d}__m{m:02d}.npy' for m in range(1, 13)]
        if all(f.exists() for f in files):
            o = np.full(n, np.nan)
            for m, f in zip(range(1, 13), files):
                a = np.load(f)
                if len(a) != (month == m).sum():
                    sys.exit(f'FATAL: {f.name} has {len(a)} rows, month {m} has {(month == m).sum()}')
                o[month == m] = a
            seeds.append(o)
    return (np.mean(seeds, axis=0), len(seeds)) if seeds else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-route', action='store_true', help='separate weights for route D')
    a = ap.parse_args()

    d = load('training', phase='DEP', columns=[TARGET, 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                                               'AOBT_3_flt'])
    y = d[TARGET].astype(float).to_numpy()
    month = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    D = route_of((d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds().to_numpy()) == 'D'
    so = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    P = D & np.isfinite(so) & (so > TAIL_CAP)
    n = len(y)

    members, names = [], []
    for fam in FAMILIES:
        o, k = family_oof(fam, month, n)
        print(f'  {fam:<9} {"complete seeds: " + str(k) if k else "not complete -- skipped"}')
        if o is not None:
            members.append(o); names.append(fam)
    if 'mix' not in names or 'nn' not in names:
        sys.exit('FATAL: need at least mix and nn for the production baseline')
    X = np.column_stack(members)

    # production structure (v9): LightGBM + network, 0.65 with an NM record, 0.5 without
    wv = np.where(D, 0.5, 0.65)
    prod = (1 - wv) * X[:, names.index('mix')] + wv * X[:, names.index('nn')]

    fit = np.isin(month, SCREEN) & ~P
    groups = [('route D', D), ('routes A/B/C', ~D)] if a.per_route else [('all', np.ones(n, bool))]
    blend = np.empty(n); weights = {}
    for gname, g in groups:
        w, _ = nnls(X[fit & g], y[fit & g])
        weights[gname] = dict(zip(names, np.round(w, 4).tolist()))
        blend[g] = X[g] @ w
        print(f'\n  weights ({gname}), fitted on months {SCREEN} outside P, sum {w.sum():.3f}:')
        print('    ' + '  '.join(f'{nm}={wt:.3f}' for nm, wt in zip(names, w)))

    print('\n  blend minus production (LGB+network, v9 weights), rows OUTSIDE P, negative = better:')
    for scope, k in (('8 held-out months', ~np.isin(month, SCREEN) & ~P),
                     ('scored months 1+7', np.isin(month, SCORED) & ~P)):
        b = paired_bootstrap(y[k], blend[k], prod[k], n_boot=500)
        r = lambda p: np.sqrt(np.mean((y[k] - p[k]) ** 2))
        print(f'    {scope:<19} {r(prod):7.2f} -> {r(blend):7.2f}   {b["diff"]:+.3f} +- {b["sd"]:.3f}  '
              f'CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]  P(worse) {100*b["p_a_worse"]:.1f}%')
    json.dump({'members': names, 'weights': weights, 'per_route': a.per_route},
              open(CACHE / 'exp66_weights.json', 'w'), indent=1)
    print(f'\n  saved cache/exp66_weights.json | peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
