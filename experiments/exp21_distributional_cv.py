"""EXP21 -- the new architectures under the full leave-one-month-out protocol.

Everything here is compared against the cached v3-mixture OOF predictions from
EXP13 (cache/exp13_oof.npy, row 2, 296.22 OOF). That file was produced with the
same folds, the same row order and the same eighteen base features, so it can be
used row-for-row in a paired bootstrap without re-fitting it -- which is the
only reason a 2.7 h architecture can be compared against it inside one session.

ARMS
----
  dist       distributional / binned-classification regression (prc.models.distributional)
  anchored   v3 mixture with its bulk regressor replaced by an anchored-residual
             regressor (prc.models.anchored)
  anchored0  the anchored-residual regressor ALONE, untrimmed, no mixture --
             present so that "did the reparameterisation help" and "did the
             mixture help" cannot be confused for one another
  mixture    the v3 mixture, re-fitted rather than loaded. Only needed if the
             cache is missing; it should reproduce 296.22.

A paired bootstrap is reported for every arm against the cached baseline, and
nothing here should be called an improvement unless its 95% interval clears
zero. The bootstrap sd of an ABSOLUTE OOF RMSE is ~47 s; paired it is ~11-17 s.

  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
    systemd-inhibit --what=idle:sleep:handle-lid-switch \
    /home/thinklex/prc2026-venv/bin/python experiments/exp21_distributional_cv.py \
      --arms anchored0,anchored
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TAIL_CAP, CACHE
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.anchored import fit_predict as anchored_fit_predict
from prc.models.dataset import assemble_base
from prc.models.distributional import DistributionalRegressor, make_edges
from prc.validation import decompose, paired_bootstrap, month_cv_oof, per_month
pd.set_option('display.width', 210)

ap = argparse.ArgumentParser()
ap.add_argument('--arms', default='anchored0,anchored,dist')
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--n-tail', type=int, default=1, help='tail bins for the distributional arm')
ap.add_argument('--leaves', type=int, default=63)
ap.add_argument('--rounds', type=int, default=300)
ap.add_argument('--tail-rep', default='sched_global')
ap.add_argument('--tail-cal', type=float, default=1.0,
                help='scalar on the tail probability mass; 1.0 = no recalibration')
ap.add_argument('--months', default='', help='comma list, for a cheap partial run')
a = ap.parse_args()

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(REG, objective='binary', num_leaves=63, min_data_in_leaf=50)
CLF['metric'] = 'auc'
EDGES = make_edges(core_width=100.0, core_hi=4000.0, n_mid=6, n_tail=a.n_tail)
TAILBIN = EDGES[:-1] >= TAIL_CAP


def _tail_anchor(Xv, yt):
    """E[y | tail, x] the way the v3 mixture defines it: the schedule stamp
    where it is itself in the tail, the training tail mean otherwise."""
    osv = Xv['off_sched'].to_numpy(dtype=float)
    return np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())


def fp_mixture(Xt, yt, Xv):
    k = yt <= TAIL_CAP
    mb = lgb.train(REG, lgb.Dataset(Xt[k], yt[k], categorical_feature=CATS),
                   num_boost_round=350)
    mc = lgb.train(CLF, lgb.Dataset(Xt, (yt > TAIL_CAP).astype(int),
                                    categorical_feature=CATS), num_boost_round=300)
    p = mc.predict(Xv)
    return np.clip(p * _tail_anchor(Xv, yt) + (1 - p) * np.clip(mb.predict(Xv), 0, None),
                   0, None)


def fp_anchored0(Xt, yt, Xv):
    return anchored_fit_predict(REG, Xt, yt, Xv, categorical_feature=CATS)


def fp_anchored(Xt, yt, Xv):
    """Same mixture as v3, bulk component reparameterised. Only the bulk changes,
    so any difference is attributable to the reparameterisation alone."""
    pb = anchored_fit_predict(REG, Xt, yt, Xv, categorical_feature=CATS, trim=TAIL_CAP)
    mc = lgb.train(CLF, lgb.Dataset(Xt, (yt > TAIL_CAP).astype(int),
                                    categorical_feature=CATS), num_boost_round=300)
    p = mc.predict(Xv)
    return np.clip(p * _tail_anchor(Xv, yt) + (1 - p) * pb, 0, None)


def fp_dist(Xt, yt, Xv):
    m = DistributionalRegressor(EDGES, num_boost_round=a.rounds,
                                params=dict(num_leaves=a.leaves, num_threads=a.threads,
                                            learning_rate=0.1))
    m.fit(Xt, yt, categorical_feature=CATS)
    P = m.predict_proba(Xv).astype(np.float64)
    if a.tail_cal != 1.0:
        tm = P[:, TAILBIN].sum(1)
        P[:, TAILBIN] *= a.tail_cal
        P[:, ~TAILBIN] *= (np.maximum(1 - a.tail_cal * tm, 0.0)
                           / np.maximum(1 - tm, 1e-12))[:, None]
    return m.expectation(P.astype(np.float32), tail_rep=a.tail_rep,
                         off_sched=Xv['off_sched'].to_numpy(dtype=float))


ARMS = {'mixture': fp_mixture, 'anchored0': fp_anchored0, 'anchored': fp_anchored,
        'dist': fp_dist}

t0 = time.time()
X, y, month = assemble_base()
months = [int(m) for m in a.months.split(',')] if a.months else None
print(f'{len(y):,} departures, {int((y>TAIL_CAP).sum())} tail rows | '
      f'K={len(EDGES)-1} n_tail={a.n_tail} | {time.time()-t0:.0f}s | '
      f'RSS {peak_rss_gb():.2f} GB', flush=True)

base = None
cached = CACHE / 'exp13_oof.npy'
if cached.exists():
    base = np.load(cached)[2]          # row 2 = C mixture, 296.22 OOF
    d = decompose(y, base)
    print(f'cached v3 mixture baseline: OOF {d["rmse"]:.2f} | bulk {d["rmse_bulk"]:.1f} '
          f'| tail {d["rmse_tail"]:.0f}')

oofs = {}
for name in [s for s in a.arms.split(',') if s]:
    print(f'\n=== {name} ===', flush=True)
    oofs[name] = month_cv_oof(X, y, month, ARMS[name], months=months)
    np.save(CACHE / f'exp21_oof_{name}.npy', oofs[name].astype(np.float32))
    ok = np.isfinite(oofs[name])
    d = decompose(y[ok], oofs[name][ok])
    print(f'  OOF RMSE {d["rmse"]:7.2f} | bulk {d["rmse_bulk"]:6.1f} '
          f'(contrib {d["contrib_bulk"]:6.1f}) | tail {d["rmse_tail"]:8.0f} '
          f'(contrib {d["contrib_tail"]:6.1f}) | n_tail {d["n_tail"]}', flush=True)
    if base is not None:
        r = paired_bootstrap(y[ok], oofs[name][ok], base[ok])
        print(f'  paired vs v3 mixture: {r["diff"]:+8.2f} +- {r["sd"]:5.2f}  '
              f'95% CI [{r["lo"]:+8.1f}, {r["hi"]:+8.1f}]  P(worse) {r["p_a_worse"]:.3f}')
        print('  ' + ('ESTABLISHED IMPROVEMENT' if r['hi'] < 0 else
                      'NOT ESTABLISHED -- interval crosses zero' if r['lo'] < 0 <= r['hi']
                      else 'ESTABLISHED REGRESSION'))
    print(per_month(y[ok], oofs[name][ok], month[ok]).round(1).to_string(index=False))

if len(oofs) > 1:
    print('\npairwise between arms')
    ks = list(oofs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            ok = np.isfinite(oofs[ks[i]]) & np.isfinite(oofs[ks[j]])
            r = paired_bootstrap(y[ok], oofs[ks[i]][ok], oofs[ks[j]][ok])
            print(f'  {ks[i]:10s} vs {ks[j]:10s}: {r["diff"]:+8.2f} +- {r["sd"]:5.2f}  '
                  f'CI [{r["lo"]:+8.1f}, {r["hi"]:+8.1f}]')

print(f'\npeak RSS {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
