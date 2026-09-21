"""EXP22 -- is the distributional model a better TAIL CLASSIFIER than the binary one?

The dev log ranks classifier precision as the single highest-value open problem:
gating P(T|x) changes bulk RMSE by nothing at any threshold, which localises the
damage to a few hundred high-confidence false positives, and the v3 mixture
reaches 82.9% precision in its top 0.01% but recovers only 29 of 149 tail rows.

EXP20 showed the distributional model loses to the mixture end to end. That does
not settle whether its tail mass is a better PROBABILITY than the binary
classifier's, and there is a reason to think it might be: the binary classifier
sees 584 positives and nothing else, whereas the softmax also sees which bin a
3,000 s or 6,000 s row fell into, so near-tail rows carry gradient information
about the tail boundary instead of being lumped in with 900 s rows.

This is a decomposition, not a new model. Sum the distributional model's tail
bins into a single P(T|x), drop it into the v3 mixture in place of the binary
classifier's, and hold everything else fixed. Whatever moves is attributable to
the probability alone.

Reads the probability matrices EXP20 cached, so it costs one reference re-fit
(~2 min) rather than another 45 minutes of softmax.

  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
    /home/thinklex/prc2026-venv/bin/python experiments/exp22_dist_as_classifier.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TAIL_CAP, CACHE
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.dataset import assemble_base
from prc.models.distributional import make_edges
from prc.validation import decompose, paired_bootstrap
pd.set_option('display.width', 210)

HOLDOUT = 7
GRIDS = {'nt10L63': 10, 'nt1L63': 1, 'nt1L127': 1}   # name -> n_tail used by EXP20

t0 = time.time()
X, y, month = assemble_base()
va, tr = month == HOLDOUT, month != HOLDOUT
Xt, yt, Xv, yv = X[tr], y[tr], X[va], y[va]
osv = Xv['off_sched'].to_numpy(dtype=float)
truth = yv > TAIL_CAP

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(REG, objective='binary', num_leaves=63, min_data_in_leaf=50)
CLF['metric'] = 'auc'

k = yt <= TAIL_CAP
bulk = np.clip(lgb.train(REG, lgb.Dataset(Xt[k], yt[k], categorical_feature=CATS),
                         num_boost_round=350).predict(Xv), 0, None)
p_bin = lgb.train(CLF, lgb.Dataset(Xt, (yt > TAIL_CAP).astype(int),
                                   categorical_feature=CATS),
                  num_boost_round=300).predict(Xv)
et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
print(f'references fitted ({time.time()-t0:.0f}s); {int(truth.sum())} tail rows in '
      f'{len(yv):,} | RSS {peak_rss_gb():.2f} GB', flush=True)


def mixture(p):
    return np.clip(p * et + (1 - p) * bulk, 0, None)


def lift(p, name):
    """Precision and recall at the operating points the dev log uses. RMSE is
    not a ranking metric, so a probability can be better ordered and still lose;
    both views are reported."""
    order = np.argsort(-p)
    rows = []
    for frac in (0.0001, 0.001, 0.01):
        n = max(1, int(round(frac * len(p))))
        top = order[:n]
        rows.append(dict(model=name, top=f'{frac*100:g}%', n=n,
                         precision=float(truth[top].mean()),
                         captured=int(truth[top].sum()), of=int(truth.sum())))
    return rows


rows, cmp_rows = lift(p_bin, 'binary clf'), []
probs = {'binary clf': p_bin}
for name, n_tail in GRIDS.items():
    f = CACHE / f'exp20_proba_{name}.npy'
    if not f.exists():
        print(f'  (missing {f.name} -- run exp20 first)')
        continue
    P = np.load(f)
    edges = make_edges(core_width=50.0 if name.endswith('w50') else 100.0,
                       core_hi=4000.0, n_mid=6, n_tail=n_tail)
    tb = edges[:-1] >= TAIL_CAP
    assert P.shape[1] == len(edges) - 1, f'{name}: {P.shape[1]} cols vs {len(edges)-1} bins'
    probs[f'dist {name}'] = P[:, tb].sum(1).astype(np.float64)
    rows += lift(probs[f'dist {name}'], f'dist {name}')

print('\nTAIL PROBABILITY QUALITY (July 2025 fold)')
print(pd.DataFrame(rows).to_string(index=False))

print('\nCALIBRATION AND END-TO-END, same mixture, only P(T|x) swapped')
act = float(truth.mean())
for name, p in probs.items():
    d = decompose(yv, mixture(p))
    r = paired_bootstrap(yv, mixture(p), mixture(p_bin))
    cmp_rows.append(dict(P_source=name, mean_p=p.mean(), ratio=p.mean() / act,
                         rmse=d['rmse'], bulk=d['rmse_bulk'], tail=d['rmse_tail'],
                         vs_binary=r['diff'], lo=r['lo'], hi=r['hi']))
print(pd.DataFrame(cmp_rows).round(4).to_string(index=False))

# An average of the two probabilities is the cheapest way to ask whether they
# are making DIFFERENT mistakes -- ensemble diversity rather than dominance.
for name, p in list(probs.items())[1:]:
    blend = 0.5 * (p + p_bin)
    r = paired_bootstrap(yv, mixture(blend), mixture(p_bin))
    print(f'  50/50 blend {name:16s} vs binary: {r["diff"]:+7.2f} '
          f'CI [{r["lo"]:+7.1f}, {r["hi"]:+7.1f}]  '
          f'RMSE {decompose(yv, mixture(blend))["rmse"]:.2f}')

print(f'\nreference mixture RMSE {decompose(yv, mixture(p_bin))["rmse"]:.2f} | '
      f'peak RSS {peak_rss_gb():.2f} GB | {(time.time()-t0)/60:.1f} min')
