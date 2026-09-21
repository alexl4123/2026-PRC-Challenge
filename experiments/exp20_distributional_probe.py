"""EXP20 -- distributional (binned-classification) regression, single-fold probe.

WHAT IS BEING TESTED
--------------------
A structurally different estimator of E[y|x]: discretise y into K bins, fit a
LightGBM softmax over the bins, and integrate the predicted distribution. See
prc/models/distributional.py for the argument; in one line, cross-entropy gives
every row a gradient bounded by 1 regardless of its label, so one model can fit
the bulk and the tail without the trimmed/untrimmed trade-off that has shaped
every model in this project so far.

WHY A SINGLE FOLD FIRST
-----------------------
A K=56 softmax grows 56 trees per iteration, so a 12-fold LOMO run is ~2.7 h on
the laptop. July is held out here because it is the fold that matters: it is the
hardest month (bulk 306.9 vs March's 208.7), it carries 120 of the 584 tail rows,
and July 2025 has more rows than January, so it dominates the official score.
A configuration that cannot compete on July will not be rescued by the other
eleven months. Config selection happens here; EXP21 runs the full protocol.

WHAT A 500k-ROW PILOT ALREADY SHOWED (scratchpad, not a headline number)
-----------------------------------------------------------------------
Total probability mass placed on the tail bins was 0.62x the actual tail rate,
so the integrated expectation systematically under-shoots exactly the rows that
dominate RMSE. Leading hypothesis: ten geometric tail bins split 584 tail rows
into ten classes of ~53, and no class has enough signal left. The binary
classifier in the v3 mixture pools all of them into one positive class and
reaches 82.9% precision in its top 0.01%. Hence the n_tail ablation below:
n_tail=1 makes the model a strict generalisation of the mixture (one pooled tail
class, K-1 bulk classes) and n_tail=10 spends resolution on tail magnitude.

Runtime ~70 min, peak ~3 GB. Run detached:
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
    systemd-inhibit --what=idle:sleep:handle-lid-switch \
    /home/thinklex/prc2026-venv/bin/python experiments/exp20_distributional_probe.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TAIL_CAP, CACHE
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.dataset import assemble_base
from prc.models.distributional import DistributionalRegressor, make_edges
from prc.validation import decompose, rmse
pd.set_option('display.width', 210)

HOLDOUT = 7
t0 = time.time()
X, y, month = assemble_base()
va, tr = month == HOLDOUT, month != HOLDOUT
Xt, yt, Xv, yv = X[tr], y[tr], X[va], y[va]
osv = Xv['off_sched'].to_numpy()
print(f'train {tr.sum():,} ({int((yt>TAIL_CAP).sum())} tail) -> val {va.sum():,} '
      f'({int((yv>TAIL_CAP).sum())} tail) | {time.time()-t0:.0f}s | '
      f'RSS {peak_rss_gb():.2f} GB', flush=True)

# --- references on the identical fold -------------------------------------
REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(REG, objective='binary', num_leaves=63, min_data_in_leaf=50)
CLF['metric'] = 'auc'

ref = {}
k = yt <= TAIL_CAP
mb = lgb.train(REG, lgb.Dataset(Xt[k], yt[k], categorical_feature=CATS), num_boost_round=350)
ref['L2 trimmed'] = np.clip(mb.predict(Xv), 0, None)
mu = lgb.train(REG, lgb.Dataset(Xt, yt, categorical_feature=CATS), num_boost_round=350)
ref['L2 untrimmed'] = np.clip(mu.predict(Xv), 0, None)
mc = lgb.train(CLF, lgb.Dataset(Xt, (yt > TAIL_CAP).astype(int), categorical_feature=CATS),
               num_boost_round=300)
p_bin = mc.predict(Xv)
et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
ref['v3 mixture'] = np.clip(p_bin * et + (1 - p_bin) * ref['L2 trimmed'], 0, None)
print(f'references fitted ({time.time()-t0:.0f}s)', flush=True)

act = float((yv > TAIL_CAP).mean())
print(f'\nbinary classifier tail mass: mean {p_bin.mean():.6f} vs actual {act:.6f} '
      f'(ratio {p_bin.mean()/act:.2f})')
for nm, p in ref.items():
    d = decompose(yv, p)
    print(f'  {nm:14s} RMSE {d["rmse"]:8.2f}  bulk {d["rmse_bulk"]:6.1f}  '
          f'tail {d["rmse_tail"]:8.0f}')

# --- the architecture under test ------------------------------------------
# n_tail is the ablation of interest: 1 pools every tail row into a single class
# (a strict generalisation of the v3 mixture), 10 spends resolution on magnitude.
# The leaves=127 row asks whether the bulk gap is capacity rather than structure.
CONFIGS = [
    ('nt10 L63 ', dict(n_tail=10), dict(num_leaves=63), 300),
    ('nt1  L63 ', dict(n_tail=1), dict(num_leaves=63), 300),
    ('nt1  L127', dict(n_tail=1), dict(num_leaves=127), 300),
]
rows = []
for name, gkw, pkw, rounds in CONFIGS:
    edges = make_edges(core_width=100.0, core_hi=4000.0, n_mid=6, **gkw)
    m = DistributionalRegressor(edges, params=dict(pkw, num_threads=6, learning_rate=0.1),
                                num_boost_round=rounds)
    t = time.time()
    m.fit(Xt, yt, categorical_feature=CATS)
    P = m.predict_proba(Xv)
    np.save(CACHE / f'exp20_proba_{name.strip().replace(" ", "")}.npy', P)
    tb = edges[:-1] >= TAIL_CAP
    tm = P[:, tb].sum(1).astype(np.float64)
    print(f'\n=== {name} K={m.n_bins} rounds={rounds} ({time.time()-t:.0f}s) ===', flush=True)
    print(f'  tail mass mean {tm.mean():.6f} vs actual {act:.6f} '
          f'(ratio {tm.mean()/act:.2f}) | >1% on {int((tm>0.01).sum()):,} rows')

    # Bulk conditional mean, renormalised over bulk bins. Separates "does the
    # model know the bulk" from "does tail mass leak into bulk predictions".
    bulk_only = (P[:, ~tb].astype(np.float64) @ m.centroids_[~tb]) / np.maximum(1 - tm, 1e-12)
    print(f'  bulk-only conditional mean: bulk RMSE '
          f'{decompose(yv, np.clip(bulk_only,0,None))["rmse_bulk"]:6.1f}')

    for rep in ('centroid', 'sched_global', 'sched_inbin'):
        # Tail-mass recalibration. If the softmax under-states rare classes the
        # fix is a scalar on the tail mass, not a different architecture; a scan
        # says whether that is all that is wrong.
        for f in (1.0, 1.5, 2.0):
            Q = P.astype(np.float64).copy()
            Q[:, tb] *= f
            Q[:, ~tb] *= np.maximum(1 - f * tm, 0.0)[:, None] / np.maximum(1 - tm, 1e-12)[:, None]
            p = m.expectation(Q.astype(np.float32), tail_rep=rep, off_sched=osv)
            d = decompose(yv, p)
            rows.append(dict(config=name, rep=rep, cal=f, rmse=d['rmse'],
                             bulk=d['rmse_bulk'], tail=d['rmse_tail']))
            print(f'  {rep:13s} cal x{f:.2f}  RMSE {d["rmse"]:8.2f}  '
                  f'bulk {d["rmse_bulk"]:6.1f}  tail {d["rmse_tail"]:8.0f}', flush=True)
    del m, P

print('\n' + '=' * 90)
res = pd.DataFrame(rows).sort_values('rmse')
print(res.round(2).to_string(index=False))
print(f'\nbest distributional {res.rmse.min():.2f} vs v3 mixture '
      f'{rmse(yv, ref["v3 mixture"]):.2f} on this fold')
print(f'peak RSS {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
