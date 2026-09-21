"""EXP04 -- build the first two submissions.

These are a PAIRED experiment, identical in every respect except one:

  v1  trained with the pathological tail REMOVED   (y <= TAIL_CAP)
  v2  trained with the tail LEFT IN

Purpose is threefold:
  1. validate the output path end to end before the deadline crunch;
  2. calibrate our local capped Jan+Jul-2025 holdout against the official
     Jan+Jul-2026 metric (they are not the same quantity);
  3. settle the trimming question with a measurement instead of an argument.
     Local holdout says trimming is worth ~46 s. If the official scores agree,
     the reasoning in EXP01 is sound and we build on it. If v2 beats v1, the
     2026 labels are tailed after all and we rethink.

Each model: best_iteration is found on the Jan+Jul 2025 holdout, then the model
is REFIT on all twelve months (more data won ~10 RMSE for a 2025 team going from
an 80/20 split to 99/1), with the round count scaled by the extra data.

Run:
  systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
    /home/thinklex/prc2026-venv/bin/python experiments/exp04_submission_v1v2.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc.config import TAIL_CAP, TARGET, ID
from prc.features import build, align_categories, CATS, FEATURES
from prc.features import NEEDED
from prc.io import load, peak_rss_gb
from prc.submit import build as build_submission
from prc.validation import score

t0 = time.time()
tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
rk = load('ranking', phase='DEP', columns=NEEDED + [ID])
print(f'train {tr.shape}  ranking {rk.shape}  ({time.time()-t0:.0f}s)')

Xtr, Xrk = build(tr), build(rk)
y = tr[TARGET].astype(float)
month = tr.MVT_TIME_UTC_mvt.dt.month
Xtr, Xrk = align_categories(Xtr, Xrk)

print('\ncategory coverage of the ranking set by the training vocabulary:')
for c in CATS:
    print(f'  {c:24s} {Xrk[c].notna().mean()*100:6.2f}%  '
          f'({Xrk[c].cat.categories.size} levels)')

val = month.isin([1, 7]).to_numpy()
trn = ~val
yv = y[val].to_numpy()
mv = month[val].to_numpy()

PARAMS = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
              deterministic=True, force_row_wise=True)


def run(tag, cap, version):
    keep_t = trn & ((y <= cap).to_numpy() if cap else np.ones(len(y), bool))
    print(f'\n=== {tag} (version v{version}) ===')
    print(f'  holdout fit on {keep_t.sum():,} rows '
          f'(dropped {int(trn.sum()-keep_t.sum())})')

    ev = yv <= TAIL_CAP
    m = lgb.train(PARAMS, lgb.Dataset(Xtr[keep_t], y[keep_t], categorical_feature=CATS),
                  num_boost_round=3000,
                  valid_sets=[lgb.Dataset(Xtr[val][ev], yv[ev], categorical_feature=CATS)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    s = score(yv, np.clip(m.predict(Xtr[val], num_iteration=m.best_iteration), 0, None),
              month=mv)
    print(f'  holdout capped RMSE = {s["rmse_capped"]:.2f}   '
          f'Jan {s["rmse_m01"]:.2f} / Jul {s["rmse_m07"]:.2f}   '
          f'(best_iter {m.best_iteration})')

    # Refit on all twelve months. Scale rounds by the extra data.
    keep_all = (y <= cap).to_numpy() if cap else np.ones(len(y), bool)
    rounds = int(round(m.best_iteration * keep_all.sum() / max(keep_t.sum(), 1)))
    t = time.time()
    full = lgb.train(PARAMS, lgb.Dataset(Xtr[keep_all], y[keep_all], categorical_feature=CATS),
                     num_boost_round=rounds)
    pred = full.predict(Xrk)
    print(f'  refit on {keep_all.sum():,} rows, {rounds} rounds, {time.time()-t:.0f}s')
    print(f'  ranking predictions: mean {pred.mean():.1f}s  median {np.median(pred):.1f}s  '
          f'p01 {np.percentile(pred,1):.0f}  p99 {np.percentile(pred,99):.0f}')
    path = build_submission(rk[ID].to_numpy(), pred, version=version)
    return s['rmse_capped'], path


r1, p1 = run('TRIMMED  (y <= 7200)', TAIL_CAP, 1)
r2, p2 = run('UNTRIMMED (all rows)', None, 2)

print('\n' + '=' * 78)
print(f'local holdout: v1 trimmed {r1:.2f}   v2 untrimmed {r2:.2f}   '
      f'delta {r2-r1:+.2f}s')
print('If the official scores show the same sign and rough magnitude, the EXP01')
print('reasoning holds. If v2 wins, the 2026 labels are tailed and we rethink.')
print(f'\npeak RSS = {peak_rss_gb():.2f} GB   total {time.time()-t0:.0f}s')
