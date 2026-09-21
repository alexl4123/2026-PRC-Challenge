"""EXP05 -- was the 'oracle floor' argument wrong?

v1 (trimmed) scored 543.67 officially against a local capped holdout of 257.57.
The submission was accepted with all 344,841 pairs used, so this is not a format
bug: the 2026 test labels DO carry a heavy tail.

That falsifies the EXP01 conclusion. Where the reasoning broke:

  `oracle_floor` assumed the tail values are UNPREDICTABLE beyond their group
  mean. That made it look like a hard lower bound on RMSE. It is not. If the
  tail values can be predicted from features, a model can go far below it.

And they partly can: for rows with y > 7200, y ~= (MVT_TIME - SCHED_TIME) in
~81% of cases, and off_sched is available at scoring time. So the leader's
239.13 is not evidence of a clean test set -- it is evidence that they PREDICT
the tail instead of ignoring it.

This experiment measures, on the Jan+Jul 2025 holdout evaluated UNCAPPED (the
metric that actually matches the leaderboard):
  A  trimmed training            -- what we just submitted
  B  untrimmed training          -- v2
  C  untrimmed + tail features   -- the hypothesis
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET
from prc.features import build, align_categories, CATS, NEEDED
from prc.io import load, peak_rss_gb
from prc.validation import rmse

tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
X = build(tr)
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()

# --- tail-oriented features (all computable on the ranking set) ---
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
X['sched_to_aobt'] = (tr.AOBT_3_flt - tr.SCHED_TIME_UTC_mvt).dt.total_seconds()
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
TAILF = ['aobt_missing', 'lobt_missing', 'off_lobt', 'sched_to_aobt', 'log_off_sched']

val = np.isin(month, [1, 7]); trn = ~val
yv = y[val]
Xa, = align_categories(X)
base = [c for c in X.columns if c not in TAILF]

P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
         min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=1.0, num_threads=8, verbose=-1, seed=0, deterministic=True,
         force_row_wise=True)

def run(tag, cols, cap):
    sel = trn & ((y <= cap) if cap else True)
    cats = [c for c in CATS if c in cols]
    t = time.time()
    m = lgb.train(P, lgb.Dataset(X.loc[sel, cols], y[sel], categorical_feature=cats),
                  num_boost_round=3000,
                  valid_sets=[lgb.Dataset(X.loc[val, cols], yv, categorical_feature=cats)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    p = np.clip(m.predict(X.loc[val, cols], num_iteration=m.best_iteration), 0, None)
    tail = yv > 7200
    print(f'{tag:44s} UNCAPPED={rmse(yv,p):7.2f}  capped={rmse(yv[~tail],p[~tail]):6.2f}  '
          f'tailRMSE={rmse(yv[tail],p[tail]):8.0f}  maxpred={p.max():6.0f}  it={m.best_iteration}  {time.time()-t:.0f}s')
    return rmse(yv, p)

print(f'holdout: {val.sum():,} rows, {int((yv>7200).sum())} of them > 7200s\n')
run('A  trimmed @7200, base feats  (= our v1)', base, 7200)
run('B  untrimmed,     base feats  (= our v2)', base, None)
run('C  untrimmed,     + tail feats', list(X.columns), None)
print(f'\npeak RSS = {peak_rss_gb():.2f} GB')
