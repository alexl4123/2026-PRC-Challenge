"""EXP03 -- the baseline ladder, cheapest model first.

The point is not to win with these. It is that a LightGBM number is
uninterpretable without knowing what a constant, a group mean and a linear model
already achieve. Everything is measured under one protocol:

  train  = 2025 excluding January and July
  holdout= January + July 2025          (matches the scored months' seasonality)
  metric = RMSE on rows with y <= TAIL_CAP, reported for Jan and Jul separately

Run:
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
    /home/thinklex/prc2026-venv/bin/python experiments/exp03_baseline_ladder.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from prc.config import TAIL_CAP
from prc.io import load, peak_rss_gb
from prc.validation import oracle_floor, rmse, score

COLS = ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt',
        'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'TAXITIME_SEC_mvt',
        'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt']

t0 = time.time()
d = load('training', phase='DEP', columns=COLS)
d['month'] = d.MVT_TIME_UTC_mvt.dt.month

# 'Pier' = the leading letter-or-digit block of the stand id. At most of these
# airports that is the terminal/pier, and it generalises to rare stands.
d['PIER'] = (d.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+|\d)', expand=False)
             .fillna('NA'))

# Offsets available at scoring time (MVT_TIME is given for ranking departures).
for src, name in [('AOBT_3_flt', 'off_aobt3'), ('SCHED_TIME_UTC_mvt', 'off_sched'),
                  ('EOBT_1_flt', 'off_eobt')]:
    d[name] = (d.MVT_TIME_UTC_mvt - d[src]).dt.total_seconds()
d['hour'] = d.MVT_TIME_UTC_mvt.dt.hour
d['dow'] = d.MVT_TIME_UTC_mvt.dt.dayofweek

val = d.month.isin([1, 7])
trn = ~val
y = d.TAXITIME_SEC_mvt.astype(float)
yt, yv = y[trn], y[val]
mv = d.month[val].values
print(f'train {trn.sum():,}   holdout {val.sum():,}   load {time.time()-t0:.0f}s')

# Training rows used for fitting: the pathological tail is excluded (EXP01).
fit = trn & (y <= TAIL_CAP)
print(f'rows used for fitting: {fit.sum():,}  (dropped {int(trn.sum()-fit.sum())} tail rows)')
print(f'\nholdout oracle floor @ {TAIL_CAP}s = {oracle_floor(yv, TAIL_CAP):.1f}s '
      f'(uncapped RMSE can never go below this)\n')

results = []


def report(name, pred):
    s = score(yv, pred, month=mv)
    results.append(dict(model=name, **s))
    print(f'{name:38s} capped={s["rmse_capped"]:7.2f}  Jan={s.get("rmse_m01",float("nan")):7.2f} '
          f' Jul={s.get("rmse_m07",float("nan")):7.2f}   (uncapped {s["rmse_uncapped"]:7.1f})')


print('--- tier 0: constants ---')
report('global mean', np.full(val.sum(), y[fit].mean()))
report('global median', np.full(val.sum(), y[fit].median()))

print('\n--- tier 1: hierarchical group means (smoothed, with fallback) ---')


def group_mean(keys, k=20.0):
    """Mean per key group, shrunk toward the global mean by k pseudo-counts,
    then applied to the holdout with a global fallback for unseen groups."""
    gm = y[fit].mean()
    g = pd.DataFrame({'k': list(zip(*[d.loc[fit, c].astype(str) for c in keys])), 'y': y[fit]})
    agg = g.groupby('k').y.agg(['sum', 'count'])
    sm = (agg['sum'] + k * gm) / (agg['count'] + k)
    vk = pd.Series(list(zip(*[d.loc[val, c].astype(str) for c in keys])), index=d.index[val])
    return vk.map(sm).fillna(gm).values


for keys in [['APT'], ['APT', 'RUNWAY_mvt'], ['APT', 'PIER'], ['APT', 'RUNWAY_mvt', 'PIER'],
             ['APT', 'RUNWAY_mvt', 'STAND_mvt'],
             ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_OPERATOR_flt']]:
    report('mean by ' + '+'.join(c.replace('_mvt', '').replace('_flt', '') for c in keys),
           group_mean(keys))

print('\n--- tier 1b: the free NM estimate ---')
gm = y[fit].mean()
report('MVT_TIME - AOBT_3 (raw)', d.loc[val, 'off_aobt3'].fillna(gm).values)

print('\n--- tier 2: ridge on one-hot categoricals + numerics ---')
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATS = ['APT', 'RUNWAY_mvt', 'PIER', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
        'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt']
NUMS = ['off_aobt3', 'off_sched', 'off_eobt', 'hour', 'dow']

enc = OneHotEncoder(handle_unknown='ignore', min_frequency=20, dtype=np.float32)
Xc_t = enc.fit_transform(d.loc[fit, CATS].astype(str))
Xc_v = enc.transform(d.loc[val, CATS].astype(str))
num_med = d.loc[fit, NUMS].median()
sc = StandardScaler()
Xn_t = sc.fit_transform(d.loc[fit, NUMS].fillna(num_med)).astype(np.float32)
Xn_v = sc.transform(d.loc[val, NUMS].fillna(num_med)).astype(np.float32)
Xt = sparse.hstack([Xc_t, sparse.csr_matrix(Xn_t)]).tocsr()
Xv = sparse.hstack([Xc_v, sparse.csr_matrix(Xn_v)]).tocsr()
print(f'  design matrix {Xt.shape}, {Xt.nnz/1e6:.1f}M nonzeros, peakRSS={peak_rss_gb():.2f} GB')
for alpha in [1.0, 10.0]:
    r = Ridge(alpha=alpha, solver='sparse_cg', max_iter=300)
    r.fit(Xt, y[fit])
    report(f'ridge one-hot (alpha={alpha:g})', np.clip(r.predict(Xv), 0, None))
del Xt, Xv, Xc_t, Xc_v

print('\n--- tier 3: LightGBM ---')
import lightgbm as lgb

LGB_CATS = ['APT', 'RUNWAY_mvt', 'PIER', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
            'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt']
FEATS = LGB_CATS + NUMS
X = d[FEATS].copy()
for c in LGB_CATS:
    X[c] = X[c].astype('category')
params = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
              deterministic=True, force_row_wise=True)
t = time.time()
m = lgb.train(params, lgb.Dataset(X[fit], y[fit], categorical_feature=LGB_CATS),
              num_boost_round=2000,
              valid_sets=[lgb.Dataset(X[val][yv <= TAIL_CAP], yv[yv <= TAIL_CAP],
                                      categorical_feature=LGB_CATS)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
report(f'LightGBM ({m.best_iteration} trees, {time.time()-t:.0f}s)',
       np.clip(m.predict(X[val], num_iteration=m.best_iteration), 0, None))

print('\n' + '=' * 96)
print(pd.DataFrame(results)[['model', 'rmse_capped', 'rmse_m01', 'rmse_m07', 'rmse_uncapped']]
      .round(2).to_string(index=False))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB   total {time.time()-t0:.0f}s')
