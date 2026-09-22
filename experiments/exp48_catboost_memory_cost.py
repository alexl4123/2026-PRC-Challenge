"""EXP48 -- did the memory-driven CatBoost settings cost accuracy, or is the
reversal about row count?

THE PUZZLE. exp43 sized CatBoost on month 4 at 600k training rows and it beat
LightGBM clearly: 204.13 against 206.89. At the full 1.9M rows on month 7 it
came back at 269.40 against LightGBM's 268.50 -- WORSE. Two changes happened
between those runs and they are confounded:

  (a) rows      600k -> 1.9M
  (b) settings  lr 0.05 x 3000 iters, border_count 128, unlimited CTR leaves
                ->  lr 0.08 x 2000 iters, border_count 96, ctr_leaf_count_limit 1e6
      every one of (b) forced by being OOM-killed twice inside the memory cap.

This isolates (b) by running the FULL-RUN SETTINGS at the PROBE'S row count and
month, so the only difference from the 204.13 measurement is the settings.

  result ~= 204   -> the memory settings are innocent; CatBoost's advantage
                     simply does not survive going from 600k to 1.9M rows, which
                     is what you would expect of ordered target statistics
                     (they help most when rows per category are scarce).
  result >> 204   -> the memory settings did the damage, and CatBoost deserves a
                     rerun on a machine that can hold the unconstrained model --
                     i.e. the cluster, not this laptop.

Runtime ~10 min, peak ~4 GB.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from prc.config import TAIL_CAP
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.altfam import FULL, config_cols, design
from prc.validation import rmse

THREADS, SEED, N_SUB, VAL_MONTH = 3, 0, 600_000, 4
PROBE_RESULT = 204.13          # exp43, same rows, same month, same seed

t0 = time.time()
X, y, month, groups = design()
cols = config_cols(groups, FULL)
cats = [c for c in CATS if c in cols]
num_cols = [c for c in cols if c not in cats]

va = month == VAL_MONTH
tr = (~va) & (y <= TAIL_CAP)
rng = np.random.default_rng(SEED)
idx_tr = rng.choice(np.flatnonzero(tr), N_SUB, replace=False)
idx_va = np.flatnonzero(va)
Xtr, Xva = X.iloc[idx_tr][cols], X.iloc[idx_va][cols]
ytr, yva = y[idx_tr], y[idx_va]
bulk = yva <= TAIL_CAP
del X


def to_pool(Xf, yf=None):
    d = {c: Xf[c].cat.codes.to_numpy().astype(np.int32) + 1 for c in cats}
    for c in num_cols:
        d[c] = Xf[c].to_numpy(dtype=np.float32, na_value=np.nan)
    return Pool(pd.DataFrame(d, columns=cats + num_cols), yf,
                cat_features=list(range(len(cats))))


ptr, pva = to_pool(Xtr, ytr), to_pool(Xva)
print(f'train {N_SUB:,}  val {len(idx_va):,} (month {VAL_MONTH})   '
      f'({time.time() - t0:.0f}s)', flush=True)
print(f'exp43 reference on exactly these rows: {PROBE_RESULT:.2f}\n', flush=True)

ARMS = [
    dict(tag='full-run settings', learning_rate=0.08, iterations=2000,
         border_count=96, ctr_leaf_count_limit=1_000_000),
    dict(tag='probe settings', learning_rate=0.05, iterations=3000,
         border_count=128, ctr_leaf_count_limit=None),
]
for arm in ARMS:
    tag = arm.pop('tag')
    ts = time.time()
    m = CatBoostRegressor(loss_function='RMSE', depth=8, max_ctr_complexity=2,
                          one_hot_max_size=16, bootstrap_type='Bernoulli',
                          subsample=0.8, rsm=0.8, l2_leaf_reg=3.0,
                          random_seed=SEED, thread_count=THREADS,
                          used_ram_limit='3gb', allow_writing_files=False,
                          verbose=0, **arm)
    m.fit(ptr)
    r = rmse(yva[bulk], np.clip(m.predict(pva), 0, None)[bulk])
    print(f'{tag:20s} bulk RMSE {r:7.2f}   vs probe {r - PROBE_RESULT:+6.2f}   '
          f'({time.time() - ts:.0f}s)  RSS {peak_rss_gb():.2f} GB', flush=True)
    del m

print(f'\npeak RSS {peak_rss_gb():.2f} GB   total {time.time() - t0:.0f}s')
