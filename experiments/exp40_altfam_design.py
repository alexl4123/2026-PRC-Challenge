"""EXP40 -- build and cache the design matrix for the exp41-46 family study.

Everything downstream (CatBoost, XGBoost, the embedding network, the blend)
reads this one cached matrix, so it is built once. It follows
cluster/cv_task.assemble() step for step, differing only in loading a slim
column subset for the congestion pass -- which is what keeps the peak inside
the laptop's budget. If the two ever drift, every number in exp41+ becomes
incomparable with the cluster grid that produced the 236.60 baseline, which is
the whole point of the comparison.

The alignment that matters is checked inside altfam._assemble: the congestion
block and the base block come from two separate load() calls, and the build
refuses to proceed unless they return the DEP rows in the same order. The
indirect confirmation is exp41, whose per-month figures reproduce the published
grid to within seed noise (m01 223.6 vs 221.8, m07 301.1 vs 299.8,
m10 229.8 vs 228.1).

Run:
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
    systemd-inhibit --what=idle:sleep:handle-lid-switch \
    /home/thinklex/prc2026-venv/bin/python experiments/exp40_altfam_design.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.models.altfam import BASE, FULL, config_cols, design

t0 = time.time()
X, y, month, groups = design(rebuild='--rebuild' in sys.argv)
print(f'design: {X.shape[0]:,} rows x {X.shape[1]} cols   ({time.time() - t0:.0f}s)')

print('\nfeature groups')
for g, cs in groups.items():
    print(f'  {g:5s} {len(cs):3d}  {cs[:6]}{" ..." if len(cs) > 6 else ""}')
print(f'  r-full  -> {len(config_cols(groups, FULL))} cols')
print(f'  c-base  -> {len(config_cols(groups, BASE))} cols')

print('\ncategorical cardinalities')
for c in X.columns:
    if str(X[c].dtype) == 'category':
        print(f'  {c:24s} {X[c].cat.categories.size:5d} levels, '
              f'{int(X[c].isna().sum()):,} unmapped')

print(f'\ntarget: n={len(y):,}  mean {y.mean():.1f}  median {np.median(y):.1f}  '
      f'max {y.max():.0f}  tail(>{TAIL_CAP}) {int((y > TAIL_CAP).sum())}')
print('rows per month: ' + '  '.join(f'{m}:{int((month == m).sum()):,}' for m in range(1, 13)))

# Memory footprint by dtype, since every later script inherits it.
mb = X.memory_usage(deep=True).sum() / 1e6
print(f'\nX in memory: {mb:.0f} MB   peak RSS {peak_rss_gb():.2f} GB   '
      f'total {time.time() - t0:.0f}s')
