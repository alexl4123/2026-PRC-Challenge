"""EXP70 -- v12 = v10 + the reviewed feature stack, arm (c), outside P.

The stack (experiments/v11/, built 2026-09-24 by the inside agent) is a residual
LightGBM on NM clock differences, destination, previous-arrival stamping and the
stand-turnover features INCLUDING look-ahead (approved by the user). It takes no base
prediction as input, so its correction is additive on any base.
Against the family-blend base, outside P: -5.73 (12 mo) / -6.12 (scored), 12/12 months.
On top of v10 (v9 + route-D corrector), measured on 2025 OOF: -4.82 [-5.35,-4.30]
(12 mo), -5.31 (scored), 12/12 months; keeping it on route-D rows too is -0.76 better.
P rows are unchanged.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
from prc.submit import build, TEAM
z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
ids, v10f = z['ids'], np.clip(z['trailD_full'], 0, None)           # v10 before rounding
st = np.load(CACHE / 'final' / 'stack_c_ranking.npz', allow_pickle=True)
assert np.array_equal(st['ids'], ids), 'stack ids != v10 ids'
v10 = pd.read_parquet(OUTPUTS / f'{TEAM}_v10.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt']
assert (np.rint(v10f).astype(int) == v10.reindex(ids).to_numpy()).all(), 'reconstructed v10 != shipped v10'
inP = st['in_P'].astype(bool); corr = st['corr']
assert np.isnan(corr[inP]).all() and np.isfinite(corr[~inP]).all(), 'corr must be NaN exactly on P'
pred = v10f + np.where(inP, 0.0, corr)
path = build(ids, pred, version=12, clip=(0.0, 200000.0))
v12 = pd.read_parquet(path).set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
d = (v12 - v10.reindex(ids).to_numpy()).astype(float)
print(f'v12 vs v10: rows changed {int((d != 0).sum()):,} | inside P changed: {int((d[inP] != 0).sum())} | '
      f'mean shift {d.mean():+.2f}s | |shift| median {np.median(np.abs(d)):.0f}s, p99 {np.percentile(np.abs(d),99):.0f}s, '
      f'max +{d.max():.0f} / {d.min():.0f}')
print(f'  corr: mean {corr[~inP].mean():+.1f}s | share of rows with |corr| > 300 s: {np.mean(np.abs(corr[~inP]) > 300):.4f}')
