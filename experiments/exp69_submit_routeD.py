"""EXP69 -- v10 = v9 + the route-D disruption corrector (major-levers agent, m11d/m11f).

Only flights with NO NM off-block record (route D) change. A small residual model uses
the airport's trailing state: the last 90 min of NM taxi estimates from flights WITH a
record, the share of flights without one, and recent arrival taxi-in -- all backward-
looking, all from the ranking file. 12-fold -0.60 [-0.94,-0.25], scored -0.63; its
real target is the Amsterdam winter disruption of 2-7 Jan 2026 (805 route-D rows lifted
by ~757 s). Month-wise CV cannot see that event, so this is a leaderboard test.
Candidate arrays: cache/major/m11f_candidates.npz (full-strength variant, clipped at 0).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
from prc.submit import build, TEAM
z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
ids, v9a, cand = z['ids'], z['v9'], np.clip(z['trailD_full'], 0, None)
v9 = pd.read_parquet(OUTPUTS / f'{TEAM}_v9.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt']
assert (np.rint(np.clip(v9a, 0, 2e5)).astype(int) == v9.reindex(ids).to_numpy()).all(), 'embedded v9 != shipped v9'
path = build(ids, cand, version=10, clip=(0.0, 200000.0))
v10 = pd.read_parquet(path).set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
d = v10 - v9.reindex(ids).to_numpy()
print(f'v10 vs v9: rows changed {int((d != 0).sum()):,} | mean shift on changed {d[d != 0].mean():+.0f}s | '
      f'max up {d.max()} s, max down {d.min()} s')
