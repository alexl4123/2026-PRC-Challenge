"""m11f -- candidate 2026 arrays for a leaderboard test of the route-D state corrector
(NOT submission files): full and half-shrunk corrections on v9, with MVT ids, ranking DEP order."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
M = CACHE / 'major'
rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'off_aobt3'])
v9 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v9.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
full = np.load(M / 'm11d_pred_trailD.npy'); half = v9 + 0.5 * (full - v9)
assert np.all(full[rk.off_aobt3.notna().to_numpy()] == v9[rk.off_aobt3.notna().to_numpy()]), 'route A must be untouched'
np.savez(M / 'm11f_candidates.npz', ids=rk.MVT_ID_mvt.to_numpy(), v9=v9, trailD_full=full, trailD_half=half)
for nm, a in [('full', full), ('half', half)]:
    dl = a - v9
    print(f'{nm}: rows moved >1 s {int((np.abs(dl)>1).sum())}, >60 s {int((np.abs(dl)>60).sum())}; mean shift on moved {dl[np.abs(dl)>1].mean():+.0f}; '
          f'sum shift^2 {(dl**2).sum():.3e}; min/max pred {a.min():.0f}/{a.max():.0f}')
