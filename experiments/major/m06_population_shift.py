"""m06 -- is the 2026 scored population different from 2025 Jan+Jul where it matters?
Route D / P counts by airport x month, off_sched in P, and v9's predictions there."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
M = CACHE / 'major'
tr = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'off_aobt3', 'off_sched', 'y', 'STAND_mvt', 'FLIGHT_mvt'])
rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['APT', 'month', 'off_aobt3', 'off_sched', 'MVT_ID_mvt', 'STAND_mvt', 'FLIGHT_mvt'])
v9 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v9.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
rk['pred'] = v9.reindex(rk.MVT_ID_mvt).to_numpy()
tr['pred'] = np.load(M / 'v9s_oof.npy')
for d in (tr, rk):
    d['D'] = d.off_aobt3.isna(); d['P'] = d.D & (d.off_sched > 7200)
    d['Dmid'] = d.D & d.off_sched.between(1800, 7200)
t25 = tr[tr.month.isin([1, 7])]
rows = []
for (a, m), g in t25.groupby(['APT', 'month'], observed=True):
    h = rk[(rk.APT == a) & (rk.month == m)]
    rows.append((a, m, len(g), len(h), g.D.mean()*100, h.D.mean()*100, g.P.sum(), h.P.sum(), g.Dmid.sum(), h.Dmid.sum(),
                 (g.y[g.P] > 7200).mean() if g.P.any() else np.nan,
                 g.pred[g.P].mean() if g.P.any() else np.nan, h.pred[h.P].mean() if h.P.any() else np.nan,
                 g.off_sched[g.P].median() if g.P.any() else np.nan, h.off_sched[h.P].median() if h.P.any() else np.nan))
R = pd.DataFrame(rows, columns=['apt', 'm', 'n25', 'n26', 'D%25', 'D%26', 'P25', 'P26', 'Dmid25', 'Dmid26', 'tailrate_P25',
                                'predP25', 'predP26', 'soP25', 'soP26'])
pd.set_option('display.width', 220)
print(R.round(2).to_string(index=False))
print('\ntotals: P 2025 scored', t25.P.sum(), ' P 2026', rk.P.sum(), ' D 2025', t25.D.sum(), ' D 2026', rk.D.sum())
# unseen stands / callsigns in 2026 vs 2025
st25 = set(zip(tr.APT.astype(str), tr.STAND_mvt.astype(str)))
rk['stand_unseen'] = [(a, s) not in st25 for a, s in zip(rk.APT.astype(str), rk.STAND_mvt.astype(str))]
t25b = t25.copy()
print('\n2026 departures with (apt,stand) unseen in 2025, by airport (%):')
print((rk.groupby('APT', observed=True).stand_unseen.mean() * 100).round(2).to_string())
