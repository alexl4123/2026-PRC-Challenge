"""m08 -- where did the P-only submissions (v5->v6->v7) move predictions in 2026, and
what does the live SSE change imply?  Diagnostic, from existing submissions only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
M = CACHE / 'major'; N = 344841
LIVE = {'v4': 322.2460, 'v5': 321.2771, 'v6': 308.1903, 'v7': 304.351, 'v8': 297.829, 'v9': 297.3414}
rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'off_aobt3', 'off_sched', 'MVT_TIME_UTC_mvt'])
S = {}
for v in LIVE:
    s = pd.read_parquet(OUTPUTS / f'understanding-accordion_{v}.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    S[v] = s.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
D = rk.off_aobt3.isna().to_numpy(); P = D & (rk.off_sched.to_numpy() > 7200)
cell = rk.APT.astype(str).to_numpy() + '-' + rk.month.astype(str).to_numpy()
for a, b in [('v4', 'v5'), ('v5', 'v6'), ('v6', 'v7')]:
    dl = S[b] - S[a]
    sdd = (dl ** 2).sum(); dsse = (LIVE[b] ** 2 - LIVE[a] ** 2) * N
    srd = (LIVE[a] ** 2 * N - LIVE[b] ** 2 * N + sdd) / 2
    print(f'\n{a}->{b}: live dSSE {dsse:.3e}; rows changed P {(np.abs(dl[P])>0.5).sum()}, notP {(np.abs(dl[~P])>0.5).sum()}; '
          f'sum d^2 {sdd:.3e} (P share {100*(dl[P]**2).sum()/sdd:.1f}%);  sum r*d = {srd:.3e};  slope r~d = {srd/sdd:.3f}')
    t = pd.DataFrame({'cell': cell[P], 'd': dl[P]}).groupby('cell').d.agg(['size', 'mean', lambda x: (x**2).sum()])
    t.columns = ['nP', 'mean_d', 'sum_d2']; t['share_d2%'] = 100 * t.sum_d2 / (dl[P] ** 2).sum()
    print(t[t['share_d2%'] > 1].sort_values('share_d2%', ascending=False).round(1).to_string())
# the EHAM January P rows: what each version served
k = P & (cell == 'EHAM-1')
print('\nEHAM Jan 2026 P rows', k.sum(), ' mean pred by version:', {v: round(S[v][k].mean()) for v in LIVE},
      ' median off_sched', np.median(rk.off_sched.to_numpy()[k]))
