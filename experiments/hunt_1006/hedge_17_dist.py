"""2026 LIRF zone P: hypothesis distances and expected hedge SSE by date-crossing group (label-free)."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load
import assemble as A
C = A.Components()
t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
rows = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
pb = 0.5 * C.fit('w2_final', 'mix')[rows] + 0.5 * C.member('seq')[rows]; so = C.off_sched[rows]
rk = load('ranking', phase='DEP', columns=['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).iloc[rows]
du = ((rk.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - rk.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()
L = C.apt[rows] == 'LIRF'; dist = np.abs(so - pb); h = p * (1 - p) * dist ** 2; tot = 235.3996 ** 2 * 344841
print(f'LIRF zone P 2026: {L.sum()} rows; |Hs - Hn| median {np.median(dist[L]):.0f} p90 {np.quantile(dist[L], .9):.0f} max {dist[L].max():.0f}')
for nm, m in (('UTC-date crossing (B)', L & (du >= 1)), ('no crossing', L & (du < 1))):
    print(f'  {nm}: rows {m.sum()}, mid-p rows {int(((p > .05) & (p < .95) & m).sum())}, expected hedge SSE {h[m].sum():.3e} ({100 * h[m].sum() / tot:.1f}% of file SSE)')
for lo, hi in ((7200, 10800), (10800, 14400), (14400, 21600), (21600, 1e7)):
    m = L & (du < 1) & (so > lo) & (so <= hi)
    print(f'    no crossing, off_sched ({lo},{hi:.0f}]: rows {m.sum()}, p mean {p[m].mean():.2f}, hedge {h[m].sum():.2e}')
