"""Statistical companion of B: LIRF zone P with off_sched > T (no date condition) -> p := 1.  2025 OOF and 2026 moves."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, f = V['y'], V['month'], V['f']
W = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
Pidx, p14, pbP, Prest, so, rome = W['Pidx'], W['p14'], W['pbP'], W['P_rest'], W['so'], W['rome']
yP = y[Pidx]; cls = np.where(np.abs(yP - so[Pidx]) <= 120, 's', np.where(yP > 43200, 'd', np.where(yP <= 7200, 'n', 'o')))
N = len(y); S = np.isin(month, [1, 7]); held = ~np.isin(month, [3, 6, 9, 12]); allm = np.ones(N, bool)
rm = lambda g, m: np.sqrt(np.mean((y[m] - g[m]) ** 2))
for T in (14400, 21600, 28800, 43200):
    m = rome & (so[Pidx] > T)
    g = f.copy(); g[Pidx] = Prest + np.clip(np.where(m, 1.0, p14) * so[Pidx] + (1 - np.where(m, 1.0, p14)) * pbP, 0, None)
    print(f'T={T}: LIRF P rows {m.sum()} classes {pd.Series(cls[m]).value_counts().to_dict()} | 12m {rm(g, allm) - rm(f, allm):+.3f} held {rm(g, held) - rm(f, held):+.3f} Jan+Jul {rm(g, S) - rm(f, S):+.3f}')

# 2025: B (UTC crossing, 1800 s margin) OR off_sched > 28,800 at LIRF
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).iloc[Pidx]
du = ((d.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()
m = rome & ((du >= 1) | (so[Pidx] > 28800))
g = f.copy(); g[Pidx] = Prest + np.clip(np.where(m, 1.0, p14) * so[Pidx] + (1 - np.where(m, 1.0, p14)) * pbP, 0, None)
ch = np.abs(g - f) > .5
print(f'B or T28800: rows {m.sum()} classes {pd.Series(cls[m]).value_counts().to_dict()} changed {ch.sum()} worse {int((ch & ((y - g) ** 2 > (y - f) ** 2 + 1)).sum())} | 12m {rm(g, allm) - rm(f, allm):+.3f} held {rm(g, held) - rm(f, held):+.3f} Jan+Jul {rm(g, S) - rm(f, S):+.3f} Jan {rm(g, month == 1) - rm(f, month == 1):+.3f} Jul {rm(g, month == 7) - rm(f, month == 7):+.3f}')
# 2026 moves
import assemble as A
from prc.io import load
C = A.Components()
t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True); t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
rows = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
pb = 0.5 * C.fit('w2_final', 'mix')[rows] + 0.5 * C.member('seq')[rows]; s6 = C.off_sched[rows]; corr = C.corr[rows]; v28 = A.shipped(C, 28)
rk = load('ranking', phase='DEP', columns=['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).iloc[rows]
du6 = ((rk.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - rk.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()
L = C.apt[rows] == 'LIRF'
for nm, mm in (('T28800 only (no crossing)', L & (du6 < 1) & (s6 > 28800)), ('B or T28800', L & ((du6 >= 1) | (s6 > 28800)))):
    o = np.rint(np.clip(np.where(mm, 1.0, p) * s6 + (1 - np.where(mm, 1.0, p)) * pb + corr, 0, A.HI)); mv = o - v28[rows]
    print(f'2026 {nm}: rows {mm.sum()}, moved {int((mv != 0).sum())}, sum move^2 {(mv ** 2).sum():.3e}, moves>1000: {np.sort(mv[np.abs(mv) > 1000]).round().tolist()}')
