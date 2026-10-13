import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load
import assemble as A
pd.set_option('display.width', 250)
C = A.Components()
t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True); t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
rows = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
pb = 0.5 * C.fit('w2_final', 'mix')[rows] + 0.5 * C.member('seq')[rows]; s6 = C.off_sched[rows]; v28 = A.shipped(C, 28)
rk = load('ranking', phase='DEP', columns=['FLIGHT_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).iloc[rows]
du6 = ((rk.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - rk.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()
L = (C.apt[rows] == 'LIRF') & ((du6 >= 1) | (s6 > 28800))
x = pd.DataFrame(dict(fl=rk.FLIGHT_mvt.astype(str).to_numpy(), ades=rk.ADES_mvt.astype(str).to_numpy(),
                      mL=rk.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.strftime('%m-%d %H:%M').to_numpy(),
                      sL=rk.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.strftime('%m-%d %H:%M').to_numpy(),
                      du=du6, p=p.round(3), off_sched=s6, pb=pb.round(), v28=v28[rows]))[L]
print(x.sort_values('p').head(12).to_string())
