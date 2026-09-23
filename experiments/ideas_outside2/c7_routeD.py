import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy')
v6=np.load(C+'exp62_oof.npy')[1]; r=y-v6
d=load('training',phase='DEP',columns=['APT','AOBT_3_flt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'])
apt=d.APT.astype(str).values
os_=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds().values
D=d.AOBT_3_flt.isna().values; P=D&(os_>7200); DnP=D&~P
stamp=np.abs(y-os_)<=60
for lab,sel in [('12 months',np.ones(len(y),bool)),('months 1+7',np.isin(mon,[1,7]))]:
    tot=np.sum(r[sel]**2)
    print(f'--- {lab}: total RMSE {np.sqrt(tot/sel.sum()):.1f}')
    for name,m in [('P',P),('D not P',DnP),('  DnP y>7200 (invisible)',DnP&(y>7200)),('  DnP stamped y<=7200',DnP&stamp&(y<=7200)),('  DnP real y<=7200',DnP&~stamp&(y<=7200)),('route A/B/C',~D)]:
        mm=m&sel
        print(f'  {name:28s} rows {mm.sum():7d}  SSE share {np.sum(r[mm]**2)/tot:.3f}  rmse {np.sqrt(np.mean(r[mm]**2)):8.1f}')
# stamped share by off_sched band in DnP, and oracle
print('\nD-not-P by off_sched band (12 months): rows, stamped share, v6 rmse, oracle rmse (stamped->off_sched, real->v6)')
for lo,hi in [(-1e9,600),(600,1200),(1200,1800),(1800,3600),(3600,7200)]:
    m=DnP&(os_>lo)&(os_<=hi)&(y<=7200)
    orc=np.where(stamp[m],os_[m],v6[m])
    print(f'  ({lo:6.0f},{hi:5.0f}] rows {m.sum():6d} stamped {stamp[m].mean():.3f}  v6 {np.sqrt(np.mean(r[m]**2)):6.1f}  oracle {np.sqrt(np.mean((y[m]-orc)**2)):6.1f}')
m=DnP&(y<=7200)
orc=v6.copy(); orc[m&stamp]=os_[m&stamp]
for lab,sel in [('12 months',np.ones(len(y),bool)),('months 1+7',np.isin(mon,[1,7]))]:
    print(f'oracle stamped-vs-real inside D-not-P (y<=7200): total RMSE {np.sqrt(np.mean(r[sel]**2)):.2f} -> {np.sqrt(np.mean((y[sel]-orc[sel])**2)):.2f}')
print('stamped share in DnP by airport:', pd.Series(stamp[DnP&(y<=7200)&(os_>1800)]).groupby(apt[DnP&(y<=7200)&(os_>1800)]).agg(['mean','size']).round(3).to_dict('index'))
