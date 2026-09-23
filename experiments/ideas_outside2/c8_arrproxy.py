import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
cols=['APT','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','TAXITIME_SEC_mvt']
f=load('training',columns=cols)
f['mon']=f.MVT_TIME_UTC_mvt.dt.month
f['stamp']=((f.BLOCK_TIME_UTC_mvt-f.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=5)
arr=f[f.PHASE_mvt.eq('ARR')]; dep=f[f.PHASE_mvt.eq('DEP')].copy()
dep['os']=(dep.MVT_TIME_UTC_mvt-dep.SCHED_TIME_UTC_mvt).dt.total_seconds()
dep['P']=dep.AOBT_3_flt.isna()&(dep.os>7200); dep['tail']=dep.TAXITIME_SEC_mvt>7200
a=arr.groupby(['APT','mon']).stamp.mean().rename('arr_stamp')
dP=dep[dep.P].groupby(['APT','mon']).agg(P_rows=('tail','size'),P_tailrate=('tail','mean'))
dd=dep.groupby(['APT','mon']).agg(dep_stamp=('stamp','mean'))
T=pd.concat([a,dP,dd],axis=1)
for ap in ['LIRF','EGLL','LFPG','LEBL','EDDM']:
    t=T.loc[ap]
    print(ap, 'corr(arr_stamp, P_tailrate) %.2f  corr(arr_stamp, dep_stamp) %.2f'%(t.arr_stamp.corr(t.P_tailrate), t.arr_stamp.corr(t.dep_stamp)))
print(T.loc['LIRF'].round(4))
