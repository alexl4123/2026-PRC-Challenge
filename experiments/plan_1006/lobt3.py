import numpy as np, pandas as pd
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
SEC=lambda s: (pd.to_datetime(s,utc=True)-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds()
cols=['PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','EOBT_1_flt','IOBT_flt','APT']
t=pd.read_parquet(C+'training.parquet',columns=cols); t=t[t.PHASE_mvt=='DEP']
b=SEC(t.BLOCK_TIME_UTC_mvt)
for c in ['LOBT_flt','AOBT_3_flt','EOBT_1_flt','IOBT_flt','SCHED_TIME_UTC_mvt']:
    d=(b-SEC(t[c])).dropna()
    print(f'2025 BLOCK-{c:18s} n {len(d):8d} q0 {d.min():9.0f} q.0001 {d.quantile(1e-4):8.0f} q.5 {d.median():6.0f} q.9999 {d.quantile(1-1e-4):8.0f} max {d.max():9.0f}  share|d|<=3606 {np.mean(np.abs(d)<=3606):.5f}')
d=(b-SEC(t.LOBT_flt)).dropna()
print('BLOCK-LOBT share at |d| in [3540,3606]: %.5f; histogram of d/600:'%np.mean((np.abs(d)>=3540)), np.histogram(d,bins=[-3607,-3000,-1800,-600,0,600,1800,3000,3607])[0])
# label-free 2025 vs 2026 comparison of NM clocks relation
k=pd.read_parquet(C+'ranking.parquet',columns=['PHASE_mvt','MVT_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','SCHED_TIME_UTC_mvt']); k=k[k.PHASE_mvt=='DEP']
for nm,x in (('2025',t),('2026',k)):
    a=(SEC(x.AOBT_3_flt)-SEC(x.LOBT_flt)).dropna(); m=(SEC(x.MVT_TIME_UTC_mvt)-SEC(x.LOBT_flt)).dropna()
    print(nm,'AOBT3-LOBT: share|.|<=3606 %.5f max %.0f min %.0f | MVT-LOBT median %.0f q.999 %.0f max %.0f'%(np.mean(np.abs(a)<=3606),a.max(),a.min(),m.median(),m.quantile(.999),m.max()))
