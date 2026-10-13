import pandas as pd, numpy as np, glob
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
cols=['PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','TAXITIME_SEC_mvt']
d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(ROOT+'training_2025-*.parquet'))]); d=d[d.PHASE_mvt=='DEP']
for nm,k in [('NM',d.LOBT_flt.notna()),('no-NM',d.LOBT_flt.isna())]:
    x=d[k]; bs=(x.BLOCK_TIME_UTC_mvt-x.SCHED_TIME_UTC_mvt).dt.total_seconds()
    print(nm, len(x), 'BLOCK-SCHED quantiles [0,.001,.01,.5,.99,.999,1]', np.quantile(bs,[0,.001,.01,.5,.99,.999,1]).round().tolist())
x=d[d.LOBT_flt.isna()]; bs=(x.BLOCK_TIME_UTC_mvt-x.SCHED_TIME_UTC_mvt).dt.total_seconds()
print('no-NM: share |BLOCK-SCHED|<=60', (bs.abs()<=60).mean(), ' y>40000:', (x.TAXITIME_SEC_mvt>40000).sum())
