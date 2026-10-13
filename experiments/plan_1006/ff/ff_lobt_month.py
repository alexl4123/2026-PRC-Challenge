import numpy as np, pandas as pd
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
SEC=lambda s: (pd.to_datetime(s,utc=True)-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds()
t=pd.read_parquet(C+'training.parquet',columns=['PHASE_mvt','MVT_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt']); t=t[t.PHASE_mvt=='DEP'].reset_index(drop=True)
z=np.load(C+'w2/v24s_oof.npz'); y,f,month=z['y'],z['f'],z['month']
mv=SEC(t.MVT_TIME_UTC_mvt).to_numpy(); lob=SEC(t.LOBT_flt).to_numpy(); ao=SEC(t.AOBT_3_flt).to_numpy()
lo=mv-lob-3606; hi=lo+7212; has=~np.isnan(lob)
g=f.copy(); g[has]=np.clip(f[has],np.maximum(lo[has],0),hi[has]); moved=has&(g!=f)
alt=np.where(moved & ~np.isnan(ao), np.clip(mv-ao,np.maximum(lo,0),hi), g)
R=lambda p,mm: np.sqrt(((y-p)[mm]**2).mean())
for m in range(1,13):
    mm=month==m; print(m,'%+.3f / %+.3f'%(R(g,mm)-R(f,mm),R(alt,mm)-R(f,mm)), int((moved&mm).sum()))
held=~np.isin(month,[3,6,9,12])
print('held %+.3f / %+.3f'%(R(g,held)-R(f,held),R(alt,held)-R(f,held)))
d=(y-g)**2-(y-alt)**2
print('aobt3 vs boundary on moved rows: better %d worse %d'%((d>0).sum(),(d<0).sum()))
print('moved up (f<lo) %d, down (f>hi) %d'%((moved&(f<lo)).sum(),(moved&(f>hi)).sum()))
