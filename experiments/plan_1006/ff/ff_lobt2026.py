import numpy as np, pandas as pd
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'; C=ROOT+'prc2026/cache/'
SEC=lambda s: (pd.to_datetime(s,utc=True)-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds()
k=pd.read_parquet(C+'ranking.parquet',columns=['PHASE_mvt','MVT_ID_mvt','MVT_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','APT']); k=k[k.PHASE_mvt=='DEP'].reset_index(drop=True)
k=k.merge(pd.read_parquet(ROOT+'prc2026/outputs/understanding-accordion_v27.parquet'),on='MVT_ID_mvt',how='left')
N=len(k); S0=243.1478; SSE0=N*S0**2
mv=SEC(k.MVT_TIME_UTC_mvt).to_numpy(); lob=SEC(k.LOBT_flt).to_numpy(); ao=SEC(k.AOBT_3_flt).to_numpy()
lo=np.maximum(mv-lob-3606,0); hi=mv-lob+3606; p=k.TAXITIME_SEC_mvt.to_numpy(float); has=~np.isnan(lob)
b=p.copy(); b[has]=np.clip(p[has],lo[has],hi[has]); m=has&(b!=p)
a=b.copy(); s=m&~np.isnan(ao); a[s]=np.clip(mv[s]-ao[s],lo[s],hi[s])
def bounds(q):
    r_lo=(p-lo)**2-(q-lo)**2; r_hi=(p-hi)**2-(q-hi)**2
    return np.minimum(r_lo,r_hi)[m].sum(), np.maximum(r_lo,r_hi)[m].sum()
for nm,q in (('boundary',b),('aobt3-in-window',a)):
    mn,mx=bounds(q); mid=((p-a)**2-(q-a)**2)[m].sum()
    f=lambda red: np.sqrt((SSE0-red)/N)
    print(f'{nm:16s} worst-case RMSE {f(mn):.2f}  best-case {f(mx):.2f}  if truth=AOBT3 point {f(mid):.2f}')
print('rows moved',m.sum(),' rows where aobt3 point differs from boundary',int((a!=b).sum()))
