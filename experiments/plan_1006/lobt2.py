import numpy as np, pandas as pd
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
C=ROOT+'prc2026/cache/'
SEC=lambda s: (pd.to_datetime(s,utc=True)-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds()
k=pd.read_parquet(C+'ranking.parquet',columns=['PHASE_mvt','MVT_ID_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','EOBT_1_flt','APT','FLIGHT_ID_mvt']); k=k[k.PHASE_mvt=='DEP'].reset_index(drop=True)
rr=pd.read_parquet(C+'drivers_0926/rows_ranking.parquet',columns=['apt','route','off_sched','month'])
print(len(k),len(rr))
sub=pd.read_parquet(ROOT+'prc2026/outputs/understanding-accordion_v27.parquet')
k=k.merge(sub,on='MVT_ID_mvt',how='left')
assert (k.APT.astype(str).values==rr.apt.astype(str).values).all()
k['route']=rr.route.astype(str).values; k['month']=rr.month.values
mv=SEC(k.MVT_TIME_UTC_mvt); lo=mv-SEC(k.LOBT_flt)-3606; hi=lo+7212
p=k.TAXITIME_SEC_mvt.astype(float)
q=np.clip(p,np.maximum(lo,0),hi); q=q.where(k.LOBT_flt.notna(),p)
k['pred']=p; k['lo']=lo; k['hi']=hi; k['move']=q-p; k['off_sched']=mv-SEC(k.SCHED_TIME_UTC_mvt); k['mvt_aobt3']=mv-SEC(k.AOBT_3_flt)
m=k.move!=0
print(k.loc[m].groupby('route').move.agg(['size',lambda v:(v**2).sum()]))
print(k.loc[m].sort_values('move',key=np.abs,ascending=False)[['APT','month','route','pred','lo','hi','off_sched','mvt_aobt3','move']].head(15).round(0).to_string())
# how was the rule in 2025 by route (OOF): share of window-moved rows per route
