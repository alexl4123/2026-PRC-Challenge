"""Stand-turnover family, decomposed and refined (inbound vs next occupant by type/operator match)."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
t0=time.time()
E=pd.Timestamp('1970-01-01',tz='UTC')
cols=['APT','STAND_mvt','FLIGHT_mvt','AIRCRAFT_TYPE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','PHASE_mvt']
full=load('training',columns=cols)
isdep=full.PHASE_mvt.eq('DEP').values
dep=full[isdep].reset_index(drop=True).drop(columns=['BLOCK_TIME_UTC_mvt'])  # leakage guard
arr=full[~isdep].reset_index(drop=True); del full
N=len(dep)
t=(dep.MVT_TIME_UTC_mvt-E).dt.total_seconds().values; ts=(dep.SCHED_TIME_UTC_mvt-E).dt.total_seconds().values
pre=lambda s: s.astype(str).str[:3].values
ev=pd.DataFrame({'k':(arr.APT.astype(str)+'|'+arr.STAND_mvt.astype(str).str.replace(r'(?<=\d)[A-Za-z]$','',regex=True)).values,
                 'tb':(arr.BLOCK_TIME_UTC_mvt-E).dt.total_seconds().values,
                 'typ':arr.AIRCRAFT_TYPE_mvt.astype(str).values,'op':pre(arr.FLIGHT_mvt)}).dropna(subset=['tb']).sort_values('tb')
ok=dep.STAND_mvt.notna().values
q=pd.DataFrame({'k':(dep.APT.astype(str)+'|'+dep.STAND_mvt.astype(str).str.replace(r'(?<=\d)[A-Za-z]$','',regex=True)).values,'t':t,'ts':ts,'i':np.arange(N),
                'typ':dep.AIRCRAFT_TYPE_mvt.astype(str).values,'op':pre(dep.FLIGHT_mvt)})[ok]
F=pd.DataFrame(index=np.arange(N))
# last arrival before takeoff, and the one before it
e1=ev.rename(columns={'tb':'tl','typ':'ltyp','op':'lop'})
m=pd.merge_asof(q.sort_values('t'),e1,left_on='t',right_on='tl',by='k',direction='backward')
F.loc[m.i.values,'sl_to']=(m.t-m.tl).values
F.loc[m.i.values,'sl_after_sched']=(m.tl-m.ts).values
F.loc[m.i.values,'sl_same_typ']=(m.ltyp==m.typ).astype(float).where(m.tl.notna()).values
F.loc[m.i.values,'sl_same_op']=(m.lop==m.op).astype(float).where(m.tl.notna()).values
# last arrival before our SCHED (inbound candidate)
m=pd.merge_asof(q.sort_values('ts'),e1,left_on='ts',right_on='tl',by='k',direction='backward')
F.loc[m.i.values,'si_before_sched']=(m.ts-m.tl).values
F.loc[m.i.values,'si_same_typ']=(m.ltyp==m.typ).astype(float).where(m.tl.notna()).values
# first arrival after our SCHED (next-occupant candidate), relative to takeoff; and its type/op match
m=pd.merge_asof(q.sort_values('ts'),e1,left_on='ts',right_on='tl',by='k',direction='forward',allow_exact_matches=False)
F.loc[m.i.values,'sn_minus_to']=(m.tl-m.t).values
F.loc[m.i.values,'sn_same_op']=(m.lop==m.op).astype(float).where(m.tl.notna()).values
F.loc[m.i.values,'sn_same_typ']=(m.ltyp==m.typ).astype(float).where(m.tl.notna()).values
# next arrival after takeoff
m=pd.merge_asof(q.sort_values('t'),e1,left_on='t',right_on='tl',by='k',direction='forward')
F.loc[m.i.values,'sa_after_to']=(m.tl-m.t).values
# next DEPARTURE from our stand (its takeoff) and previous departure (their takeoffs are public)
dq=pd.DataFrame({'k':q.k.values,'td':q.t.values}).sort_values('td')
m=pd.merge_asof(q.sort_values('t'),dq,left_on='t',right_on='td',by='k',direction='forward',allow_exact_matches=False)
F.loc[m.i.values,'sd_next_to']=(m.td-m.t).values
m=pd.merge_asof(q.sort_values('t'),dq,left_on='t',right_on='td',by='k',direction='backward',allow_exact_matches=False)
F.loc[m.i.values,'sd_prev_to']=(m.t-m.td).values
F.to_parquet(OUT+'c4_standnorm.parquet')
print('built %.0fs'%(time.time()-t0),flush=True)
