"""Flight-level physical features NOT in the bulk model: runway busy-period position (queueing),
stand next-occupant (arrival in-block at our stand before our takeoff), destination.
Test = LOMO residual booster on the v6 clean-bulk residual, vs a null booster with only (pred, APT, hour)."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
from scipy.stats import spearmanr
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
t0=time.time()
E=pd.Timestamp('1970-01-01',tz='UTC')
cols=['APT','STAND_mvt','RUNWAY_mvt','ADES_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','TAXITIME_SEC_mvt','PHASE_mvt']
full=load('training',columns=cols)
full['t']=(full.MVT_TIME_UTC_mvt-E).dt.total_seconds()
full['tb']=(full.BLOCK_TIME_UTC_mvt-E).dt.total_seconds()
full['ts']=(full.SCHED_TIME_UTC_mvt-E).dt.total_seconds()
isdep=full.PHASE_mvt.eq('DEP').values
dep=full[isdep].reset_index(drop=True); arr=full[~isdep].reset_index(drop=True); del full
dep=dep.drop(columns=['BLOCK_TIME_UTC_mvt','tb'])   # leakage guard: never touch departure block time
y=dep.TAXITIME_SEC_mvt.astype(float).values; N=len(y)
assert np.allclose(y,np.load(C+'altfam_y.npy'))
mon=np.load(C+'altfam_month.npy')
v6=np.load(C+'exp62_oof.npy')[1]; res=y-v6
off_sched=dep.t.values-dep.ts.values
P=dep.AOBT_3_flt.isna().values&(off_sched>7200)
clean=(y<=7200)&~P
F=pd.DataFrame(index=dep.index)
# ---- runway busy period (takeoff sequence per APT|RUNWAY) ----
k=dep.APT.astype(str)+'|'+dep.RUNWAY_mvt.astype(str)
s=pd.DataFrame({'k':k.values,'t':dep.t.values,'i':np.arange(N)}).sort_values(['k','t'])
same=s.k.values[1:]==s.k.values[:-1]
gap=np.r_[np.inf,np.where(same,np.diff(s.t.values),np.inf)]
gapn=np.r_[np.where(same,np.diff(s.t.values),np.inf),np.inf]
for G in (120,180):
    newbp=gap>G
    bpid=np.cumsum(newbp)
    pos=s.groupby(bpid).cumcount().values
    start=pd.Series(s.t.values).groupby(bpid).transform('first').values
    size=pd.Series(bpid).map(pd.Series(bpid).value_counts()).values
    F.loc[s.i.values,f'bp{G}_pos']=pos
    F.loc[s.i.values,f'bp{G}_elapsed']=s.t.values-start
    F.loc[s.i.values,f'bp{G}_remaining']=size-pos-1
g2=pd.Series(np.where(np.isfinite(gap),gap,np.nan))
F.loc[s.i.values,'gap_prev']=g2.values
F.loc[s.i.values,'gap_next']=np.where(np.isfinite(gapn),gapn,np.nan)
F.loc[s.i.values,'gap_prev3_mean']=g2.groupby(s.k.values).transform(lambda x: x.rolling(3,min_periods=1).mean()).values
# ---- stand next occupant: last arrival in-block at our stand before our takeoff ----
ka=arr.APT.astype(str)+'|'+arr.STAND_mvt.astype(str)
ev=pd.DataFrame({'k':ka.values,'tb':arr.tb.values}).dropna().sort_values('tb')
q=pd.DataFrame({'k':(dep.APT.astype(str)+'|'+dep.STAND_mvt.astype(str)).values,'t':dep.t.values,'ts':dep.ts.values,'i':np.arange(N)})
q=q[dep.STAND_mvt.notna().values].sort_values('t')
m=pd.merge_asof(q,ev.rename(columns={'tb':'tl'}),left_on='t',right_on='tl',by='k',direction='backward')
F.loc[m.i.values,'stand_lastarr_to_takeoff']=(m.t-m.tl).values
F.loc[m.i.values,'stand_lastarr_after_sched']=(m.tl-m.ts).values
m2=pd.merge_asof(q,ev.rename(columns={'tb':'tn'}),left_on='t',right_on='tn',by='k',direction='forward')
F.loc[m2.i.values,'stand_nextarr_after_takeoff']=(m2.tn-m2.t).values
# arrivals at our stand between our sched and our takeoff
ev['one']=1
F['stand_arr_sched_to_to']=np.nan
# use cumulative counts via asof on sorted times
ev2=ev.sort_values(['k','tb']).copy(); ev2['c']=ev2.groupby('k').cumcount()+1
a1=pd.merge_asof(q.sort_values('t'),ev2[['k','tb','c']].sort_values('tb'),left_on='t',right_on='tb',by='k',direction='backward')
a0=pd.merge_asof(q.sort_values('ts'),ev2[['k','tb','c']].sort_values('tb'),left_on='ts',right_on='tb',by='k',direction='backward')
c1=pd.Series(a1.c.fillna(0).values,index=a1.i.values); c0=pd.Series(a0.c.fillna(0).values,index=a0.i.values)
F.loc[c1.index,'stand_arr_sched_to_to']=(c1-c0.reindex(c1.index)).values
F['ades']=pd.Categorical(dep.ADES_mvt.astype(str))
F['apt']=pd.Categorical(dep.APT.astype(str))
F['pred']=v6
F['hour']=dep.MVT_TIME_UTC_mvt.dt.hour.values
F['off_sched']=off_sched
F['aobt_missing']=dep.AOBT_3_flt.isna().values.astype(int)
F.to_parquet(OUT+'c2_features.parquet')
print('features built %.0fs rss %.2f'%(time.time()-t0,peak_rss_gb()),flush=True)
# ---- univariate: Spearman with residual on clean bulk, and with y ----
newf=[c for c in F.columns if c.startswith(('bp','gap','stand'))]
cb=np.where(clean)[0]
sub=np.random.default_rng(0).choice(cb,400000,replace=False)
print('\nSpearman on clean bulk (400k sample): feature  cover  rho(resid)  rho(y)  | median within-apt |rho(resid)|')
for c in newf:
    x=F[c].values[sub]; ok=np.isfinite(x)
    r1=spearmanr(x[ok],res[sub][ok]).correlation; r2=spearmanr(x[ok],y[sub][ok]).correlation
    wa=[]
    for a in np.unique(dep.APT.values[sub]):
        mm=ok&(dep.APT.values[sub]==a)
        if mm.sum()>1000: wa.append(abs(spearmanr(x[mm],res[sub][mm]).correlation))
    print(f'  {c:28s} {ok.mean():.3f} {r1:+.3f} {r2:+.3f} | {np.median(wa):.3f}')
# stand next-occupant lower bound: when last arrival at our stand came AFTER our sched and < 3600 s before takeoff
lb=F.stand_lastarr_to_takeoff.values; after=F.stand_lastarr_after_sched.values>0
mm=clean&np.isfinite(lb)&(lb<3600)&after
print(f'\nnext-occupant candidates (arr at our stand after our SCHED, <1h before our takeoff): {mm.sum()} rows ({mm.sum()/clean.sum():.3%})')
print(f'  y >= lb holds on {np.mean(y[mm]>=lb[mm]):.3f}; v6 pred < lb on {np.mean(v6[mm]<lb[mm]):.3f}; resid mean {res[mm].mean():+.1f}, rmse {np.sqrt(np.mean(res[mm]**2)):.1f}; SSE share {np.sum(res[mm]**2)/np.sum(res[clean]**2):.4f}')
vi=mm&(v6<lb)
print(f'  rows where v6 violates the bound: {vi.sum()}, resid mean {res[vi].mean():+.1f}; clipping pred up to lb there: clean-bulk RMSE {np.sqrt(np.mean(res[clean]**2)):.3f} -> ',end='')
r2=res.copy(); r2[vi]=y[vi]-lb[vi]; print(f'{np.sqrt(np.mean(r2[clean]**2)):.3f}')
print('peak rss %.2f GB, %.0fs'%(peak_rss_gb(),time.time()-t0))
