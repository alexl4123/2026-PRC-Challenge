"""Ceilings for group-level residual corrections on the CLEAN bulk (y<=7200, outside P).
LOMO shrunk group-mean residual (what a group-level static feature could add beyond v6 / v8-like),
in-sample oracle, and LOO within-month state bins."""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
t0=time.time()
cols=['APT','STAND_mvt','RUNWAY_mvt','FLIGHT_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt',
      'MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','TAXITIME_SEC_mvt']
d=load('training',phase='DEP',columns=cols)
y=d.TAXITIME_SEC_mvt.astype(float).values
mon=np.load(C+'altfam_month.npy')
assert np.allclose(y,np.load(C+'altfam_y.npy'))
o=np.load(C+'exp62_oof.npy'); b=np.load(C+'blend_oof_mix_nn3.npy')
v6=o[1]; v8l=0.5*o[1]+0.5*b[1]
off_sched=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds().values
P=d.AOBT_3_flt.isna().values&(off_sched>7200)
clean=(y<=7200)&~P
N=len(y); print('rows',N,'clean',clean.sum())
day=d.MVT_TIME_UTC_mvt.dt.floor('D').values.astype('datetime64[D]').astype(np.int64)
hour=d.MVT_TIME_UTC_mvt.dt.hour.values
pier=d.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+|\d)',expand=False).fillna('NA').values
apt=d.APT.astype(str).values
def key(*arrs):
    s=pd.Series(arrs[0]).astype(str)
    for a in arrs[1:]: s=s+'|'+pd.Series(a).astype(str)
    return pd.factorize(s)[0]
keys={
 'apt_stand_rwy': key(apt,d.STAND_mvt.values,d.RUNWAY_mvt.values),
 'apt_stand':     key(apt,d.STAND_mvt.values),
 'apt_rwy':       key(apt,d.RUNWAY_mvt.values),
 'apt_callsign':  key(apt,d.FLIGHT_mvt.values),
 'apt_ades':      key(apt,d.ADES_mvt.values),
 'apt_type_rwy':  key(apt,d.AIRCRAFT_TYPE_mvt.values,d.RUNWAY_mvt.values),
 'apt_op_stand':  key(apt,d.AIRCRAFT_OPERATOR_flt.values,d.STAND_mvt.values),
 'apt_callsign_stand': key(apt,d.FLIGHT_mvt.values,d.STAND_mvt.values),
}
def rmse(r,m): return np.sqrt(np.mean(r[m]**2))
# day-cluster bootstrap of delta RMSE (over the clean bulk, scored as contribution over all N rows is too diluted; report clean-bulk RMSE)
rng=np.random.default_rng(0)
cd=clean.nonzero()[0]; dcl=pd.factorize(day[cd].astype(str)+apt[cd])[0]
def boot(r0,r1,B=200):
    e0=pd.Series(r0[cd]**2).groupby(dcl).sum().values; e1=pd.Series(r1[cd]**2).groupby(dcl).sum().values
    n=pd.Series(np.ones(len(cd))).groupby(dcl).sum().values
    G=len(n); out=[]
    for _ in range(B):
        i=rng.integers(0,G,G); out.append(np.sqrt(e1[i].sum()/n[i].sum())-np.sqrt(e0[i].sum()/n[i].sum()))
    return np.percentile(out,[2.5,97.5])
def lomo_corr(res,g,k):
    """shrunk mean of residual in group g from OTHER months, only clean rows contribute"""
    corr=np.zeros(N)
    df=pd.DataFrame({'g':g[clean],'m':mon[clean],'r':res[clean]})
    tot=df.groupby('g')['r'].agg(['sum','count'])
    per=df.groupby(['g','m'])['r'].agg(['sum','count'])
    S=tot['sum'].reindex(g).values; n=tot['count'].reindex(g).values
    idx=pd.MultiIndex.from_arrays([g,mon])
    Sm=per['sum'].reindex(idx).fillna(0).values; nm=per['count'].reindex(idx).fillna(0).values
    S=np.nan_to_num(S)-Sm; n=np.nan_to_num(n)-nm
    return S/(n+k), n
for name,res in [('v6',y-v6),('v8like',y-v8l)]:
    base=rmse(res,clean)
    print(f'\n=== {name}: clean-bulk RMSE {base:.2f}')
    for kn,g in keys.items():
        line=f'  {kn:20s}'
        for k in [20,100,400]:
            c,n=lomo_corr(res,g,k)
            r1=res-c
            line+=f'  k={k:<3d} {rmse(r1,clean)-base:+.3f}'
        # in-sample oracle (not LOO; optimistic)
        s=pd.Series(res[clean]).groupby(g[clean]).transform('mean').values
        r2=res.copy(); r2[clean]=res[clean]-s
        line+=f' | in-sample oracle {rmse(r2,clean)-base:+.2f} | ngroups {len(np.unique(g[clean]))}'
        print(line,flush=True)
    # coverage in other months for stand-rwy pairs
    c,n=lomo_corr(res,keys['apt_stand_rwy'],100)
    for lo,hi in [(0,0),(1,19),(20,99),(100,999),(1000,1e9)]:
        m=clean&(n>=lo)&(n<=hi)
        print(f'   pair rows-in-other-months [{lo},{hi}]: rows {m.sum()/clean.sum():.4f}  SSE share {np.sum(res[m]**2)/np.sum(res[clean]**2):.4f}  rmse {rmse(res,m):.1f}  mean resid {res[m].mean():+.1f}')
    # LOO within-month state bins
    for bn,g in [('apt_day_hour',key(apt,day,hour)),('apt_day_hour_rwy',key(apt,day,hour,d.RUNWAY_mvt.values)),
                 ('apt_day_hour_pier',key(apt,day,hour,pier)),('apt_day_hour_pier_rwy',key(apt,day,hour,pier,d.RUNWAY_mvt.values))]:
        s=pd.Series(res[clean]).groupby(g[clean]); S=s.transform('sum').values; n=s.transform('count').values
        line=f'  LOO {bn:24s}'
        for k in [0,3,10]:
            loo=(S-res[clean])/np.maximum(n-1+k,1); r2=res.copy(); r2[clean]=res[clean]-loo
            line+=f' k={k} {rmse(r2,clean)-base:+.3f}'
        print(line,flush=True)
    if name=='v6':
        for kn in ['apt_stand_rwy','apt_callsign']:
            c,n=lomo_corr(res,keys[kn],100); ci=boot(res,res-c)
            print(f'  CI {kn} k=100: [{ci[0]:+.3f},{ci[1]:+.3f}]')
print('peak rss %.2f GB, %.0fs'%(peak_rss_gb(),time.time()-t0))
