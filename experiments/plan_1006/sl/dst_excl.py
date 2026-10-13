import pandas as pd, numpy as np, glob
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
C=ROOT+'prc2026/cache/'
cols=['MVT_ID_mvt','PHASE_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','LOBT_flt']
def dl(d):
    dep=d[d.PHASE_mvt=='DEP'].reset_index(drop=True); dep['i']=np.arange(len(dep)); arr=d[d.PHASE_mvt=='ARR']
    a=arr[arr.LOBT_flt.notna()][['FLIGHT_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','LOBT_flt']].rename(columns={'MVT_TIME_UTC_mvt':'am','LOBT_flt':'dlobt'})
    m=dep[dep.LOBT_flt.isna()][['i','FLIGHT_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt']].merge(a,on=['FLIGHT_mvt','ADEP_mvt','ADES_mvt'])
    t=(m.am-m.MVT_TIME_UTC_mvt).dt.total_seconds(); m=m[(t>0)&(t<86400)].assign(t=t).sort_values('t').drop_duplicates('i')
    off=np.full(len(dep),np.nan); off[m.i.to_numpy()]=(m.MVT_TIME_UTC_mvt-m.dlobt).dt.total_seconds().to_numpy()
    return dep, off
d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(ROOT+'training_2025-*.parquet'))])
dep,off=dl(d)
V=np.load(C+'w2/v24s_oof.npz',allow_pickle=True); y,f,month,P=V['y'],V['f'],V['month'],V['P']
k=~np.isnan(off); lo=off-3606; hi=off+3606
inside=k&(f>lo)&(f<hi)
print('2025 no-NM deps with dst LOBT', k.sum(), ' in P', (k&P).sum(), ' OOF pred inside excluded window', inside.sum())
# move to nearest boundary
g=f.copy(); g[inside]=np.where(f[inside]-lo[inside]<hi[inside]-f[inside], np.maximum(lo[inside],0), hi[inside])
sse=(y-f)**2-(y-g)**2
print(' nearest-boundary move: SSE saved %.3e, rows better %d worse %d'%(sse[inside].sum(), (sse[inside]>0).sum(), (sse[inside]<0).sum()))
r=lambda h,m_: np.sqrt(np.mean((y[m_]-h[m_])**2)); held=~np.isin(month,[3,6,9,12]); S=np.isin(month,[1,7])
print(' held %+.3f scored %+.3f'%(r(g,held)-r(f,held), r(g,S)-r(f,S)))
rk=pd.read_parquet(ROOT+'ranking.parquet',columns=cols); dep2,off2=dl(rk)
s=pd.read_parquet(ROOT+'prc2026/outputs/understanding-accordion_v27.parquet'); p=dep2[['MVT_ID_mvt']].merge(s,on='MVT_ID_mvt',how='left').TAXITIME_SEC_mvt.to_numpy(float)
k2=~np.isnan(off2); ins2=k2&(p>off2-3606)&(p<off2+3606)
print('2026 no-NM deps with dst LOBT', k2.sum(), ' v27 pred inside excluded window', ins2.sum(), ' preds', np.round(p[ins2]).tolist()[:20], ' windows lo', np.round(off2[ins2]-3606).tolist()[:20])
