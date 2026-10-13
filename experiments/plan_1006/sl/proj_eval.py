import pandas as pd, numpy as np, glob, sys
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
sys.path.insert(0,ROOT+'prc2026')
from prc.validation import block_bootstrap
C=ROOT+'prc2026/cache/'
d=pd.concat([pd.read_parquet(f,columns=['PHASE_mvt','MVT_TIME_UTC_mvt','LOBT_flt']) for f in sorted(glob.glob(ROOT+'training_2025-*.parquet'))])
d=d[d.PHASE_mvt=='DEP'].reset_index(drop=True)
offl=(d.MVT_TIME_UTC_mvt-d.LOBT_flt).dt.total_seconds().to_numpy()
V=np.load(C+'w2/v24s_oof.npz',allow_pickle=True)
y,month,P,D,f=V['y'],V['month'],V['P'],V['D'],V['f']
rw=pd.read_parquet(C+'drivers_0926/rows_training.parquet',columns=['apt','date_loc'])
block=pd.factorize(rw.apt.astype(str)+'_'+rw.date_loc.astype(str))[0]
has=~np.isnan(offl); lo=np.where(has,np.maximum(offl-3606,0),np.nan); hiB=offl+3606
fc=f.copy(); fc[has]=np.clip(f[has],lo[has],hiB[has])
held=~np.isin(month,[3,6,9,12]); S=np.isin(month,[1,7])
r=lambda g,k: np.sqrt(np.mean((y[k]-g[k])**2))
ch=(fc!=f)&(f<np.nan_to_num(lo,nan=-1))
g=fc.copy(); dels={}
for m in range(1,13):
    tr=ch&(month!=m); dl=float(np.mean((y-lo)[tr])); dels[m]=round(dl)
    te=ch&(month==m); g[te]=np.minimum(lo[te]+dl, hiB[te])
print('LOMO deltas', dels)
sse=(y-fc)**2-(y-g)**2
print('delta-variant vs projection: held %+.3f scored %+.3f Jan %+.3f Jul %+.3f; rows worse %d of %d'%(r(g,held)-r(fc,held), r(g,S)-r(fc,S), r(g,month==1)-r(fc,month==1), r(g,month==7)-r(fc,month==7),(sse<-1e-6).sum(), ch.sum()))
for nm,k in [('held',held),('scored',S)]:
    b=block_bootstrap(y[k],g[k],fc[k],block[k],n_boot=1000); print(' ',nm, {kk:round(v,3) for kk,v in b.items()})
hi_=np.where(held)[0]; keep=np.ones(len(y),bool); keep[hi_[np.argsort(sse[held])[::-1][:100]]]=False
print('  R5 (vs projection) w/o top100 held %+.4f'%(r(g,held&keep)-r(fc,held&keep)), ' months better', sum(r(g,month==m)<r(fc,month==m) for m in range(1,13)))
# where are the binding rows: route/tail
print('binding rows: y>7200 share %.2f, off_lobt median %.0f, apt counts'%(np.mean(y[ch]>7200), np.median(offl[ch])), rw.apt[ch].value_counts().to_dict())
