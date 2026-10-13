import pandas as pd, numpy as np, glob, sys
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
C=ROOT+'prc2026/cache/'
cols=['PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','TAXITIME_SEC_mvt']
d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(ROOT+'training_2025-*.parquet'))])
d=d[d.PHASE_mvt=='DEP'].reset_index(drop=True)
rw=pd.read_parquet(C+'drivers_0926/rows_training.parquet',columns=['apt','off_sched','y','route','date_loc'])
print(len(d),len(rw))
osch=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
print('off_sched match', np.nanmax(np.abs(osch-rw.off_sched.to_numpy())), 'y match', np.nanmax(np.abs(d.TAXITIME_SEC_mvt.to_numpy()-rw.y.to_numpy())))
offl=(d.MVT_TIME_UTC_mvt-d.LOBT_flt).dt.total_seconds().to_numpy()
V=np.load(C+'w2/v24s_oof.npz',allow_pickle=True)
y,month,P,D,f=V['y'],V['month'],V['P'],V['D'],V['f']
print('y vs npz', np.nanmax(np.abs(y-rw.y.to_numpy())))
has=~np.isnan(offl)
lo=offl-3606; hi=offl+3606
fc=f.copy(); fc[has]=np.clip(f[has],np.maximum(lo[has],0),hi[has])
ch=fc!=f
print('LOBT present', has.mean(), ' in P', (has&P).sum(), ' in D', (has&D).sum(), ' A/B/C w/o LOBT', (~has&~P&~D).sum())
print('rows changed by projection', ch.sum(), ' by month', np.bincount(month[ch].astype(int),minlength=13)[1:])
held=~np.isin(month,[3,6,9,12]); S=np.isin(month,[1,7])
r=lambda g,k: np.sqrt(np.mean((y[k]-g[k])**2))
for nm,k in [('12m',np.ones(len(y),bool)),('held',held),('scored',S),('Jan',month==1),('Jul',month==7)]:
    print(nm, round(r(f,k),3), '->', round(r(fc,k),3), 'delta', round(r(fc,k)-r(f,k),4))
sse=(y-f)**2-(y-fc)**2
print('SSE saved', sse[ch].sum(), ' n rows', ch.sum(), ' any worse?', (sse[ch]<-1e-6).sum())
idx=np.where(ch)[0]; o=np.argsort(-sse[idx])[:15]
print(pd.DataFrame({'apt':rw.apt.to_numpy()[idx[o]],'m':month[idx[o]],'route':rw.route.to_numpy()[idx[o]],'y':y[idx[o]],'f':f[idx[o]].round(),'fc':fc[idx[o]].round(),'off_lobt':offl[idx[o]],'off_sched':osch[idx[o]],'P':P[idx[o]],'D':D[idx[o]]}))
# distribution of violations
viol_lo=(f<lo)&has; viol_hi=(f>hi)&has
print('pred below lower bound', viol_lo.sum(), ' above upper', viol_hi.sum())
