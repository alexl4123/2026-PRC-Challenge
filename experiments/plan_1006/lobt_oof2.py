import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
from prc.config import CACHE
tr=load('training', phase='DEP', columns=['MVT_TIME_UTC_mvt','LOBT_flt','TAXITIME_SEC_mvt','ADEP_mvt','SCHED_TIME_UTC_mvt']).reset_index(drop=True)
month=tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
ml=(tr.MVT_TIME_UTC_mvt-tr.LOBT_flt).dt.total_seconds().to_numpy()
y=tr.TAXITIME_SEC_mvt.astype(float).to_numpy()
i=np.where(ml>40000)[0]; print('analogue rows', i, month[i], ml[i], y[i], (tr.MVT_TIME_UTC_mvt-tr.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()[i])
lo=np.clip(ml-3606,0,None); hi=ml+3606
fams=['mix','routed','lintree','nn','xgboost','perapt','w3']
P=[]
for fam in fams:
    a=np.full(len(tr),np.nan)
    for m in range(1,13):
        a[month==m]=np.load(CACHE/'cv'/'adsb2'/f'{fam}__seed00__m{m:02d}.npy').astype(float)
    P.append(a)
P=np.array(P); print('member OOF on analogue row:', dict(zip(fams, P[:,i[0]].round(0))))
avg=np.clip(P.mean(0),0,None)
q=np.where(np.isnan(ml),avg,np.clip(avg,lo,hi))
bad=q!=avg; jj=np.isin(month,[1,7])
r=lambda v,m: np.sqrt(np.mean((v[m]-y[m])**2))
print(f'equal-weight 7-member avg: viol={bad.sum()} (Jan+Jul {(bad&jj).sum()}); 12m {r(avg,np.ones(len(y),bool)):.3f}->{r(q,np.ones(len(y),bool)):.3f}; Jan+Jul {r(avg,jj):.3f}->{r(q,jj):.3f}; held-out(not 1,7) {r(avg,~jj):.3f}->{r(q,~jj):.3f}')
d=(avg-y)**2-(q-y)**2; o=np.sort(d[bad])[::-1]
print('gain concentration: top1 share', o[0]/o.sum(), 'top10', o[:10].sum()/o.sum(), 'n gaining rows', (o>0).sum())
