import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
from prc.config import CACHE
tr=load('training', phase='DEP', columns=['MVT_TIME_UTC_mvt','LOBT_flt','TAXITIME_SEC_mvt','ADEP_mvt']).reset_index(drop=True)
month=tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
ml=(tr.MVT_TIME_UTC_mvt-tr.LOBT_flt).dt.total_seconds().to_numpy()
y=tr.TAXITIME_SEC_mvt.astype(float).to_numpy()
lo=np.clip(ml-3606,0,None); hi=ml+3606
for fam in ['mix','routed','lintree','nn','xgboost','perapt','w3']:
    P=np.full(len(tr),np.nan); ok=True
    for m in range(1,13):
        f=CACHE/'cv'/'adsb2'/f'{fam}__seed00__m{m:02d}.npy'
        if not f.exists(): ok=False; break
        a=np.load(f).astype(float); idx=np.where(month==m)[0]
        if len(a)!=len(idx): print(fam,m,'len mismatch',len(a),len(idx)); ok=False; break
        P[idx]=a
    if not ok: continue
    q=np.where(np.isnan(ml),P,np.clip(P,lo,hi)); q=np.clip(q,0,None); p0=np.clip(P,0,None)
    bad=q!=p0
    def r(v,msk): return np.sqrt(np.mean((v[msk]-y[msk])**2))
    jj=np.isin(month,[1,7])
    print(f'{fam:8s} viol={bad.sum():4d} (JanJul {(bad&jj).sum():3d}) 12m {r(p0,np.ones_like(bad)):.3f}->{r(q,np.ones_like(bad)):.3f}  Jan+Jul {r(p0,jj):.3f}->{r(q,jj):.3f}  worse rows={(np.abs(q-y)>np.abs(p0-y)).sum()}')
