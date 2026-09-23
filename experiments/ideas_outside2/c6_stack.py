"""Per-airport stacking of the LGB and NN bulk members, LOMO, on the clean bulk. Also per-airport linear recalibration."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy')
o=np.load(C+'exp62_oof.npy'); b=np.load(C+'blend_oof_mix_nn3.npy').astype(float)
d=load('training',phase='DEP',columns=['APT','AOBT_3_flt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'])
apt=d.APT.astype(str).values
os_=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds().values
routeD=d.AOBT_3_flt.isna().values
P=routeD&(os_>7200); clean=(y<=7200)&~P
L,Nn=b[0],b[1]; base=0.5*L+0.5*Nn
def rm(p,m): return np.sqrt(np.mean((y[m]-p[m])**2))
print('clean bulk: LGB %.2f  NN %.2f  50/50 %.2f  corr(resid) %.3f'%(rm(L,clean),rm(Nn,clean),rm(base,clean),np.corrcoef(y[clean]-L[clean],y[clean]-Nn[clean])[0,1]))
arms={}
# A: global LOMO optimal weight; B: per-airport weight; C: per-airport x route(A/D) weight; D: per-airport linear recal a+b*pred of 50/50
out={k:base.copy() for k in ['glob_w','apt_w','aptroute_w','apt_recal','apt_w_recal']}
seg_ar=np.char.add(apt.astype('U4'),np.where(routeD,'D','A'))
for m in range(1,13):
    tr=clean&(mon!=m); te=mon==m
    # global w
    dl=L-Nn; r=y-Nn
    w=np.sum(dl[tr]*r[tr])/np.sum(dl[tr]**2); out['glob_w'][te]=Nn[te]+w*dl[te]
    for seg,key in [(apt,'apt_w'),(seg_ar,'aptroute_w')]:
        for s in np.unique(seg):
            a=tr&(seg==s); t=te&(seg==s)
            if a.sum()<100: continue
            w=np.clip(np.sum(dl[a]*r[a])/np.sum(dl[a]**2),0,1); out[key][t]=Nn[t]+w*dl[t]
    for s in np.unique(apt):
        a=tr&(apt==s); t=te&(apt==s)
        X=np.c_[np.ones(a.sum()),base[a]]; coef=np.linalg.lstsq(X,y[a],rcond=None)[0]
        out['apt_recal'][t]=coef[0]+coef[1]*base[t]
        X=np.c_[np.ones(a.sum()),L[a],Nn[a]]; coef=np.linalg.lstsq(X,y[a],rcond=None)[0]
        out['apt_w_recal'][t]=coef[0]+coef[1]*L[t]+coef[2]*Nn[t]
sc=clean&np.isin(mon,[1,7])
print('%-12s %9s %9s'%('arm','12m','m1+7'))
print('%-12s %9.3f %9.3f'%('50/50',rm(base,clean),rm(base,sc)))
for k,p in out.items(): print('%-12s %+9.3f %+9.3f'%(k,rm(p,clean)-rm(base,clean),rm(p,sc)-rm(base,sc)))
