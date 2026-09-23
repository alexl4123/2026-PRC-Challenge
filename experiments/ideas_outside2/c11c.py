"""Blend-member screen: global LGB (seed0) + {global seed1 | per-airport LGB | log-target LGB w/ per-airport smearing}.
Features: 72 base + stand. 300k clean-bulk training subsample from the other months; test = clean bulk of month m."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
t0=time.time()
X=pd.read_parquet(C+'altfam_X.parquet'); S=pd.read_parquet(OUT+'c4_stand.parquet')
stand=['sl_to','sl_after_sched','sl_same_typ','sl_same_op','si_before_sched','si_same_typ','sn_minus_to','sn_same_op','sn_same_typ','sa_after_to','sd_next_to','sd_prev_to']
for c in stand: X[c]=S[c].values.astype('float32')
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy')
P=(X.aobt_missing.values==1)&(X.off_sched.values>7200); clean=(y<=7200)&~P
cats=['APT','RUNWAY_mvt','PIER','STAND_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt','MARKET_SEGMENT_flt','WK_TBL_CAT_flt']
cols=list(X.columns)
PAR=dict(objective='regression',learning_rate=0.05,num_leaves=127,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,
         bagging_freq=1,lambda_l2=1.0,num_threads=2,verbose=-1,deterministic=True,force_row_wise=True)
R=int(sys.argv[3])
m=int(sys.argv[1]); rng=np.random.default_rng(0)
tr=np.sort(rng.choice(np.where(clean&(mon!=m))[0],int(sys.argv[2]),replace=False)); te=np.where(clean&(mon==m))[0]
Xtr,Xte=X.iloc[tr],X.iloc[te]; apt_tr=Xtr.APT.astype(str).values; apt_te=Xte.APT.astype(str).values
pr={}
pr['g0']=lgb.train(dict(PAR,seed=0),lgb.Dataset(Xtr,y[tr],categorical_feature=cats),R).predict(Xte); print('g0 %.0fs'%(time.time()-t0),flush=True)
pr['g1']=lgb.train(dict(PAR,seed=7,bagging_seed=8,feature_fraction_seed=9),lgb.Dataset(Xtr,y[tr],categorical_feature=cats),10*R).predict(Xte); print('g1 %.0fs'%(time.time()-t0),flush=True)
pa=np.zeros(len(te))
for a in np.unique(apt_tr):
    i=apt_tr==a; j=apt_te==a
    bst=lgb.train(dict(PAR,seed=0),lgb.Dataset(Xtr[i].drop(columns=['APT']),y[tr][i],categorical_feature=[c for c in cats if c!='APT']),R)
    pa[j]=bst.predict(Xte[j].drop(columns=['APT']))
pr['apt']=pa; print('apt %.0fs'%(time.time()-t0),flush=True)
yt=y[te]; rm=lambda p: np.sqrt(np.mean((yt-p)**2))
print(f'm{m:02d} solo: '+'  '.join(f'{k} {rm(v):.2f}' for k,v in pr.items()))
for k in ['g1','apt']:
    r0=yt-pr['g0']; r1=yt-pr[k]
    best=min(((w,rm((1-w)*pr['g0']+w*pr[k])) for w in np.linspace(0,1,11)),key=lambda t:t[1])
    print(f'  g0+{k}: corr(resid) {np.corrcoef(r0,r1)[0,1]:.3f}  50/50 {rm(0.5*pr["g0"]+0.5*pr[k])-rm(pr["g0"]):+.2f}  best w {best[0]:.1f} {best[1]-rm(pr["g0"]):+.2f} (in-sample w)')
print('peak rss %.2f GB, %.0fs'%(peak_rss_gb(),time.time()-t0))
print('  3-way g0+g1+apt equal: %+.2f'%(rm((pr['g0']+pr['g1']+pr['apt'])/3)-rm(pr['g0'])))

print('  gbig(=g1, 10x rounds)+apt 50/50: %+.2f vs g0'%(rm(0.5*pr['g1']+0.5*pr['apt'])-rm(pr['g0'])))
print('  corr(gbig resid, apt resid) %.3f'%np.corrcoef(yt-pr['g1'],yt-pr['apt'])[0,1])
