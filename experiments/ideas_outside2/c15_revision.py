"""NASA-2nd-place analogue: revision of the off-block estimate (IOBT -> EOBT_1 -> LOBT, vs SCHED and AOBT_3). Residual booster vs null."""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
from scipy.stats import spearmanr
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
d=load('training',phase='DEP',columns=['APT','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','LOBT_flt','IOBT_flt'])
F=pd.read_parquet(OUT+'c2_features.parquet',columns=['apt','hour','off_sched','aobt_missing'])
sec=lambda a,b:(d[a]-d[b]).dt.total_seconds().values
F['rev_lobt_eobt']=sec('LOBT_flt','EOBT_1_flt'); F['rev_lobt_iobt']=sec('LOBT_flt','IOBT_flt'); F['rev_eobt_iobt']=sec('EOBT_1_flt','IOBT_flt')
F['iobt_sched']=sec('IOBT_flt','SCHED_TIME_UTC_mvt'); F['lobt_sched']=sec('LOBT_flt','SCHED_TIME_UTC_mvt'); F['aobt_lobt']=sec('AOBT_3_flt','LOBT_flt'); F['off_iobt']=sec('MVT_TIME_UTC_mvt','IOBT_flt')
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy'); v6=np.load(C+'exp62_oof.npy')[1]; res=y-v6; F['pred']=v6
P=(F.aobt_missing.values==1)&(F.off_sched.values>7200); clean=(y<=7200)&~P
newf=['rev_lobt_eobt','rev_lobt_iobt','rev_eobt_iobt','iobt_sched','lobt_sched','aobt_lobt','off_iobt']
sub=np.random.default_rng(0).choice(np.where(clean)[0],400000,replace=False)
for c in newf:
    x=F[c].values[sub]; ok=np.isfinite(x)
    print(f'  {c:14s} cover {ok.mean():.3f} rho(resid) {spearmanr(x[ok],res[sub][ok]).correlation:+.3f} rho(y) {spearmanr(x[ok],y[sub][ok]).correlation:+.3f} nonzero {np.mean(x[ok]!=0):.3f}')
null=['pred','apt','hour','off_sched','aobt_missing']
PAR=dict(objective='regression',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,num_threads=2,verbose=-1,seed=0)
rng=np.random.default_rng(0); tot={'null':0,'rev':0}; nr=0
for m in (1,7):
    tr=rng.choice(np.where(clean&(mon!=m))[0],300000,replace=False); te=np.where(clean&(mon==m))[0]; nr+=len(te)
    for a,cols in [('null',null),('rev',null+newf)]:
        b=lgb.train(PAR,lgb.Dataset(F.iloc[tr][cols],res[tr],categorical_feature=['apt']),200); tot[a]+=np.sum((res[te]-b.predict(F.iloc[te][cols]))**2)
print('revision family incremental over null (m1+7): %+.3f'%(np.sqrt(tot['rev']/nr)-np.sqrt(tot['null']/nr)))
