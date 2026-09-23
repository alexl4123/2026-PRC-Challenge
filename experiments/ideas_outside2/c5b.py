import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
t0=time.time()
F=pd.concat([pd.read_parquet(OUT+'c4_stand.parquet'),pd.read_parquet(OUT+'c2_features.parquet',columns=['apt','hour','off_sched','aobt_missing','stand_arr_sched_to_to'])],axis=1)
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy')
o=np.load(C+'exp62_oof.npy'); b=np.load(C+'blend_oof_mix_nn3.npy')
pred={'v6':o[1],'v8like':0.5*o[1]+0.5*b[1]}[sys.argv[1]]
F['pred']=pred; res=y-pred
P=(F.aobt_missing.values==1)&(F.off_sched.values>7200); clean=(y<=7200)&~P
null=['pred','apt','hour','off_sched','aobt_missing']
stand=['sl_to','sl_after_sched','sl_same_typ','sl_same_op','si_before_sched','si_same_typ','sn_minus_to','sn_same_op','sn_same_typ','sa_after_to','stand_arr_sched_to_to','sd_next_to','sd_prev_to']
PAR=dict(objective='regression',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.9,
         bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,num_threads=2,verbose=-1,seed=0,cat_smooth=50,min_data_per_group=200)
rng=np.random.default_rng(1)
rows=[]
for m in [int(x) for x in sys.argv[2].split(',')]:
    tr=rng.choice(np.where(clean&(mon!=m))[0],300000,replace=False); te=np.where(clean&(mon==m))[0]
    out={}
    for a,cols in [('null',null),('stand',null+stand)]:
        bst=lgb.train(PAR,lgb.Dataset(F.iloc[tr][cols],res[tr],categorical_feature=['apt']),num_boost_round=200)
        out[a]=res[te]-bst.predict(F.iloc[te][cols])
    rows.append(pd.DataFrame({'D':F.aobt_missing.values[te],'os':F.off_sched.values[te],'m':m,'apt':F.apt.values[te].astype(str),'y':y[te],'r0':res[te],'rn':out['null'],'rs':out['stand']}))
    print(f"m{m:02d} base {np.sqrt(np.mean(res[te]**2)):.2f} null {np.sqrt(np.mean(out['null']**2))-np.sqrt(np.mean(res[te]**2)):+.2f} stand {np.sqrt(np.mean(out['stand']**2))-np.sqrt(np.mean(res[te]**2)):+.2f} stand-null {np.sqrt(np.mean(out['stand']**2))-np.sqrt(np.mean(out['null']**2)):+.2f}",flush=True)
R=pd.concat(rows)
R['band']=pd.cut(R.y,[-1,900,1800,3600,7200],labels=['<15m','15-30m','30-60m','1-2h'])
tot=np.sum(R.r0**2)
g=R.groupby('band',observed=True).apply(lambda d: pd.Series({'rows':len(d)/len(R),'dSSE_stand_vs_base_%':100*(np.sum(d.rs**2)-np.sum(d.r0**2))/tot,'dSSE_stand_vs_null_%':100*(np.sum(d.rs**2)-np.sum(d.rn**2))/tot}))
print(g.round(3))
g=R.groupby('apt').apply(lambda d: pd.Series({'rows':len(d),'rmse0':np.sqrt(np.mean(d.r0**2)),'d_stand':np.sqrt(np.mean(d.rs**2))-np.sqrt(np.mean(d.r0**2)),'d_stand_vs_null':np.sqrt(np.mean(d.rs**2))-np.sqrt(np.mean(d.rn**2))}))
print(g.round(2))
# day-cluster bootstrap for stand vs base and stand vs null
R['d0']=R.rs**2-R.r0**2; R['dn']=R.rs**2-R.rn**2
g=R.groupby('D').apply(lambda d: pd.Series({'rows':len(d),'rmse0':np.sqrt(np.mean(d.r0**2)),'d_stand':np.sqrt(np.mean(d.rs**2))-np.sqrt(np.mean(d.r0**2)),'d_stand_vs_null':np.sqrt(np.mean(d.rs**2))-np.sqrt(np.mean(d.rn**2)),'dSSE_share_of_total_%':100*(np.sum(d.rs**2)-np.sum(d.r0**2))/tot}))
print(g.round(2))
print('peak rss %.2f GB, %.0fs'%(peak_rss_gb(),time.time()-t0))
