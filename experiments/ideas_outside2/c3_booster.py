"""LOMO residual booster: does a feature family carry information beyond v6 (and v8-like)?
null = (pred, apt, hour, off_sched, aobt_missing); arms add families. 4 held-out months, 600k-row training subsample."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
t0=time.time()
F=pd.read_parquet(OUT+'c2_features.parquet')
E=pd.Timestamp('1970-01-01',tz='UTC')
dep=load('training',phase='DEP',columns=['APT','RUNWAY_mvt','MVT_TIME_UTC_mvt','FLIGHT_mvt'])
t=(dep.MVT_TIME_UTC_mvt-E).dt.total_seconds().values; N=len(t)
# forward counts: takeoffs on our runway / airport in (t, t+w]
for key,name in [(dep.APT.astype(str)+'|'+dep.RUNWAY_mvt.astype(str),'rwy'),(dep.APT.astype(str),'apt')]:
    s=pd.DataFrame({'k':key.values,'t':t,'i':np.arange(N)}).sort_values(['k','t'])
    kc=pd.factorize(s.k)[0]; tt=s.t.values
    # position within key via searchsorted on combined (kc, t)
    comb=kc*1e10+tt
    for w in (300,600,900):
        hi=np.searchsorted(comb,comb+w,side='right'); lo=np.searchsorted(comb,comb,side='right')
        F.loc[s.i.values,f'fwd_{name}_{w//60}m']=hi-lo
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy')
o=np.load(C+'exp62_oof.npy'); b=np.load(C+'blend_oof_mix_nn3.npy')
P=(F.aobt_missing.values==1)&(F.off_sched.values>7200)
clean=(y<=7200)&~P
fam={
 'null': ['pred','apt','hour','off_sched','aobt_missing'],
}
fam['busy']=fam['null']+['bp120_pos','bp120_elapsed','bp120_remaining','bp180_pos','bp180_elapsed','bp180_remaining','gap_prev','gap_next','gap_prev3_mean']
fam['fwd']=fam['null']+['fwd_rwy_5m','fwd_rwy_10m','fwd_rwy_15m','fwd_apt_5m','fwd_apt_10m','fwd_apt_15m','gap_next','bp180_remaining','bp120_remaining']
fam['stand']=fam['null']+['stand_lastarr_to_takeoff','stand_lastarr_after_sched','stand_nextarr_after_takeoff','stand_arr_sched_to_to']
fam['ades']=fam['null']+['ades']
fam['all']=sorted(set(sum(fam.values(),[])))
PAR=dict(objective='regression',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.9,
         bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,num_threads=2,verbose=-1,seed=0,cat_smooth=50,min_data_per_group=200)
rng=np.random.default_rng(0)
BASES={'v6':o[1],'v8like':0.5*o[1]+0.5*b[1]}
MONTHS=[int(x) for x in sys.argv[2].split(',')]
NTR=int(sys.argv[3]) if len(sys.argv)>3 else 300000
for bname,pred in [(sys.argv[1],BASES[sys.argv[1]])]:
    F['pred']=pred; res=y-pred
    print(f'\n=== base {bname}',flush=True)
    tot={a:[0,0] for a in fam}; base_sse=0; nrows=0
    for m in MONTHS:
        tr=np.where(clean&(mon!=m))[0]; tr=rng.choice(tr,NTR,replace=False)
        te=np.where(clean&(mon==m))[0]
        b0=np.sum(res[te]**2); base_sse+=b0; nrows+=len(te)
        line=f'  m{m:02d} base {np.sqrt(b0/len(te)):.2f}'
        for a,cols in fam.items():
            ds=lgb.Dataset(F.iloc[tr][cols],res[tr],categorical_feature=[c for c in ['apt','ades'] if c in cols],free_raw_data=True)
            bst=lgb.train(PAR,ds,num_boost_round=200)
            c=bst.predict(F.iloc[te][cols])
            s1=np.sum((res[te]-c)**2); tot[a][0]+=s1
            line+=f' | {a} {np.sqrt(s1/len(te))-np.sqrt(b0/len(te)):+.2f}'
        print(line,flush=True)
    b=np.sqrt(base_sse/nrows)
    print('  POOLED: base %.2f  '%b+'  '.join(f'{a} {np.sqrt(tot[a][0]/nrows)-b:+.3f}' for a in fam))
    print('  incremental over null: '+'  '.join(f'{a} {np.sqrt(tot[a][0]/nrows)-np.sqrt(tot["null"][0]/nrows):+.3f}' for a in fam if a!='null'),flush=True)
print('peak rss %.2f GB, %.0fs'%(peak_rss_gb(),time.time()-t0))
