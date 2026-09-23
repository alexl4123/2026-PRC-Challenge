"""Stand-turnover family, decomposed and refined (inbound vs next occupant by type/operator match)."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
t0=time.time()
E=pd.Timestamp('1970-01-01',tz='UTC')
cols=['APT','STAND_mvt','FLIGHT_mvt','AIRCRAFT_TYPE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','PHASE_mvt']
full=load('training',columns=cols)
isdep=full.PHASE_mvt.eq('DEP').values
dep=full[isdep].reset_index(drop=True).drop(columns=['BLOCK_TIME_UTC_mvt'])  # leakage guard
arr=full[~isdep].reset_index(drop=True); del full
N=len(dep)
t=(dep.MVT_TIME_UTC_mvt-E).dt.total_seconds().values; ts=(dep.SCHED_TIME_UTC_mvt-E).dt.total_seconds().values
pre=lambda s: s.astype(str).str[:3].values
ev=pd.DataFrame({'k':(arr.APT.astype(str)+'|'+arr.STAND_mvt.astype(str)).values,
                 'tb':(arr.BLOCK_TIME_UTC_mvt-E).dt.total_seconds().values,
                 'typ':arr.AIRCRAFT_TYPE_mvt.astype(str).values,'op':pre(arr.FLIGHT_mvt)}).dropna(subset=['tb']).sort_values('tb')
ok=dep.STAND_mvt.notna().values
q=pd.DataFrame({'k':(dep.APT.astype(str)+'|'+dep.STAND_mvt.astype(str)).values,'t':t,'ts':ts,'i':np.arange(N),
                'typ':dep.AIRCRAFT_TYPE_mvt.astype(str).values,'op':pre(dep.FLIGHT_mvt)})[ok]
F=pd.DataFrame(index=np.arange(N))
# last arrival before takeoff, and the one before it
e1=ev.rename(columns={'tb':'tl','typ':'ltyp','op':'lop'})
m=pd.merge_asof(q.sort_values('t'),e1,left_on='t',right_on='tl',by='k',direction='backward')
F.loc[m.i.values,'sl_to']=(m.t-m.tl).values
F.loc[m.i.values,'sl_after_sched']=(m.tl-m.ts).values
F.loc[m.i.values,'sl_same_typ']=(m.ltyp==m.typ).astype(float).where(m.tl.notna()).values
F.loc[m.i.values,'sl_same_op']=(m.lop==m.op).astype(float).where(m.tl.notna()).values
# last arrival before our SCHED (inbound candidate)
m=pd.merge_asof(q.sort_values('ts'),e1,left_on='ts',right_on='tl',by='k',direction='backward')
F.loc[m.i.values,'si_before_sched']=(m.ts-m.tl).values
F.loc[m.i.values,'si_same_typ']=(m.ltyp==m.typ).astype(float).where(m.tl.notna()).values
# first arrival after our SCHED (next-occupant candidate), relative to takeoff; and its type/op match
m=pd.merge_asof(q.sort_values('ts'),e1,left_on='ts',right_on='tl',by='k',direction='forward',allow_exact_matches=False)
F.loc[m.i.values,'sn_minus_to']=(m.tl-m.t).values
F.loc[m.i.values,'sn_same_op']=(m.lop==m.op).astype(float).where(m.tl.notna()).values
F.loc[m.i.values,'sn_same_typ']=(m.ltyp==m.typ).astype(float).where(m.tl.notna()).values
# next arrival after takeoff
m=pd.merge_asof(q.sort_values('t'),e1,left_on='t',right_on='tl',by='k',direction='forward')
F.loc[m.i.values,'sa_after_to']=(m.tl-m.t).values
# next DEPARTURE from our stand (its takeoff) and previous departure (their takeoffs are public)
dq=pd.DataFrame({'k':q.k.values,'td':q.t.values}).sort_values('td')
m=pd.merge_asof(q.sort_values('t'),dq,left_on='t',right_on='td',by='k',direction='forward',allow_exact_matches=False)
F.loc[m.i.values,'sd_next_to']=(m.td-m.t).values
m=pd.merge_asof(q.sort_values('t'),dq,left_on='t',right_on='td',by='k',direction='backward',allow_exact_matches=False)
F.loc[m.i.values,'sd_prev_to']=(m.t-m.td).values
F.to_parquet(OUT+'c4_stand.parquet')
print('built %.0fs'%(time.time()-t0),flush=True)
G=pd.read_parquet(OUT+'c2_features.parquet',columns=['apt','hour','off_sched','aobt_missing','stand_lastarr_to_takeoff','stand_lastarr_after_sched','stand_nextarr_after_takeoff','stand_arr_sched_to_to'])
F=pd.concat([F,G],axis=1)
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy')
o=np.load(C+'exp62_oof.npy'); F['pred']=o[1]; res=y-o[1]
P=(F.aobt_missing.values==1)&(F.off_sched.values>7200); clean=(y<=7200)&~P
null=['pred','apt','hour','off_sched','aobt_missing']
fam={'null':null,
 'A_lastarr':null+['stand_lastarr_to_takeoff','stand_lastarr_after_sched'],
 'B_nextarr':null+['stand_nextarr_after_takeoff'],
 'C_count':null+['stand_arr_sched_to_to'],
 'stand4':null+['stand_lastarr_to_takeoff','stand_lastarr_after_sched','stand_nextarr_after_takeoff','stand_arr_sched_to_to'],
 'stand_ref':null+['sl_to','sl_after_sched','sl_same_typ','sl_same_op','si_before_sched','si_same_typ','sn_minus_to','sn_same_op','sn_same_typ','sa_after_to','stand_arr_sched_to_to'],
 'stand_ref_dep':null+['sl_to','sl_after_sched','sl_same_typ','sl_same_op','si_before_sched','si_same_typ','sn_minus_to','sn_same_op','sn_same_typ','sa_after_to','stand_arr_sched_to_to','sd_next_to','sd_prev_to'],
}
PAR=dict(objective='regression',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.9,
         bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,num_threads=2,verbose=-1,seed=0,cat_smooth=50,min_data_per_group=200)
rng=np.random.default_rng(0)
tot={a:0 for a in fam}; bs=0; nr=0
for m in [int(x) for x in sys.argv[1].split(',')]:
    tr=rng.choice(np.where(clean&(mon!=m))[0],300000,replace=False); te=np.where(clean&(mon==m))[0]
    b0=np.sum(res[te]**2); bs+=b0; nr+=len(te); line=f'm{m:02d} base {np.sqrt(b0/len(te)):.2f}'
    for a,cols in fam.items():
        bst=lgb.train(PAR,lgb.Dataset(F.iloc[tr][cols],res[tr],categorical_feature=['apt']),num_boost_round=200)
        s1=np.sum((res[te]-bst.predict(F.iloc[te][cols]))**2); tot[a]+=s1
        line+=f' | {a} {np.sqrt(s1/len(te))-np.sqrt(b0/len(te)):+.2f}'
        if a=='stand_ref_dep' and m==int(sys.argv[1].split(',')[0]):
            imp=pd.Series(bst.feature_importance('gain'),index=cols).sort_values(ascending=False); imp=imp/imp.sum()
    print(line,flush=True)
print('incremental over null (pooled): '+'  '.join(f'{a} {np.sqrt(tot[a]/nr)-np.sqrt(tot["null"]/nr):+.3f}' for a in fam if a!='null'))
print('gain share stand_ref_dep:',imp.round(3).to_dict())
print('peak rss %.2f GB, %.0fs'%(peak_rss_gb(),time.time()-t0))
