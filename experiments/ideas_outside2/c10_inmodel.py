"""Stand-turnover (+ optional forward-runway) features INSIDE a bulk LightGBM (72 v5/v6 features), subsample screen.
Arms: base72 vs base72+stand (vs base72+stand+fwd). Train on a 400k clean-bulk subsample of the other 11 months, test month m clean bulk."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
t0=time.time()
X=pd.read_parquet(C+'altfam_X.parquet')
S=pd.read_parquet(OUT+'c4_stand.parquet')
stand=['sl_to','sl_after_sched','sl_same_typ','sl_same_op','si_before_sched','si_same_typ','sn_minus_to','sn_same_op','sn_same_typ','sa_after_to','sd_next_to','sd_prev_to']
extra=sys.argv[2] if len(sys.argv)>2 else 'stand'
for c in stand: X[c]=S[c].values.astype('float32')
if extra=='rev':
    from prc.io import load as _ld
    d=_ld('training',phase='DEP',columns=['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','LOBT_flt','IOBT_flt'])
    sec=lambda a,b:(d[a]-d[b]).dt.total_seconds().values.astype('float32')
    REV={'rev_lobt_eobt':sec('LOBT_flt','EOBT_1_flt'),'rev_eobt_iobt':sec('EOBT_1_flt','IOBT_flt'),'lobt_sched':sec('LOBT_flt','SCHED_TIME_UTC_mvt'),'aobt_lobt':sec('AOBT_3_flt','LOBT_flt'),'aobt_eobt':sec('AOBT_3_flt','EOBT_1_flt')}
    for k,v in REV.items(): X[k]=v
if extra=='overlap':
    from prc.io import load as _ld
    d=_ld('training',phase='DEP',columns=['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','LOBT_flt','IOBT_flt'])
    sec=lambda a,b:(d[a]-d[b]).dt.total_seconds().values.astype('float32')
    REV={'rev_lobt_eobt':sec('LOBT_flt','EOBT_1_flt'),'rev_eobt_iobt':sec('EOBT_1_flt','IOBT_flt'),'lobt_sched':sec('LOBT_flt','SCHED_TIME_UTC_mvt'),'aobt_lobt':sec('AOBT_3_flt','LOBT_flt'),'aobt_eobt':sec('AOBT_3_flt','EOBT_1_flt')}
    for k,v in REV.items(): X[k]=v
    ST=pd.read_parquet('/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/ideas_inside/stampfeat_training.parquet')
    STC=['pa_stamped','pa_det','pa_inb_m_sibt','pa_ldg_m_sibt','aday_stamp','aday_n','opday_stamp','opday_n','nm_delay']
    for c in STC: X['ins_'+c]=ST[c].values.astype('float32')
if extra=='norm':
    Sn=pd.read_parquet(OUT+'c4_standnorm.parquet')
    for c in stand: X[c+'_n']=Sn[c].values.astype('float32')
if extra=='fwd':
    G=pd.read_parquet(OUT+'c2_features.parquet',columns=['gap_next','bp120_remaining','bp180_remaining','bp180_pos','bp180_elapsed'])
    for c in G.columns: X[c]=G[c].values.astype('float32')
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy')
P=(X.aobt_missing.values==1)&(X.off_sched.values>7200); clean=(y<=7200)&~P
cats=['APT','RUNWAY_mvt','PIER','STAND_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt','MARKET_SEGMENT_flt','WK_TBL_CAT_flt']
base=[c for c in X.columns if c not in stand and not c.endswith('_n') and not c.startswith(('rev_','lobt_sched','aobt_lobt','aobt_eobt','ins_')) and c not in ('gap_next','bp120_remaining','bp180_remaining','bp180_pos','bp180_elapsed')]
assert len(base)==72, len(base)
arms={'base72':base,'+stand':base+stand}
if extra=='overlap': arms={'+inside(a,c)':base+list(REV)+['ins_'+c for c in STC],'+inside(a,c)+stand':base+list(REV)+['ins_'+c for c in STC]+stand}
if extra=='rev': arms={'+stand':base+stand,'+stand+rev':base+stand+list(REV)}
if extra=='norm': arms={'+stand':base+stand,'+stand+norm':base+stand+[c+'_n' for c in stand]}
if extra=='fwd': arms={'+stand':base+stand,'+stand+fwd':base+stand+['gap_next','bp120_remaining','bp180_remaining','bp180_pos','bp180_elapsed']}
PAR=dict(objective='regression',learning_rate=0.05,num_leaves=255,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,
         bagging_freq=1,lambda_l2=1.0,num_threads=2,verbose=-1,seed=0,deterministic=True,force_row_wise=True)
rng=np.random.default_rng(0)
res={}
for m in [int(v) for v in sys.argv[1].split(',')]:
    tr=np.sort(rng.choice(np.where(clean&(mon!=m))[0],400000,replace=False)); te=np.where(clean&(mon==m))[0]
    for a,cols in arms.items():
        bst=lgb.train(PAR,lgb.Dataset(X.iloc[tr][cols],y[tr],categorical_feature=[c for c in cats if c in cols]),num_boost_round=500)
        p=bst.predict(X.iloc[te][cols]); res[(m,a)]=(y[te]-p)
        print(f'm{m:02d} {a:12s} clean-bulk RMSE {np.sqrt(np.mean((y[te]-p)**2)):.2f}  ({time.time()-t0:.0f}s)',flush=True)
        if a!=list(arms)[0]:
            imp=pd.Series(bst.feature_importance('gain'),index=cols); imp=imp/imp.sum()
            print('   gain share of new features: %.3f; top: %s'%(imp[[c for c in cols if c not in arms[list(arms)[0]]]].sum(), imp.sort_values(ascending=False).head(8).round(3).to_dict()))
a0,a1=list(arms)
for m in sorted({k[0] for k in res}):
    r0,r1=res[(m,a0)],res[(m,a1)]
    # day-agnostic row bootstrap of the RMSE difference
    d=[];n=len(r0);rg=np.random.default_rng(1)
    for _ in range(200):
        i=rg.integers(0,n,n); d.append(np.sqrt(np.mean(r1[i]**2))-np.sqrt(np.mean(r0[i]**2)))
    print(f'm{m:02d} {a1} - {a0}: {np.sqrt(np.mean(r1**2))-np.sqrt(np.mean(r0**2)):+.2f}  95% row-bootstrap [{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}]')
print('peak rss %.2f GB, %.0fs'%(peak_rss_gb(),time.time()-t0))
