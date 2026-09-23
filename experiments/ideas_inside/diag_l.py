"""P2 specialist test: route D & 1800 < off_sched <= 7200; label = artefact (|y - off_sched| < 60).
pred inside P2 = q*off_sched + (1-q)*base ; base = v6 or v8~ ; LOMO, TE in-fold via prc.p_features.te_features."""
import sys, time
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from common import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from prc.p_features import te_features, FAMILIES
t0 = time.time()
cand = pd.read_parquet(f'{D}/p2_cand_training.parquet'); pool = pd.read_parquet(f'{D}/p2_pool_training.parquet')
d = load_dep(['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
month_all = pd.to_datetime(M, unit='s').month.to_numpy(); P = d.AOBT_3_flt.isna().to_numpy() & (M - S > 7200)
Pidx = cand.index.to_numpy(); cand = cand.reset_index(drop=True)
yP = y[Pidx]; osP = cand.off_sched.to_numpy(float); art = (np.abs(yP - osP) < 60).astype(int); monP = cand.month.to_numpy()
pool_art = (np.abs(y[pool['pos'].to_numpy()] - pool.off_sched.to_numpy()) < 60).astype(float)
v6, v7, bl = preds(); v8 = np.where(P, v7, 0.5 * v7 + 0.5 * bl[1])
pb = np.load(f'{R}/exp62_pb.npy').astype(float)
EXTRA = [c for c in FAMILIES['all'] if c != 'sched_hour']
base_cols = ['apt_c', 'pier_c', 'stand_c', 'off_sched', 'sched_hour']
XB = pd.DataFrame(dict(apt_c=pd.Categorical(cand.apt).codes, pier_c=pd.Categorical(cand.k_pier).codes, stand_c=pd.Categorical(cand.k_stand).codes,
                       off_sched=osP, sched_hour=cand.sched_hour.to_numpy()))
XB = pd.concat([XB, cand[[c for c in EXTRA if not c.startswith('te_')]]], axis=1)
prm = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.9, bagging_fraction=0.9,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, deterministic=True, force_row_wise=True, num_threads=2)
arms = {'tbl': None, 'gbm_base': (base_cols, 'none'), 'gbm_x': (base_cols + EXTRA, 'none'), 'gbm_x_wabs': (base_cols + EXTRA, 'abs')}
Q = {k: np.full(len(cand), np.nan) for k in arms}
buckets = np.array([1800, 2700, 3600, 4800, 6000, 7201])
for m in range(1, 13):
    va, tr = monP == m, monP != m
    Xf = XB.copy(); pk = (pool.month != m).to_numpy()
    te_tr, te_va = te_features(cand[tr], art[tr], pool[pk], pool_art[pk], cand[va])
    for c in te_tr.columns:
        col = np.full(len(cand), np.nan); col[tr], col[va] = te_tr[c].to_numpy(), te_va[c].to_numpy(); Xf[c] = col
    # table: airport x off_sched bucket, Beta shrink (w=8) to airport rate
    b = np.digitize(osP, buckets); g = art[tr].mean()
    ar = pd.Series(art[tr]).groupby(cand.apt[tr].to_numpy()).agg(['sum', 'count'])
    arate = ((ar['sum'] + 8 * g) / (ar['count'] + 8)).to_dict()
    tb = pd.DataFrame(dict(a=cand.apt[tr].to_numpy(), b=b[tr], t=art[tr])).groupby(['a', 'b']).t.agg(['sum', 'count'])
    for i in np.where(va)[0]:
        a_ = cand.apt[i]; pa = arate.get(a_, g)
        s, n = (tb.loc[(a_, b[i])] if (a_, b[i]) in tb.index else (0, 0))
        Q['tbl'][i] = (s + 8 * pa) / (n + 8)
    gap = np.maximum(osP - pb[Pidx], 1.0)
    for nm, spec in arms.items():
        if spec is None: continue
        cols, wm = spec; w = gap / gap[tr].mean() if wm == 'abs' else np.ones(len(cand))
        mono = [1 if c == 'off_sched' else 0 for c in cols]
        ps = [lgb.train(dict(prm, seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2, monotone_constraints=mono),
                        lgb.Dataset(Xf.loc[tr, cols], art[tr], weight=w[tr], categorical_feature=['apt_c', 'pier_c', 'stand_c']), 200).predict(Xf.loc[va, cols])
              for s in (0, 1, 2)]
        Q[nm][va] = np.mean(ps, axis=0)
print(f'fitted {time.time()-t0:.0f}s peak {peak():.2f}; P2 rows {len(cand)}, artefacts {art.sum()}, LIRF rows {(cand.apt=="LIRF").sum()}')
np.save(f'{D}/diag_l_Q.npy', np.vstack([Q[k] for k in arms]))
sc = np.isin(month_all, [1, 7]); lir = cand.apt.to_numpy() == 'LIRF'
for bn, base in [('v6', v6), ('v8~', v8)]:
    print(f'--- base {bn}')
    for nm in arms:
        q = Q[nm]; new = base.copy(); new[Pidx] = q * osP + (1 - q) * base[Pidx]
        b1 = boot(y, new, base, 300); b2 = boot(y[sc], new[sc], base[sc], 300)
        print(f'  {nm:11s} AUC {roc_auc_score(art, q):.3f} LIRF {roc_auc_score(art[lir], q[lir]):.3f} Brier {np.mean((q-art)**2):.4f} | 12m d {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored d {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]')
print('base rate Brier', np.mean((art - art.mean()) ** 2))
