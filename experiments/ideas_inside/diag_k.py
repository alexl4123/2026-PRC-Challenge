from common import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
cols = ['APT', 'FLIGHT_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'ADES_mvt',
        'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt']
tr = load_dep(cols + ['TAXITIME_SEC_mvt']); rk = load_dep(cols, 'ranking')
# 1. unseen categories in 2026 (ranking) relative to all of 2025, per airport
print('=== share of 2026 ranking DEP rows whose category was never seen in 2025 (per airport)')
out = {}
for c in ['STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt', 'ADES_mvt']:
    ktr = set(K(S_(tr.APT), S_(tr[c]))); krk = K(S_(rk.APT), S_(rk[c]))
    out[c] = pd.Series(~np.isin(krk, list(ktr))).groupby(S_(rk.APT)).mean() * 100
print(pd.DataFrame(out).round(2).to_string())
# 2. adversarial validation: 2025 Jan+Jul vs 2026 Jan+Jul, per airport, on the model's observable inputs
def fe(d):
    M = d.MVT_TIME_UTC_mvt.to_numpy()
    f = pd.DataFrame(dict(off_aobt3=M - d.AOBT_3_flt, off_sched=M - d.SCHED_TIME_UTC_mvt, off_eobt=M - d.EOBT_1_flt,
                          off_lobt=M - d.LOBT_flt, nm_delay=d.AOBT_3_flt - d.SCHED_TIME_UTC_mvt,
                          hour=pd.to_datetime(M, unit='s').hour, month=pd.to_datetime(M, unit='s').month))
    for c in ['STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']:
        f[c] = S_(d[c])
    f['apt'] = S_(d.APT)
    return f
a = fe(tr); a = a[np.isin(a.month, [1, 7])]; a['is26'] = 0
b = fe(rk); b['is26'] = 1
ab = pd.concat([a, b], ignore_index=True)
for c in ['STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']:
    ab[c] = pd.Categorical(ab[c]).codes
feats = ['off_aobt3', 'off_sched', 'off_eobt', 'off_lobt', 'nm_delay', 'hour', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']
print('\n=== adversarial AUC 2025 vs 2026 (same months), per airport x month, 2-fold, + top features')
prm = dict(objective='binary', learning_rate=0.1, num_leaves=31, min_data_in_leaf=100, verbose=-1, num_threads=2, seed=0)
for ap in sorted(ab.apt.unique()):
    for mm in [1, 7]:
        s = ab[(ab.apt == ap) & (ab.month == mm)].reset_index(drop=True)
        rng = np.random.default_rng(0); fold = rng.integers(0, 2, len(s)); pr = np.zeros(len(s)); gains = 0
        for k in [0, 1]:
            m = lgb.train(prm, lgb.Dataset(s.loc[fold != k, feats], s.is26[fold != k], categorical_feature=['STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']), 100)
            pr[fold == k] = m.predict(s.loc[fold == k, feats]); gains = gains + m.feature_importance('gain')
        g = pd.Series(gains, index=feats); g = (g / g.sum()).sort_values(ascending=False)
        print(f'  {ap} m{mm}: AUC {roc_auc_score(s.is26, pr):.3f}  top {g.head(3).round(2).to_dict()}')
# 3. route D, not P, 1800 < off_sched <= 7200 : artefact share and oracle
y = tr.TAXITIME_SEC_mvt.to_numpy(); M = tr.MVT_TIME_UTC_mvt.to_numpy(); S = tr.SCHED_TIME_UTC_mvt.to_numpy()
os_ = M - S; miss = tr.AOBT_3_flt.isna().to_numpy(); month = pd.to_datetime(M, unit='s').month.to_numpy()
v6, v7, bl = preds()
pop = miss & (os_ > 1800) & (os_ <= 7200)
art = np.abs(y - os_) < 60
print(f'\n=== route D, 1800<off_sched<=7200: {pop.sum():,} rows, artefact share {art[pop].mean():.4f}, SSE share {100*((y-v6)[pop]**2).sum()/((y-v6)**2).sum():.2f}%')
print('   artefact share by airport:', (pd.Series(art[pop]).groupby(S_(tr.APT)[pop]).mean() * 100).round(1).to_dict())
new = v6.copy(); k = pop & art; new[k] = os_[k]
pbv = np.load(f'{R}/exp62_pb.npy').astype(float); k2 = pop & ~art; new2 = new.copy(); new2[k2] = pbv[k2]
sc = np.isin(month, [1, 7])
print(f'   oracle (artefact rows := off_sched): 12m {rmse(y, v6):.3f} -> {rmse(y, new):.3f}; scored {rmse(y[sc], v6[sc]):.3f} -> {rmse(y[sc], new[sc]):.3f}')
print(f'   v6 mean pred on artefact rows {v6[k].mean():.0f} vs y {y[k].mean():.0f}; on real rows {v6[k2].mean():.0f} vs y {y[k2].mean():.0f}')
print('peak', peak())
