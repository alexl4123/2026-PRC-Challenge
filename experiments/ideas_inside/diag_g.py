"""Stacked residual test for per-flight STAMPING propensity in route A delayed flights.
arm null : residual model on inputs v6 already has (apt, nm_delay, off_aobt3, off_sched, pb, hour)
arm full : null + stamping features (rotation, same-day arrival stamping, nested callsign/stand/operator TEs)
usage: diag_g.py <arm> ; saves corrections to diag_g_<arm>.npy"""
import sys, time
from common import *
import lightgbm as lgb
arm = sys.argv[1]
t0 = time.time()
d = load_dep(['APT', 'FLIGHT_mvt', 'STAND_mvt', 'AIRCRAFT_OPERATOR_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
B = d.BLOCK_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
month = pd.to_datetime(M, unit='s').month.to_numpy()
hour = pd.to_datetime(M, unit='s').hour.to_numpy()
apt = S_(d.APT)
v6, v7, bl = preds()
r = y - v6
dl = A3 - S
pop = (~np.isnan(A3)) & (dl > 300) & (M - A3 > 0) & (M - A3 <= 7200)
lab_mask = (~np.isnan(A3)) & (dl > 600) & (M - A3 > 0)
lab = (np.abs(B - S) < 60).astype(float)
print(f'pop rows {pop.sum():,}; label rows {lab_mask.sum():,}; stamp rate {lab[lab_mask].mean():.4f}', flush=True)
X = pd.DataFrame(dict(apt=pd.Categorical(apt).codes, nm_delay=dl, off_aobt3=M - A3, off_sched=M - S, pb=v6, hour=hour))
if arm == 'full':
    f = pd.read_parquet(f'{D}/stampfeat_training.parquet')
    for c in ['pa_stamped', 'pa_det', 'pa_inb_m_sibt', 'pa_ldg_m_sibt', 'aday_stamp', 'aday_n', 'opday_stamp', 'opday_n']:
        X[c] = f[c].to_numpy()
    del f
    enc = {}
    for nm, key in [('cs', K(apt, S_(d.FLIGHT_mvt))), ('stand', K(apt, S_(d.STAND_mvt))), ('op', K(apt, S_(d.AIRCRAFT_OPERATOR_flt)))]:
        c, u = pd.factorize(key)
        Sm = np.zeros((len(u), 13)); Nm = np.zeros((len(u), 13))
        np.add.at(Sm, (c[lab_mask], month[lab_mask]), lab[lab_mask]); np.add.at(Nm, (c[lab_mask], month[lab_mask]), 1)
        enc[nm] = (c, Sm, Nm)
    prior = lab[lab_mask].mean()
del d
feat_cols = list(X.columns) + (['te_cs', 'te_stand', 'te_op'] if arm == 'full' else [])
corr = np.full(len(y), np.nan)
params = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=0, deterministic=True, force_row_wise=True)
rng = np.random.default_rng(0)
for m in range(1, 13):
    va = pop & (month == m); tr = pop & (month != m) & (y <= 7200)
    tri = np.where(tr)[0]
    if len(tri) > 500_000:
        tri = np.sort(rng.choice(tri, 500_000, replace=False))
    vai = np.where(va)[0]
    Xm = X.iloc[np.concatenate([tri, vai])].copy()
    if arm == 'full':
        ii = np.concatenate([tri, vai]); k = month[ii]
        for nm, (c, Sm, Nm) in enc.items():
            cc = c[ii]
            s = Sm[cc].sum(1) - Sm[cc, m] - np.where(k != m, Sm[cc, k], 0)
            n = Nm[cc].sum(1) - Nm[cc, m] - np.where(k != m, Nm[cc, k], 0)
            Xm[f'te_{nm}'] = (s + 20 * prior) / (n + 20)
    Xtr, Xva = Xm.iloc[:len(tri)], Xm.iloc[len(tri):]
    mdl = lgb.train(params, lgb.Dataset(Xtr[feat_cols], np.clip(r[tri], -3000, 3000), categorical_feature=['apt']), num_boost_round=200)
    corr[vai] = mdl.predict(Xva[feat_cols])
    if m == 1 and arm == 'full':
        imp = pd.Series(mdl.feature_importance('gain'), index=feat_cols).sort_values(ascending=False)
        print((imp / imp.sum()).round(3).to_string())
    print(f'fold {m:02d} tr {len(tri):,} va {len(vai):,} {time.time()-t0:.0f}s peak {peak():.2f}', flush=True)
np.save(f'{D}/diag_g_{arm}.npy', corr)
print('done', time.time() - t0)
