"""Generic stacked-residual harness on ALL rows outside P (v6 residual, LOMO, train subsample 500k).
usage: diag_s.py <arm>   arms: null stamp clocks ident all"""
import sys, time
from common import *
import lightgbm as lgb
arm = sys.argv[1]; base_name = sys.argv[2] if len(sys.argv) > 2 else 'v6'
t0 = time.time()
d = load_dep(['APT', 'FLIGHT_mvt', 'ADES_mvt', 'STAND_mvt', 'AIRCRAFT_OPERATOR_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
              'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
B = d.BLOCK_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy(); E1 = d.EOBT_1_flt.to_numpy(); LO = d.LOBT_flt.to_numpy(); IO = d.IOBT_flt.to_numpy()
month = pd.to_datetime(M, unit='s').month.to_numpy(); hour = pd.to_datetime(M, unit='s').hour.to_numpy()
apt = S_(d.APT)
P = np.isnan(A3) & (M - S > 7200)
v6, v7, bl = preds()
base = v6 if base_name == 'v6' else np.where(P, v7, 0.5 * v7 + 0.5 * bl[1])
r = y - base
pop = ~P
X = pd.DataFrame(dict(apt=pd.Categorical(apt).codes, off_aobt3=M - A3, off_sched=M - S, pb=base, hour=hour, miss=np.isnan(A3).astype(int)))
nested = {}
if arm in ('stamp', 'all', 'stamp_noday', 'all_nocs'):
    f = pd.read_parquet(f'{D}/stampfeat_training.parquet')
    for c in ['pa_stamped', 'pa_inb_m_sibt', 'pa_ldg_m_sibt', 'aday_stamp', 'aday_n', 'opday_stamp', 'opday_n']:
        X[c] = f[c].to_numpy()
    X['nm_delay'] = A3 - S
    lab_mask = (~np.isnan(A3)) & (A3 - S > 600) & (M - A3 > 0); lab = (np.abs(B - S) < 60).astype(float)
    for nm, key in [('te_cs_st', K(apt, S_(d.FLIGHT_mvt))), ('te_op_st', K(apt, S_(d.AIRCRAFT_OPERATOR_flt))), ('te_stand_st', K(apt, S_(d.STAND_mvt)))]:
        nested[nm] = (key, lab_mask, lab, 20.0, lab[lab_mask].mean())
if arm in ('clocks', 'all', 'all_nocs'):
    X['nm_delay'] = A3 - S; X['off_iobt'] = M - IO; X['lobt_m_iobt'] = LO - IO; X['eobt_m_lobt'] = E1 - LO
    X['a3_m_eobt'] = A3 - E1; X['a3_m_lobt'] = A3 - LO; X['eobt_m_sched'] = E1 - S; X['iobt_m_sched'] = IO - S
if arm in ('ident', 'all', 'all_nocs'):
    X['ades'] = pd.Categorical(S_(d.ADES_mvt)).codes
    src = pop & (y <= 7200); rc = np.clip(r, -3000, 3000)
    for nm, key in [('te_cs_r', K(apt, S_(d.FLIGHT_mvt))), ('te_ades_r', K(apt, S_(d.ADES_mvt)))]:
        nested[nm] = (key, src, rc, 20.0, 0.0)
del d
if arm == 'stamp_noday':
    X = X.drop(columns=['aday_n', 'opday_n'])
if arm == 'all_nocs':
    nested = {k: v for k, v in nested.items() if k not in ('te_cs_st', 'te_cs_r')}
enc = {}
for nm, (key, msk, lab, w, prior) in nested.items():
    c, u = pd.factorize(key); Sm = np.zeros((len(u), 13)); Nm = np.zeros((len(u), 13))
    np.add.at(Sm, (c[msk], month[msk]), lab[msk]); np.add.at(Nm, (c[msk], month[msk]), 1)
    enc[nm] = (c, Sm, Nm, w, prior)
cats = ['apt'] + (['ades'] if 'ades' in X.columns else [])
feat = list(X.columns) + list(enc)
corr = np.zeros(len(y)); rng = np.random.default_rng(0)
prm = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=0, deterministic=True, force_row_wise=True, cat_smooth=50, max_cat_to_onehot=4)
for m in range(1, 13):
    tri = np.where(pop & (month != m) & (y <= 7200))[0]; tri = np.sort(rng.choice(tri, 500_000, replace=False))
    vai = np.where(pop & (month == m))[0]; ii = np.concatenate([tri, vai]); k = month[ii]
    Xm = X.iloc[ii].copy()
    for nm, (c, Sm, Nm, w, prior) in enc.items():
        cc = c[ii]
        s = Sm[cc].sum(1) - Sm[cc, m] - np.where(k != m, Sm[cc, k], 0); n = Nm[cc].sum(1) - Nm[cc, m] - np.where(k != m, Nm[cc, k], 0)
        Xm[nm] = (s + w * prior) / (n + w)
    mdl = lgb.train(prm, lgb.Dataset(Xm.iloc[:len(tri)][feat], np.clip(r[tri], -3000, 3000), categorical_feature=cats), 200)
    corr[vai] = mdl.predict(Xm.iloc[len(tri):][feat])
    if m == 7:
        imp = pd.Series(mdl.feature_importance('gain'), index=feat); print('gain share (fold 7):', (imp / imp.sum()).sort_values(ascending=False).round(3).head(14).to_dict())
np.save(f'{D}/diag_s_{arm}_{base_name}.npy', corr)
print(f'{arm} done {time.time()-t0:.0f}s peak {peak():.2f}')
