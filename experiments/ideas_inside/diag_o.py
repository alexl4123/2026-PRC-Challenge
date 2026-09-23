"""Additivity: v8~ + bulk-feature stack (all) + P2 specialist (gbm_x) + weighted P specialist (x_w_sq)."""
from common import *
d = load_dep(['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
month = pd.to_datetime(M, unit='s').month.to_numpy(); sc = np.isin(month, [1, 7])
P = d.AOBT_3_flt.isna().to_numpy() & (M - S > 7200); Pi = np.where(P)[0]
v6, v7, bl = preds(); v8 = np.where(P, v7, 0.5 * v7 + 0.5 * bl[1])
pb = np.load(f'{R}/exp62_pb.npy').astype(float); pb8 = np.where(P, pb, 0.5 * pb + 0.5 * bl[1])
A = v8 + np.load(f'{D}/diag_s_all_v8.npy')
cand = pd.read_parquet(f'{D}/p2_cand_training.parquet'); P2 = cand.index.to_numpy(); os2 = cand.off_sched.to_numpy(float)
q = np.load(f'{D}/diag_l_Q.npy')[2]                        # gbm_x
AP2 = A.copy(); AP2[P2] = q * os2 + (1 - q) * A[P2]
spec = np.load(f'{D}/diag_e_specP.npy')[2]                  # x_w_sq (row order of ARMS in diag_e)
osP = (M - S)[Pi]
AP2W = AP2.copy(); AP2W[Pi] = np.clip(spec * osP + (1 - spec) * pb[Pi], 0, None)
V8W = v8.copy(); V8W[Pi] = AP2W[Pi]
for nm, new, ref in [('A vs v8~', A, v8), ('A+P2 vs A', AP2, A), ('A+P2 vs v8~', AP2, v8), ('W alone vs v8~', V8W, v8),
                     ('A+P2+W vs v8~', AP2W, v8)]:
    b1 = boot(y, new, ref, 300); b2 = boot(y[sc], new[sc], ref[sc], 300)
    print(f'{nm:16s} 12m {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]')
print('RMSE v8~', rmse(y, v8), ' A+P2+W', rmse(y, AP2W))
