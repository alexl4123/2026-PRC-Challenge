from common import *
d = load_dep(['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
month = pd.to_datetime(M, unit='s').month.to_numpy(); sc = np.isin(month, [1, 7]); apt = S_(d.APT)
P = d.AOBT_3_flt.isna().to_numpy() & (M - S > 7200)
v6, v7, b = preds()
v8 = np.where(P, v7, 0.5 * v7 + 0.5 * b[1])
C = {a: np.nan_to_num(np.load(f'{D}/diag_g_{a}.npy')) for a in ['null', 'full']}
for bn, base in [('v6', v6), ('v8~', v8)]:
    print(f'--- base {bn}: 12m {rmse(y, base):.3f} scored {rmse(y[sc], base[sc]):.3f}')
    for a, c in C.items():
        new = base + c
        b1 = boot(y, new, base, 300); b2 = boot(y[sc], new[sc], base[sc], 300)
        print(f'  {a:5s} 12m d {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored d {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]')
    nf = base + C['full']; nn = base + C['null']
    b1 = boot(y, nf, nn, 300); b2 = boot(y[sc], nf[sc], nn[sc], 300)
    print(f'  full-null 12m d {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored d {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]')
    nf_s = (y - nf) ** 2; nn_s = (y - nn) ** 2
    print('  full-null by airport, dRMSE(12m, global scale):', {a: round(float(np.sqrt(((nf_s - nn_s) * (apt == a)).sum() / len(y) + rmse(y, nn) ** 2) - rmse(y, nn)), 3) for a in np.unique(apt)})
