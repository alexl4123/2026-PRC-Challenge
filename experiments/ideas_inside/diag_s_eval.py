import sys
from common import *
base_name = sys.argv[1]; arms = sys.argv[2].split(',')
d = load_dep(['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
month = pd.to_datetime(M, unit='s').month.to_numpy(); sc = np.isin(month, [1, 7]); apt = S_(d.APT)
P = d.AOBT_3_flt.isna().to_numpy() & (M - S > 7200)
v6, v7, b = preds(); base = v6 if base_name == 'v6' else np.where(P, v7, 0.5 * v7 + 0.5 * b[1])
nul = base + np.load(f'{D}/diag_s_null_{base_name}.npy')
print(f'base {base_name}: 12m {rmse(y, base):.3f}  scored {rmse(y[sc], base[sc]):.3f}')
b1 = boot(y, nul, base, 300); b2 = boot(y[sc], nul[sc], base[sc], 300)
print(f'  null - base      12m {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]')
for a in arms:
    new = base + np.load(f'{D}/diag_s_{a}_{base_name}.npy')
    b1 = boot(y, new, nul, 300); b2 = boot(y[sc], new[sc], nul[sc], 300); b3 = boot(y, new, base, 300)
    per = {ap: round(float(np.sqrt((((y - new) ** 2 - (y - nul) ** 2) * (apt == ap)).sum() / len(y) + rmse(y, nul) ** 2) - rmse(y, nul)), 2) for ap in np.unique(apt)}
    print(f'  {a:6s} - null   12m {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}] | vs base 12m {b3[0]:+.3f}')
    print(f'         by airport (12m): {per}')
