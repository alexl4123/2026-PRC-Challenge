import sys; sys.path.insert(0, '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/ideas_inside')
from common import *
d = load_dep(['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
P = np.isnan(A3) & (M - S > 7200); mon = pd.to_datetime(M, unit='s').month.to_numpy(); sc = np.isin(mon, [1, 7])
e53 = np.load(f'{R}/exp53_oof.npy'); o = np.load(f'{R}/exp62_oof.npy'); v6, v7, bl = preds()
print('exp53 rows RMSE', [round(rmse(y, e53[i]), 3) for i in range(len(e53))], '| exp62 v5', round(rmse(y, o[0]), 3))
nP = ~P
print('corr(exp53[1], exp62 v5) outside P, residuals:', np.corrcoef((y - e53[1])[nP & (y <= 7200)], (y - o[0])[nP & (y <= 7200)])[0, 1].round(4))
lgb2 = 0.5 * (e53[1] + o[0])
print('outside-P RMSE: 1-seed', round(rmse(y[nP], o[0][nP]), 3), ' exp53', round(rmse(y[nP], e53[1][nP]), 3), ' 2-seed avg', round(rmse(y[nP], lgb2[nP]), 3))
v8s = np.where(P, v7, 0.5 * v7 + 0.5 * bl[1])
w = np.where(np.isnan(A3), 0.5, 0.65)
v9s2 = np.where(P, v7, w * bl[1] + (1 - w) * lgb2)
v8s2 = np.where(P, v7, 0.5 * bl[1] + 0.5 * lgb2)
for nm, a in [('v8~ (1-seed, w .5)', v8s), ('v8~2 (2-seed, w .5)', v8s2), ('v9~2 (2-seed, w .65/.5)', v9s2)]:
    print(f'{nm:24s} 12m {rmse(y, a):.3f} scored {rmse(y[sc], a[sc]):.3f}')
np.save('base_v9s2.npy', v9s2)
