from common import *
d = load_dep(['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); month = pd.to_datetime(M, unit='s').month.to_numpy()
P = d.AOBT_3_flt.isna().to_numpy() & (M - d.SCHED_TIME_UTC_mvt.to_numpy() > 7200)
v6, v7, bl = preds()
nets = np.zeros((3, len(y)))
for s in range(3):
    for m in range(1, 13):
        nets[s, month == m] = np.load(f'{R}/cv/study/nn__seed{s:02d}__m{m:02d}.npy')
print('check: mean of 3 seeds vs blend_oof row1 max|diff|', np.abs(nets.mean(0) - bl[1]).max())
o = ~P; L = bl[0]
res = []
for k, combos in [(1, [[0], [1], [2]]), (2, [[0, 1], [0, 2], [1, 2]]), (3, [[0, 1, 2]])]:
    solo = np.mean([rmse(y[o], nets[c].mean(0)[o]) for c in combos])
    blend = np.mean([rmse(y[o], 0.5 * L[o] + 0.5 * nets[c].mean(0)[o]) for c in combos])
    res.append((k, solo, blend)); print(f'{k} net seeds: net solo outsideP {solo:.3f}  50/50 with 1-seed LGB {blend:.3f}')
r1 = np.corrcoef((y - nets[0])[o & (y <= 7200)], (y - nets[1])[o & (y <= 7200)])[0, 1]
print('net seed-to-seed residual corr (bulk):', round(r1, 4))
# 1/k extrapolation of the blend: MSE(k) = a + b/k fitted on k=1,3
a_ = (3 * res[2][2] ** 2 - res[0][2] ** 2) / 2; b_ = res[0][2] ** 2 - a_
for k in [5, 10]:
    print(f'extrapolated blend with {k} net seeds: {np.sqrt(a_ + b_ / k):.3f}  (vs 3 seeds {res[2][2]:.3f})')
