"""In-model test of the stand family, c10 protocol (400k clean-bulk subsample, 255 leaves, 500 rounds), month m.
 base72 | A: train cont / val cont (outside2's number) | B: same model, val 'pair' (what their ranking build ships)
 | C: train isocap / val isocap (the fix).  usage: rv_test.py <month> [ntrain]"""
import sys, time
from common import *
import lightgbm as lgb
m = int(sys.argv[1]); ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
t0 = time.time()
X = pd.read_parquet(f'{R}/altfam_X.parquet'); y = np.load(f'{R}/altfam_y.npy'); mon = np.load(f'{R}/altfam_month.npy')
P = (X.aobt_missing.values == 1) & (X.off_sched.values > 7200); clean = (y <= 7200) & ~P
rng = np.random.default_rng(100 + m)
tr = np.sort(rng.choice(np.where(clean & (mon != m))[0], ntr, replace=False)); te = np.where(clean & (mon == m))[0]
base = list(X.columns); assert len(base) == 72
X = X.iloc[np.concatenate([tr, te])].reset_index(drop=True); ntr_ = len(tr)
COLS = ['sl_to', 'sl_after_sched', 'sl_same_typ', 'sl_same_op', 'si_before_sched', 'si_same_typ', 'sn_minus_to', 'sn_same_op', 'sn_same_typ', 'sa_after_to', 'sd_next_to', 'sd_prev_to']
F = {md: pd.read_parquet(f'{D}/rv_stand_{md}.parquet').iloc[np.concatenate([tr, te])].reset_index(drop=True) for md in ['cont', 'pair', 'isocap']}
cats = ['APT', 'RUNWAY_mvt', 'PIER', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt']
PAR = dict(objective='regression', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, seed=0, deterministic=True, force_row_wise=True)
def fit(Xa):
    return lgb.train(PAR, lgb.Dataset(Xa.iloc[:ntr_], y[tr], categorical_feature=cats), 500)
yt = y[te]; pr = {}
b = fit(X); pr['base72'] = b.predict(X.iloc[ntr_:]); print(f'base {time.time()-t0:.0f}s', flush=True)
XA = pd.concat([X, F['cont'][COLS]], axis=1); bA = fit(XA); pr['A cont/cont'] = bA.predict(XA.iloc[ntr_:])
XB = pd.concat([X, F['pair'][COLS]], axis=1); pr['B cont/pair'] = bA.predict(XB.iloc[ntr_:]); del XA, XB
XC = pd.concat([X, F['isocap'][COLS]], axis=1); bC = fit(XC); pr['C isocap'] = bC.predict(XC.iloc[ntr_:]); del XC
print(f'fits {time.time()-t0:.0f}s peak {peak():.2f}', flush=True)
np.savez(f'{D}/rv_test_m{m:02d}_{ntr}.npz', te=te, **{k.split()[0]: v for k, v in pr.items()})
rm = lambda p, k=slice(None): np.sqrt(np.mean((yt[k] - p[k]) ** 2))
def bt(a, b_, k):
    ra, rb = (yt[k] - a[k]) ** 2, (yt[k] - b_[k]) ** 2; g = np.random.default_rng(1); n = len(ra); d = []
    for _ in range(200):
        i = g.integers(0, n, n); d.append(np.sqrt(ra[i].mean()) - np.sqrt(rb[i].mean()))
    return np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5)
print(f'm{m:02d} ntrain {ntr}: base72 clean-bulk {rm(pr["base72"]):.2f}')
for k in ['A cont/cont', 'B cont/pair', 'C isocap']:
    r = bt(pr[k], pr['base72'], slice(None))
    print(f'  {k:12s} {rm(pr[k]):.2f}  d {r[0]:+.2f} [{r[1]:+.2f},{r[2]:+.2f}]')
# where it lands (arm A and C vs base): route, stamped class, next-occupant bound
d = load_dep(['SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'MVT_TIME_UTC_mvt'])
S, B, A3, M = [d[c].to_numpy()[te] for c in ['SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'MVT_TIME_UTC_mvt']]
slc = F['cont']['sl_to'].to_numpy()[ntr_:]; sla = F['cont']['sl_after_sched'].to_numpy()[ntr_:]
routeD = np.isnan(A3); stamped = (~routeD) & (A3 - S > 600) & (np.abs(B - S) < 60)
nextocc = np.isfinite(slc) & (slc < yt)          # last arrival before takeoff parked AFTER our true off-block (label used for analysis only)
jump = np.any([np.abs(F['pair'][c].to_numpy()[ntr_:]) > 30 * 86400 for c in ['sl_to', 'si_before_sched', 'sn_minus_to', 'sa_after_to', 'sd_next_to', 'sd_prev_to']], axis=0)
N = len(yt); base_sse = ((yt - pr['base72']) ** 2)
for k in ['A cont/cont', 'C isocap', 'B cont/pair']:
    ds = (yt - pr[k]) ** 2 - base_sse
    tot = np.sqrt((base_sse.sum() + ds.sum()) / N) - np.sqrt(base_sse.sum() / N)
    parts = {nm: np.sqrt((base_sse.sum() + ds[msk].sum()) / N) - np.sqrt(base_sse.sum() / N) for nm, msk in
             [('routeD', routeD), ('routeA stamped@SCHED', stamped), ('next-occupant rows', nextocc), ('other', ~routeD & ~stamped & ~nextocc), ('pair-jump rows', jump)]}
    print(f'  {k:12s} total {tot:+.2f} | ' + ' | '.join(f'{a} {v:+.2f}' for a, v in parts.items()))
print(f'  shares: routeD {routeD.mean():.4f} stamped {stamped.mean():.4f} next-occupant {nextocc.mean():.4f} jump {jump.mean():.4f}')
print('peak', peak(), f'{time.time()-t0:.0f}s')
