"""m11d -- the state lever confined to ROUTE D (no NM record), where m11 found the gain.

A route-D-only residual corrector: fitted on route-D bulk-served rows (outside P, or P
with specialist p < 0.1; y <= 7200), applied to the same rows only.  Route A is untouched,
so the +0.8 s stacking penalty m11's all-rows null showed cannot arise there.
Arms: null_D (v9, off_sched, APT, hour) vs trail_D (+ trailing state).  Also a
shrunk variant (half the correction) as a robustness check.
Then fitted on all 2025 and applied to 2026 route-D rows (m11d_pred_trailD.npy, ranking
DEP order; NOT a submission).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE, OUTPUTS
from prc.validation import paired_bootstrap

M = CACHE / 'major'
t0 = time.time()
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'y'])
f = pd.read_parquet(M / 'm11_state_training.parquet')
y = d.y.to_numpy(); mon = d.month.to_numpy(); n = len(y); v9 = np.load(M / 'v9s_oof.npy')
D = d.off_aobt3.isna().to_numpy(); P = D & (d.off_sched.to_numpy() > 7200)
sp = np.load(CACHE / 'exp62_spec_P.npy')[1]; pP = np.zeros(n); pP[np.where(P)[0]] = sp
R = D & (~P | (pP < 0.1))
APTS = sorted(d.APT.astype(str).unique())
def frame(dd, v, ff):
    return pd.concat([pd.DataFrame({'v9': v, 'off_sched': dd.off_sched.to_numpy(),
                                    'APT': pd.Categorical(dd.APT.astype(str), categories=APTS),
                                    'hour': dd.MVT_TIME_UTC_mvt.dt.hour.to_numpy()}), ff.reset_index(drop=True)], axis=1)
X = frame(d, v9, f)
ARMS = {'null_D': ['v9', 'off_sched', 'APT', 'hour'],
        'trail_D': ['v9', 'off_sched', 'APT', 'hour', 'hpT', 'hpT75', 'hpTn', 'arrT', 'dshT', 'hpT_rel', 'hpT75_rel', 'arrT_rel']}
PAR = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20,
           num_threads=4, verbose=-1, seed=0)
NR = 300
fitm = R & (y <= 7200); r = y - v9
pred = {}
for arm, F in ARMS.items():
    c = np.zeros(n)
    for m in range(1, 13):
        tr = fitm & (mon != m); te = R & (mon == m)
        c[te] = lgb.train(PAR, lgb.Dataset(X.loc[tr, F], r[tr]), NR).predict(X.loc[te, F])
    pred[arm] = v9 + c
    pred[arm + '_half'] = v9 + 0.5 * c
S = np.isin(mon, [1, 7])
for arm in ['null_D', 'trail_D', 'trail_D_half']:
    for sc, k in [('12m', np.ones(n, bool)), ('scored', S), ('m1', mon == 1), ('m2', mon == 2), ('m7', mon == 7)]:
        b = paired_bootstrap(y[k], pred[arm][k], v9[k], n_boot=1000)
        print(f'  {arm:13s} - v9~ [{sc:6s}] {b["diff"]:+.3f} +- {b["sd"]:.3f} CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]')
nf = ~(mon == 2)
b = paired_bootstrap(y[nf], pred['trail_D'][nf], v9[nf], n_boot=1000)
print(f'  trail_D - v9~ [11m, Feb excluded] {b["diff"]:+.3f} +- {b["sd"]:.3f} CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]')
e0 = (y - v9) ** 2; e1 = (y - pred['trail_D']) ** 2
print('  per month trail_D - v9~:', ' '.join(f'{m}:{np.sqrt(e1[mon==m].mean())-np.sqrt(e0[mon==m].mean()):+.2f}' for m in range(1, 13)))
print('  route-D bulk-served rows RMSE: v9~ %.1f  trail_D %.1f' % (np.sqrt(e0[R].mean()), np.sqrt(e1[R].mean())))
np.save(M / 'm11d_oof_trailD.npy', pred['trail_D'])

# ---- 2026 application ----------------------------------------------------------------
rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched'])
fr = pd.read_parquet(M / 'm11_state_ranking.parquet')
v9r = pd.read_parquet(OUTPUTS / 'understanding-accordion_v9.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
Dr = rk.off_aobt3.isna().to_numpy(); Pr = Dr & (rk.off_sched.to_numpy() > 7200)
z = np.load(CACHE / 'final' / 'spec_x_ranking.npz'); ids = pd.Series(np.arange(len(z['ids'])), index=z['ids'])
pr = z['p_spec'][ids.reindex(rk.MVT_ID_mvt).to_numpy()]; pr = np.where(np.isfinite(pr), pr, 0.0)
Rr = Dr & (~Pr | (pr < 0.1))
Xr = frame(rk, v9r, fr)
F = ARMS['trail_D']
bst = lgb.train(PAR, lgb.Dataset(X.loc[fitm, F], r[fitm]), NR)
cr = np.zeros(len(rk)); cr[Rr] = bst.predict(Xr.loc[Rr, F])
np.save(M / 'm11d_pred_trailD.npy', v9r + cr)
cell = rk.APT.astype(str).to_numpy() + '-' + rk.month.astype(str).to_numpy()
t = pd.DataFrame({'cell': cell[Rr], 'c': cr[Rr]}).groupby('cell').c.agg(['size', 'mean', lambda x: np.mean(np.abs(x)), lambda x: (x ** 2).sum()])
t.columns = ['n', 'mean_shift', 'mean_abs', 'sum_sq']; t['share%'] = 100 * t.sum_sq / t.sum_sq.sum()
print('\n2026 route-D shifts (trail_D, fitted on all 2025):')
print(t.sort_values('share%', ascending=False).head(10).round(1).to_string())
print('total 2026 sum shift^2 %.3e ; rows moved >60 s: %d' % ((cr ** 2).sum(), (np.abs(cr) > 60).sum()))
print(f'{time.time()-t0:.0f}s')
