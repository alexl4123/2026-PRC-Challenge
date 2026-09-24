"""m15 -- explicit stamping mixture for LIRF route-A delayed flights (the one place route-A
stamping is concentrated: q = 7% vs <= 1% elsewhere; remaining oracle -7.4 s 12m / -8.5 scored).

Population L: LIRF, NM record present (route A), off_sched > 1800 s.
  q(x)   : P(|y - off_sched| <= 120)   LightGBM classifier, 12-fold LOMO
  m_ns(x): E[y | not stamped]          LightGBM regressor on non-stamped rows (y <= 7200), LOMO
  pred   = q*off_sched + (1-q)*m_ns    replaces v9~ on L only
Compared with v9~ by paired bootstrap on all rows (12 months and scored months), plus a
null arm that refits m_ns alone (q := 0) to separate the mixture from a LIRF re-fit.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from prc.config import CACHE
from prc.validation import paired_bootstrap

M = CACHE / 'major'
t0 = time.time()
d = pd.read_parquet(M / 'dep_training.parquet')
st8 = pd.read_parquet(M / 'm11_state_training.parquet')
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy(); mon = d.month.to_numpy()
v9 = np.load(M / 'v9s_oof.npy'); n = len(y)
L = (d.APT.astype(str).to_numpy() == 'LIRF') & np.isfinite(oa) & (oa > 0) & (oa <= 7200) & (so > 1800)
print('L rows', L.sum(), 'stamped rate', np.mean(np.abs(y[L] - so[L]) <= 120).round(4))
X = pd.DataFrame({'off_sched': so, 'off_aobt3': oa, 'off_eobt1': d.off_eobt1, 'off_lobt': d.off_lobt,
                  'off_iobt': d.off_iobt, 'dnm': so - oa, 'eobt_sched': so - d.off_eobt1.to_numpy(),
                  'hour': d.MVT_TIME_UTC_mvt.dt.hour, 'dow': d.MVT_TIME_UTC_mvt.dt.dayofweek,
                  'sched_min': d.SCHED_TIME_UTC_mvt.dt.minute})
for c in ['STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'ADES_mvt']:
    X[c] = d[c].astype(str)
X['PIER'] = d.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+|\d)', expand=False).fillna('NA')
X['CSP'] = d.FLIGHT_mvt.astype(str).str[:3]
X = pd.concat([X, st8[['hpT', 'hpT_rel', 'dshT', 'arrT_rel']]], axis=1)
X = X[L].reset_index(drop=True)
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = X[c].astype('category')
yL = y[L]; soL = so[L]; mL = mon[L]; stL = (np.abs(yL - soL) <= 120).astype(int)
PC = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, cat_smooth=20, num_threads=4, verbose=-1, seed=0)
PR = dict(objective='regression', learning_rate=0.03, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, cat_smooth=20, num_threads=4, verbose=-1, seed=0)
q = np.zeros(len(yL)); mns = np.zeros(len(yL))
for m in range(1, 13):
    tr = mL != m; te = mL == m
    if te.sum() == 0:
        continue
    c = lgb.train(PC, lgb.Dataset(X[tr], stL[tr]), 400); q[te] = c.predict(X[te])
    trn = tr & (stL == 0) & (yL <= 7200)
    r = lgb.train(PR, lgb.Dataset(X[trn], yL[trn]), 600); mns[te] = r.predict(X[te])
print('fits', f'{time.time()-t0:.0f}s')
print('AUC stamped (LOMO) %.3f ; Brier %.4f vs base-rate %.4f' % (
    roc_auc_score(stL, q), np.mean((q - stL) ** 2), np.mean((stL.mean() - stL) ** 2)))
pred_mix = v9.copy(); pred_mix[L] = q * soL + (1 - q) * mns
pred_null = v9.copy(); pred_null[L] = mns
np.save(M / 'm15_oof_mix.npy', pred_mix)
S = np.isin(mon, [1, 7])
for nm, a in [('mixture', pred_mix), ('null (m_ns only)', pred_null)]:
    for sc, k in [('12m', np.ones(n, bool)), ('scored', S)]:
        b = paired_bootstrap(y[k], a[k], v9[k], n_boot=1000)
        print(f'  {nm:18s} - v9~ [{sc:6s}] {b["diff"]:+.3f} +- {b["sd"]:.3f} CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]')
b = paired_bootstrap(y, pred_mix, pred_null, n_boot=1000)
print(f'  mixture - null [12m] {b["diff"]:+.3f} +- {b["sd"]:.3f} CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]')
e0 = (y - v9) ** 2; e1 = (y - pred_mix) ** 2
print('  per month (mixture - v9~):', ' '.join(f'{m}:{np.sqrt(e1[mon==m].mean())-np.sqrt(e0[mon==m].mean()):+.2f}' for m in range(1, 13)))
print('  L rows: RMSE v9~ %.1f  mixture %.1f  null %.1f' % (np.sqrt(e0[L].mean()), np.sqrt(e1[L].mean()), np.sqrt(((y - pred_null)[L] ** 2).mean())))
