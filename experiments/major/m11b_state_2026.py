"""m11b -- what would the m11 state booster do to the 2026 predictions?  (No labels.)
Fits the booster arms on all 2025 bulk-served rows (target y - v9~ OOF) and applies them to
v9's shipped 2026 predictions on bulk-served rows.  Writes the candidate prediction arrays
(ranking DEP order) to cache/major/m11b_pred_<arm>.npy -- NOT a submission file.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE, OUTPUTS
from prc.io import peak_rss_gb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_state_features import build

M = CACHE / 'major'
t0 = time.time()
ftr = build('training'); frk = build('ranking')
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'y'])
rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched'])
y = d.y.to_numpy(); v9 = np.load(M / 'v9s_oof.npy')
D = d.off_aobt3.isna().to_numpy(); P = D & (d.off_sched.to_numpy() > 7200)
sp = np.load(CACHE / 'exp62_spec_P.npy')[1]; pP = np.zeros(len(y)); pP[np.where(P)[0]] = sp
fit = (~P | (pP < 0.1)) & (y <= 7200)
sub = pd.read_parquet(OUTPUTS / 'understanding-accordion_v9.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
v9r = sub.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
Dr = rk.off_aobt3.isna().to_numpy(); Pr = Dr & (rk.off_sched.to_numpy() > 7200)
z = np.load(CACHE / 'final' / 'spec_x_ranking.npz')
ids = pd.Series(np.arange(len(z['ids'])), index=z['ids'])
pr = z['p_spec'][ids.reindex(rk.MVT_ID_mvt).to_numpy()]
pr = np.where(np.isfinite(pr), pr, 0.0)
served_r = ~Pr | (pr < 0.1)
print('2026 bulk-served rows', served_r.sum(), 'of', len(rk), '; P rows with p<0.1:', (Pr & (pr < 0.1)).sum())

def frame(dd, v, f, Dm):
    X = pd.DataFrame({'v9': v, 'off_sched': dd.off_sched.to_numpy(), 'off_aobt3': dd.off_aobt3.to_numpy(),
                      'D': Dm.astype(int), 'APT': pd.Categorical(dd.APT.astype(str), categories=sorted(d.APT.astype(str).unique())),
                      'hour': dd.MVT_TIME_UTC_mvt.dt.hour.to_numpy()})
    return pd.concat([X, f.reset_index(drop=True)], axis=1)

Xtr = frame(d, v9, ftr, D); Xrk = frame(rk, v9r, frk, Dr)
ARMS = {'null': ['v9', 'off_sched', 'off_aobt3', 'D', 'APT', 'hour'],
        'trail': ['v9', 'off_sched', 'off_aobt3', 'D', 'APT', 'hour', 'hpT', 'hpT75', 'hpTn', 'arrT', 'dshT',
                  'hpT_rel', 'hpT75_rel', 'arrT_rel']}
PAR = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
           feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10,
           num_threads=4, verbose=-1, seed=0)
out = {}
for arm, F in ARMS.items():
    b = lgb.train(PAR, lgb.Dataset(Xtr.loc[fit, F], (y - v9)[fit]), 250)
    c = np.zeros(len(rk)); c[served_r] = b.predict(Xrk.loc[served_r, F])
    out[arm] = v9r + c
    np.save(M / f'm11b_pred_{arm}.npy', out[arm])
    print(arm, f'{time.time()-t0:.0f}s', flush=True)
sh = out['trail'] - out['null']
cell = rk.APT.astype(str).to_numpy() + '-' + rk.month.astype(str).to_numpy()
t = pd.DataFrame({'cell': cell, 'D': Dr, 'sh': sh, 'abs': np.abs(sh)})
g = t.groupby(['cell', 'D']).agg(n=('sh', 'size'), mean_shift=('sh', 'mean'), mean_abs=('abs', 'mean'),
                                  p95_abs=('abs', lambda x: np.percentile(x, 95)), sum_sq=('sh', lambda x: (x ** 2).sum()))
g['share_sq%'] = 100 * g.sum_sq / g.sum_sq.sum()
pd.set_option('display.width', 200)
print('\ntrail minus null, 2026 (cells with >= 2% of the squared shift):')
print(g[g['share_sq%'] >= 2].sort_values('share_sq%', ascending=False).round(1).to_string())
k = (cell == 'EHAM-1') & Dr
print('\nEHAM Jan 2026 route D: n', k.sum(), ' mean v9', v9r[k].mean().round(0), ' null', out['null'][k].mean().round(0),
      ' trail', out['trail'][k].mean().round(0))
print('peak', round(peak_rss_gb(), 2))
