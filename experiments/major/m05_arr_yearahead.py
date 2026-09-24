"""m05 -- how much does a 2025-trained model degrade ONE YEAR AHEAD? Measured on the
only labelled 2026 movements we have: the ranking file's ARRIVALS (provided data).

LOMO inside 2025 (what all our validation measures) vs train-on-2025 -> test-2026,
for January and July, arrival taxi-in, LightGBM, three feature sets that differ in
how much they memorise identities (stand / operator). If the 2026 error exceeds the
2025 LOMO error by more than the change in the target's own spread, our LOMO
numbers are optimistic about the leaderboard, and the size says by how much.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE
from prc.io import peak_rss_gb

M = CACHE / 'major'
cols = ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt',
        'MVT_TIME_UTC_mvt', 'off_sched', 'off_block', 'month', 'y']
tr = pd.read_parquet(M / 'arr_training.parquet', columns=cols); tr['yr'] = 2025
rk = pd.read_parquet(M / 'arr_ranking.parquet', columns=cols); rk['yr'] = 2026
d = pd.concat([tr, rk], ignore_index=True); del tr, rk
print('arrival label check: y == -off_block ?', np.nanmean(np.isclose(d.y, -d.off_block, atol=1)).round(4))
d = d[d.y.between(0, 3600) & d.off_sched.notna()].reset_index(drop=True)
d['PIER'] = d.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+|\d)', expand=False).fillna('NA')
d['CSP'] = d.FLIGHT_mvt.astype(str).str[:3]
t = d.MVT_TIME_UTC_mvt
d['hour'] = t.dt.hour; d['dow'] = t.dt.dayofweek
# arrivals landing in the previous 30 min at the same airport (simple congestion, no labels)
d = d.sort_values(['APT', 'MVT_TIME_UTC_mvt']).reset_index(drop=True)
ts = d.MVT_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
n30 = np.zeros(len(d))
for a, idx in d.groupby('APT', observed=True).indices.items():
    s = ts[idx]; n30[idx] = np.arange(len(s)) - np.searchsorted(s, s - 1800)
d['n30'] = n30
CAT = ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'PIER', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'CSP']
for c in CAT:
    d[c] = d[c].astype(str).astype('category')
SETS = {'full(stand+op+cs)': ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'PIER', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'CSP', 'hour', 'dow', 'off_sched', 'n30'],
        'no stand':          ['APT', 'RUNWAY_mvt', 'PIER', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'hour', 'dow', 'off_sched', 'n30'],
        'no stand/op/cs':    ['APT', 'RUNWAY_mvt', 'PIER', 'AIRCRAFT_TYPE_mvt', 'hour', 'dow', 'off_sched', 'n30']}
PAR = dict(objective='regression', learning_rate=0.08, num_leaves=255, min_data_in_leaf=100,
           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, cat_l2=10,
           num_threads=4, verbose=-1, seed=0)
y = d.y.to_numpy(float); yr = d.yr.to_numpy(); mon = d.month.to_numpy(); apt = d.APT.astype(str).to_numpy()
res = []
for nm, F in SETS.items():
    t0 = time.time()
    X = d[F]
    tr25 = (yr == 2025)
    m_all = lgb.train(PAR, lgb.Dataset(X[tr25], y[tr25]), 400)
    for m in [1, 7]:
        te26 = (yr == 2026) & (mon == m)
        p26 = m_all.predict(X[te26])
        trl = tr25 & (mon != m); te25 = tr25 & (mon == m)
        ml = lgb.train(PAR, lgb.Dataset(X[trl], y[trl]), 400)
        p25 = ml.predict(X[te25])
        for a in ['ALL'] + sorted(set(apt)):
            k25 = np.ones(te25.sum(), bool) if a == 'ALL' else apt[te25] == a
            k26 = np.ones(te26.sum(), bool) if a == 'ALL' else apt[te26] == a
            r25 = np.sqrt(np.mean((y[te25][k25] - p25[k25]) ** 2)); r26 = np.sqrt(np.mean((y[te26][k26] - p26[k26]) ** 2))
            s25 = y[te25][k25].std(); s26 = y[te26][k26].std()
            b26 = np.mean(p26[k26] - y[te26][k26])
            res.append((nm, m, a, r25, s25, r26, s26, b26, 1 - r25**2/s25**2, 1 - r26**2/s26**2))
    print(nm, f'{time.time()-t0:.0f}s', 'peak', round(peak_rss_gb(), 2), flush=True)
R = pd.DataFrame(res, columns=['set', 'month', 'apt', 'lomo25_rmse', 'sd25', 'ahead26_rmse', 'sd26', 'bias26', 'R2_25', 'R2_26'])
R.to_csv(M / 'm05_arr_yearahead.csv', index=False)
pd.set_option('display.width', 200)
print(R[R.apt == 'ALL'].round(3).to_string(index=False))
print(R[R.set == 'full(stand+op+cs)'].round(3).to_string(index=False))
