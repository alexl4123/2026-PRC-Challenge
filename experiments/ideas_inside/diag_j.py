"""Blend diversity from TARGET FORM (route A, reduced features, subsampled). Members:
 direct  : y
 direct2 : y, different seed + different subsample (diversity CONTROL)
 resid   : y - off_aobt3 (anchored), added back
 log     : log(y), back-transformed with per-fold smearing
usage: diag_j.py <months comma>  -> saves diag_j_m<MM>.npz"""
import sys, time
from common import *
import lightgbm as lgb
months = [int(x) for x in sys.argv[1].split(',')]
t0 = time.time()
d = load_dep(['APT', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
              'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
month = pd.to_datetime(M, unit='s').month.to_numpy()
oa = M - A3
A = np.isfinite(oa) & (oa > 0) & (oa <= 7200)
X = pd.DataFrame({c: pd.Categorical(S_(d[c])).codes for c in ['APT', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']})
X['off_aobt3'] = oa; X['off_sched'] = M - S; X['off_eobt'] = M - d.EOBT_1_flt.to_numpy(); X['off_lobt'] = M - d.LOBT_flt.to_numpy()
X['hour'] = pd.to_datetime(M, unit='s').hour; X['dow'] = pd.to_datetime(M, unit='s').dayofweek
cats = ['APT', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']
del d
prm = dict(objective='regression', learning_rate=0.07, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, verbose=-1, num_threads=2, deterministic=True, force_row_wise=True, cat_smooth=20)
for m in months:
    va = np.where(A & (month == m))[0]
    pool = np.where(A & (month != m) & (y <= 7200) & (y > 0))[0]
    out = {'idx': va}
    for name, seed in [('direct', 0), ('direct2', 1), ('resid', 0), ('log', 0)]:
        rng = np.random.default_rng(seed); tri = np.sort(rng.choice(pool, 400_000, replace=False))
        yt = y[tri]
        if name == 'resid': yt = yt - oa[tri]
        if name == 'log': yt = np.log(yt)
        mdl = lgb.train(dict(prm, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2),
                        lgb.Dataset(X.iloc[tri], yt, categorical_feature=cats), 300)
        p = mdl.predict(X.iloc[va])
        if name == 'resid': p = p + oa[va]
        if name == 'log':
            smear = np.mean(np.exp(yt - mdl.predict(X.iloc[tri])))      # Duan smearing, training rows
            p = np.exp(p) * smear
        out[name] = p
        print(f'm{m:02d} {name} {time.time()-t0:.0f}s', flush=True)
    np.savez(f'{D}/diag_j_m{m:02d}.npz', **out)
print('peak', peak())
