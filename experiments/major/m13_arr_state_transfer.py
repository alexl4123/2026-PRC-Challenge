"""m13 -- do transductive STATE features rescue the January-2026 disruption?  Tested where we
have 2026 labels: the ranking file's arrivals (provided data).

Arrival taxi-in model (as m05 'full'), train 2025 -> test 2026 Jan/Jul, and 2025 LOMO Jan/Jul,
with and without trailing airport-state features that are also available for departures:
  dep side (no labels): hpT / hpT_rel (median NM proxy of route-A departures, trailing 90 min),
                        dshT (share of departures without an NM record, trailing 3 h)
  arr side:             arrT / arrT_rel (median taxi-in of arrivals in-blocked in the trailing
                        60 min before our LANDING) -- the same-phase analogue of hpT
If EHAM-Jan-2026 improves a lot while 2025 LOMO barely moves, state features are a
disruption lever that LOMO cannot see.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE
from prc.io import peak_rss_gb

M = CACHE / 'major'


def dep_state_at(dep, q_apt, q_t):
    """trailing dep-side state evaluated at times q_t (epoch s) for airports q_apt."""
    t = dep.MVT_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
    oa = dep.off_aobt3.to_numpy(); apt = dep.APT.astype(str).to_numpy()
    okA = np.isfinite(oa) & (oa > 0) & (oa <= 7200); Dm = ~np.isfinite(oa)
    hpT = np.full(len(q_t), np.nan); dsh = np.full(len(q_t), np.nan)
    for a in np.unique(q_apt):
        qi = np.where(q_apt == a)[0]; qt = q_t[qi]
        di = np.where(apt == a)[0]; di = di[np.argsort(t[di])]; tt = t[di]
        A = okA[di]; tA = tt[A]; oA = oa[di][A]
        cD = np.concatenate([[0], np.cumsum(Dm[di])])
        lo = np.searchsorted(tA, qt - 5400, side='right'); hi = np.searchsorted(tA, qt, side='right')
        lo3 = np.searchsorted(tt, qt - 10800, side='right'); hi3 = np.searchsorted(tt, qt, side='right')
        for j in range(len(qi)):
            w = oA[lo[j]:hi[j]]
            if len(w) >= 3: hpT[qi[j]] = np.median(w)
            n3 = hi3[j] - lo3[j]
            if n3 >= 5: dsh[qi[j]] = (cD[hi3[j]] - cD[lo3[j]]) / n3
    return hpT, dsh


def arr_state(arr):
    """median taxi-in of other arrivals in-blocked in (t_land - 60 min, t_land)."""
    tl = arr.MVT_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
    tb = arr.BLOCK_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
    y = arr.y.to_numpy(); apt = arr.APT.astype(str).to_numpy(); out = np.full(len(arr), np.nan)
    for a in np.unique(apt):
        i = np.where(apt == a)[0]; o = i[np.argsort(tb[i])]; tbs = tb[o]; ys = y[o]
        lo = np.searchsorted(tbs, tl[i] - 3600, side='right'); hi = np.searchsorted(tbs, tl[i], side='left')
        for j in range(len(i)):
            w = ys[lo[j]:hi[j]]
            if len(w) >= 3: out[i[j]] = np.median(w)
    return out


t0 = time.time()
cols = ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt',
        'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'off_sched', 'month', 'y']
parts = []
for yr, w in [(2025, 'training'), (2026, 'ranking')]:
    a = pd.read_parquet(M / f'arr_{w}.parquet', columns=cols)
    dep = pd.read_parquet(M / f'dep_{w}.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'off_aobt3'])
    a = a[a.y.between(0, 3600) & a.off_sched.notna()].reset_index(drop=True)
    qt = a.MVT_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
    a['hpT'], a['dshT'] = dep_state_at(dep, a.APT.astype(str).to_numpy(), qt)
    a['arrT'] = arr_state(a)
    a['yr'] = yr
    parts.append(a)
d = pd.concat(parts, ignore_index=True); del parts
for c in ['hpT', 'arrT']:
    d[c + '_rel'] = d[c] / d.groupby('APT', observed=True)[c].transform('median')
print('features', f'{time.time()-t0:.0f}s', flush=True)
d['PIER'] = d.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+|\d)', expand=False).fillna('NA')
d['CSP'] = d.FLIGHT_mvt.astype(str).str[:3]
d['hour'] = d.MVT_TIME_UTC_mvt.dt.hour; d['dow'] = d.MVT_TIME_UTC_mvt.dt.dayofweek
for c in ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'PIER', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'CSP']:
    d[c] = d[c].astype(str).astype('category')
BASE = ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'PIER', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'CSP', 'hour', 'dow', 'off_sched']
SETS = {'base': BASE, '+dep state': BASE + ['hpT', 'hpT_rel', 'dshT'], '+dep+arr state': BASE + ['hpT', 'hpT_rel', 'dshT', 'arrT', 'arrT_rel']}
PAR = dict(objective='regression', learning_rate=0.08, num_leaves=255, min_data_in_leaf=100,
           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, cat_l2=10,
           num_threads=4, verbose=-1, seed=0)
y = d.y.to_numpy(float); yr = d.yr.to_numpy(); mon = d.month.to_numpy(); apt = d.APT.astype(str).to_numpy()
res = []
for nm, F in SETS.items():
    X = d[F]; tr25 = yr == 2025
    m_all = lgb.train(PAR, lgb.Dataset(X[tr25], y[tr25]), 400)
    for m in [1, 7]:
        te26 = (yr == 2026) & (mon == m); p26 = m_all.predict(X[te26])
        trl = tr25 & (mon != m); te25 = tr25 & (mon == m)
        p25 = lgb.train(PAR, lgb.Dataset(X[trl], y[trl]), 400).predict(X[te25])
        for a in ['ALL', 'EHAM', 'ALL-but-EHAM']:
            k25 = np.ones(te25.sum(), bool) if a == 'ALL' else ((apt[te25] == 'EHAM') if a == 'EHAM' else (apt[te25] != 'EHAM'))
            k26 = np.ones(te26.sum(), bool) if a == 'ALL' else ((apt[te26] == 'EHAM') if a == 'EHAM' else (apt[te26] != 'EHAM'))
            res.append((nm, m, a, np.sqrt(np.mean((y[te25][k25] - p25[k25]) ** 2)), np.sqrt(np.mean((y[te26][k26] - p26[k26]) ** 2))))
    print(nm, f'{time.time()-t0:.0f}s', 'peak', round(peak_rss_gb(), 2), flush=True)
R = pd.DataFrame(res, columns=['set', 'month', 'scope', 'lomo25', 'ahead26'])
R.to_csv(M / 'm13_arr_state.csv', index=False)
print(R.round(2).to_string(index=False))
