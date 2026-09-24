"""m11 -- transductive CURRENT-STATE features, tested as a residual booster on v9~.

Motivation (m07, m10): January 2026 holds an out-of-distribution disruption (EHAM
2-7 Jan: route-D share 20-40 %, 502 P rows vs 72 in Jan 2025; same-hour route-A NM
proxy 2,000-3,500 s).  In 2025, for route-D non-stamped rows, the same-hour route-A
proxy correlates 0.51 with y while v9's prediction correlates 0.38.  v9 has trailing
COUNTS only, nothing that says how long the neighbours are taxiing right now.

Features (no departure labels anywhere; arrivals' in-block times are provided data):
  hpT   median NM proxy (MVT-AOBT_3, route A) of departures at the same airport that
        took off in (t-90min, t]                                    -- trailing
  hpT75 its 75th percentile;  hpTn  count
  arrT  median arrival taxi-in, same airport, in-block in (t-60min, t]  -- trailing
  dshT  share of departures without an NM record, same airport, (t-3h, t]
  hpC   median route-A proxy, takeoff within t +- 60 min  (CENTRED -> uses flights that
        took off AFTER ours = look-ahead in time; reported as a separate arm)
Arms (12-fold LOMO residual booster, target y - v9~, same learner in every arm):
  null   : v9 pred, off_sched, off_aobt3, routeD flag, APT, hour     (stacking control)
  trail  : null + hpT, hpT75, hpTn, arrT, dshT
  centre : trail + hpC                                              (LOOK-AHEAD)
Rows: the booster is fitted and applied only where v9 serves the bulk (outside P, or
inside P with specialist p < 0.1); LIRF-P decisions are untouched.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE, OUTPUTS
from prc.validation import paired_bootstrap
from prc.io import peak_rss_gb

M = CACHE / 'major'


def state_features(dep, arr):
    dep = dep.copy()
    dep['_i'] = np.arange(len(dep))
    t = dep.MVT_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
    oa = dep.off_aobt3.to_numpy()
    okA = np.isfinite(oa) & (oa > 0) & (oa <= 7200)
    out = {k: np.full(len(dep), np.nan) for k in ['hpT', 'hpT75', 'hpTn', 'hpC', 'dshT', 'arrT']}
    ta = (arr.BLOCK_TIME_UTC_mvt.astype('int64') // 10**6).to_numpy()
    ya = arr.y.to_numpy(); aa = arr.APT.astype(str).to_numpy()
    apt = dep.APT.astype(str).to_numpy()
    for a in np.unique(apt):
        idx = np.where(apt == a)[0]; idx = idx[np.argsort(t[idx])]
        tt = t[idx]; A = okA[idx]; oo = oa[idx]; Dm = ~np.isfinite(oo)
        tA = tt[A]; oA = oo[A]
        lo = np.searchsorted(tA, tt - 5400, side='right'); hi = np.searchsorted(tA, tt, side='right')
        loC = np.searchsorted(tA, tt - 3600); hiC = np.searchsorted(tA, tt + 3600, side='right')
        lo3 = np.searchsorted(tt, tt - 10800, side='right'); hi3 = np.searchsorted(tt, tt, side='right')
        cD = np.concatenate([[0], np.cumsum(Dm)])
        ka = np.where(aa == a)[0]; ka = ka[np.argsort(ta[ka])]; tka = ta[ka]; yka = ya[ka]
        loa = np.searchsorted(tka, tt - 3600, side='right'); hia = np.searchsorted(tka, tt, side='right')
        for j in range(len(idx)):
            w = oA[lo[j]:hi[j]]
            if len(w) >= 3:
                out['hpT'][idx[j]] = np.median(w); out['hpT75'][idx[j]] = np.percentile(w, 75)
            out['hpTn'][idx[j]] = len(w)
            w = oA[loC[j]:hiC[j]]
            if len(w) >= 3:
                out['hpC'][idx[j]] = np.median(w)
            n3 = hi3[j] - lo3[j]
            if n3 >= 5:
                out['dshT'][idx[j]] = (cD[hi3[j]] - cD[lo3[j]]) / n3
            w = yka[loa[j]:hia[j]]
            if len(w) >= 3:
                out['arrT'][idx[j]] = np.median(w)
    f = pd.DataFrame(out)
    for c in ['hpT', 'hpT75', 'hpC', 'arrT']:     # relative to the airport's own typical level
        f[c + '_rel'] = f[c] / pd.Series(f[c]).groupby(apt).transform('median').to_numpy()
    return f


def build(which):
    p = M / f'm11_state_{which}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    dep = pd.read_parquet(M / f'dep_{which}.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'off_aobt3'])
    arr = pd.read_parquet(M / f'arr_{which}.parquet', columns=['APT', 'BLOCK_TIME_UTC_mvt', 'y'])
    arr = arr[arr.y.between(0, 7200)]
    f = state_features(dep, arr)
    f.to_parquet(p)
    return f


if __name__ == '__main__':
    t0 = time.time()
    ftr = build('training'); frk = build('ranking')
    print('features built', f'{time.time()-t0:.0f}s', 'peak', round(peak_rss_gb(), 2), flush=True)
    d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'y'])
    y = d.y.to_numpy(); mon = d.month.to_numpy(); n = len(y)
    v9 = np.load(M / 'v9s_oof.npy')
    D = d.off_aobt3.isna().to_numpy(); P = D & (d.off_sched.to_numpy() > 7200)
    sp = np.load(CACHE / 'exp62_spec_P.npy')[1]            # v7 specialist p on P rows (DEP order of P)
    pP = np.zeros(n); pP[np.where(P)[0]] = sp
    served_bulk = ~P | (pP < 0.1)
    X = pd.DataFrame({'v9': v9, 'off_sched': d.off_sched, 'off_aobt3': d.off_aobt3, 'D': D.astype(int),
                      'APT': d.APT.astype(str).astype('category'), 'hour': d.MVT_TIME_UTC_mvt.dt.hour})
    X = pd.concat([X, ftr], axis=1)
    ARMS = {'null': ['v9', 'off_sched', 'off_aobt3', 'D', 'APT', 'hour'],
            'trail': ['v9', 'off_sched', 'off_aobt3', 'D', 'APT', 'hour', 'hpT', 'hpT75', 'hpTn', 'arrT', 'dshT',
                      'hpT_rel', 'hpT75_rel', 'arrT_rel'],
            'centre': ['v9', 'off_sched', 'off_aobt3', 'D', 'APT', 'hour', 'hpT', 'hpT75', 'hpTn', 'arrT', 'dshT',
                       'hpT_rel', 'hpT75_rel', 'arrT_rel', 'hpC', 'hpC_rel']}
    PAR = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
               feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10,
               num_threads=4, verbose=-1, seed=0)
    fit_rows = served_bulk & (y <= 7200)
    r = y - v9
    preds = {}
    for arm, F in ARMS.items():
        corr = np.zeros(n)
        for m in range(1, 13):
            tr = fit_rows & (mon != m); te = served_bulk & (mon == m)
            b = lgb.train(PAR, lgb.Dataset(X.loc[tr, F], r[tr]), 250)
            corr[te] = b.predict(X.loc[te, F])
        preds[arm] = v9 + corr
        np.save(M / f'm11_oof_{arm}.npy', preds[arm])
        print(arm, f'{time.time()-t0:.0f}s', 'peak', round(peak_rss_gb(), 2), flush=True)
    S = np.isin(mon, [1, 7])
    def rm(a, k): return np.sqrt(np.mean((y[k] - a[k]) ** 2))
    for arm in ARMS:
        print(f'\n=== {arm}: RMSE 12m {rm(preds[arm], np.ones(n, bool)):.3f} (v9~ {rm(v9, np.ones(n, bool)):.3f})  '
              f'scored {rm(preds[arm], S):.3f} (v9~ {rm(v9, S):.3f})')
        for nm, k in [('12m', np.ones(n, bool)), ('scored', S), ('m1', mon == 1), ('m7', mon == 7)]:
            b = paired_bootstrap(y[k], preds[arm][k], v9[k], n_boot=500)
            print(f'   {arm} - v9~ [{nm:6s}] {b["diff"]:+.3f} +- {b["sd"]:.3f}  CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]')
    for arm in ['trail', 'centre']:
        for nm, k in [('12m', np.ones(n, bool)), ('scored', S)]:
            b = paired_bootstrap(y[k], preds[arm][k], preds['null'][k], n_boot=500)
            print(f'   {arm} - null [{nm:6s}] {b["diff"]:+.3f} +- {b["sd"]:.3f}  CI [{b["lo"]:+.3f}, {b["hi"]:+.3f}]')
        # where: SSE change by route / D rows
        e0 = (y - preds['null']) ** 2; e1 = (y - preds[arm]) ** 2
        for nm, k in [('routeA', ~D), ('routeD notP', D & ~P), ('P', P)]:
            print(f'      {arm} vs null SSE change {nm:12s}: {(e1[k].sum()-e0[k].sum())/1e8:+.2f}e8 (12m)')
    # month-by-month for trail vs null
    e0 = (y - preds['null']) ** 2; e1 = (y - preds['trail']) ** 2
    print('   trail-null per month (RMSE diff):', ' '.join(
        f'{m}:{np.sqrt(e1[mon==m].mean())-np.sqrt(e0[mon==m].mean()):+.2f}' for m in range(1, 13)))
    print('total', f'{time.time()-t0:.0f}s', 'peak', round(peak_rss_gb(), 2))
