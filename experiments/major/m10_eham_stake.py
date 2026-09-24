"""m10 -- what is at stake in the January 2026 disruption days, if route-D flights taxied
like the route-A flights that took off in the same airport-hour?  (Stand-in truth = the
NM proxy of same airport, same day, takeoff within +-1 h, route A. An ASSUMPTION, stated.)
Also: how does y relate to the same-hour route-A proxy for route-D rows IN 2025 (labelled)?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
M = CACHE / 'major'; N = 344841

def hour_proxy(d):
    """median NM proxy of route-A departures at the same airport with takeoff within +-60 min (excl. self)."""
    d = d.sort_values('MVT_TIME_UTC_mvt').reset_index(drop=True)
    t = d.MVT_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
    oa = d.off_aobt3.to_numpy(); res = np.full(len(d), np.nan); cnt = np.zeros(len(d))
    for apt, idx in d.groupby('APT', observed=True).indices.items():
        tt = t[idx]; aa = oa[idx]
        lo = np.searchsorted(tt, tt - 3600); hi = np.searchsorted(tt, tt + 3600, side='right')
        for j in range(len(idx)):
            w = aa[lo[j]:hi[j]]; w = w[np.isfinite(w) & (w > 0) & (w <= 7200)]
            if len(w) >= 3:
                res[idx[j]] = np.median(w); cnt[idx[j]] = len(w)
    d['hp'] = res; d['hp_n'] = cnt
    return d

rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched'])
v9 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v9.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
rk['pred'] = v9.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
rk = hour_proxy(rk)
rk['D'] = rk.off_aobt3.isna(); rk['P'] = rk.D & (rk.off_sched > 7200)

tr = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'y'])
tr['pred'] = np.load(M / 'v9s_oof.npy')
tr = hour_proxy(tr)
tr['D'] = tr.off_aobt3.isna(); tr['P'] = tr.D & (tr.off_sched > 7200)
# 2025, labelled: for route-D NON-stamped rows (|y - off_sched| > 120), how does y track hp?
k = tr.D & ((tr.y - tr.off_sched).abs() > 120) & (tr.y <= 7200) & tr.hp.notna()
print('2025 route-D non-stamped rows', k.sum(), ' corr(y, hp) %.3f  corr(y, pred) %.3f  corr(y-pred, hp-median_hp_apt) %.3f' % (
    np.corrcoef(tr.y[k], tr.hp[k])[0, 1], np.corrcoef(tr.y[k], tr.pred[k])[0, 1],
    np.corrcoef((tr.y - tr.pred)[k], (tr.hp - tr.groupby('APT', observed=True).hp.transform('median'))[k])[0, 1]))
for q in [(0, 900), (900, 1200), (1200, 1600), (1600, 2400), (2400, 9e9)]:
    kk = k & tr.hp.between(*q)
    if kk.sum() > 20:
        print(f'   hp in {q}: n={kk.sum():6d}  mean y {tr.y[kk].mean():6.0f}  mean pred {tr.pred[kk].mean():6.0f}  mean hp {tr.hp[kk].mean():6.0f}')
# 2026: route-D rows by hp band -- the predictions we served
k6 = rk.D & rk.hp.notna()
print('\n2026 route-D rows by same-hour route-A proxy band (no labels):')
for q in [(0, 900), (900, 1200), (1200, 1600), (1600, 2400), (2400, 9e9)]:
    kk = k6 & rk.hp.between(*q)
    print(f'   hp in {q}: n={kk.sum():6d}  (P {(kk & rk.P).sum():5d})  mean pred {rk.pred[kk & ~rk.P].mean():6.0f} (notP) '
          f'{rk.pred[kk & rk.P].mean():6.0f} (P)  mean hp {rk.hp[kk].mean():6.0f}   by apt: ' +
          ', '.join(f'{a}:{n}' for a, n in rk[kk].APT.astype(str).value_counts().head(4).items()))
