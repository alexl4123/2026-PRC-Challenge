"""s13 -- where the error lives NOW: the v11 stand-in (v11~) and its error pools.

v10~ = v9~ + route-D corrector (cache/major/m11d_oof_trailD.npy, full, clipped at 0)
v11~ = v10~ + stack arm (c) correction outside P (cache/v11/oof_c.npy; read-only)
(the main session's own additivity check used the same construction: -4.82 12m / -5.31 scored)

Then: pool anatomy of v11~ (12 months, scored months), per airport, and a rough re-test of the
LIRF route-A stamping mixture (major m15) on top of v11~, applied as its delta vs v9~.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.validation import paired_bootstrap

M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy(); mon = d.month.to_numpy()
apt = d.APT.astype(str).to_numpy(); n = len(y)
D = ~np.isfinite(oa); P = D & (so > 7200); S = np.isin(mon, [1, 7])
v9 = np.load(M / 'v9s_oof.npy')
v10 = np.clip(np.load(M / 'm11d_oof_trailD.npy'), 0, None)
c = np.load(CACHE / 'v11' / 'oof_c.npy').astype(float)
v11 = np.where(P, v10, np.clip(v10 + np.nan_to_num(c), 0, None))
r = lambda a, m: float(np.sqrt(np.mean((y[m] - a[m]) ** 2)))
al = np.ones(n, bool)
for nm, a in [('v9~', v9), ('v10~', v10), ('v11~', v11)]:
    print(f'{nm:5s} 12m {r(a, al):.2f}  scored {r(a, S):.2f}')
pb = paired_bootstrap(y[S], v11[S], v9[S]); print('v11~ - v9~ scored', {k: round(v, 3) for k, v in pb.items()})
pb = paired_bootstrap(y, v11, v9); print('v11~ - v9~ 12m   ', {k: round(v, 3) for k, v in pb.items()})

stamped = np.abs(y - so) <= 120
cls = np.full(n, 'A_normal', dtype=object)
cls[~D & (y > 1800) & (y <= 7200) & ~stamped] = 'A_band_phys'
cls[~D & (y > 1800) & (y <= 7200) & stamped] = 'A_band_stamped'
cls[~D & (y > 7200)] = 'A_tail'
cls[D & ~P & (y <= 7200)] = 'D_notP_bulk'
cls[D & ~P & (y > 7200)] = 'D_notP_tail'
cls[P & (y <= 7200)] = 'P_false_alarm_zone'
cls[P & (y > 7200)] = 'P_tail'
for scope, m in [('12 months', al), ('scored 1+7', S)]:
    rows = []
    for nm, a in [('v9~', v9), ('v11~', v11)]:
        e2 = (y - a) ** 2; tot = e2[m].sum(); reach = tot - e2[m & (cls == 'D_notP_tail')].sum()
        for cc in ['A_normal', 'A_band_phys', 'A_band_stamped', 'A_tail', 'D_notP_bulk', 'D_notP_tail', 'P_false_alarm_zone', 'P_tail']:
            k = m & (cls == cc)
            rows.append(dict(model=nm, cell=cc, n=k.sum(), sse_pct=100 * e2[k].sum() / tot,
                             pct_of_reachable=np.nan if cc == 'D_notP_tail' else 100 * e2[k].sum() / reach,
                             rmse=np.sqrt(e2[k].mean()), d_if_perfect=np.sqrt((tot - e2[k].sum()) / m.sum()) - np.sqrt(tot / m.sum())))
    t = pd.DataFrame(rows)
    print(f'\n=== {scope}: pools, v9~ vs v11~')
    print(t.pivot(index='cell', columns='model', values=['n', 'sse_pct', 'pct_of_reachable', 'rmse', 'd_if_perfect']).round(2).to_string())

print('\n=== v11~ scored months: % of reachable SSE (excl. D_notP_tail) by airport x cell')
e2 = (y - v11) ** 2; k = S & (cls != 'D_notP_tail'); tot = e2[k].sum()
tab = pd.crosstab(apt[k], cls[k], values=e2[k], aggfunc='sum').fillna(0) / tot * 100
tab['ALL'] = tab.sum(axis=1)
print(tab.round(2).to_string())
print('\n=== v11~ 12 months: % of reachable SSE by airport x cell')
k = cls != 'D_notP_tail'; tot = e2[k].sum()
tab = pd.crosstab(apt[k], cls[k], values=e2[k], aggfunc='sum').fillna(0) / tot * 100
tab['ALL'] = tab.sum(axis=1)
print(tab.round(2).to_string())

# LIRF route-A stamping mixture on top: apply m15's delta (mix - v9~) to v11~
mix = np.load(M / 'm15_oof_mix.npy')
delta = mix - v9
v11m = np.clip(v11 + delta, 0, None)
for scope, m in [('12m', al), ('scored', S)]:
    pb = paired_bootstrap(y[m], v11m[m], v11[m])
    print(f'm15 delta on top of v11~ ({scope}): {pb["diff"]:+.3f} [{pb["lo"]:+.3f},{pb["hi"]:+.3f}]   (vs v9~ it was -0.98 12m / -0.74 scored)')
print('months better:', sum(r(v11m, mon == mm) < r(v11, mon == mm) for mm in range(1, 13)), '/ 12')
np.save(CACHE / 'strategy' / 'v11s_oof.npy', v11)
