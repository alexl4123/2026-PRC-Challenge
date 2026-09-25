"""r01 -- where the error lives in the v15 STRUCTURE (2025 OOF), read-only from existing caches.

v15~ is built exactly as experiments/w2/w2_routeD_P.py builds its baseline + D1 + P1 (seeds 0-2):
  A/B/C outside P : W2 blend (cache/w2/w2_eval.npz 'new')
  route D, not P  : W2 blend + route-D corrector delta (v10~ - v9~)
  P               : p14 (TabICL own at Rome, pooled elsewhere) * off_sched + (1 - p14) * pb_new,
                    pb_new = 0.5 W2 LightGBM bulk + 0.5 W2 network bulk
Pools as in strategy/s13. Also: per-airport shares, row concentration, bands of y, and the
'reachable' scored-month RMSE (excluding route-D-not-P tail rows = clock errors with no observable).
Nothing is written except stdout (and, if R01_OUT is set, v15s_oof.npy into that directory).

    systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 nice -n 5 \
        python experiments/research_0925/r01_v15_anatomy.py
"""
import os, sys, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'w2'))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.validation import paired_bootstrap
from w2_routeD_P import bulk_oof

M = CACHE / 'major'
b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'].astype(float), b['month']
z = np.load(CACHE / 'w2' / 'w2_eval.npz'); seeds = list(z['seeds']); print('w2_eval seeds', seeds)
v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
v9 = np.load(M / 'v9s_oof.npy').astype(float)
v10 = np.clip(np.load(M / 'm11d_oof_trailD.npy').astype(float), 0, None)
A = ~D & ~P; DnP = D & ~P
w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
Pidx = w4['Pidx']; assert np.array_equal(np.flatnonzero(P), Pidx)
rome = w4['apt'] == 'LIRF'
p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
assert np.array_equal(d.month.to_numpy(), month) and np.allclose(d.y.to_numpy(), y)
so_all = d.off_sched.to_numpy().astype(float); apt = d.APT.astype(str).to_numpy()
so = so_all[Pidx]
pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)

base = v11.copy()
base[A] = np.clip(z['new'][A], 0, None)
base[Pidx] = v11[Pidx] + (mixP(p14, pb_old) - mixP(p_cur, pb_old))
v15 = base.copy()
v15[DnP] = np.clip(z['new'][DnP] + (v10[DnP] - v9[DnP]), 0, None)
pb_new = 0.5 * bulk_oof('mix', seeds, month)[Pidx] + 0.5 * bulk_oof('nn', seeds, month)[Pidx]
v15[Pidx] = base[Pidx] + (mixP(p14, pb_new) - mixP(p14, pb_old))

S = np.isin(month, [1, 7]); al = np.ones(len(y), bool)
r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
print(f'v14~ (base) 12m {r(base, al):.2f} scored {r(base, S):.2f} | v15~ 12m {r(v15, al):.2f} scored {r(v15, S):.2f} '
      f'| v11~ 12m {r(v11, al):.2f} scored {r(v11, S):.2f}')

stamped = np.abs(y - so_all) <= 120
cls = np.full(len(y), 'A_normal', dtype=object)
cls[~D & (y > 1800) & (y <= 7200) & ~stamped] = 'A_band_phys'
cls[~D & (y > 1800) & (y <= 7200) & stamped] = 'A_band_stamped'
cls[~D & (y > 7200)] = 'A_tail'
cls[D & ~P & (y <= 7200)] = 'D_notP_bulk'
cls[D & ~P & (y > 7200)] = 'D_notP_tail'
cls[P & (y <= 7200)] = 'P_false_alarm_zone'
cls[P & (y > 7200)] = 'P_tail'
CELLS = ['A_normal', 'A_band_phys', 'A_band_stamped', 'A_tail', 'D_notP_bulk', 'D_notP_tail', 'P_false_alarm_zone', 'P_tail']

unreach = cls == 'D_notP_tail'
print(f'reachable (excl. D_notP_tail): v15~ 12m {r(v15, ~unreach):.2f} scored {r(v15, S & ~unreach):.2f}; '
      f'v11~ 12m {r(v11, ~unreach):.2f} scored {r(v11, S & ~unreach):.2f}')
for mm in (1, 7):
    k = month == mm
    print(f'  month {mm}: all {r(v15, k):.2f}  reachable {r(v15, k & ~unreach):.2f}  unreachable rows {int((k & unreach).sum())}')

e15 = (y - v15) ** 2; e11 = (y - v11) ** 2
for scope, m in [('12 months', al), ('scored 1+7', S)]:
    tot15 = e15[m & ~unreach].sum(); tot11 = e11[m & ~unreach].sum()
    rows = []
    for cc in CELLS:
        k = m & (cls == cc)
        rows.append(dict(cell=cc, n=int(k.sum()),
                         pct_reach_v11=np.nan if cc == 'D_notP_tail' else 100 * e11[k].sum() / tot11,
                         pct_reach_v15=np.nan if cc == 'D_notP_tail' else 100 * e15[k].sum() / tot15,
                         rmse_v11=np.sqrt(e11[k].mean()), rmse_v15=np.sqrt(e15[k].mean()),
                         sse_change_pct=100 * (e15[k].sum() / e11[k].sum() - 1),
                         d_if_perfect=np.nan if cc == 'D_notP_tail' else
                         np.sqrt((tot15 - e15[k].sum()) / (m & ~unreach).sum()) - np.sqrt(tot15 / (m & ~unreach).sum())))
    print(f'\n=== {scope}: pools (share of REACHABLE SSE), v11~ -> v15~')
    print(pd.DataFrame(rows).round(2).to_string(index=False))

print('\n=== v15~ scored months: % of reachable SSE by airport x cell')
k = S & ~unreach; tot = e15[k].sum()
tab = pd.crosstab(apt[k], cls[k], values=e15[k], aggfunc='sum').fillna(0) / tot * 100
tab['ALL'] = tab.sum(axis=1); tab['rmse_all'] = [r(v15, S & ~unreach & (apt == a)) for a in tab.index]
tab['rmse_normal'] = [r(v15, S & (cls == 'A_normal') & (apt == a)) for a in tab.index]
tab['n'] = [int((S & ~unreach & (apt == a)).sum()) for a in tab.index]
print(tab.round(2).sort_values('ALL', ascending=False).to_string())
print('\n=== v15~ 12 months: % of reachable SSE by airport')
k = ~unreach; tot = e15[k].sum()
print((pd.Series(e15[k]).groupby(apt[k]).sum() / tot * 100).round(2).sort_values(ascending=False).to_string())

print('\n=== row concentration (v15~, reachable): share of SSE in the top-k rows')
for scope, m in [('12m', ~unreach), ('scored', S & ~unreach)]:
    e = np.sort(e15[m])[::-1]; tot = e.sum()
    print(f'  {scope}: ' + '  '.join(f'top {kk}: {100 * e[:kk].sum() / tot:.1f}%' for kk in (10, 100, 1000, 10000))
          + f'  (n={m.sum():,})')

print('\n=== v15~ by y band (scored months, reachable): rows, share of SSE, RMSE, mean residual y - pred')
bands = [(-1e9, 300), (300, 600), (600, 900), (900, 1200), (1200, 1800), (1800, 3600), (3600, 7200), (7200, 1e9)]
k0 = S & ~unreach; tot = e15[k0].sum()
for lo, hi in bands:
    k = k0 & (y > lo) & (y <= hi)
    print(f'  ({lo:>6.0f},{hi:>6.0f}]  n {k.sum():7,d}  SSE {100 * e15[k].sum() / tot:5.1f}%  RMSE {r(v15, k):7.1f}  '
          f'bias {np.mean(y[k] - v15[k]):+8.1f}')

print('\n=== mean residual (y - v15~) by airport x month (12 months, A_normal rows) -- a year-shift proxy')
k = cls == 'A_normal'
print(pd.crosstab(apt[k], month[k], values=(y - v15)[k], aggfunc='mean').round(1).to_string())

# live comparison: live v15 = 275.31. 2025 scored-month reachable RMSE vs live.
print('\npeak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
if os.environ.get('R01_OUT'):   # optional: save the stand-in OUTSIDE the repo (e.g. a scratch dir)
    np.save(os.path.join(os.environ['R01_OUT'], 'v15s_oof.npy'), v15.astype(np.float32))
