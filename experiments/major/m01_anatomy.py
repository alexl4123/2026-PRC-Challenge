"""m01 -- where our squared error lives (v9 stand-in), and the class oracles.

v9~ : inside P = v7 recipe (exp62_oof row 2); outside P = 0.35*v7 + 0.65*net-mixture
      on routes A/B/C, 0.5/0.5 on route D (cache/blend_oof_mix_nn3.npy = 3-seed net
      mixture OOF). Same approximation the ideas_inside agent used for v8.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.validation import paired_bootstrap
from prc.io import peak_rss_gb

M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet',
                    columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3', 'off_eobt1', 'off_lobt',
                             'off_iobt', 'MVT_ID_mvt', 'FLIGHT_ID_mvt'])
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy()
mon = d.month.to_numpy(); apt = d.APT.astype(str).to_numpy()
n = len(y)
o = np.load(CACHE / 'exp62_oof.npy'); bb = np.load(CACHE / 'blend_oof_mix_nn3.npy').astype(float); print('blend shape', bb.shape, 'lgb-mix row0 vs v6 maxdiff', np.nanmax(np.abs(bb[0]-o[1]))); b = bb[1]
v5, v6, v7 = o[0], o[1], o[2]
D = ~np.isfinite(oa)
routeA = np.isfinite(oa) & (oa > 0) & (oa <= 7200)
P = D & np.isfinite(so) & (so > 7200)
w = np.where(D, 0.5, 0.65)
v9 = np.where(P, v7, (1 - w) * v7 + w * b)
np.save(M / 'v9s_oof.npy', v9)
S = np.isin(mon, [1, 7])


def r(a, m=None):
    m = np.ones(n, bool) if m is None else m
    return float(np.sqrt(np.mean((y[m] - a[m]) ** 2)))


print('RMSE 12m / scored:  v5 %.2f/%.2f  v6 %.2f/%.2f  v7 %.2f/%.2f  net %.2f/%.2f  v9~ %.2f/%.2f' % (
    r(v5), r(v5, S), r(v6), r(v6, S), r(v7), r(v7, S), r(b), r(b, S), r(v9), r(v9, S)))
pb = paired_bootstrap(y[S], v9[S], v7[S]); print('v9~ - v7 scored', {k: round(v, 3) for k, v in pb.items()})

# ---- decomposition --------------------------------------------------------
stamped = np.abs(y - so) <= 120
cls = np.full(n, 'A_normal', dtype=object)
nd = ~D
cls[nd & (y > 1800) & (y <= 7200) & stamped] = 'A_delayband_stampedSCHED'
cls[nd & (y > 1800) & (y <= 7200) & ~stamped] = 'A_delayband_other'
cls[nd & (y > 7200)] = 'A_tail'
cls[D & ~P & (y <= 7200)] = 'D_notP_bulk'
cls[D & ~P & (y > 7200)] = 'D_notP_tail'
cls[P & (y <= 7200)] = 'P_bulk(false-alarm zone)'
cls[P & (y > 7200)] = 'P_tail'
e2 = (y - v9) ** 2
for scope, m in [('12 months', np.ones(n, bool)), ('scored 1+7', S)]:
    print(f'\n=== SSE anatomy, v9~, {scope}: RMSE {r(v9, m):.2f}, n {m.sum()}')
    tot = e2[m].sum()
    rows = []
    for c in sorted(set(cls)):
        k = m & (cls == c)
        rows.append((c, k.sum(), 100 * k.sum() / m.sum(), 100 * e2[k].sum() / tot,
                     np.sqrt(e2[k].mean()) if k.any() else np.nan,
                     # RMSE if this cell's error were zero
                     np.sqrt((tot - e2[k].sum()) / m.sum()) - r(v9, m)))
    print(pd.DataFrame(rows, columns=['cell', 'n', 'rows%', 'SSE%', 'RMSE', 'dRMSE_if_zero']).round(2).to_string(index=False))

# ---- by true-y band x route (scored months) ---------------------------------
bands = [0, 600, 900, 1200, 1800, 2700, 3600, 7200, 20000, 1e9]
bi = np.digitize(y, bands) - 1
print('\n=== scored months, by y band: rows, SSE%, bias(pred-y), RMSE, dRMSE if zero')
tot = e2[S].sum()
for j in range(len(bands) - 1):
    k = S & (bi == j)
    if k.any():
        print(f'  {bands[j]:>6.0f}-{bands[j+1]:<8.0f} n={k.sum():7d} SSE%={100*e2[k].sum()/tot:6.2f} '
              f'bias={np.mean(v9[k]-y[k]):8.1f} rmse={np.sqrt(e2[k].mean()):8.1f} '
              f'd0={np.sqrt((tot-e2[k].sum())/S.sum())-r(v9,S):7.2f}')

# ---- the top rows ---------------------------------------------------------------
for scope, m in [('12 months', np.ones(n, bool)), ('scored', S)]:
    tot = e2[m].sum(); srt = np.sort(e2[m])[::-1]
    print(f'\n{scope}: share of SSE in top-k rows: ' + ', '.join(
        f'top{k}={100*srt[:k].sum()/tot:.1f}%' for k in [5, 10, 20, 50, 100, 300, 1000, 3000, 10000]))
    for k in [0, 5, 10, 20, 50, 100, 300, 1000]:
        print(f'   drop top {k:5d} rows -> RMSE {np.sqrt(srt[k:].sum()/m.sum()):.2f}')

# ---- clean bulk by airport, scored months -------------------------------------
print('\n=== scored months, outside P, y<=7200 (clean bulk) by airport: RMSE v9~ / sd(y) / R2')
k0 = S & ~P & (y <= 7200)
print('   ALL', round(r(v9, k0), 1))
for a in sorted(set(apt)):
    k = k0 & (apt == a)
    print(f'   {a}  n={k.sum():6d} rmse={r(v9,k):6.1f} sd={y[k].std():6.1f} R2={1-np.mean((y[k]-v9[k])**2)/y[k].var():.3f}')
print('peak', round(peak_rss_gb(), 2))
