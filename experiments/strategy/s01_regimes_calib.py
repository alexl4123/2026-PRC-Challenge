"""s01 -- per-airport clock regimes (Q3) and a nested post-hoc recalibration of v9~.

Part A: for each airport, how good is the NM clock (AOBT_3) as a proxy for the
airport's off-block, and where does the v9~ error sit (route A normal / delay band,
route D, P)?  This is the evidence for any airport clustering.

Part B: is v9~ miscalibrated per airport x route?  Nested LOMO: for held-out month m,
fit y ~ f(pred) on the OTHER 11 months' OOF rows (outside P, y <= 7200), apply to m.
Variants: linear per (APT, route), isotonic per (APT, route).  Paired bootstrap vs v9~.

Inputs : cache/major/dep_training.parquet, cache/major/v9s_oof.npy
Outputs: cache/strategy/s01.log (stdout)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from prc.config import CACHE
from prc.validation import paired_bootstrap
from prc.io import peak_rss_gb

M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3', 'off_block'])
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy()
mon = d.month.to_numpy(); apt = d.APT.astype(str).to_numpy()
v9 = np.load(M / 'v9s_oof.npy')
n = len(y)
D = ~np.isfinite(oa)
A = np.isfinite(oa) & (oa > 0) & (oa <= 7200)
P = D & np.isfinite(so) & (so > 7200)
S = np.isin(mon, [1, 7])
e2 = (y - v9) ** 2
APTS = sorted(set(apt))


def r(a, m):
    return float(np.sqrt(np.mean((y[m] - a[m]) ** 2)))


print('=== A. clock regimes per airport (route A = AOBT_3 present, 0<off_aobt3<=7200; bulk y<=7200)')
rows = []
tot12 = e2.sum(); totS = e2[S].sum()
for a in APTS:
    ka = apt == a
    kA = ka & A & (y <= 7200)
    dlt = y[kA] - oa[kA]
    band = kA & (y > 1800)
    rows.append(dict(
        apt=a, n=ka.sum(), routeA=100 * (ka & A).sum() / ka.sum(), routeD=100 * (ka & D).sum() / ka.sum(),
        corr_y_oa=np.corrcoef(y[kA], oa[kA])[0, 1], sd_oa=oa[kA].std(), sd_y=y[kA].std(),
        med_abs_y_minus_oa=np.median(np.abs(dlt)), within60=100 * np.mean(np.abs(dlt) <= 60),
        band_pct=100 * band.sum() / kA.sum(),
        band_proxy_sees=100 * np.mean(oa[band] > 1800) if band.any() else np.nan,
        rmseA_v9=r(v9, kA), R2A=1 - np.mean((y[kA] - v9[kA]) ** 2) / y[kA].var(),
        sse12_pct=100 * e2[ka].sum() / tot12, sseS_pct=100 * e2[ka & S].sum() / totS))
t = pd.DataFrame(rows).round(3)
print(t.to_string(index=False))

# SSE by airport x cell on the scored months
cls = np.full(n, 'A_normal', dtype=object)
nd = ~D
cls[nd & (y > 1800) & (y <= 7200)] = 'A_band'
cls[nd & (y > 7200)] = 'A_tail'
cls[D & ~P & (y <= 7200)] = 'D_bulk'
cls[D & ~P & (y > 7200)] = 'D_tail'
cls[P] = 'P'
print('\n=== scored months (2025 Jan+Jul): % of total SSE by airport x cell')
tab = pd.crosstab(apt[S], cls[S], values=e2[S], aggfunc='sum').fillna(0) / totS * 100
tab['ALL'] = tab.sum(1)
print(tab.round(2).to_string())
print('\n=== 12 months: % of total SSE by airport x cell')
tab = pd.crosstab(apt, cls, values=e2, aggfunc='sum').fillna(0) / tot12 * 100
tab['ALL'] = tab.sum(1)
print(tab.round(2).to_string())

print('\n=== B. calibration of v9~: E[y|pred decile] outside P, y<=7200 (12 months)')
k = ~P & (y <= 7200)
q = pd.qcut(v9[k], 10, labels=False, duplicates='drop')
cal = pd.DataFrame({'pred': v9[k], 'y': y[k], 'q': q}).groupby('q').agg(pred=('pred', 'mean'), y=('y', 'mean'), n=('y', 'size'))
print(cal.round(1).to_string())

route = np.where(A, 'A', np.where(D, 'D', 'BC'))
out_lin = v9.copy(); out_iso = v9.copy()
for m in range(1, 13):
    te = (mon == m) & ~P
    tr = (mon != m) & ~P & (y <= 7200) & np.isfinite(v9)
    for a in APTS:
        for rt in ['A', 'D', 'BC']:
            ktr = tr & (apt == a) & (route == rt)
            kte = te & (apt == a) & (route == rt)
            if kte.sum() == 0 or ktr.sum() < 200:
                continue
            x = v9[ktr]; yy = y[ktr]
            b = np.polyfit(x, yy, 1)
            out_lin[kte] = np.polyval(b, v9[kte])
            iso = IsotonicRegression(out_of_bounds='clip').fit(x, yy)
            out_iso[kte] = iso.predict(v9[kte])
out_lin = np.clip(out_lin, 0, None); out_iso = np.clip(out_iso, 0, None)
for nm, arr in [('linear per apt x route', out_lin), ('isotonic per apt x route', out_iso),
                ('50/50 v9~ + isotonic', 0.5 * v9 + 0.5 * out_iso)]:
    all_ = np.ones(n, bool)
    pb12 = paired_bootstrap(y, arr, v9); pbS = paired_bootstrap(y[S], arr[S], v9[S])
    permo = [r(arr, mon == m) - r(v9, mon == m) for m in range(1, 13)]
    print(f'{nm:28s} 12m {r(arr, all_):.2f} vs {r(v9, all_):.2f}  d {pb12["diff"]:+.3f} [{pb12["lo"]:+.3f},{pb12["hi"]:+.3f}]'
          f' | scored d {pbS["diff"]:+.3f} [{pbS["lo"]:+.3f},{pbS["hi"]:+.3f}] | months neg {sum(p < 0 for p in permo)}/12')
print('peak', round(peak_rss_gb(), 2), 'GB')
