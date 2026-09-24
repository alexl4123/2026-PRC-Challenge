"""m02 -- which clock does the airport's BLOCK equal, per error cell? And the oracle
of knowing it. Question: is the delay band (y 1800-7200, NM record present)
physics (a genuinely long taxi) or recording (BLOCK copied from a planned clock)?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE

M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet')
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy()
mon = d.month.to_numpy(); apt = d.APT.astype(str).to_numpy()
v9 = np.load(M / 'v9s_oof.npy')
n = len(y); S = np.isin(mon, [1, 7])
D = ~np.isfinite(oa); P = D & (so > 7200)
# BLOCK = MVT - y, clock c = MVT - off_c  ->  BLOCK - c = off_c - y
clocks = {'SCHED': so, 'AOBT3': oa, 'EOBT1': d.off_eobt1.to_numpy(), 'LOBT': d.off_lobt.to_numpy(),
          'IOBT': d.off_iobt.to_numpy()}
dist = np.column_stack([np.abs(v - y) for v in clocks.values()])
dist = np.where(np.isfinite(dist), dist, np.inf)
names = np.array(list(clocks))
near = dist.argmin(1); dmin = dist.min(1)
lab = np.where(dmin <= 60, names[near], np.where(dmin <= 300, 'w300:' + names[near], 'none'))

cls = np.full(n, 'A_normal', dtype=object)
st = np.abs(y - so) <= 120
cls[~D & (y > 1800) & (y <= 7200) & st] = 'A_band_stSCHED'
cls[~D & (y > 1800) & (y <= 7200) & ~st] = 'A_band_other'
cls[~D & (y > 7200)] = 'A_tail'
cls[D & ~P & (y <= 7200)] = 'D_notP_bulk'
cls[D & ~P & (y > 7200)] = 'D_notP_tail'
cls[P & (y <= 7200)] = 'P_bulk'
cls[P & (y > 7200)] = 'P_tail'
for c in ['A_normal', 'A_band_other', 'A_band_stSCHED', 'D_notP_bulk', 'P_bulk', 'P_tail']:
    k = cls == c
    vc = pd.Series(lab[k]).value_counts(normalize=True).head(8)
    print(f'{c:16s} n={k.sum():8d}  ' + '  '.join(f'{i}:{100*v:.1f}%' for i, v in vc.items()))

# The delay band 'other': how does y relate to the NM proxy?
k = cls == 'A_band_other'
r = y[k] - oa[k]
print('\nA_band_other: y - off_aobt3 quantiles', np.percentile(r, [5, 25, 50, 75, 95]).round(0))
print('  off_aobt3 quantiles', np.percentile(oa[k], [5, 25, 50, 75, 95]).round(0))
print('  off_sched quantiles', np.percentile(so[k], [5, 25, 50, 75, 95]).round(0))
print('  share with |y-off_aobt3|<=300:', round(np.mean(np.abs(r) <= 300), 3))
print('  by airport: n, share |y-oa|<=300, median y-oa, median oa, median y')
for a in sorted(set(apt)):
    kk = k & (apt == a)
    rr = y[kk] - oa[kk]
    print(f'   {a} n={kk.sum():6d} near={np.mean(np.abs(rr)<=300):.3f} med(y-oa)={np.median(rr):7.0f} '
          f'med oa={np.median(oa[kk]):6.0f} med y={np.median(y[kk]):6.0f}  SSE%sc={100*((y-v9)**2)[kk&S].sum()/((y-v9)**2)[S].sum():.2f}')

# ORACLE: if the anchor clock were known (within 60 s), predict MVT - clock (+ the clock's
# median offset); else keep v9.  Separately per cell.
pred_or = v9.copy()
off = {nm: np.nanmedian((v - y)[lab == nm]) if (lab == nm).any() else 0 for nm, v in clocks.items()}
for j, nm in enumerate(names):
    kk = (lab == nm)
    pred_or[kk] = list(clocks.values())[j][kk] - off[nm]
e_b = (y - v9) ** 2; e_o = (y - pred_or) ** 2
for scope, m in [('12m', np.ones(n, bool)), ('scored', S)]:
    rb = np.sqrt(e_b[m].mean()); ro = np.sqrt(e_o[m].mean())
    print(f'\nORACLE anchor-known (<=60 s) {scope}: {rb:.2f} -> {ro:.2f} ({ro-rb:+.2f})')
    for c in sorted(set(cls)):
        kk = m & (cls == c)
        alt = e_b[m].sum() - e_b[kk].sum() + e_o[kk].sum()
        print(f'   only in {c:16s}: {np.sqrt(alt/m.sum())-rb:+7.2f}')
