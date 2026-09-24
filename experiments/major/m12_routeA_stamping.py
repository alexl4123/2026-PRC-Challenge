"""m12 -- route-A stamping (BLOCK at SCHED although NM saw a normal pushback): where is it
concentrated?  If the rate q is low everywhere, the (1-q) algebra caps any classifier."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
v9 = np.load(M / 'v9s_oof.npy'); y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy()
A = np.isfinite(oa) & (oa > 0) & (oa <= 7200)
dnm = so - oa                        # NM-observed departure delay (AOBT_3 - SCHED)
st = np.abs(y - so) <= 120
e2 = (y - v9) ** 2; tot = e2.sum(); n = len(y)
print('route A, by NM delay band: n, stamped rate q, SSE share of stamped rows, oracle dRMSE (stamped rows -> off_sched)')
for lo, hi in [(600, 1200), (1200, 1800), (1800, 3600), (3600, 7200), (7200, 1e9)]:
    k = A & (dnm > lo) & (dnm <= hi)
    ks = k & st
    alt = tot - e2[ks].sum() + ((y - so) ** 2)[ks].sum()
    print(f'  dnm {lo:5.0f}-{hi:<9.0f} n={k.sum():7d} q={100*st[k].mean():5.2f}%  SSE%={100*e2[ks].sum()/tot:5.2f}  oracle {np.sqrt(alt/n)-np.sqrt(tot/n):+.2f}')
k = A & (dnm > 1200)
print('\nroute A, NM delay > 1200 s, by airport: n, q%, oracle dRMSE (12m), mean v9 - y on stamped rows')
for a in sorted(d.APT.astype(str).unique()):
    kk = k & (d.APT.astype(str).to_numpy() == a); ks = kk & st
    alt = tot - e2[ks].sum() + ((y - so) ** 2)[ks].sum()
    print(f'  {a} n={kk.sum():7d} q={100*st[kk].mean():5.2f}%  oracle {np.sqrt(alt/n)-np.sqrt(tot/n):+.2f}  bias on stamped {np.mean(v9[ks]-y[ks]) if ks.any() else 0:8.0f}')
# the (1-q) value of a PERFECTLY CALIBRATED but non-discriminating probability, per airport, vs oracle
