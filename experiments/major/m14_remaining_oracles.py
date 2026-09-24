"""m14 -- oracle ceilings REMAINING on top of v9~ (hidden class known, everything else fixed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy(); mon = d.month.to_numpy()
apt = d.APT.astype(str).to_numpy(); n = len(y)
v9 = np.load(M / 'v9s_oof.npy'); pb = np.load(CACHE / 'exp62_pb.npy').astype(float)
D = ~np.isfinite(oa); P = D & (so > 7200); A = ~D
st = np.abs(y - so) <= 120
S = np.isin(mon, [1, 7])
def g(pred, m): return np.sqrt(np.mean((y[m] - pred[m]) ** 2))
orc = {}
o = v9.copy(); k = P; o[k] = np.where(st[k] | (y[k] > 7200), so[k], pb[k]); orc['P: artefact vs real known'] = o
o = v9.copy(); k = A & st & (so > 1800); o[k] = so[k]; orc['route A: stamped-at-SCHED known (all apts)'] = o
o = v9.copy(); k = A & st & (so > 1800) & (apt == 'LIRF'); o[k] = so[k]; orc['route A: stamped known, LIRF only'] = o
o = v9.copy(); k = D & ~P & st & (so > 1800); o[k] = so[k]; orc['route D notP: stamped known'] = o
o = v9.copy()
for kk, val in [(P, np.where(st | (y > 7200), so, pb)), (A & st & (so > 1800), so), (D & ~P & st & (so > 1800), so)]:
    o[kk] = val[kk]
orc['ALL recording classes known'] = o
print(f'v9~ baseline: 12m {g(v9, np.ones(n, bool)):.2f}  scored {g(v9, S):.2f}')
for nm, o in orc.items():
    print(f'  {nm:45s} 12m {g(o, np.ones(n, bool)) - g(v9, np.ones(n, bool)):+7.2f}   scored {g(o, S) - g(v9, S):+7.2f}')
