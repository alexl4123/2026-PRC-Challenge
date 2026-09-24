"""s10 -- the tail inside P: what classifier quality would capture HALF of the remaining oracle?

Inside P the mixture predicts  p*et + (1-p)*pb  (et = off_sched, pb = bulk regressor, p = v7's
specialist, exp62 arm b).  Replace p by  p_a = (1-a)*p + a*1[y>7200]  (a = 0 current, a = 1 oracle)
and report the RMSE change on 12 months and on the scored months, with AUC / Brier of p_a, so that
'capture half' translates into a required discrimination level.  Everything outside P stays v9~.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from prc.config import CACHE

M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy(); mon = d.month.to_numpy()
apt = d.APT.astype(str).to_numpy()
P = ~np.isfinite(oa) & (so > 7200)
Pidx = np.flatnonzero(P)
v9 = np.load(M / 'v9s_oof.npy')
pb = np.load(CACHE / 'exp62_pb.npy').astype(float)
p = np.load(CACHE / 'exp62_spec_P.npy')[1]
assert len(p) == len(Pidx)
yP = y[Pidx]; etP = so[Pidx]; pbP = pb[Pidx]; t = (yP > 7200).astype(float)
recon = np.clip(p * etP + (1 - p) * pbP, 0, None)
print('reconstruction vs v9~ inside P: max |diff|', np.nanmax(np.abs(recon - v9[Pidx])).round(3))
S = np.isin(mon, [1, 7]); n = len(y)
base_sse = (y - v9) ** 2
lirf = apt[Pidx] == 'LIRF'


def total(pa):
    pr = np.clip(pa * etP + (1 - pa) * pbP, 0, None)
    e = base_sse.copy(); e[Pidx] = (yP - pr) ** 2
    return np.sqrt(e.mean()), np.sqrt(e[S].mean())


r0 = total(p); r1 = total(t)
print(f'current  12m {r0[0]:.2f}  scored {r0[1]:.2f}   | oracle 12m {r1[0]:.2f} ({r1[0]-r0[0]:+.2f})  scored {r1[1]:.2f} ({r1[1]-r0[1]:+.2f})')
print(f'P rows {len(Pidx)}, tails {int(t.sum())}; LIRF P {lirf.sum()} (tails {int(t[lirf].sum())})')
print(f'current AUC all P {roc_auc_score(t, p):.3f}  LIRF {roc_auc_score(t[lirf], p[lirf]):.3f}  non-LIRF {roc_auc_score(t[~lirf], p[~lirf]):.3f}; '
      f'Brier {np.mean((p-t)**2):.4f}')
print('\n  a     12m    d12m   scored   dS    AUC(all)  AUC(LIRF)  Brier')
for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    pa = (1 - a) * p + a * t
    r = total(pa)
    print(f'{a:4.1f} {r[0]:8.2f} {r[0]-r0[0]:+7.2f} {r[1]:8.2f} {r[1]-r0[1]:+7.2f}   {roc_auc_score(t, pa):.3f}     '
          f'{roc_auc_score(t[lirf], pa[lirf]):.3f}   {np.mean((pa-t)**2):.4f}')
# where is the remaining P error: false alarms vs missed tails, by airport
pr = recon
e = (yP - pr) ** 2
print('\nremaining P SSE (12m) split: missed tails (t=1) %.1f%%, false alarms (t=0) %.1f%%' % (100 * e[t == 1].sum() / e.sum(), 100 * e[t == 0].sum() / e.sum()))
tab = pd.DataFrame({'apt': apt[Pidx], 't': t, 'e': e}).groupby(['apt', 't']).e.sum().unstack(fill_value=0) / e.sum() * 100
print(tab.round(1).to_string())
