"""s02 -- the moderate-delay band (1800 < y <= 7200, NM record present): what physically
happened, and how much of it can the (approved, look-ahead) stand-turnover signal see?

Diagnostic only: uses the departure's true BLOCK time to classify rows, never as a feature.
For each departure: is there an ARRIVAL at the same (airport, stand) whose in-block falls
in (our BLOCK, our takeoff)?  If yes, the next occupant physically parked on our stand
while we were still 'taxiing' -> the long taxi is physical and the stand-turnover family
observes it.  Also (BLOCK, AOBT_3): the aircraft had physically left the stand before NM's
off-block, i.e. NM's clock is late (the 'NM-blind' mechanism).

Observable version (what a feature can use): an arrival at our stand with in-block in
(our SCHED, our takeoff).  Its lift for the delay band and the SSE share it covers.

Inputs : cache/major/{dep,arr}_training.parquet, cache/major/v9s_oof.npy
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import peak_rss_gb

M = CACHE / 'major'
dep = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'STAND_mvt', 'month', 'y', 'off_sched', 'off_aobt3',
                                                          'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'])
arr = pd.read_parquet(M / 'arr_training.parquet', columns=['APT', 'STAND_mvt', 'BLOCK_TIME_UTC_mvt'])
v9 = np.load(M / 'v9s_oof.npy')
y = dep.y.to_numpy(); oa = dep.off_aobt3.to_numpy(); so = dep.off_sched.to_numpy()
mon = dep.month.to_numpy(); apt = dep.APT.astype(str).to_numpy()
n = len(y)
D = ~np.isfinite(oa); A = ~D
P = D & (so > 7200)
S = np.isin(mon, [1, 7])
e2 = (y - v9) ** 2


def sec(s):
    s = s.dt.tz_convert('UTC').dt.tz_localize(None) if getattr(s.dt, 'tz', None) is not None else s
    v = s.astype('datetime64[us]').astype('int64') // 1_000_000
    return np.where(s.isna().to_numpy(), -1, v.to_numpy())


# key codes over the union of (APT, STAND)
kd = (dep.APT.astype(str) + '|' + dep.STAND_mvt.astype(object).fillna('NA').astype(str))
ka = (arr.APT.astype(str) + '|' + arr.STAND_mvt.astype(object).fillna('NA').astype(str))
codes, _ = pd.factorize(pd.concat([kd, ka], ignore_index=True))
cd = codes[:len(kd)].astype(np.int64); ca = codes[len(kd):].astype(np.int64)
okd = ~pd.isna(dep.STAND_mvt).to_numpy()
tin = sec(arr.BLOCK_TIME_UTC_mvt)
oka = ~pd.isna(arr.STAND_mvt).to_numpy() & (tin > 0)
BIG = np.int64(10 ** 10)
code_a = np.sort(ca[oka] * BIG + tin[oka])
t_to = sec(dep.MVT_TIME_UTC_mvt); t_bl = sec(dep.BLOCK_TIME_UTC_mvt); t_sc = sec(dep.SCHED_TIME_UTC_mvt)
t_aobt = t_to - np.where(np.isfinite(oa), oa, 0).astype(np.int64)


def n_between(t0, t1):
    """number of arrivals at our stand with in-block in (t0, t1)."""
    lo = np.searchsorted(code_a, cd * BIG + t0, side='right')
    hi = np.searchsorted(code_a, cd * BIG + t1, side='left')
    out = np.where(okd & (t1 > t0), hi - lo, 0)
    return out


nx_bl_to = n_between(t_bl, t_to) > 0            # next occupant parked while we 'taxied'
nx_bl_aobt = n_between(t_bl, t_aobt) > 0        # ... and before NM's off-block
obs = n_between(t_sc, t_to) > 0                 # OBSERVABLE: an arrival parked on our stand after our SCHED, before our takeoff
obs30 = n_between(t_to - 1800, t_to) > 0        # observable: an arrival parked on our stand in the 30 min before our takeoff

stamped = np.abs(y - so) <= 120
cls = np.full(n, 'other', dtype=object)
cls[A & (y <= 1800)] = 'A_normal'
cls[A & (y > 1800) & (y <= 7200) & ~stamped] = 'A_band_phys'
cls[A & (y > 1800) & (y <= 7200) & stamped] = 'A_band_stamped'
cls[D & ~P & (y <= 7200)] = 'D_bulk'

for scope, m in [('12 months', np.ones(n, bool)), ('scored 1+7', S)]:
    tot = e2[m].sum()
    print(f'\n=== {scope}: share of rows with an arrival at our stand in the window  (and SSE share of the cell)')
    rows = []
    for c in ['A_normal', 'A_band_phys', 'A_band_stamped', 'D_bulk']:
        k = m & (cls == c)
        rows.append(dict(cell=c, n=k.sum(), sse_pct_total=100 * e2[k].sum() / tot,
                         in_BLOCK_TO=100 * nx_bl_to[k].mean(), in_BLOCK_AOBT=100 * nx_bl_aobt[k].mean(),
                         obs_SCHED_TO=100 * obs[k].mean(), obs_30min=100 * obs30[k].mean(),
                         sse_share_obs=100 * e2[k & obs].sum() / max(e2[k].sum(), 1),
                         sse_share_phys_seen=100 * e2[k & nx_bl_to].sum() / max(e2[k].sum(), 1)))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

print('\n=== per airport, route-A delay band (physical), 12 months')
rows = []
for a in sorted(set(apt)):
    k = (apt == a) & (cls == 'A_band_phys'); kn = (apt == a) & (cls == 'A_normal')
    gap = (t_aobt[k] - t_bl[k])
    rows.append(dict(apt=a, n=k.sum(), med_AOBT_minus_BLOCK=np.median(gap), p25=np.percentile(gap, 25), p75=np.percentile(gap, 75),
                     band_nx_bl_to=100 * nx_bl_to[k].mean(), band_nx_bl_aobt=100 * nx_bl_aobt[k].mean(),
                     band_obs=100 * obs[k].mean(), normal_obs=100 * obs[kn].mean(),
                     lift_obs=obs[k].mean() / max(obs[kn].mean(), 1e-9),
                     band_resid_obs=np.mean(y[k & obs] - v9[k & obs]) if (k & obs).any() else np.nan,
                     band_resid_notobs=np.mean(y[k & ~obs] - v9[k & ~obs])))
print(pd.DataFrame(rows).round(2).to_string(index=False))

# how much a single observable flag could buy if used perfectly as a bin-mean residual correction
# (nested LOMO, per airport x route x flag x off_sched band) -- a crude ceiling for the look-ahead flag alone
route = np.where(A, 'A', 'D')
sob = np.digitize(np.nan_to_num(so, nan=-1), [0, 600, 1200, 1800, 3600, 7200])
grp = pd.Series(apt).astype(str) + route + obs.astype(int).astype(str) + obs30.astype(int).astype(str) + sob.astype(str)
g = grp.to_numpy()
res = y - v9
corr = np.zeros(n)
fit = ~P & (y <= 7200)
for mm in range(1, 13):
    tr = fit & (mon != mm)
    means = pd.Series(res[tr]).groupby(g[tr]).agg(['mean', 'size'])
    sh = means['mean'] * means['size'] / (means['size'] + 200)       # shrunk
    te = (mon == mm) & ~P
    corr[te] = pd.Series(g[te]).map(sh).fillna(0).to_numpy()
adj = np.clip(v9 + corr, 0, None)
r = lambda a, m: np.sqrt(np.mean((y[m] - a[m]) ** 2))
al = np.ones(n, bool)
print(f'\nobservable-flag bin correction (nested): 12m {r(adj, al):.2f} vs {r(v9, al):.2f} ({r(adj, al)-r(v9, al):+.2f}); '
      f'scored {r(adj, S):.2f} vs {r(v9, S):.2f} ({r(adj, S)-r(v9, S):+.2f})')
print('peak', round(peak_rss_gb(), 2), 'GB')
