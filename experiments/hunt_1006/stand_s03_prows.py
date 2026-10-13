"""Zone-P / route-D rows of 2025: where is the recorded BLOCK relative to arrivals at the same stand?"""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
t = pd.read_parquet(H + 'stand_ev_training.parquet', columns=['id', 'PHASE_mvt', 'APT', 'STAND_mvt', 'mvt', 'blk', 'sch', 'aobt3', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt'])
dep = t[t.PHASE_mvt == 'DEP'].reset_index(drop=True); arr = t[t.PHASE_mvt == 'ARR'].reset_index(drop=True)
o = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
assert np.array_equal(o.MVT_ID_mvt.to_numpy(), dep.id.to_numpy()), 'order'
z = np.load(R + 'cache/w2/v28s_oof.npz')
P, D, f, y = z['P'].astype(bool), z['D'].astype(bool), z['f'], z['y']
dep['y'] = dep.mvt - dep.blk
assert np.allclose(dep.y.to_numpy(), y)
dep['key'] = dep.APT.astype(str) + '|' + dep.STAND_mvt.astype(str)
arr['key'] = arr.APT.astype(str) + '|' + arr.STAND_mvt.astype(str)
arr = arr[np.isfinite(arr.blk) & arr.STAND_mvt.notna()].sort_values(['key', 'blk'])
A = {k: g.blk.to_numpy() for k, g in arr.groupby('key', sort=False)}
AT = {k: g.AIRCRAFT_TYPE_mvt.astype(str).to_numpy() for k, g in arr.groupby('key', sort=False)}


def arrivals_between(key, lo, hi):
    a = A.get(key)
    if a is None:
        return np.empty(0)
    return a[(a > lo) & (a < hi)]


rows = np.flatnonzero(D)
out = []
for i in rows:
    r = dep.iloc[i]
    a = A.get(r.key, np.empty(0))
    T, S, B = r.mvt, r.sch, r.blk
    j = np.searchsorted(a, T)  # arrivals strictly before T: a[:j]
    am = a[j - 1] if j > 0 else np.nan          # latest arrival before take-off
    am2 = a[j - 2] if j > 1 else np.nan
    n_sched_T = ((a > S) & (a < T)).sum()
    nxt_after_B = a[np.searchsorted(a, B, side='right')] if np.searchsorted(a, B, side='right') < len(a) else np.nan
    prev_before_B = a[np.searchsorted(a, B) - 1] if np.searchsorted(a, B) > 0 else np.nan
    out.append((i, r.APT, P[i], T - S, r.y, f[i], T - am, T - am2, n_sched_T, nxt_after_B - B, B - prev_before_B))
q = pd.DataFrame(out, columns=['i', 'APT', 'P', 'off_sched', 'y', 'f', 'T_am', 'T_am2', 'n_arr_sched_T', 'next_arr_after_B', 'B_minus_prev_arr'])
q.to_parquet(H + 'stand_s03_D.parquet')
q['B_sched'] = q.off_sched - q.y  # B - SCHED
q['stamp'] = q.B_sched.abs() <= 60
print('route D rows', len(q), 'P', q.P.sum())
for nm, g in (('P', q[q.P]), ('D~P', q[~q.P])):
    print('=====', nm, len(g))
    print('B-SCHED quantiles', np.percentile(g.B_sched, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
    print('stamp share (|B-SCHED|<=60)', g.stamp.mean())
    print('crosstab n_arr in (SCHED,T) capped 3 vs stamp')
    print(pd.crosstab(g.n_arr_sched_T.clip(0, 3), g.stamp))
    # arrival at stand between B and T exists? (next arrival after B before T)
    g2 = g.assign(arr_in_taxi=g.next_arr_after_B < g.y)
    print('arrival at stand during (B,T):', g2.arr_in_taxi.mean())
    print(pd.crosstab(g2.arr_in_taxi, g2.stamp))
