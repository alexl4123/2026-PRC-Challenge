"""MVT_ID assignment order: within (APT, SCHED) groups, does ID order follow BLOCK, take-off or flight number?"""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
t = pd.read_parquet(H + 'stand_ev_training.parquet', columns=['id', 'PHASE_mvt', 'APT', 'mvt', 'blk', 'sch', 'FLIGHT_mvt', 'STAND_mvt'])
t = t.sort_values('id').reset_index(drop=True)
# global layout: runs of APT along the id axis
apt = t.APT.astype(str).to_numpy()
print('APT changes along id order:', int((apt[1:] != apt[:-1]).sum()), 'of', len(t) - 1)
sch = t.sch.to_numpy(); same = apt[1:] == apt[:-1]
print('sched nondecr along id within APT (both phases):', np.mean((sch[1:] >= sch[:-1])[same]))
day = (t.sch // 86400).to_numpy()
print('sched day nondecr along id within APT:', np.mean((day[1:] >= day[:-1])[same]))
# where SCHED decreases, by how much
d = (sch[1:] - sch[:-1])[same]
print('negative sched steps quantiles (s):', np.percentile(d[d < 0], [0, 1, 10, 50, 90, 99, 100]) if (d < 0).any() else None)
dep = t[t.PHASE_mvt == 'DEP'].copy()
dep['fn'] = dep.FLIGHT_mvt.astype(str)
g = dep.groupby(['APT', 'sch'], observed=True)
dep['gs'] = g.id.transform('size')
m = dep[dep.gs >= 2].sort_values(['APT', 'sch', 'id'])
k = (m.APT.astype(str) + m.sch.astype(str)).to_numpy()
s2 = k[1:] == k[:-1]
for col in ('blk', 'mvt'):
    v = m[col].to_numpy(); print(f'within (APT,SCHED) groups, consecutive ids: {col} increasing share', np.mean((v[1:] > v[:-1])[s2]), 'ties', np.mean((v[1:] == v[:-1])[s2]))
fn = m.fn.to_numpy(); print('flight number alphabetical increasing share', np.mean((fn[1:] > fn[:-1])[s2]))
st = m.STAND_mvt.astype(str).to_numpy(); print('stand alphabetical increasing share', np.mean((st[1:] > st[:-1])[s2]))
# per airport for blk vs mvt
m['nb'] = np.r_[np.nan, np.where(s2, (m.blk.to_numpy()[1:] > m.blk.to_numpy()[:-1]).astype(float), np.nan)]
m['nm'] = np.r_[np.nan, np.where(s2, (m.mvt.to_numpy()[1:] > m.mvt.to_numpy()[:-1]).astype(float), np.nan)]
print(m.groupby('APT', observed=True)[['nb', 'nm']].mean())
