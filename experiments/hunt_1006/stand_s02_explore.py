"""Basic structure: FLIGHT_ID linkage, stand alternation in 2025 recorded data."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
t = pd.read_parquet(H + 'stand_ev_training.parquet')
print(t.head(3).T)
dep = t[t.PHASE_mvt == 'DEP']; arr = t[t.PHASE_mvt == 'ARR']
print('dep', len(dep), 'arr', len(arr))
print('FLIGHT_ID unique dep', dep.FLIGHT_ID_mvt.nunique(), 'arr', arr.FLIGHT_ID_mvt.nunique())
both = set(dep.FLIGHT_ID_mvt.dropna()) & set(arr.FLIGHT_ID_mvt.dropna())
print('FLIGHT_ID in both phases', len(both))
# same FLIGHT_ID ARR & DEP at the same airport?
k = dep[['FLIGHT_ID_mvt', 'APT']].merge(arr[['FLIGHT_ID_mvt', 'APT']], on=['FLIGHT_ID_mvt', 'APT'])
print('FLIGHT_ID shared at same APT', len(k))
print('stand nan dep', dep.STAND_mvt.isna().mean(), 'arr', arr.STAND_mvt.isna().mean())
print('sample FLIGHT_mvt dep', dep.FLIGHT_mvt.head(5).tolist(), 'callsign', dep.CALLSIGN_flt.head(5).tolist())
print('sample FLIGHT_mvt arr', arr.FLIGHT_mvt.head(5).tolist(), 'callsign', arr.CALLSIGN_flt.head(5).tolist())
# arrival block - mvt
ab = arr.blk - arr.mvt
print('arr taxi-in quantiles', np.nanpercentile(ab, [0, 0.1, 1, 50, 99, 99.9, 100]))

# alternation per stand: events sorted by time
ev = pd.concat([
    pd.DataFrame({'APT': dep.APT.astype(str).values, 'S': dep.STAND_mvt.astype(str).values, 't': dep.blk.values, 'k': 1}),
    pd.DataFrame({'APT': arr.APT.astype(str).values, 'S': arr.STAND_mvt.astype(str).values, 't': arr.blk.values, 'k': 0})])
ev = ev[(ev.S != 'nan') & np.isfinite(ev.t)].sort_values(['APT', 'S', 't', 'k']).reset_index(drop=True)
same = (ev.APT.values[1:] == ev.APT.values[:-1]) & (ev.S.values[1:] == ev.S.values[:-1])
kk = ev.k.values
rep = same & (kk[1:] == kk[:-1])
print('pairs', same.sum(), 'repeat same kind', rep.sum(), rep.sum() / same.sum())
r = pd.DataFrame({'APT': ev.APT.values[1:][same], 'rep': rep[same], 'rk': kk[1:][same]})
print(r.groupby('APT').agg(n=('rep', 'size'), rep=('rep', 'mean')))
print('repeat DEP-DEP share', (rep & (kk[1:] == 1)).sum(), 'ARR-ARR', (rep & (kk[1:] == 0)).sum())
