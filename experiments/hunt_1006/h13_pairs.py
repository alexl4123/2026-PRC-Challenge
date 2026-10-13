import numpy as np, pandas as pd
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 400)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'SCHED', 'MVT', 'BLOCK', 'LOBT', 'AOBT3', 'STAND_mvt', 'RUNWAY_mvt', 'CALLSIGN_flt', 'AIRCRAFT_TYPE_mvt'])
d['APT'] = d.APT.astype(str); d['y'] = d.MVT - d.BLOCK
g = d.dropna(subset=['FLIGHT_mvt']).copy()
g['n'] = g.groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True).MVT.transform('size'); g = g[g.n == 2].sort_values(['APT', 'FLIGHT_mvt', 'SCHED', 'MVT'])
a = g.groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True)
p = pd.DataFrame({'APT': a.APT.first(), 'nm1': a.LOBT.apply(lambda x: x.notna().iloc[0]), 'nm2': a.LOBT.apply(lambda x: x.notna().iloc[1]),
                  'y1': a.y.first(), 'y2': a.y.last(), 'dMVT': a.MVT.last() - a.MVT.first(), 'dBLOCK': a.BLOCK.last() - a.BLOCK.first(),
                  'off_sched1': a.MVT.first() - a.SCHED.first(), 'stand_eq': a.STAND_mvt.apply(lambda x: x.iloc[0] == x.iloc[1]),
                  'rwy_eq': a.RUNWAY_mvt.apply(lambda x: x.iloc[0] == x.iloc[1]), 'mid1': a.MVT_ID_mvt.first(), 'mid2': a.MVT_ID_mvt.last(),
                  'cs1': a.CALLSIGN_flt.first(), 'cs2': a.CALLSIGN_flt.last()}).reset_index(drop=True)
print(p.APT.value_counts().to_string())
print(p.groupby(['nm1', 'nm2']).size())
p['cls'] = np.where(p.dMVT < 120, 'sameMVT', 'later')
print(p.groupby(['APT', 'cls']).size())
print(p[p.cls == 'sameMVT'].head(60).to_string())
print(p[p.cls == 'later'].head(40).to_string())
