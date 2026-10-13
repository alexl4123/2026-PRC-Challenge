"""Mechanism check: for flights between two of the 10 airports, BLOCK - LOBT(arrival record) of route-A/B/C departures stops at
+-3606 and the clean route-D departures start right after it (one continuous distribution split by the join window)."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
m = pd.read_parquet(H + 'h22_union25.parquet')
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'ADES_mvt', 'FLIGHT_RULE_mvt', 'BLOCK', 'LOBT', 'FLIGHT_ID_mvt'])
a = pd.read_parquet(H + 'tr_arr.parquet', columns=['FLIGHT_ID_mvt', 'LOBT'])
# route A/B/C departures whose NM flight also has an arrival row at one of the 10 airports (same FLIGHT_ID)
x = d[d.LOBT.notna() & (d.FLIGHT_RULE_mvt.astype(str) == 'I')].merge(a.dropna(subset=['FLIGHT_ID_mvt']).drop_duplicates('FLIGHT_ID_mvt'), on='FLIGHT_ID_mvt', suffixes=('', '_a'))
bl_abc = (x.BLOCK - x.LOBT_a).to_numpy()
bl_d = (m.BLOCK - m.a_LOBT).to_numpy()
edges = [-1e9, -7200, -5400, -4500, -4000, -3800, -3700, -3606, -3500, -3400, -3000, 0, 3000, 3400, 3500, 3606, 3700, 3800, 4000, 4500, 5400, 7200, 1e9]
h1 = np.histogram(bl_abc, edges)[0]; h2 = np.histogram(bl_d, edges)[0]
t = pd.DataFrame({'bin': [f'({edges[k]:.0f},{edges[k + 1]:.0f}]' for k in range(len(edges) - 1)], 'route A/B/C (linked)': h1, 'clean route D': h2,
                  'width_s': np.diff(edges)})
print(f'route A/B/C IFR departures with an NM arrival row (same FLIGHT_ID): {len(x)}; clean route-D linked: {len(m)}')
print(t.to_string(index=False))
print('per-second density just inside (3500,3606] A/B/C:', round(h1[14] / 106, 2), '  just outside (3606,3700] route D:', round(h2[15] / 94, 2), ' (3700,3800]', round(h2[16] / 100, 2))
print('negative side: (-3606,-3500] A/B/C', round(h1[7] / 106, 2), '/s;  (-3700,-3606] route D', round(h2[6] / 94, 2), '/s')
