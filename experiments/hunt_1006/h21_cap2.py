"""LEBL/LEMD cap: arrivals' taxi-in in 2025 and 2026 (label-free consistency), OOF effect, 2026 moves."""
import numpy as np, pandas as pd
pd.set_option('display.width', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
for yr in ('tr', 'ra'):
    a = pd.read_parquet(H + f'{yr}_arr.parquet', columns=['APT', 'MVT', 'BLOCK']); a['APT'] = a.APT.astype(str); a['ti'] = a.BLOCK - a.MVT
    g = a.groupby('APT').ti
    print(yr, 'ARR taxi-in: max by APT', g.max().to_dict())
    print(yr, '   count > 3606 by APT', g.apply(lambda s: int((s > 3606).sum())).to_dict(), '  count in (3000,3606]', g.apply(lambda s: int(((s > 3000) & (s <= 3606)).sum())).to_dict())
    print(yr, '   min by APT', g.min().to_dict())
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'MVT', 'BLOCK']); d['APT'] = d.APT.astype(str); d['y'] = d.MVT - d.BLOCK
print('2025 DEP y in (3590, 3620] at LEBL/LEMD:', d[d.APT.isin(['LEBL', 'LEMD']) & (d.y > 3590)].y.sort_values().tolist())
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']
sp = d.APT.isin(['LEBL', 'LEMD']).to_numpy()
for cap in (3606, 3612):
    g2 = f.copy(); k = sp & (f > cap); g2[k] = cap
    r = lambda h, kk: np.sqrt(np.mean((y[kk] - h[kk]) ** 2)); ALL = np.ones(len(y), bool); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
    dd = (y - g2) ** 2 - (y - f) ** 2
    print(f'OOF cap {cap}: rows changed {k.sum()}, worse {(dd > 0).sum()}, 12m {r(g2, ALL) - r(f, ALL):+.5f}, held {r(g2, held) - r(f, held):+.5f}, Jan+Jul {r(g2, S) - r(f, S):+.5f}')
r6 = pd.read_parquet(H + 'ra_dep.parquet', columns=['APT', 'MVT_ID_mvt', 'LOBT', 'MVT', 'SCHED']); r6['APT'] = r6.APT.astype(str)
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet'); r6 = r6.merge(v, on='MVT_ID_mvt')
for cap in (3606, 3612):
    k = r6.APT.isin(['LEBL', 'LEMD']) & (r6.TAXITIME_SEC_mvt > cap)
    s = (r6.TAXITIME_SEC_mvt[k] - cap)
    print(f'2026 cap {cap}: rows {k.sum()}, moves {s.tolist()}, routes nm={r6[k].LOBT.notna().tolist()}, off_sched {(r6[k].MVT - r6[k].SCHED).tolist()}, bound {np.sqrt(235.3996 ** 2 - (s ** 2).sum() / 344841) - 235.3996:+.5f}')
