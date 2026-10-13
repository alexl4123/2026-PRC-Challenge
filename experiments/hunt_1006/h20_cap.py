"""Is y capped at ~3600 s at some airports (data filter)?"""
import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'MVT', 'BLOCK', 'SCHED', 'LOBT', 'AOBT3'])
d['APT'] = d.APT.astype(str); d['y'] = d.MVT - d.BLOCK
bins = [2400, 2700, 3000, 3300, 3500, 3600, 3606, 3700, 3900, 4200, 4800, 6000, 1e9]
print(pd.crosstab(d.APT, pd.cut(d.y, bins)).to_string())
d['mon'] = pd.to_datetime(d.MVT, unit='s').dt.month
for a in ['LEBL', 'LEMD']:
    x = d[d.APT == a]
    print(a, 'y max by month', x.groupby('mon').y.max().to_dict())
    print(a, 'MVT - AOBT3 > 3606 (label-free) rows:', ((x.MVT - x.AOBT3) > 3606).sum(), ' MVT - SCHED > 3606:', ((x.MVT - x.SCHED) > 3606).sum(), ' of', len(x))
    k = x[(x.MVT - x.AOBT3) > 3606]
    print('   those rows: y quantiles', k.y.quantile([0, .5, .9, 1]).round().to_dict(), ' (MVT-AOBT3) quantiles', (k.MVT - k.AOBT3).quantile([0, .5, .9, 1]).round().to_dict())
for a in ['EDDF', 'EDDM', 'LSZH', 'EHAM']:
    x = d[d.APT == a]; k = x[(x.MVT - x.AOBT3) > 3606]
    print(a, 'MVT - AOBT3 > 3606 rows', len(k), ' y>3606 among them', (k.y > 3606).sum())
r = pd.read_parquet(H + 'ra_dep.parquet', columns=['APT', 'MVT', 'SCHED', 'AOBT3', 'LOBT', 'MVT_ID_mvt']); r['APT'] = r.APT.astype(str)
v = pd.read_parquet('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/outputs/understanding-accordion_v28.parquet')
r = r.merge(v, on='MVT_ID_mvt')
for a in ['LEBL', 'LEMD']:
    x = r[r.APT == a]
    print(f'2026 {a}: rows {len(x)}, v28 > 3606: {(x.TAXITIME_SEC_mvt > 3606).sum()}, max {x.TAXITIME_SEC_mvt.max()}, MVT-AOBT3 > 3606: {((x.MVT - x.AOBT3) > 3606).sum()}, MVT-SCHED > 3606: {((x.MVT - x.SCHED) > 3606).sum()}')
    print('   v28 > 3000:', x[x.TAXITIME_SEC_mvt > 3000].TAXITIME_SEC_mvt.sort_values().tolist())
