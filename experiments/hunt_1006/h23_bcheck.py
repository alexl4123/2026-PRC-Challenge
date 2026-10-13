import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
m = pd.read_parquet(H + 'h22_union25.parquet')
b = m[~m.onesided & m.stamp_excl]
print('2025 B rows', len(b), ' with OOF f > lo:', (b.f > b.lo).sum(), ' with f - lo > 1000:', (b.f - b.lo > 1000).sum())
print(b[b.f - b.lo > 1000][['APT', 'FLIGHT_mvt', 'month', 'y', 'f', 'lo', 'hi', 'off_sched', 'off_alobt']].sort_values('f').to_string())
print('B rows by APT:', b.APT.value_counts().to_dict())
# Rome SCHED-2h mode in 2025: route D, y - off_sched in [7100, 7300]
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'MVT', 'BLOCK', 'SCHED', 'LOBT', 'FLIGHT_mvt']); d['APT'] = d.APT.astype(str)
d['y'] = d.MVT - d.BLOCK; d['os'] = d.MVT - d.SCHED; d['date'] = pd.to_datetime(d.MVT, unit='s').dt.date
k = d[(d.y - d.os).between(7100, 7300)]
print('\n2025 rows with BLOCK = SCHED - 2h (+-100 s):', len(k), ' route D:', k.LOBT.isna().sum()); print(k.groupby(['APT', 'date']).size().to_string())
k = d[(d.y - d.os).between(3500, 3700) & d.LOBT.isna()]
print('\n2025 route-D rows with BLOCK = SCHED - 1h (+-100 s):', len(k)); print(k.groupby(['APT']).size().to_string())
m6 = pd.read_parquet(H + 'h22_union26.parquet')
x = m6[m6.MVT_ID_mvt == 213148806.0]
ts = lambda s: pd.to_datetime(s, unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
for c in ['SCHED', 'a_LOBT', 'a_AOBT3', 'MVT', 'a_MVT']: x[c + '_t'] = ts(x[c])
print(x[['APT', 'FLIGHT_mvt', 'ADES_mvt', 'SCHED_t', 'a_LOBT_t', 'a_AOBT3_t', 'MVT_t', 'a_MVT_t', 'f', 'lo', 'hi']].T.to_string())
