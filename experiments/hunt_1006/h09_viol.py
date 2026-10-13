import numpy as np, pandas as pd
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
m = pd.read_parquet(H + 'h07_link25.parquet')
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'MVT', 'BLOCK', 'LOBT', 'CALLSIGN_flt'])
used = set(d.FLIGHT_ID_mvt.dropna().unique())
m['fid_used_by_dep'] = m.a_FLIGHT_ID_mvt.isin(used)
m['excl'] = (m.BLOCK - m.a_LOBT).abs() > 3606
print(pd.crosstab(m.fid_used_by_dep, m.excl))
ts = lambda s: pd.to_datetime(s, unit='s').dt.strftime('%m-%d %H:%M:%S')
v = m[~m.excl].copy()
for c in ['MVT', 'BLOCK', 'SCHED', 'a_LOBT', 'a_AOBT3', 'a_IOBT', 'a_EOBT1']: v[c + '_t'] = ts(v[c])
print(v[['APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'ADES_mvt', 'FLIGHT_RULE_mvt', 'fid_used_by_dep', 'SCHED_t', 'BLOCK_t', 'MVT_t', 'a_LOBT_t', 'a_IOBT_t', 'a_EOBT1_t', 'a_AOBT3_t']].to_string())
# other departure rows with the same callsign (CALLSIGN_flt) as the arrival's, same airport, within 1 day, that DO have NM
dd = d[d.LOBT.notna()][['APT', 'CALLSIGN_flt', 'MVT', 'LOBT', 'FLIGHT_ID_mvt']].copy(); dd['APT'] = dd.APT.astype(str); dd['CALLSIGN_flt'] = dd.CALLSIGN_flt.astype(str)
j = v[['i', 'APT', 'a_CALLSIGN_flt', 'MVT', 'a_LOBT', 'a_FLIGHT_ID_mvt']].merge(dd, left_on=['APT', 'a_CALLSIGN_flt'], right_on=['APT', 'CALLSIGN_flt'], suffixes=('', '_o'))
j = j[(j.MVT_o - j.MVT).abs() < 86400]
print('\nviolators: other NM departures with the same callsign within 1 day:'); print(j.assign(dt=j.MVT_o - j.MVT, lobt_eq=j.LOBT == j.a_LOBT, fid_eq=j.FLIGHT_ID_mvt == j.a_FLIGHT_ID_mvt)[['APT', 'a_CALLSIGN_flt', 'dt', 'lobt_eq', 'fid_eq']].to_string())
