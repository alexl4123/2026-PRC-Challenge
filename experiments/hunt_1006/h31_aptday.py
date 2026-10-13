import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
d = pd.read_parquet(H + 'ra_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT', 'LOBT', 'FLIGHT_RULE_mvt']); d['APT'] = d.APT.astype(str)
d['date'] = pd.to_datetime(d.MVT, unit='s').dt.date; d['D'] = d.LOBT.isna() & (d.FLIGHT_RULE_mvt.astype(str) == 'I')
s = d.groupby(['APT', 'date']).D.mean().rename('share')
c = pd.read_csv(H + 'cand_moves.csv'); c = c.merge(d[['MVT_ID_mvt', 'date']], on='MVT_ID_mvt').merge(s.reset_index(), on=['APT', 'date'])
print(c[['APT', 'FLIGHT_mvt', 'date', 'share']].sort_values('share', ascending=False).head(12).to_string())
t = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'MVT', 'LOBT', 'FLIGHT_RULE_mvt']); t['APT'] = t.APT.astype(str)
t['date'] = pd.to_datetime(t.MVT, unit='s').dt.date; t['D'] = t.LOBT.isna() & (t.FLIGHT_RULE_mvt.astype(str) == 'I')
s5 = t.groupby(['APT', 'date']).D.mean()
print('2025 airport-day share q.99', round(s5.quantile(.99), 4), ' max', round(s5.max(), 4), ' | 2026 q.99', round(s.quantile(.99), 4))
