"""Label-free: per-day share of departures without an NM record (route D) in 2025 and 2026; days of the candidate rows."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
out = {}
for yr in ('tr', 'ra'):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT', 'LOBT', 'FLIGHT_RULE_mvt'])
    d['date'] = pd.to_datetime(d.MVT, unit='s').dt.date; d['D'] = d.LOBT.isna() & (d.FLIGHT_RULE_mvt.astype(str) == 'I')
    s = d.groupby('date').D.mean(); out[yr] = (d, s)
    print(yr, 'per-day IFR route-D share: median', round(s.median(), 4), ' q.95', round(s.quantile(.95), 4), ' max', round(s.max(), 4), ' at', s.idxmax())
d6, s6 = out['ra']
m6 = pd.read_parquet(H + 'h22_union26.parquet')
k = m6[(m6.AB != m6.f)].copy(); k['date'] = pd.to_datetime(k.MVT, unit='s').dt.date
print('candidate-row days and that day\'s IFR route-D share (2026):')
print(k.groupby('date').size().to_frame('rows').join(s6.rename('routeD_share')).to_string())
a = d6.groupby(['date', 'APT']).D.mean().unstack()
print('2026 top-10 airport-days by IFR route-D share:'); print(a.stack().sort_values(ascending=False).head(10).to_string())
d5, s5 = out['tr']; a5 = d5.groupby(['date', 'APT']).D.mean().unstack()
print('2025 top-10 airport-days:'); print(a5.stack().sort_values(ascending=False).head(10).to_string())
