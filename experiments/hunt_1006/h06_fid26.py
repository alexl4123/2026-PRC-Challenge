import numpy as np, pandas as pd
pd.set_option('display.width', 250)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
d = pd.read_parquet(H + 'ra_dep.parquet', columns=['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'MVT', 'SCHED', 'LOBT', 'STAND_mvt'])
v = pd.read_parquet('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/outputs/understanding-accordion_v28.parquet')
d = d.merge(v, on='MVT_ID_mvt', how='left'); d['off_sched'] = d.MVT - d.SCHED
w = d[d.LOBT.isna() & d.FLIGHT_ID_mvt.notna()]
print(w[['APT', 'FLIGHT_mvt', 'ADES_mvt', 'off_sched', 'TAXITIME_SEC_mvt', 'STAND_mvt']].to_string())
print(pd.to_datetime(w.MVT, unit='s'))
