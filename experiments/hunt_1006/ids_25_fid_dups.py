"""ids_25 -- DEP rows sharing a FLIGHT_ID (same NM flight on two departure rows): 2025 labels, 2026 v28 preds."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
c = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'STAND_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'LOBT_flt', 'y', 'off_lobt']
for w in ['training', 'ranking']:
    d = pd.read_parquet(H + f'cache/major/dep_{w}.parquet', columns=c)
    dd = d[d.FLIGHT_ID_mvt.notna() & d.FLIGHT_ID_mvt.duplicated(keep=False)].sort_values(['FLIGHT_ID_mvt', 'MVT_ID_mvt'])
    if w == 'ranking':
        v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet')
        dd = dd.merge(v, on='MVT_ID_mvt')
    print('=====', w); print(dd.to_string())
