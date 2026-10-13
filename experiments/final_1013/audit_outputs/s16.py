import pandas as pd, numpy as np
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
C = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt', 'STAND_mvt', 'AOBT_3_flt']
for f in ['cache/major/dep_training.parquet', 'cache_final/major/dep_ranking.parquet', 'cache/major/dep_ranking.parquet']:
    d = pd.read_parquet(R + f, columns=C); k = d.duplicated(['APT', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt'], keep=False)
    k2 = d.duplicated(['APT', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt', 'FLIGHT_mvt', 'STAND_mvt'], keep=False)
    print(f, len(d), 'dup(APT,MVT,RWY)', int(k.sum()), f'{1e4*k.mean():.2f}/10k', 'fully identical flight+stand', int(k2.sum()))
    if 'final' in f:
        x = d[k].sort_values(['APT', 'MVT_TIME_UTC_mvt']); print(x.APT.value_counts().head(5).to_dict()); print(x.head(6).to_string())
