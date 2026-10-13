import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
cols = ['MVT_ID_mvt','PHASE_mvt','APT','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','LOBT_flt','EOBT_1_flt','IOBT_flt','ARVT_3_flt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_ID_mvt','CALLSIGN_flt','ADES_mvt']
a = pd.read_parquet(R+'cache/ranking.parquet', columns=cols).set_index('MVT_ID_mvt')
b = pd.read_parquet(R+'cache_final/ranking.parquet', columns=cols).set_index('MVT_ID_mvt')
print('initial', len(a), 'final', len(b), 'dup ids final', b.index.duplicated().sum())
b = b.loc[a.index]
for c in cols[1:]:
    x, y = a[c], b[c]
    if x.dtype.kind == 'M':
        print(c, x.dtype, y.dtype)
        d = (y - x).dt.total_seconds()
        ne = ~((d == 0) | (x.isna() & y.isna()))
        print(f'  {c:20s} differ {ne.mean():.4f}; diff quantiles', d[ne].quantile([0,.01,.1,.5,.9,.99,1]).round(3).tolist() if ne.any() else '')
        if ne.any():
            print('   by phase', ne.groupby(a.PHASE_mvt).mean().round(4).to_dict())
            print('   by apt', ne.groupby(a.APT).mean().round(3).to_dict())
    else:
        ne = ~((x.astype(str) == y.astype(str)))
        print(f'  {c:20s} differ {ne.mean():.4f}')
print(a.MVT_TIME_UTC_mvt.head(3).tolist(), b.MVT_TIME_UTC_mvt.head(3).tolist())
print('sub-second share initial', (a.MVT_TIME_UTC_mvt.dt.microsecond>0).mean(), 'final', (b.MVT_TIME_UTC_mvt.dt.microsecond>0).mean())
print('seconds!=0 share initial', (a.MVT_TIME_UTC_mvt.dt.second>0).mean(), 'final', (b.MVT_TIME_UTC_mvt.dt.second>0).mean())
