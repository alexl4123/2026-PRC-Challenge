"""SKEPTIC: Rome rule B (UTC-date crossing -> stamp) support check: carrier mix and model p of 2025 crossing rows vs RYR9ZM;
RYR zone-P class rates at LIRF; T rule (>8h) support; VOE87GA analogs."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
x = pd.read_parquet(R + 'cache/hunt/hedge_lirfP.parquet')
W = np.load(R + 'cache/w2/v24s_oof.npz', allow_pickle=True)
Pidx, p14 = W['Pidx'], W['p14']
dt = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
pm = pd.Series(p14, index=dt.MVT_ID_mvt.to_numpy()[Pidx])
x['p'] = pm.reindex(x.MVT_ID_mvt).to_numpy()
x['car'] = x.FLIGHT_mvt.astype(str).str[:3]
x['du'] = ((x.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - x.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
print('cls values', x.cls.value_counts().to_dict())
B = x[x.du >= 1]
print('\n2025 crossing rows (B):', len(B)); print(pd.crosstab(B.car, B.cls))
print('p14 of crossing rows by class:'); print(B.groupby('cls').p.describe())
print('crossing rows with p < 0.6:'); print(B[B.p < 0.6][['FLIGHT_mvt', 'off_sched', 'y', 'f', 'p', 'cls']])
print('\n2025 LIRF zone P by carrier (top) x class:'); t = pd.crosstab(x.car, x.cls); t['n'] = t.sum(1); print(t.sort_values('n', ascending=False).head(12))
r = x[x.car == 'RYR']
print('\nRYR LIRF zone P rows by off_sched band x class:'); print(pd.crosstab(pd.cut(r.off_sched, [7200, 10800, 14400, 21600, 28800, 1e6]), r.cls))
print('\nall LIRF zone P by off_sched band x class (no crossing):'); nc = x[x.du < 1]
print(pd.crosstab(pd.cut(nc.off_sched, [7200, 10800, 14400, 21600, 28800, 43200, 1e6]), nc.cls))
print('\nLIRF zone P local hour of MVT (crossing rows) by class:')
B = B.assign(hr=B.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.hour); print(pd.crosstab(B.hr, B.cls))
nc = nc.assign(hr=nc.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.hour)
print('no-crossing rows departing 00-05 local by class:'); print(pd.crosstab(nc[nc.hr < 6].hr, nc[nc.hr < 6].cls))
print('p14 vs class calibration, all LIRF zone P (bins):'); print(pd.crosstab(pd.cut(x.p, [0, .2, .4, .6, .8, .95, 1.0]), x.cls))
