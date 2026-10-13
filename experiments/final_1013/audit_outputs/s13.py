import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
C = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'off_sched', 'off_aobt3', 'off_lobt', 'STAND_mvt', 'RUNWAY_mvt']
a = pd.read_parquet(R + 'cache/major/dep_ranking.parquet', columns=C); b = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=C)
m = a.merge(b, on='MVT_ID_mvt', suffixes=('_i', '_f')); print('initial rows', len(a), 'matched in final', len(m))
for c in []:
    x, y = m[c + '_i'], m[c + '_f']; ne = ~((x == y) | (x.isna() & y.isna())); print(c, 'differs on', int(ne.sum()))
f = pd.read_parquet(R + 'outputs_final/understanding-accordion_final.parquet'); v = pd.read_parquet(R + 'outputs/understanding-accordion_v29.parquet')
d = f.merge(v, on='MVT_ID_mvt', suffixes=('_f', '_v')).merge(b, on='MVT_ID_mvt'); d['d'] = d.TAXITIME_SEC_mvt_f - d.TAXITIME_SEC_mvt_v
d['D'] = d.AOBT_3_flt.isna(); d['P'] = d.D & (d.off_sched > 7200); d['dd'] = d.MVT_TIME_UTC_mvt.dt.strftime('%m-%d')
x = d[(d.APT == 'LIRF') & d.dd.isin(['07-01', '07-02'])]
print(x.groupby(['dd', 'D', 'P']).d.agg(['size', 'mean', lambda s: np.sqrt((s ** 2).mean()), lambda s: s.abs().max()]).round(1).to_string())
print(x.loc[x.d.abs().sort_values().index[-8:], ['dd', 'D', 'P', 'off_sched', 'off_aobt3', 'TAXITIME_SEC_mvt_v', 'TAXITIME_SEC_mvt_f']].to_string())
y = d[d.dd.isin(['07-01', '07-02', '07-03', '07-04'])]; print('\nrms diff by APT for 07-01..07-04 (route D / not):'); print(y.groupby(['APT', 'D']).d.agg(lambda s: np.sqrt((s ** 2).mean())).unstack().round(1).to_string())
