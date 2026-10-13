import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
dep = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AIRCRAFT_OPERATOR_flt', 'off_aobt3', 'off_sched'])
arr = pd.read_parquet(R + 'cache_final/major/arr_ranking.parquet', columns=['APT', 'MVT_TIME_UTC_mvt'])
for apt, day, hrs in (('EDDF', '2026-02-03', range(10, 19)), ('EDDM', '2026-06-07', range(16, 22)), ('EDDM', '2026-07-18', range(13, 19)), ('LSZH', '2026-06-21', range(2, 8)), ('LSZH', '2026-06-30', range(17, 22)), ('LEMD', '2026-01-28', range(7, 13))):
    d = dep[(dep.APT == apt) & (dep.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d') == day)]; a = arr[(arr.APT == apt) & (arr.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d') == day)]
    s = d.groupby(d.MVT_TIME_UTC_mvt.dt.hour).agg(dep=('MVT_ID_mvt', 'size'), sched_dep=('off_sched', 'size'), offs_med=('off_sched', 'median'), px_med=('off_aobt3', 'median'))
    s['arr'] = a.groupby(a.MVT_TIME_UTC_mvt.dt.hour).size(); sc = d.groupby(d.SCHED_TIME_UTC_mvt.dt.hour).size(); s['by_sched_hour'] = sc
    print(apt, day); print(s.reindex(list(hrs)).fillna(0).astype(int).T.to_string())
# EDDF 02-12 operator mix
d = dep[dep.APT == 'EDDF']; d = d.assign(dd=d.MVT_TIME_UTC_mvt.dt.strftime('%m-%d'))
top = d.AIRCRAFT_OPERATOR_flt.value_counts().index[0]
print('EDDF top-operator departures per day 02-09..02-15:', d[d.AIRCRAFT_OPERATOR_flt == top].groupby('dd').size().loc['02-09':'02-15'].to_dict(), ' others:', d[d.AIRCRAFT_OPERATOR_flt != top].groupby('dd').size().loc['02-09':'02-15'].to_dict())
# final vs v29 on overlap
f = pd.read_parquet(R + 'outputs_final/understanding-accordion_final.parquet'); v = pd.read_parquet(R + 'outputs/understanding-accordion_v29.parquet')
m = f.merge(v, on='MVT_ID_mvt', suffixes=('_f', '_v')); print('\nv29 rows', len(v), 'overlap with final ids', len(m))
if len(m):
    m = m.merge(dep[['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt']], on='MVT_ID_mvt'); m['d'] = m.TAXITIME_SEC_mvt_f - m.TAXITIME_SEC_mvt_v
    print('diff final - v29: rms %.1f, mean %.1f, rows changed %d' % (np.sqrt((m.d ** 2).mean()), m.d.mean(), (m.d != 0).sum()))
    g = m.groupby(['APT', m.MVT_TIME_UTC_mvt.dt.strftime('%m-%d')]).d.agg(lambda s: np.sqrt((s ** 2).mean())); print('largest airport-day rms diffs:', g.sort_values().tail(8).round(1).to_dict())
