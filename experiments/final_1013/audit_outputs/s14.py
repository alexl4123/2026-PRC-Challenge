import numpy as np, pandas as pd, pyarrow.parquet as pq
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
S = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
pd.set_option('display.width', 250)
C = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt', 'off_sched', 'off_aobt3']
a = pd.read_parquet(R + 'cache/major/dep_ranking.parquet', columns=C); b = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=C)
m = a.merge(b, on='MVT_ID_mvt', suffixes=('_i', '_f'))
dt = (m.MVT_TIME_UTC_mvt_f - m.MVT_TIME_UTC_mvt_i).dt.total_seconds()
print('MVT_TIME final - initial (s):', dt.describe(percentiles=[.01, .1, .5, .9, .99]).round(3).to_dict())
print('value counts of rounded diff:', dt.round(0).value_counts().head(10).to_dict())
ds = (m.SCHED_TIME_UTC_mvt_f - m.SCHED_TIME_UTC_mvt_i).dt.total_seconds(); print('SCHED diff:', ds.describe().round(3).to_dict())
print(m[['MVT_TIME_UTC_mvt_i', 'MVT_TIME_UTC_mvt_f', 'AOBT_3_flt_i', 'off_aobt3_i', 'off_aobt3_f']].head(5).to_string())
print('by APT mean diff:', dt.groupby(m.APT_i.astype(str)).agg(['mean', 'min', 'max']).round(2).to_string())
# raw organiser files
for f in ['ranking.parquet', 'final_ranking.parquet']:
    p = pq.ParquetFile(S + f); print(f, p.metadata.num_rows, p.schema_arrow)
