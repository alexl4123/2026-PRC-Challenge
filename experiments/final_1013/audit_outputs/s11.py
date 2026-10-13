import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
for f, nm in (('cache_final/major/dep_ranking.parquet', 'DEP26'), ('cache_final/major/arr_ranking.parquet', 'ARR26')):
    d = pd.read_parquet(R + f, columns=['APT', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt', 'off_aobt3'] if 'dep' in f else ['APT', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt'])
    d = d[(d.APT == 'LFPG') & (d.MVT_TIME_UTC_mvt.dt.month.isin([1, 2, 6, 7]))]
    per = np.select([d.MVT_TIME_UTC_mvt.dt.month <= 2, d.MVT_TIME_UTC_mvt < pd.Timestamp('2026-06-08', tz='UTC')], ['JanFeb', 'Jun1-7'], 'Jun8-Jul31')
    print(nm, 'runway share %'); print((100 * pd.crosstab(per, d.RUNWAY_mvt, normalize='index')).round(1).to_string())
    if 'dep' in f:
        print('px by runway & period'); print(d.assign(p=per, px=d.off_aobt3.clip(0, 7200)).pivot_table(index='p', columns='RUNWAY_mvt', values='px', aggfunc='mean').round(0).to_string())
