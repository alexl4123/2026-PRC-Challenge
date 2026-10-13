import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
cols = ['PHASE_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'EOBT_1_flt']
f = [('PHASE_mvt', '==', 'DEP'), ('APT', '==', 'LTFM')]
tr = pd.read_parquet(R + 'cache/training.parquet', columns=cols, filters=f)
rk = pd.read_parquet(R + 'cache_final/ranking.parquet', columns=cols, filters=f)
d = pd.concat([tr, rk], ignore_index=True)
d['off_a3'] = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
d['y'] = (d.MVT_TIME_UTC_mvt - d.BLOCK_TIME_UTC_mvt).dt.total_seconds()
d['blk_m_a3'] = (d.BLOCK_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
d['a3_m_eobt'] = (d.AOBT_3_flt - d.EOBT_1_flt).dt.total_seconds()
d['wk'] = d.MVT_TIME_UTC_mvt.dt.to_period('W')
g = d.groupby('wk').agg(n=('y', 'size'), off_a3=('off_a3', 'median'), y=('y', 'median'), blk_m_a3=('blk_m_a3', 'median'),
                        a3_eq_blk=('blk_m_a3', lambda s: (s.abs() < 60).mean()), a3_m_eobt=('a3_m_eobt', 'median'))
print(g.round(2).to_string())
