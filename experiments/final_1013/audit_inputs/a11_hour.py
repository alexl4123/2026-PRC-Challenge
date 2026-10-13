import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
out = {}
for name, kp, ap in [('2025', 'cache/training.parquet', 'cache/adsb_lol/features_training.parquet'), ('2026', 'cache_final/ranking.parquet', 'cache_final/adsb_lol/features_ranking.parquet')]:
    k = pd.read_parquet(R+kp, columns=['MVT_ID_mvt','PHASE_mvt','APT','MVT_TIME_UTC_mvt'], filters=[('PHASE_mvt','==','DEP')])
    a = pd.read_parquet(R+ap, columns=['MVT_ID_mvt','adsb_matched','adsb_off_first']).drop_duplicates('MVT_ID_mvt')
    m = k.merge(a, on='MVT_ID_mvt'); m = m[m.APT.astype(str) != 'LTFM']
    m = m[m.MVT_TIME_UTC_mvt.dt.month.isin([1, 2, 6, 7])]
    m['ym'] = m.MVT_TIME_UTC_mvt.dt.strftime('%y-%m'); m['h'] = m.MVT_TIME_UTC_mvt.dt.hour
    m['mat'] = m.adsb_matched.eq(1); m['gnd'] = m.adsb_off_first.notna()
    print(name, 'matched by hour (rows) x month'); print(m.groupby(['h', 'ym']).mat.mean().unstack().round(2).T.to_string())
    print(name, 'ground by hour x month'); print(m.groupby(['h', 'ym']).gnd.mean().unstack().round(2).T.to_string())
