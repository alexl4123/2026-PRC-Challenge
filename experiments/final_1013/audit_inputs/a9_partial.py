import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for name, kp, ap in [('2025', 'cache/training.parquet', 'cache/adsb_lol/features_training.parquet'), ('2026', 'cache_final/ranking.parquet', 'cache_final/adsb_lol/features_ranking.parquet')]:
    k = pd.read_parquet(R+kp, columns=['MVT_ID_mvt','PHASE_mvt','APT','MVT_TIME_UTC_mvt'], filters=[('PHASE_mvt','==','DEP')])
    a = pd.read_parquet(R+ap, columns=['MVT_ID_mvt','adsb_matched','adsb_off_first']).drop_duplicates('MVT_ID_mvt')
    m = k.merge(a, on='MVT_ID_mvt', how='left'); m = m[m.APT.astype(str) != 'LTFM']
    m['day'] = m.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d'); m['blk'] = m.MVT_TIME_UTC_mvt.dt.hour // 3
    m['mat'] = m.adsb_matched.eq(1); m['gnd'] = m.adsb_off_first.notna()
    h = m.groupby(['day','blk']).agg(mat=('mat','mean'), gnd=('gnd','mean'), n=('mat','size')).reset_index()
    h = h[h.blk.between(2, 7)]     # 06-24 UTC, where traffic is dense
    dd = h.groupby('day').agg(mat_min=('mat','min'), mat_max=('mat','max'), gnd_min=('gnd','min'), gnd_max=('gnd','max'))
    day = m.groupby('day').agg(n=('mat','size'), mat=('mat','mean'), gnd=('gnd','mean'))
    dd = dd.join(day)
    bad = dd[(dd.mat_min < 0.5 * dd.mat_max) | (dd.mat < 0.8) | (dd.gnd < 0.5 * dd.gnd.median())]
    print(f'\n{name}: days (all airports except LTFM) with a 3-h block matched < half the best block, or day matched < 0.8, or ground share < half the median: {len(bad)} of {len(dd)}')
    print(bad.round(2).to_string())
    print(name, 'median day matched', dd.mat.median().round(3), 'ground', dd.gnd.median().round(3))
