import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
k = pd.read_parquet(R+'cache_final/ranking.parquet', columns=['MVT_ID_mvt','PHASE_mvt','APT','MVT_TIME_UTC_mvt'], filters=[('PHASE_mvt','==','DEP')])
a = pd.read_parquet(R+'cache_final/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt','adsb_matched','adsb_off_first','adsb_apt_day_ground','adsb_apt_day_best'])
m = k.merge(a, on='MVT_ID_mvt'); m['APT'] = m.APT.astype(str)
m['day'] = m.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d'); m['h'] = m.MVT_TIME_UTC_mvt.dt.hour
m['mat'] = m.adsb_matched.eq(1); m['gnd'] = m.adsb_off_first.notna()
for d in ['2026-06-06', '2026-01-23', '2026-06-11']:
    x = m[(m.day == d) & (m.APT != 'LTFM')]
    print(d, 'hourly matched:', x.groupby('h').mat.mean().round(2).to_dict())
x = m[(m.day == '2026-06-06')]
print('06-06 deps with hour<=9 (all apts incl LTFM):', int((x.h <= 9).sum()), 'of', len(x), '; hour<9:', int((x.h < 9).sum()))
x = m[(m.day == '2026-01-23') & (m.APT != 'LTFM')]
print('01-23 per-apt matched by hour block', x.groupby(['APT', x.h // 3]).mat.mean().unstack().round(2).to_string())
x = m[m.day == '2026-06-11']
print('06-11 deps', len(x), 'apt_day_ground by apt', x.groupby('APT').adsb_apt_day_ground.mean().round(2).to_dict())
print('06-11 deps per apt', x.groupby('APT').size().to_dict())
l = m[(m.APT == 'LIRF')]
print('LIRF deps Jun', int((l.MVT_TIME_UTC_mvt.dt.month == 6).sum()), 'Jul', int((l.MVT_TIME_UTC_mvt.dt.month == 7).sum()))
print('LIRF apt_day_ground by month', l.groupby(l.MVT_TIME_UTC_mvt.dt.month).adsb_apt_day_ground.mean().round(3).to_dict())
e = m[(m.APT == 'EDDF')]
print('EDDF ground share by month', e.groupby(e.MVT_TIME_UTC_mvt.dt.month).gnd.mean().round(3).to_dict(), 'n Feb', int((e.MVT_TIME_UTC_mvt.dt.month == 2).sum()))
# EDDF ground share by stand prefix Feb vs Jan to see whether a receiver area was lost
st = pd.read_parquet(R+'cache_final/ranking.parquet', columns=['MVT_ID_mvt','STAND_mvt'])
e = e.merge(st, on='MVT_ID_mvt'); e['pier'] = e.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+)', expand=False)
print('EDDF ground share by pier, Jan(1-28) vs Feb vs Jun', e.assign(p=np.where(e.day < '2026-01-29', 'Jan', np.where(e.day < '2026-03-01', 'Feb', 'JunJul'))).groupby(['pier', 'p']).gnd.mean().unstack().round(2).to_string())
