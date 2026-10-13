import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 40)
k = pd.read_parquet(R+'cache_final/ranking.parquet', columns=['MVT_ID_mvt','PHASE_mvt','APT','MVT_TIME_UTC_mvt'], filters=[('PHASE_mvt','==','DEP')])
a = pd.read_parquet(R+'cache_final/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt','adsb_matched','adsb_has_ground','adsb_off_first','adsb_off_best','adsb_n_pts','adsb_apt_day_ground'])
m = k.merge(a, on='MVT_ID_mvt', how='left')
m['APT'] = m.APT.astype(str); m['day'] = m.MVT_TIME_UTC_mvt.dt.strftime('%m-%d'); m['h'] = m.MVT_TIME_UTC_mvt.dt.hour
m['mat'] = m.adsb_matched.eq(1); m['gnd'] = m.adsb_off_first.notna()
sus = ['01-28','01-29','01-30','06-05','06-06','06-07','06-08','06-10','06-11','06-12','06-22','06-23','06-24','06-25','06-26']
s = m[m.day.isin(sus)]
print('matched share'); print(s.groupby(['day','APT']).mat.mean().unstack().round(2))
print('ground-track share (off_first notna)'); print(s.groupby(['day','APT']).gnd.mean().unstack().round(2))
for d in ['01-29','06-11','06-25','06-06']:
    x = m[m.day==d]
    print('hourly matched', d); print(x.groupby([pd.cut(x.h,[-1,5,9,13,17,21,24])]).mat.mean().round(2).to_dict())
l = m[(m.APT=='LIRF')]
print('LIRF daily matched / ground by month'); 
t = l.groupby(['day']).agg(mat=('mat','mean'), gnd=('gnd','mean'))
print(t[t.index.str[:2].isin(['05','06','07'])].round(2).T.to_string())
print(t[t.index.str[:2].isin(['01','02'])].round(2).T.to_string())
e = m[(m.APT=='EDDF')&(m.day.str[:2]=='02')]
print('EDDF Feb daily ground'); print(e.groupby('day').gnd.mean().round(2).to_dict())
e = m[(m.APT=='EDDF')&(m.day.str[:2]=='01')]
print('EDDF Jan daily ground'); print(e.groupby('day').gnd.mean().round(2).to_dict())
# initial ranking (Jan/Jul): LIRF Jul ground
ai = pd.read_parquet(R+'cache/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt','adsb_off_first'])
ki = m[m.APT=='LIRF'][['MVT_ID_mvt','day']].merge(ai, on='MVT_ID_mvt')
print('LIRF ground share in INITIAL file by month', ki.groupby(ki.day.str[:2]).adsb_off_first.apply(lambda s: s.notna().mean()).round(3).to_dict())
