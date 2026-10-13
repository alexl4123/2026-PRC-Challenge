import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
cols = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'off_eobt1', 'STAND_mvt']
rk = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=cols).merge(pd.read_parquet(R + 'outputs_final/understanding-accordion_final.parquet'), on='MVT_ID_mvt')
rk = rk.merge(pd.read_parquet(R + 'cache_final/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt', 'adsb_matched']), on='MVT_ID_mvt')
tr = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=cols + ['y']); tr['f'] = np.load(R + 'cache/w2/v28s_oof.npz')['f']
tr['month'] = tr.MVT_TIME_UTC_mvt.dt.month; rk['month'] = rk.MVT_TIME_UTC_mvt.dt.month
bins = [-1e9, 300, 450, 600, 750, 900, 1050, 1200, 1500, 2000, 1e9]
for apt in ['LFPG', 'EDDF']:
    a = tr[(tr.APT == apt) & tr.month.isin([6, 7]) & tr.off_aobt3.notna()]; b = rk[(rk.APT == apt) & rk.month.isin([6, 7]) & rk.off_aobt3.notna()]
    ga = a.groupby(pd.cut(a.off_aobt3, bins), observed=True).agg(n25=('y', 'size'), y25=('y', 'mean'), f25=('f', 'mean'))
    gb = b.groupby(pd.cut(b.off_aobt3, bins), observed=True).agg(n26=('TAXITIME_SEC_mvt', 'size'), p26=('TAXITIME_SEC_mvt', 'mean'))
    print(f'\n{apt} Jun+Jul, by off_aobt3 bin'); print(ga.join(gb).round(0).to_string())
    print('quantiles off_aobt3 2025', a.off_aobt3.quantile([.1, .25, .5, .75, .9]).round(0).tolist(), '2026', b.off_aobt3.quantile([.1, .25, .5, .75, .9]).round(0).tolist())
    print('quantiles off_sched 2025', a.off_sched.quantile([.1, .5, .9]).round(0).tolist(), '2026', b.off_sched.quantile([.1, .5, .9]).round(0).tolist())
    # AOBT relative to schedule: off_sched - off_aobt3 = AOBT3 - SCHED
    print('AOBT3-SCHED quantiles 2025', (a.off_sched - a.off_aobt3).quantile([.1, .5, .9]).round(0).tolist(), '2026', (b.off_sched - b.off_aobt3).quantile([.1, .5, .9]).round(0).tolist())
    # AOBT3 seconds-of-minute granularity
    print('share off_aobt3 % 60 == MVT sec (AOBT on whole minute) 2025', round(((a.MVT_TIME_UTC_mvt.dt.second - a.off_aobt3) % 60 == 0).mean(), 3), '2026', round(((b.MVT_TIME_UTC_mvt.dt.second - b.off_aobt3) % 60 == 0).mean(), 3))
# hourly ADS-B match on low days (all airports except LTFM)
x = rk[rk.APT != 'LTFM']; x = x.assign(d=x.MVT_TIME_UTC_mvt.dt.strftime('%m-%d'), h=x.MVT_TIME_UTC_mvt.dt.hour)
days = ['01-22', '01-23', '01-29', '06-05', '06-06', '06-07', '06-11', '06-23', '06-25', '06-26']
t = x[x.d.isin(days)].groupby(['d', 'h']).adsb_matched.mean().unstack().round(2)
print('\nhourly ADS-B match share (all airports ex LTFM), UTC hour columns'); print(t.to_string())
t2 = x[x.d == '06-11'].groupby(['APT', 'h']).adsb_matched.mean().unstack().round(2); print('\n06-11 per APT hourly'); print(t2.to_string())
