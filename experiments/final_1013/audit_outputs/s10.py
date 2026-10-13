import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
rk = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_lobt', 'off_eobt1', 'off_sched'])
rk = rk.merge(pd.read_parquet(R + 'outputs_final/understanding-accordion_final.parquet'), on='MVT_ID_mvt').merge(
    pd.read_parquet(R + 'cache_final/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt', 'adsb_has_ground', 'adsb_off_best', 'adsb_off_push', 'adsb_off_move']), on='MVT_ID_mvt')
x = rk[(rk.APT == 'LFPG') & rk.off_aobt3.notna()].copy(); x['dd'] = x.MVT_TIME_UTC_mvt.dt.strftime('%m-%d')
x['px'] = x.off_aobt3.clip(0, 7200); x['gap'] = x.TAXITIME_SEC_mvt - x.px
x['aobt_lobt'] = x.off_lobt - x.off_aobt3; x['aobt_eobt'] = x.off_eobt1 - x.off_aobt3
g = x.groupby('dd').agg(n=('px', 'size'), px=('px', 'mean'), pred=('TAXITIME_SEC_mvt', 'mean'), gap=('gap', 'mean'), aobt_minus_lobt=('aobt_lobt', 'median'),
                        aobt_minus_eobt=('aobt_eobt', 'median'), gr=('adsb_has_ground', 'mean'))
a = x[(x.adsb_has_ground == 1) & x.adsb_off_best.notna()]
g['px_minus_adsb'] = (a.off_aobt3 - a.adsb_off_best).groupby(a.dd).median(); g['pred_minus_adsb'] = (a.TAXITIME_SEC_mvt - a.adsb_off_best).groupby(a.dd).median()
g['adsb_taxi'] = a.adsb_off_best.groupby(a.dd).median(); g['n_adsb'] = a.groupby('dd').size()
print(g.loc['05-25':'06-14'].round(0).to_string() if False else g[(g.index >= '06-01') & (g.index <= '06-12')].round(0).to_string())
for lab, sel in (('Jan-Feb', g.index < '03'), ('Jun1-4', (g.index >= '06-01') & (g.index <= '06-04')), ('Jun5-Jul31', g.index >= '06-05')):
    print(lab, g[sel][['px', 'pred', 'gap', 'px_minus_adsb', 'pred_minus_adsb', 'adsb_taxi', 'aobt_minus_eobt']].median().round(0).to_dict(), 'n_adsb', int(g[sel].n_adsb.sum()))
