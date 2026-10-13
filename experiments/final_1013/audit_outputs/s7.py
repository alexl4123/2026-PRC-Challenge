import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_outputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
C = ['MVT_ID_mvt', 'adsb_has_ground', 'adsb_off_best', 'adsb_off_move', 'adsb_off_push']
rk = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'off_aobt3'])
rk = rk.merge(pd.read_parquet(R + 'outputs_final/understanding-accordion_final.parquet'), on='MVT_ID_mvt').merge(pd.read_parquet(R + 'cache_final/adsb_lol/features_ranking.parquet', columns=C), on='MVT_ID_mvt')
tr = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'y'])
ft = pd.read_parquet(R + 'cache/adsb_lol/features_training.parquet', columns=C)
tr['f'] = np.load(R + 'cache/w2/v28s_oof.npz')['f']
tr = tr.merge(ft, on='MVT_ID_mvt', how='left'); print('training rows', len(tr))
rk['month'] = rk.MVT_TIME_UTC_mvt.dt.month; tr['month'] = tr.MVT_TIME_UTC_mvt.dt.month
print('adsb_off_best sample stats 2025 vs y:', tr[['adsb_off_best', 'y']].dropna().corr().iloc[0, 1].round(3), tr.adsb_off_best.describe().round(0).to_dict())
rows = []
for nm, d, v in (('2025', tr, 'y'), ('2026', rk, 'TAXITIME_SEC_mvt')):
    x = d[d.month.isin([1, 2, 6, 7]) & (d.adsb_has_ground == 1) & d.adsb_off_best.notna() & d.off_aobt3.notna()]
    g = x.groupby(['APT', 'month'])
    rows.append(pd.DataFrame({f'n{nm}': g.size(), f'v{nm}': g[v].median(), f'adsb{nm}': g.adsb_off_best.median(), f'px{nm}': g.off_aobt3.median(),
                              f'v-adsb{nm}': (x[v] - x.adsb_off_best).groupby([x.APT, x.month]).median(), f'px-adsb{nm}': (x.off_aobt3 - x.adsb_off_best).groupby([x.APT, x.month]).median()}))
print('rows with ADS-B ground + AOBT3; medians; v = y (2025) or pred (2026); adsb = adsb_off_best')
print(pd.concat(rows, axis=1).round(0).to_string())

# redo daily ADS-B match/ground with ID merge
F2 = pd.read_parquet(R + 'cache/adsb_lol/features_training.parquet', columns=['MVT_ID_mvt', 'adsb_matched'])
tr = tr.merge(F2, on='MVT_ID_mvt', how='left')
rk = rk.merge(pd.read_parquet(R + 'cache_final/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt', 'adsb_matched']), on='MVT_ID_mvt')
for nm, d in (('2025', tr[tr.month.isin([1, 2, 6, 7])]), ('2026', rk)):
    g = d.groupby(['APT', d.MVT_TIME_UTC_mvt.dt.floor('D')])
    dd = pd.DataFrame({'m': g.adsb_matched.mean(), 'gr': g.adsb_has_ground.mean()})
    print(nm, 'daily matched share per APT [min, 1%, median]:'); print(dd.groupby('APT').agg(mmin=('m', 'min'), m1=('m', lambda s: s.quantile(.01)), mmed=('m', 'median'), gmin=('gr', 'min'), gmed=('gr', 'median')).round(3).T.to_string())
    if nm == '2026':
        allm = d.groupby(d.MVT_TIME_UTC_mvt.dt.floor('D')).adsb_matched.mean()
        print('2026 all-airport daily match share, lowest 10:'); print(allm.sort_values().head(10).round(3).to_string())
        low = dd[dd.m < 0.75].reset_index(); low = low[low.APT != 'LTFM']; print('2026 airport-days with match < 0.75:', len(low)); print(low.round(3).to_string())
    else:
        allm = d.groupby(d.MVT_TIME_UTC_mvt.dt.floor('D')).adsb_matched.mean(); print('2025 all-airport daily match share, lowest 5:'); print(allm.sort_values().head(5).round(3).to_string())
