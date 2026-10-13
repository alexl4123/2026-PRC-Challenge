import numpy as np, pandas as pd, re
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_outputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 60)
rk = pd.read_parquet(O + 'rk.parquet').merge(pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'off_aobt3']), on='MVT_ID_mvt')
rk = rk.merge(pd.read_parquet(R + 'cache_final/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt', 'adsb_matched']), on='MVT_ID_mvt')
tr = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'y', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt'])
tr['f'] = np.load(R + 'cache/w2/v28s_oof.npz')['f']; tr['month'] = tr.MVT_TIME_UTC_mvt.dt.month
# LFPG daily: mean(pred - clip(off_aobt3)) in 2026 May-Jul vs 2025; find step
for nm, d, v in (('2025', tr, 'y'), ('2026', rk, 'TAXITIME_SEC_mvt')):
    x = d[(d.APT == 'LFPG') & d.off_aobt3.notna()]; x = x.assign(g=x[v] - x.off_aobt3.clip(0, 7200), dd=x.MVT_TIME_UTC_mvt.dt.strftime('%m-%d'))
    s = x.groupby('dd').g.mean()
    print(nm, 'LFPG daily mean (v - px), weekly blocks:'); print(s.groupby(np.arange(len(s)) // 7).agg(lambda q: f'{q.index[0]}..{q.index[-1]}: {q.mean():.0f}').to_string())
# LFPG stand family composition Jun/Jul
fam = lambda s: re.sub(r'\d.*$', '', str(s))
a = tr[(tr.APT == 'LFPG') & tr.month.isin([6, 7])].copy(); b = rk[(rk.APT == 'LFPG') & rk.month.isin([6, 7])].copy()
a['sf'] = a.STAND_mvt.map(fam); b['sf'] = b.STAND_mvt.map(fam)
ga = a.groupby('sf').agg(n25=('y', 'size'), y25=('y', 'mean'), f25=('f', 'mean'), px25=('off_aobt3', lambda q: q.clip(0, 7200).mean()))
gb = b.groupby('sf').agg(n26=('TAXITIME_SEC_mvt', 'size'), p26=('TAXITIME_SEC_mvt', 'mean'), px26=('off_aobt3', lambda q: q.clip(0, 7200).mean()))
G = ga.join(gb, how='outer'); G = G[(G.n25.fillna(0) + G.n26.fillna(0)) > 300]
G['gap25'] = G.y25 - G.px25; G['gap26'] = G.p26 - G.px26; print('\nLFPG Jun+Jul by stand prefix'); print(G.round(0).to_string())
# Effect proxy on low-ADS-B days: component disagreement d_aw for unmatched vs matched rows, low days vs other days (ex LTFM)
ids = np.load(R + 'cache_final/adsb2_final/ranking_ids.npy', allow_pickle=True); assert np.array_equal(ids, rk.MVT_ID_mvt.to_numpy())
m3 = lambda p: np.mean([np.load(p.format(s)).astype(float) for s in range(3)], axis=0)
rk['amix'] = m3(R + 'cache_final/adsb2_final/mix__seed{:02d}.npy'); rk['w2'] = m3(R + 'cache_final/w2_final/mix__seed{:02d}.npy')
rk['seq'] = m3(R + 'cache_final/seq_ep40_final/seq__seed{:02d}.npy')
rk['dd'] = rk.MVT_TIME_UTC_mvt.dt.strftime('%m-%d')
x = rk[rk.APT != 'LTFM']; low = ['01-23', '01-29', '06-06', '06-07', '06-11', '06-25']
x = x.assign(low=x.dd.isin(low))
t = x.groupby(['low', 'adsb_matched']).apply(lambda g: pd.Series({'n': len(g), 'mean(amix-w2)': (g.amix - g.w2).mean(), 'rms(amix-w2)': np.sqrt(((g.amix - g.w2) ** 2).mean()),
                                                                     'mean(seq-w2)': (g.seq - g.w2).mean(), 'rms(seq-w2)': np.sqrt(((g.seq - g.w2) ** 2).mean()), 'mean(pred-w2)': (g.TAXITIME_SEC_mvt - g.w2).mean()}))
print('\nnon-LTFM rows: low-ADS-B days vs other days, by matched flag'); print(t.round(1).to_string())
print('rows on low days (ex LTFM):', int(x.low.sum()), 'unmatched on low days:', int((x.low & (x.adsb_matched == 0)).sum()),
      'expected unmatched at normal rate:', int(x.low.sum() * (1 - x[~x.low].adsb_matched.mean())))
per = x[x.low].groupby('dd').apply(lambda g: pd.Series({'n': len(g), 'unmatched': int((g.adsb_matched == 0).sum()), 'mean(pred-w2)': (g.TAXITIME_SEC_mvt - g.w2).mean()}))
print(per.round(1).to_string()); print('other days mean(pred-w2):', round((x[~x.low].TAXITIME_SEC_mvt - x[~x.low].w2).mean(), 1))
