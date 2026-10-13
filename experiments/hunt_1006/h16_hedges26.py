"""Where does v28 hedge most in 2026 (zone P)? Label-free; looks for structure on the top hedge rows."""
import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 400)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
c = np.load(R + 'cache/final/lgb_components.npz', allow_pickle=True); ids = c['ids']; inP = c['in_P'].astype(bool)
t1 = np.load(R + 'cache/strategy/tfm_P_lirf_ranking.npz', allow_pickle=True); t2 = np.load(R + 'cache/tfm_w4/pool_nonrome_ranking.npz', allow_pickle=True)
pos = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
d = pd.read_parquet(H + 'ra_dep.parquet', columns=['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'MVT', 'SCHED', 'STAND_mvt', 'FLIGHT_RULE_mvt'])
assert np.array_equal(d.MVT_ID_mvt.to_numpy(), ids)
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(ids).to_numpy()
d['f'] = v; d['P'] = inP; d['p'] = np.nan; d.loc[d.index[pos], 'p'] = p
d['off_sched'] = d.MVT - d.SCHED
d['pb'] = (d.f - d.p * d.off_sched) / (1 - d.p)
d['hedge'] = d.p * (1 - d.p) * (d.off_sched - d.pb) ** 2
L = pd.read_parquet(H + 'h10_link26.parquet', columns=['MVT_ID_mvt', 'a_APT', 'a_LOBT', 'clean', 'fid_used'])
d = d.merge(L, on='MVT_ID_mvt', how='left')
d['off_alobt'] = d.MVT - d.a_LOBT
ts = lambda s: pd.to_datetime(s, unit='s').dt.strftime('%m-%d %H:%M')
d['MVT_t'] = ts(d.MVT); d['SCHED_t'] = ts(d.SCHED)
P = d[d.P].sort_values('hedge', ascending=False)
print('zone P rows', len(P), ' sum hedge var', f'{P.hedge.sum():.4g}', ' (=RMSE^2 share', f'{P.hedge.sum() / 344841:.1f} s^2 of {235.4 ** 2:.0f})')
print(P.groupby('APT').hedge.agg(['size', 'sum']).sort_values('sum', ascending=False).to_string())
print(P[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'FLIGHT_RULE_mvt', 'SCHED_t', 'MVT_t', 'off_sched', 'p', 'pb', 'f', 'hedge', 'a_APT', 'clean', 'off_alobt', 'FLIGHT_ID_mvt']].head(60).to_string())
print('\nP rows with a linked NM arrival:', P.a_APT.notna().sum(), ' clean', (P.clean == True).sum(), ' hedge var on them', f'{P[P.a_APT.notna()].hedge.sum():.4g}')
