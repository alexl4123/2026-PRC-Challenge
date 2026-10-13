"""Zone P at airports that never stamped in 2025: size of v28's stamping hedge in 2026 (label-free) and the 2025 OOF of removing it."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
c = np.load(R + 'cache/final/lgb_components.npz', allow_pickle=True); ids = c['ids']; inP = c['in_P'].astype(bool)
t1 = np.load(R + 'cache/strategy/tfm_P_lirf_ranking.npz', allow_pickle=True); t2 = np.load(R + 'cache/tfm_w4/pool_nonrome_ranking.npz', allow_pickle=True)
pos = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
d = pd.read_parquet(H + 'ra_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT', 'SCHED']); d['APT'] = d.APT.astype(str)
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(ids).to_numpy().astype(float)
z = np.load(R + 'cache/major/m11f_candidates.npz', allow_pickle=True); corr = np.clip(z['trailD_full'], 0, None) - z['v9']
pp = np.full(len(ids), np.nan); pp[pos] = p
os_ = (d.MVT - d.SCHED).to_numpy()
pb = (v - corr - pp * os_) / (1 - pp)          # implied non-stamp branch (rint ignored)
mv = (pb + corr) - v                             # move if p -> 0
k = inP & (d.APT != 'LIRF').to_numpy()
t = pd.DataFrame({'APT': d.APT[k], 'p': pp[k], 'mv': mv[k], 'f': v[k], 'os': os_[k]})
print(t.groupby('APT').agg(n=('p', 'size'), p_mean=('p', 'mean'), p_max=('p', 'max'), mv_mean=('mv', 'mean'), mv_min=('mv', 'min'), sumsq=('mv', lambda s: (s ** 2).sum())).round(3).to_string())
print('total sum sq of the hedge on non-LIRF P rows:', f'{(t.mv ** 2).sum():.4g}', '  -> if every such row is unstamped and pb is unbiased, expected RMSE change ~', f'{-(t.mv ** 2).sum() / (2 * 235.4 * 344841):+.3f}')
