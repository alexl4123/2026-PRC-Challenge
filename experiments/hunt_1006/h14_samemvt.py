"""Same-flight duplicate rows with (nearly) the same take-off (same APT, FLIGHT_mvt, SCHED, |dMVT| < 120 s): order of the two BLOCKs."""
import numpy as np, pandas as pd
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 400)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; yy = V['y']; month = V['month']; P = V['P']
def pairs(yr):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'SCHED', 'MVT', 'LOBT'] + (['BLOCK'] if yr == 'tr' else []))
    d['i'] = np.arange(len(d)); d['APT'] = d.APT.astype(str)
    if yr == 'tr': d['f'] = f; d['y'] = d.MVT - d.BLOCK; d['P'] = P
    else: d = d.merge(pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'}), on='MVT_ID_mvt', how='left')
    g = d.dropna(subset=['FLIGHT_mvt']); g = g[g.groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True).MVT.transform('size') == 2]
    a = g[g.LOBT.notna()]; b = g[g.LOBT.isna()]
    m = a.merge(b, on=['APT', 'FLIGHT_mvt', 'SCHED'], suffixes=('_nm', '_d'))
    m = m[(m.MVT_nm - m.MVT_d).abs() < 120].copy(); m['off_sched_d'] = m.MVT_d - m.SCHED; m['off_lobt_nm'] = m.MVT_nm - m.LOBT_nm
    return m
m = pairs('tr')
m['ok'] = (m.y_nm - (m.MVT_nm - m.MVT_d)) >= m.y_d - 12      # BLOCK_nm <= BLOCK_d (+12 jitter)
print('2025 same-take-off pairs (one NM, one not):', len(m), ' by APT', m.APT.value_counts().to_dict())
print(' BLOCK_nm <= BLOCK_d (+12):', m.ok.sum(), ' violations', (~m.ok).sum())
print(' noNM-row y quantiles', m.y_d.quantile([0, .5, .9, 1]).round().to_dict(), '  NM-row y quantiles', m.y_nm.quantile([0, .5, .9, 1]).round().to_dict())
print(' NM row stamped (|y - off_sched| <= 6):', ((m.y_nm - (m.MVT_nm - m.SCHED)).abs() <= 6).sum(), '  noNM row stamped:', ((m.y_d - m.off_sched_d).abs() <= 6).sum())
print(' noNM rows in P:', m.P_d.sum(), ' OOF f_d quantiles', m.f_d.quantile([.5, .9, 1]).round().to_dict(), ' OOF |f_d - y_d| > 1500:', ((m.f_d - m.y_d).abs() > 1500).sum())
print(m[~m.ok][['APT', 'FLIGHT_mvt', 'y_nm', 'y_d', 'MVT_nm', 'MVT_d']].to_string())
# pairwise projection: constraint  BLOCK_nm <= BLOCK_d  <=>  y_nm - y_d >= MVT_nm - MVT_d - 12 =: c
c = (m.MVT_nm - m.MVT_d - 12).to_numpy(); fn = m.f_nm.to_numpy(); fd = m.f_d.to_numpy(); viol = fn - fd < c
s = (c - (fn - fd)) / 2; fn2 = np.where(viol, fn + s, fn); fd2 = np.where(viol, fd - s, fd)
g2 = f.copy(); g2[m.i_nm.to_numpy()] = fn2; g2[m.i_d.to_numpy()] = fd2
r = lambda h, k: np.sqrt(np.mean((yy[k] - h[k]) ** 2)); ALL = np.ones(len(yy), bool); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
dd = (yy - g2) ** 2 - (yy - f) ** 2
print(f'OOF pair projection: pairs violating {viol.sum()}, rows changed {(g2 != f).sum()}, rows worse {(dd > 0).sum()}, SSE {dd.sum():+.4g}, 12m {r(g2, ALL) - r(f, ALL):+.4f}, held {r(g2, held) - r(f, held):+.4f}, Jan+Jul {r(g2, S) - r(f, S):+.4f}')
m6 = pairs('ra')
c6 = (m6.MVT_nm - m6.MVT_d - 12).to_numpy(); v6 = (m6.f_nm - m6.f_d).to_numpy() < c6
print('\n2026 same-take-off pairs:', len(m6), ' v28 violating BLOCK_nm <= BLOCK_d:', v6.sum())
s6 = (c6 - (m6.f_nm - m6.f_d).to_numpy()) / 2
print(m6[v6][['APT', 'FLIGHT_mvt', 'f_nm', 'f_d', 'off_sched_d', 'off_lobt_nm']].assign(shift=s6[v6]).to_string())
print(' guaranteed SSE reduction >= 2*sum(shift^2) =', f'{2 * (s6[v6] ** 2).sum():.4g}', ' RMSE bound', np.sqrt(235.3996 ** 2 - 2 * (s6[v6] ** 2).sum() / 344841) - 235.3996)
print(m6[['APT', 'FLIGHT_mvt', 'f_nm', 'f_d', 'off_sched_d', 'off_lobt_nm']].to_string())
