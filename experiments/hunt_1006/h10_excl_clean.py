"""Exclusion rule on the clean subset: IFR route-D departure, own NM arrival record whose FLIGHT_ID no departure row carries."""
import numpy as np, pandas as pd
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']; P = V['P']
out = {}
for yr in ('tr', 'ra'):
    m = pd.read_parquet(H + f'h07_link{"25" if yr == "tr" else "26"}.parquet')
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=['FLIGHT_ID_mvt'])
    m['fid_used'] = m.a_FLIGHT_ID_mvt.isin(set(d.FLIGHT_ID_mvt.dropna().unique()))
    m['ifr'] = m.FLIGHT_RULE_mvt.astype(str) == 'I'
    m['clean'] = m.ifr & ~m.fid_used
    m['lo'] = m.off_alobt - 3606; m['hi'] = m.off_alobt + 3606; m['onesided'] = m.lo < -12
    if yr == 'tr':
        m['f'] = f[m.i.to_numpy()]; m['P'] = P[m.i.to_numpy()]; m['month'] = month[m.i.to_numpy()]
        m['y'] = m.MVT - m.BLOCK; m['excl'] = (m.y < m.lo) | (m.y > m.hi); m['side'] = np.where(m.y > m.hi, 'above', np.where(m.y < m.lo, 'below', 'INSIDE'))
    else:
        v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
        m = m.merge(v, on='MVT_ID_mvt', how='left')
    m['inside'] = (m.f > m.lo) & (m.f < m.hi)
    out[yr] = m
m = out['tr']; m6 = out['ra']
print('2025 linked', len(m), '| crosstab clean x side:'); print(pd.crosstab([m.ifr, m.fid_used], m.side))
c = m[m.clean]
print('\nCLEAN 2025: n', len(c), ' exceptions', (c.side == 'INSIDE').sum(), ' one-sided', c.onesided.sum(), ' (of which y>hi', ((c.side == 'above') & c.onesided).sum(), ')')
print(' two-sided rows: side by off_alobt bins'); tw = c[~c.onesided]
print(pd.crosstab(pd.cut(tw.off_alobt, [3594, 4000, 4500, 5000, 6000, 8000, 12000, 1e6]), tw.side))
print(' clean by APT x side'); print(pd.crosstab(c.APT, c.side))
# OOF effect: one-sided projection on clean rows
r = lambda h, k: np.sqrt(np.mean((y[k] - h[k]) ** 2)); ALL = np.ones(len(y), bool)
held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7]); J = month == 1; Jl = month == 7
def eff(name, rows, newv):
    g2 = f.copy(); g2[rows] = newv; dd = (y - g2) ** 2 - (y - f) ** 2
    print(f'OOF {name}: rows changed {(g2 != f).sum()}, worse {(dd > 0).sum()}, SSE {dd.sum():+.4g}, 12m {r(g2, ALL) - r(f, ALL):+.4f} held {r(g2, held) - r(f, held):+.4f} '
          f'Jan+Jul {r(g2, S) - r(f, S):+.4f} Jan {r(g2, J) - r(f, J):+.4f} Jul {r(g2, Jl) - r(f, Jl):+.4f}')
    return g2
k = c[c.onesided & c.inside]
eff('clean one-sided -> max(f, hi)', k.i.to_numpy(), np.maximum(k.f, k.hi).to_numpy())
print(k[['APT', 'FLIGHT_mvt', 'y', 'f', 'hi', 'P', 'month']].to_string())
k2 = c[~c.onesided & c.inside]
eff('clean two-sided -> lo (assume below)', k2.i.to_numpy(), np.maximum(k2.lo, 0).to_numpy())
eff('clean two-sided -> nearest edge', k2.i.to_numpy(), np.where(k2.f - k2.lo < k2.hi - k2.f, np.maximum(k2.lo, 0), k2.hi))
print(k2[['APT', 'FLIGHT_mvt', 'y', 'f', 'lo', 'hi', 'side', 'P', 'month']].to_string())
# 2026
c6 = m6[m6.clean]
print('\n2026 linked', len(m6), ' clean', len(c6), ' clean one-sided', c6.onesided.sum(), ' v28 inside: one-sided', (c6.onesided & c6.inside).sum(), ' two-sided', (~c6.onesided & c6.inside).sum())
k6 = c6[c6.onesided & c6.inside]; sh = (np.maximum(k6.f, k6.hi) - k6.f)
print(k6[['APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'a_APT', 'f', 'off_alobt', 'hi', 'MVT_ID_mvt']].assign(move=sh).to_string())
print(' one-sided: moves', sh.round().tolist(), ' sumsq', f'{(sh ** 2).sum():.4g}', ' bound', np.sqrt(235.3996 ** 2 - (sh ** 2).sum() / 344841))
k62 = c6[~c6.onesided & c6.inside]
print(' two-sided inside rows:'); print(k62[['APT', 'FLIGHT_mvt', 'f', 'off_alobt', 'lo', 'hi']].to_string())
m.to_parquet(H + 'h10_link25.parquet'); m6.to_parquet(H + 'h10_link26.parquet')
