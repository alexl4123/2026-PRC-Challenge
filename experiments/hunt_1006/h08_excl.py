"""Exclusion window from the own NM arrival record, by subset; OOF effect (2025) and v28 moves (2026)."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
H = R + 'cache/hunt/'
m = pd.read_parquet(H + 'h07_link25.parquet'); m6 = pd.read_parquet(H + 'h07_link26.parquet')
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']; P = V['P']
m['f'] = f[m.i.to_numpy()]; m['P'] = P[m.i.to_numpy()]; m['month'] = month[m.i.to_numpy()]
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet')
m6 = m6.merge(v.rename(columns={'TAXITIME_SEC_mvt': 'f'}), on='MVT_ID_mvt', how='left')
for x in (m, m6):
    x['lo'] = x.off_alobt - 3606; x['hi'] = x.off_alobt + 3606
    x['onesided'] = x.lo < -12          # lower ray infeasible (y >= -12): BLOCK must be > 1 h BEFORE LOBT => y > hi
    x['strict'] = x.cs_eq & (x.FLIGHT_RULE_mvt.astype(str) == 'I') & (x.APT != x.ADES_mvt) & x.type_eq
    x['inside'] = (x.f > x.lo) & (x.f < x.hi)
m['excl'] = (m.BLOCK - m.a_LOBT).abs() > 3606
for nm, k in [('all', m.index == m.index), ('strict', m.strict), ('onesided', m.onesided), ('strict&onesided', m.strict & m.onesided)]:
    g = m[k]
    print(f'2025 {nm:16s} n={len(g):4d} exclusion holds {g.excl.mean():.4f} violations {(~g.excl).sum():3d} | OOF pred inside window {g.inside.sum():3d} '
          f'(of which violations {(g.inside & ~g.excl).sum()})')
print('\n2025 one-sided rows (all): y, f, hi, excl')
print(m[m.onesided][['APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'FLIGHT_RULE_mvt', 'strict', 'P', 'y', 'f', 'off_alobt', 'hi', 'excl', 'month']].sort_values('y').to_string())
# OOF effect of one-sided projection f -> max(f, hi) on strict&onesided and on onesided
r = lambda h, k: np.sqrt(np.mean((y[k] - h[k]) ** 2))
held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
for nm, k in [('onesided', m.onesided), ('strict&onesided', m.strict & m.onesided)]:
    g = m[k & m.inside]; g2 = f.copy(); g2[g.i.to_numpy()] = np.maximum(g.f.to_numpy(), g.hi.to_numpy())
    dd = (y - g2) ** 2 - (y - f) ** 2; ch = g.i.to_numpy()
    print(f'OOF {nm}: rows changed {len(ch)}, worse {(dd[ch] > 0).sum()}, SSE change {dd[ch].sum():.4g}, 12m {r(g2, np.ones(len(y), bool)) - r(f, np.ones(len(y), bool)):+.4f} '
          f'held {r(g2, held) - r(f, held):+.4f} Jan+Jul {r(g2, S) - r(f, S):+.4f}')
print('\n2026 linked:', len(m6), ' strict', m6.strict.sum(), ' onesided', m6.onesided.sum(), ' strict&onesided', (m6.strict & m6.onesided).sum())
print(' v28 inside excluded window: all', m6.inside.sum(), ' strict', (m6.inside & m6.strict).sum(), ' onesided', (m6.inside & m6.onesided).sum(), ' strict&onesided', (m6.inside & m6.strict & m6.onesided).sum())
print(m6[m6.onesided][['APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'FLIGHT_RULE_mvt', 'strict', 'f', 'off_alobt', 'hi', 'inside']].to_string())
sh = (np.maximum(m6.f, m6.hi) - m6.f)[m6.onesided & m6.inside]
print(' one-sided moves (s):', sh.round().tolist(), ' sum sq', f'{(sh ** 2).sum():.4g}', ' bound RMSE', np.sqrt(235.3996 ** 2 - (sh ** 2).sum() / 344841))
