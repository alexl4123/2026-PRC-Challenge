"""Clean exclusion rule with the union of two link keys (airport flight number, NM callsign). 2025 exactness, OOF, 2026 moves."""
import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 400)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT', 'SCHED', 'LOBT', 'AOBT3', 'CALLSIGN_flt', 'FLIGHT_RULE_mvt']
def links(yr):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=C + (['BLOCK'] if yr == 'tr' else []))
    a = pd.read_parquet(H + f'{yr}_arr.parquet', columns=C + ['BLOCK'])
    for x in (d, a):
        for c in ['APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt']: x[c] = x[c].astype(str)
    d['i'] = np.arange(len(d)); used = set(d.FLIGHT_ID_mvt.dropna().unique())
    D = d[d.LOBT.isna() & (d.FLIGHT_RULE_mvt.astype(str) == 'I') & (d.APT != d.ADES_mvt)]
    A = a[a.LOBT.notna() & ~a.FLIGHT_ID_mvt.isin(used)][['FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'APT', 'MVT', 'LOBT', 'AOBT3', 'FLIGHT_ID_mvt']]
    A = A.rename(columns={c: 'a_' + c for c in A.columns})
    out = []
    for key in ('a_FLIGHT_mvt', 'a_CALLSIGN_flt'):
        m = D.merge(A, left_on=['FLIGHT_mvt', 'APT', 'ADES_mvt'], right_on=[key, 'a_ADEP_mvt', 'a_APT'])
        m['t'] = m.a_MVT - m.MVT; m = m[(m.t > 0) & (m.t < 86400)]; m['key'] = key; out.append(m)
    m = pd.concat(out).sort_values('t').drop_duplicates('i').reset_index(drop=True)     # nearest NM arrival after take-off, either key
    m['off_alobt'] = m.MVT - m.a_LOBT; m['lo'] = m.off_alobt - 3606; m['hi'] = m.off_alobt + 3606; m['onesided'] = m.lo < -12
    m['off_sched'] = m.MVT - m.SCHED; m['stamp_excl'] = (m.off_sched >= m.lo) & (m.off_sched <= m.hi)
    m['month'] = pd.to_datetime(m.MVT, unit='s').dt.month
    if yr == 'tr': m['y'] = m.MVT - m.BLOCK; m['side'] = np.where(m.y > m.hi, 'above', np.where(m.y < m.lo, 'below', 'INSIDE'))
    return d, m
d, m = links('tr')
print('2025 union clean links:', len(m), ' by key', m.key.value_counts().to_dict())
print(pd.crosstab([m.onesided, m.stamp_excl], m.side))
print('per month n / exceptions:', m.groupby('month').side.apply(lambda s: f'{len(s)}/{(s == "INSIDE").sum()}').to_dict())
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']
m['f'] = f[m.i.to_numpy()]
r = lambda h, k: np.sqrt(np.mean((y[k] - h[k]) ** 2)); ALL = np.ones(len(y), bool); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
J = month == 1; Jl = month == 7
def eff(name, sel, newv):
    g2 = f.copy(); g2[m.i[sel].to_numpy()] = newv[sel]; dd = (y - g2) ** 2 - (y - f) ** 2
    mb = sum(r(g2, month == k) < r(f, month == k) for k in range(1, 13)); mw = sum(r(g2, month == k) > r(f, month == k) for k in range(1, 13))
    print(f'OOF {name}: changed {(g2 != f).sum()}, worse {(dd > 0).sum()}, SSE {dd.sum():+.4g}, 12m {r(g2, ALL) - r(f, ALL):+.4f} held {r(g2, held) - r(f, held):+.4f} '
          f'Jan+Jul {r(g2, S) - r(f, S):+.4f} Jan {r(g2, J) - r(f, J):+.4f} Jul {r(g2, Jl) - r(f, Jl):+.4f} months better/worse {mb}/{mw}')
    return g2
A = np.maximum(m.f, m.hi).to_numpy(); B = np.minimum(m.f, np.maximum(m.lo, 0)).to_numpy()
eff('A one-sided -> max(f,hi)', m.onesided.to_numpy(), A)
eff('B two-sided & stamp excluded -> min(f,lo)', (~m.onesided & m.stamp_excl).to_numpy(), B)
AB = np.where(m.onesided, A, np.where(m.stamp_excl, B, m.f))
eff('A+B', np.ones(len(m), bool), AB)
E1 = np.where(m.onesided & (m.off_sched > m.hi) & (m.APT == 'LIRF'), np.maximum(m.off_sched, A), AB)
eff('A+B+E1(LIRF one-sided stamp -> off_sched) [E]', np.ones(len(m), bool), E1)
m.to_parquet(H + 'h22_union25.parquet')
d6, m6 = links('ra')
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
m6 = m6.merge(v, on='MVT_ID_mvt', how='left')
m6['A'] = np.maximum(m6.f, m6.hi); m6['B'] = np.minimum(m6.f, np.maximum(m6.lo, 0))
m6['AB'] = np.where(m6.onesided, m6.A, np.where(m6.stamp_excl, m6.B, m6.f))
m6['E1'] = np.where(m6.onesided & (m6.off_sched > m6.hi) & (m6.APT == 'LIRF'), np.maximum(m6.off_sched, m6.A), m6.AB)
print('\n2026 union clean links:', len(m6), ' by key', m6.key.value_counts().to_dict(), ' one-sided', m6.onesided.sum(), ' two-sided stamp-excl', (~m6.onesided & m6.stamp_excl).sum())
k = m6[(m6.AB != m6.f) | (m6.E1 != m6.f)]
print(k[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'a_APT', 'key', 'onesided', 'stamp_excl', 'f', 'lo', 'hi', 'off_sched', 'AB', 'E1']].sort_values(['onesided', 'APT']).to_string())
for c in ['A', 'AB', 'E1']:
    sel = m6.onesided if c == 'A' else np.ones(len(m6), bool)
    s = (m6[c] - m6.f)[sel] if c != 'A' else (m6.A - m6.f)[m6.onesided]
    print(f' 2026 {c}: rows moved {(s != 0).sum()}, sum sq {(s ** 2).sum():.4g}, RMSE bound/estimate {np.sqrt(235.3996 ** 2 - (s ** 2).sum() / 344841) - 235.3996:+.4f}')
sB = (m6.B - m6.f)[~m6.onesided & m6.stamp_excl]; print(f' 2026 B only: rows {(sB != 0).sum()}, sum sq {(sB ** 2).sum():.4g}, bound {np.sqrt(235.3996 ** 2 - (sB ** 2).sum() / 344841) - 235.3996:+.4f}')
m6.to_parquet(H + 'h22_union26.parquet')
