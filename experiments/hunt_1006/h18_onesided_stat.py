"""[E] statistical extension for clean one-sided rows: predict the stamped value off_sched when off_sched > hi."""
import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 200)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']
m = pd.read_parquet(H + 'h10_link25.parquet'); m6 = pd.read_parquet(H + 'h10_link26.parquet')
for x in (m, m6): x['off_sched'] = x.MVT - x.SCHED; x['circ'] = x.APT == x.ADES_mvt
c = m[m.clean & m.onesided & ~m.circ].copy(); c6 = m6[m6.clean & m6.onesided & ~m6.circ].copy()
print('2025 clean one-sided (non-circular):', len(c), ' circular clean rows 2025:', (m.clean & m.circ).sum(), ' 2026:', (m6.clean & m6.circ).sum())
c['stamp_ok'] = c.off_sched > c.hi; c['stamped'] = (c.y - c.off_sched).abs() <= 6
print(pd.crosstab([c.APT, c.stamp_ok], c.stamped))
r = lambda h, k: np.sqrt(np.mean((y[k] - h[k]) ** 2)); ALL = np.ones(len(y), bool); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
J = month == 1; Jl = month == 7
def eff(name, rows, newv):
    g2 = f.copy(); g2[rows] = newv; dd = (y - g2) ** 2 - (y - f) ** 2
    print(f'OOF {name}: changed {(g2 != f).sum()}, worse {(dd > 0).sum()}, SSE {dd.sum():+.4g}, 12m {r(g2, ALL) - r(f, ALL):+.4f} held {r(g2, held) - r(f, held):+.4f} '
          f'Jan+Jul {r(g2, S) - r(f, S):+.4f} Jan {r(g2, J) - r(f, J):+.4f} Jul {r(g2, Jl) - r(f, Jl):+.4f}  months better/worse {sum(r(g2, month == k) < r(f, month == k) for k in range(1, 13))}/{sum(r(g2, month == k) > r(f, month == k) for k in range(1, 13))}')
base = np.maximum(c.f, c.hi)
eff('A: max(f, hi) [guaranteed]', c.i.to_numpy(), base.to_numpy())
e1 = np.where(c.stamp_ok & (c.APT == 'LIRF'), c.off_sched, base); eff('A + LIRF stamp_ok -> off_sched', c.i.to_numpy(), e1)
e2 = np.where(c.stamp_ok, c.off_sched, base); eff('A + all stamp_ok -> off_sched', c.i.to_numpy(), e2)
e3 = np.where(c.stamp_ok, (c.off_sched + c.hi) / 2, base); eff('A + all stamp_ok -> mid(hi, off_sched)', c.i.to_numpy(), e3)
e4 = np.where(c.stamp_ok & (c.APT == 'LIRF'), c.off_sched, np.where(c.stamp_ok, (c.off_sched + c.hi) / 2, base)); eff('A + LIRF->off_sched, others->mid', c.i.to_numpy(), e4)
# 2026
c6['stamp_ok'] = c6.off_sched > c6.hi
c6['A'] = np.maximum(c6.f, c6.hi); c6['E1'] = np.where(c6.stamp_ok & (c6.APT == 'LIRF'), np.maximum(c6.off_sched, c6.A), c6.A)
c6['E4'] = np.where(c6.stamp_ok & (c6.APT == 'LIRF'), np.maximum(c6.off_sched, c6.A), np.where(c6.stamp_ok, np.maximum((c6.off_sched + c6.hi) / 2, c6.A), c6.A))
print(c6[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'f', 'hi', 'off_sched', 'stamp_ok', 'A', 'E1', 'E4']].to_string())
for k in ['A', 'E1', 'E4']:
    s = c6[k] - c6.f; print(f' 2026 {k}: rows moved {(s != 0).sum()}, sum sq move {(s ** 2).sum():.4g}')
