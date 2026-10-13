import numpy as np, pandas as pd
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']
m = pd.read_parquet(H + 'h10_link25.parquet'); m6 = pd.read_parquet(H + 'h10_link26.parquet')
for x in (m, m6):
    x['off_sched'] = x.MVT - x.SCHED; x['stamp_excl'] = (x.off_sched >= x.lo) & (x.off_sched <= x.hi)
c = m[m.clean].copy(); c6 = m6[m6.clean].copy()
print('2025 clean: cs_eq False rows', (~c.cs_eq).sum(), '(all hold)')
tw = c[~c.onesided]
print('2025 clean two-sided: n', len(tw), ' stamping excluded (off_sched in window)', tw.stamp_excl.sum(), ' off_sched > hi', (tw.off_sched > tw.hi).sum(), ' off_sched < lo', (tw.off_sched < tw.lo).sum())
print(pd.crosstab(np.where(tw.off_sched > tw.hi, 'sched>hi', np.where(tw.stamp_excl, 'sched in win', 'sched<lo')), tw.side))
print('  rows with off_sched > hi:'); print(tw[tw.off_sched > tw.hi][['APT', 'FLIGHT_mvt', 'y', 'f', 'off_sched', 'lo', 'hi', 'side']].to_string())
os_ = c[c.onesided]
print('\n2025 one-sided (all above): y - hi quantiles', (os_.y - os_.hi).quantile([0, .1, .25, .5, .75, .9, 1]).round().to_dict())
print(' y - hi by APT median', (os_.y - os_.hi).groupby(os_.APT).agg(['size', 'median', 'min']).to_string())
print(' off_sched vs window for one-sided: sched<lo', (os_.off_sched < os_.lo).sum(), ' in', os_.stamp_excl.sum(), ' >hi', (os_.off_sched > os_.hi).sum())
print(os_[['APT', 'FLIGHT_mvt', 'y', 'f', 'off_sched', 'off_alobt', 'mvt_aobt', 'hi']].sort_values('APT').to_string())
# 2026 two-sided inside rows: off_sched position
t6 = c6[~c6.onesided & c6.inside].copy()
t6['move_to_lo'] = np.maximum(t6.lo, 0) - t6.f
print('\n2026 clean two-sided inside rows:'); print(t6[['APT', 'FLIGHT_mvt', 'f', 'off_sched', 'off_alobt', 'lo', 'hi', 'stamp_excl', 'move_to_lo', 'MVT_ID_mvt']].to_string())
for nm, k in [('all', t6.index == t6.index), ('stamp excluded', t6.stamp_excl), ('sched>hi', t6.off_sched > t6.hi)]:
    s = t6[k].move_to_lo
    print(f'  {nm}: n {k.sum()}, sumsq {(s ** 2).sum():.4g}, bound if rule holds {np.sqrt(235.3996 ** 2 - (s ** 2).sum() / 344841) - 235.3996:+.4f}')
o6 = c6[c6.onesided]
print('\n2026 clean one-sided (all 21):'); print(o6[['APT', 'FLIGHT_mvt', 'f', 'off_sched', 'off_alobt', 'mvt_aobt', 'hi', 'inside']].to_string())
# OOF: two-sided split
r = lambda h, k: np.sqrt(np.mean((y[k] - h[k]) ** 2)); ALL = np.ones(len(y), bool)
held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7]); J = month == 1; Jl = month == 7
def eff(name, rows, newv):
    g2 = f.copy(); g2[rows] = newv; dd = (y - g2) ** 2 - (y - f) ** 2
    print(f'OOF {name}: rows changed {(g2 != f).sum()}, worse {(dd > 0).sum()}, SSE {dd.sum():+.4g}, 12m {r(g2, ALL) - r(f, ALL):+.4f} held {r(g2, held) - r(f, held):+.4f} '
          f'Jan+Jul {r(g2, S) - r(f, S):+.4f} Jan {r(g2, J) - r(f, J):+.4f} Jul {r(g2, Jl) - r(f, Jl):+.4f}')
k2 = tw[tw.inside & tw.stamp_excl]; eff('two-sided, stamping excluded -> lo', k2.i.to_numpy(), np.maximum(k2.lo, 0).to_numpy())
k3 = tw[tw.inside & ~tw.stamp_excl]; eff('two-sided, sched outside win -> lo', k3.i.to_numpy(), np.maximum(k3.lo, 0).to_numpy())
k1 = os_[os_.inside]
eff('one-sided -> hi (guaranteed)', k1.i.to_numpy(), np.maximum(k1.f, k1.hi).to_numpy())
for cc in (100, 300, 500):
    eff(f'one-sided -> max(f, hi+{cc}) [stat]', os_.i.to_numpy(), np.maximum(os_.f, os_.hi + cc).to_numpy())
both = np.r_[k1.i.to_numpy(), k2.i.to_numpy()]; vals = np.r_[np.maximum(k1.f, k1.hi).to_numpy(), np.maximum(k2.lo, 0).to_numpy()]
eff('one-sided -> hi  +  two-sided stamp-excl -> lo', both, vals)
