"""2025: how exact are the stand-sequence bounds on the RECORDED block?"""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
c = pd.read_parquet(H + 'stand_ctx_training.parquet')
c['y'] = c['T'] - c.blk
c['Tam'] = c['T'] - c.a_m
c['B_am'] = c.blk - c.a_m            # >0: we left after the latest pre-take-off arrival (it was ours or earlier)
c['D'] = ~np.isfinite(c.aobt3)
print('N', len(c), 'a_m finite', np.isfinite(c.a_m).sum())
bins = [-1, 120, 300, 600, 900, 1200, 1800, 2700, 3600, 7200, 14400, 86400, 1e9]
c['bTam'] = pd.cut(c.Tam, bins)
g = c[np.isfinite(c.a_m)].groupby('bTam', observed=True)
print('\n[A] latest arrival at our stand before take-off (a_m): share with recorded BLOCK < a_m (an arrival during our taxi)')
print(g.agg(n=('y', 'size'), B_lt_am=('B_am', lambda x: (x < 0).mean()), B_lt_am_n=('B_am', lambda x: int((x < 0).sum())),
            mlink=('m_link', 'mean')).to_string())
print('\n[A by APT] T-a_m <= 600 s: n, share B<a_m (the L bound y >= T-a_m holds) ')
s = c[c.Tam <= 600]
print(s.groupby('APT').agg(n=('y', 'size'), hold=('B_am', lambda x: (x < 0).mean()), exc=('B_am', lambda x: int((x >= 0).sum()))).to_string())
print('exceptions detail T-a_m<=600 & B>=a_m: y quantiles', np.percentile(s[s.B_am >= 0].y, [0, 5, 50, 95, 100]) if (s.B_am >= 0).any() else None)
print(' their T-a_m quantiles', np.percentile(s[s.B_am >= 0].Tam, [0, 5, 50, 95, 100]) if (s.B_am >= 0).any() else None)
# with linkage: a_m has a different type/operator AND the next departure at the stand has a_m's type/operator
print('\n[B] a_m not ours by type/operator (m_link False): share B<a_m by T-a_m bin')
g = c[np.isfinite(c.a_m) & ~c.m_link].groupby('bTam', observed=True)
print(g.agg(n=('y', 'size'), B_lt_am=('B_am', lambda x: (x < 0).mean()), exc=('B_am', lambda x: int((x >= 0).sum()))).to_string())
print('\n[C] a_m not ours by type/op AND next stand departure matches a_m type/op (a_m = next rotation)')
g = c[np.isfinite(c.a_m) & ~c.m_link & c.nxt_link].groupby('bTam', observed=True)
print(g.agg(n=('y', 'size'), B_lt_am=('B_am', lambda x: (x < 0).mean()), exc=('B_am', lambda x: int((x >= 0).sum()))).to_string())
print('\n[D] U bound via own link: a_lk = latest same type/op arrival before T, with n_after_lk==0 (a_m is it); share B > a_lk')
c['B_alk'] = c.blk - c.a_lk
for cond, nm in ((c.n_after_lk == 0, 'n_after==0'), (c.n_after_lk >= 1, 'n_after>=1')):
    g = c[np.isfinite(c.a_lk) & cond].copy(); g['bT'] = pd.cut(g['T'] - g.a_lk, bins)
    print(nm); print(g.groupby('bT', observed=True).agg(n=('y', 'size'), B_gt=('B_alk', lambda x: (x > 0).mean()), exc=('B_alk', lambda x: int((x <= 0).sum()))).to_string())
c[['Tam', 'B_am', 'B_alk']].describe()
