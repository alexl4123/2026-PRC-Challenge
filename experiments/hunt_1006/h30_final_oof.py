"""Per-rule 2025 OOF effects (v28-structure OOF), identical code path to h25_candidate.rules()."""
import sys; sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt')
import numpy as np, pandas as pd
import importlib.util
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
src = open(H + 'h25_candidate.py').read().split('# ---- 2025 OOF')[0]
ns = {}; exec(src, ns); rules = ns['rules']; W = ns['W']; CAP = ns['CAP']
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']
g, e, info, d = rules('tr', f)
i = info.i.to_numpy()
A = f.copy(); A[i[info.one]] = g[i[info.one]]
B = f.copy(); B[i[info.B]] = g[i[info.B]]
sp = d.APT.isin(['LEBL', 'LEMD']).to_numpy(); Cc = f.copy(); Cc[sp] = np.minimum(f[sp], CAP)
r = lambda h, k: np.sqrt(np.mean((y[k] - h[k]) ** 2)); ALL = np.ones(len(y), bool); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
for nm, h in (('A one-sided (exact)', A), ('B two-sided stamp-excl', B), ('C LEBL/LEMD cap', Cc), ('A+B+C', g), ('A+B+C+E1 [E]', e)):
    dd = (y - h) ** 2 - (y - f) ** 2; ch = h != f; gain = -dd[ch]
    top = np.sort(gain)[::-1][:10].sum() / gain.sum() if gain.sum() > 0 else float('nan')
    print(f'{nm:24s}: changed {ch.sum():4d} (Jan+Jul {(ch & S).sum():3d}), worse {(dd > 0).sum()}, 12m {r(h, ALL) - r(f, ALL):+.4f}, held {r(h, held) - r(f, held):+.4f}, '
          f'Jan+Jul {r(h, S) - r(f, S):+.4f}, Jan {r(h, month == 1) - r(f, month == 1):+.4f}, Jul {r(h, month == 7) - r(f, month == 7):+.4f}, '
          f'months better {sum(r(h, month == k) < r(f, month == k) - 1e-12 for k in range(1, 13))}/12, top-10 share {top:.2f}')
