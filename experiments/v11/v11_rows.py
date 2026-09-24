"""How concentrated is each arm's gain? Top rows by |dSSE|, and the 12-month / February effect without them."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from prc.config import CACHE
OUT = CACHE / 'v11'
b = np.load(OUT / 'base.npz'); base, y, month, P = b['base'], b['y'], b['month'], b['P']; o = ~P
rm = lambda p, k: np.sqrt(np.mean((y[k] - p[k]) ** 2))
for a, ref in [('null', None), ('a', None), ('c', None), ('c', 'b')]:
    pa = base + np.nan_to_num(np.load(OUT / f'oof_{a}.npy').astype(float))
    pr = base if ref is None else base + np.nan_to_num(np.load(OUT / f'oof_{ref}.npy').astype(float))
    d = (y - pa) ** 2 - (y - pr) ** 2; d[P] = 0
    top = np.argsort(-np.abs(d))[:5]
    keep = o.copy(); keep[top] = False
    print(f'{a}-{ref or "base"}: 12m {rm(pa, o) - rm(pr, o):+.3f} | without its top-5 |dSSE| rows {rm(pa, keep) - rm(pr, keep):+.3f} | '
          f'Feb {rm(pa, o & (month == 2)) - rm(pr, o & (month == 2)):+.2f} -> {rm(pa, keep & (month == 2)) - rm(pr, keep & (month == 2)):+.2f}')
    print('    top rows: ' + '; '.join(f'm{month[i]} y {y[i]:.0f} ref {pr[i]:.0f} arm {pa[i]:.0f}' for i in top))
