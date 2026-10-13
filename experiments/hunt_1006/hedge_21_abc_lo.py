"""A/B/C rows where the LOBT window's lower edge excludes a normal taxi (lo = off_lobt - 3606 > 2000 s).
What are they (stamp / BLOCK~LOBT / long taxi), how well does v28 do, and would 'stamp if SCHED is inside the window' help?"""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250)
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P, D, f, f27, ol = V['y'], V['month'], V['P'], V['D'], V['f'], V['f27'], V['off_lobt']
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'off_sched'])
so = d.off_sched.to_numpy(); apt = d.APT.astype(str).to_numpy()
lo = ol - 3606; hi = ol + 3606
m = np.isfinite(ol) & (lo > 2000)
tot = ((y - f) ** 2).sum()
print(f'A/B/C rows with lo > 2000: {m.sum()}; SSE share {((y - f) ** 2)[m].sum() / tot:.4f}; projected by the window {int((m & (f != f27)).sum())}')
stamp = np.abs(y - so) <= 60
bl = np.abs((ol - y)) <= 600   # BLOCK within 10 min of LOBT
print(f'  stamped {stamp[m].mean():.3f}  BLOCK~LOBT(+-600) {bl[m].mean():.3f}  y-lo quantiles {np.quantile((y - lo)[m], [.1, .5, .9]).round().tolist()}')
s_in = m & (so >= lo) & (so <= hi)
print(f'  of them SCHED inside window: {s_in.sum()}, stamped {stamp[s_in].mean():.3f}; by airport {pd.Series(apt[s_in]).value_counts().head(5).to_dict()}')
S = np.isin(month, [1, 7]); held = ~np.isin(month, [3, 6, 9, 12]); allm = np.ones(len(y), bool)
rm = lambda g, k: np.sqrt(np.mean((y[k] - g[k]) ** 2))
for nm, val in (('-> off_sched (stamp) when SCHED in window', so), ('-> MVT - LOBT - 251 (median BLOCK-LOBT)', ol - 251)):
    g = f.copy(); g[s_in] = np.clip(val[s_in], lo[s_in], hi[s_in])
    ch = s_in & (np.abs(g - f) > .5)
    print(f'  rule {nm}: changed {ch.sum()} worse {int((ch & ((y - g) ** 2 > (y - f) ** 2)).sum())} | 12m {rm(g, allm) - rm(f, allm):+.3f} held {rm(g, held) - rm(f, held):+.3f} Jan+Jul {rm(g, S) - rm(f, S):+.3f}')
