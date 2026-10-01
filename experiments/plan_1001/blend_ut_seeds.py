import sys
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import nnls
from prc.config import CACHE
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
SCREEN = [3, 6, 9, 12]
def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o
V = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, month, P, D, f24, corr = V['y'], V['month'], V['P'], V['D'], V['f'], V['corr']
A, DnP = ~D & ~P, D & ~P
fit = np.isin(month, SCREEN) & ~P
held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
for s in (0, 1, 2):
    M = {f: oof(CACHE / 'cv' / 'adsb2', f, [s], month) for f in FAMS}
    B = np.array([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], month, '.bulk.npy') for v in POOL]); sd = B.std(0); del B
    M['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], month) for v in POOL], axis=0)
    dl = oof(CACHE / 'cv' / 'seq', 'seq', [s], month) - oof(CACHE / 'cv' / 'seq', 'seqnc', [s], month)
    X = np.column_stack([M[f] for f in FAMS + ['seq']])
    q = np.quantile(sd[fit], [1/3, 2/3]); b = np.digitize(sd, q)
    fo, fn = f24.copy(), f24.copy()
    for g, gD, isD in ((~D, A, False), (D, DnP, True)):
        w, _ = nnls(X[fit & g], y[fit & g]); fo[gD] = np.clip(X[gD] @ w + (0.5 * corr[gD] if isD else 0), 0, None)
        for c in (0, 1, 2):
            gg = g & (b == c); w, _ = nnls(X[fit & gg], y[fit & gg]); base = X @ w; k = fit & gg
            lam = float(np.dot(dl[k], (y - base)[k]) / np.dot(dl[k], dl[k])); sel = gD & (b == c)
            fn[sel] = np.clip(base[sel] + lam * dl[sel] + (0.5 * corr[sel] if isD else 0), 0, None)
    print(f'seed {s}: held {r(fn, held) - r(fo, held):+.3f}  scored {r(fn, S) - r(fo, S):+.3f}  Jan {r(fn, month == 1) - r(fo, month == 1):+.3f}  Jul {r(fn, month == 7) - r(fo, month == 7):+.3f}')
