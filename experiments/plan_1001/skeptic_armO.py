"""Blend-level member-seed gain 1 -> 3 seeds (seq pool fixed at 27 fits), to bound arm O (3 -> 6) by the
variance model: MSE(3) - MSE(6) ~ (MSE(1) - MSE(3)) / 4. Read-only, no training."""
import sys
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import nnls
from prc.config import CACHE
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
V = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, month, P, D, f24, corr = V['y'], V['month'], V['P'], V['D'], V['f'], V['corr']
def oof(folder, fam, seeds):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}.npy') / len(seeds)
    return o
seq = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2]) for v in POOL], axis=0)
fit = np.isin(month, [3, 6, 9, 12]); held = ~fit; S = np.isin(month, [1, 7])
A, DnP = ~D & ~P, D & ~P
def blend(X):
    out = f24.copy(); fm = fit & ~P
    for g, gD, isD in ((~D, A, False), (D, DnP, True)):
        w, _ = nnls(X[fm & g], y[fm & g]); out[gD] = np.clip(X[gD] @ w + (0.5 * corr[gD] if isD else 0), 0, None)
    return out
mse = lambda g, k: float(np.mean((y[k] - g[k]) ** 2))
R = lambda g, k: float(np.sqrt(mse(g, k)))
per = {s: {f: oof(CACHE / 'cv' / 'adsb2', f, [s]) for f in FAMS} for s in (0, 1, 2)}
avg = {f: np.mean([per[s][f] for s in (0, 1, 2)], axis=0) for f in FAMS}
b3 = blend(np.column_stack([avg[f] for f in FAMS] + [seq]))
print(f'3-seed members blend: held {R(b3, held):.3f} scored {R(b3, S):.3f} (v24 {R(f24, held):.3f} / {R(f24, S):.3f})')
m1h, m1s = [], []
for s in (0, 1, 2):
    b1 = blend(np.column_stack([per[s][f] for f in FAMS] + [seq]))
    m1h.append(mse(b1, held)); m1s.append(mse(b1, S))
    print(f'  members seed {s} only: held {R(b1, held):.3f} scored {R(b1, S):.3f}')
for nm, m1, k in (('held', np.mean(m1h), held), ('scored', np.mean(m1s), S)):
    m3 = mse(b3, k); s2 = 1.5 * (m1 - m3); m6 = m3 - s2 / 6
    print(f'{nm}: RMSE(1 seed, mean MSE) {np.sqrt(m1):.3f} -> 3 seeds {np.sqrt(m3):.3f}  (gain {np.sqrt(m3)-np.sqrt(m1):+.3f});'
          f'  variance-model 3 -> 6 seeds {np.sqrt(m6) - np.sqrt(m3):+.3f}  (inf seeds {np.sqrt(m3 - s2/3) - np.sqrt(m3):+.3f})')
