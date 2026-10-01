"""Robustness of the post-hoc airport-weight screen: per member seed (R4-style) and Jan<->Jul cross-fit (GREKI scheme)."""
import sys
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
z = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, f, month, P, D = z['y'], z['f'], z['month'], z['P'], z['D']
apt = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt']).apt.astype(str).to_numpy().astype(str)
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
def oof(folder, fam, seeds):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}.npy') / len(seeds)
    return o
seq = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2]) for v in POOL], axis=0)
A = ~D & ~P; fit = np.isin(month, [3, 6, 9, 12]); held = ~fit; sc = np.isin(month, [1, 7])
R = lambda g, k: np.sqrt(((y - g) ** 2)[k].mean())
APTS = sorted(set(apt))
def variants(X, fitmask):
    w0, _ = nnls(X[fitmask & A], y[fitmask & A]); g0 = f.copy(); g0[A] = np.clip(X[A] @ w0, 0, None)
    gd = g0.copy(); gb = g0.copy()
    for a_ in APTS:
        m_ = A & (apt == a_); w, _ = nnls(X[fitmask & m_], y[fitmask & m_])
        gd[m_] = np.clip(X[m_] @ (0.5 * w0 + 0.5 * w), 0, None)
        if a_ == 'LTFM': gb[m_] = np.clip(X[m_] @ w, 0, None)
    return g0, gd, gb
for s in (0, 1, 2):
    X = np.column_stack([oof(CACHE / 'cv' / 'adsb2', fm, [s]) for fm in FAMS] + [seq])
    g0, gd, gb = variants(X, fit)
    print(f'seed {s}: shared held {R(g0, held):.3f} | (d) held {R(gd, held) - R(g0, held):+.3f} scored {R(gd, sc) - R(g0, sc):+.3f} '
          f'Jan {R(gd, month == 1) - R(g0, month == 1):+.3f} Jul {R(gd, month == 7) - R(g0, month == 7):+.3f} | (b) LTFM held {R(gb, held) - R(g0, held):+.3f} '
          f'scored {R(gb, sc) - R(g0, sc):+.3f}')
X = np.column_stack([oof(CACHE / 'cv' / 'adsb2', fm, [0, 1, 2]) for fm in FAMS] + [seq])
for fitm, evm in ((1, 7), (7, 1)):
    g0, gd, gb = variants(X, month == fitm)
    k = month == evm
    print(f'fit on month {fitm}, judge month {evm}: (d) {R(gd, k) - R(g0, k):+.3f}  (b) {R(gb, k) - R(g0, k):+.3f}  '
          f'| LTFM rows (d) {R(gd, k & (apt == "LTFM")) - R(g0, k & (apt == "LTFM")):+.3f}')
