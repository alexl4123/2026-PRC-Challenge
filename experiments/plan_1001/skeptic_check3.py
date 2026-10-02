"""Skeptic re-check (read-only, no training) of today's post-hoc blend screens.
1. R5 done correctly: drop the 100 most-improved HELD-OUT rows (blend_ut.py dropped the global top-100, some of
   which sit in the screening months where the weights were fitted).
2. GREKI-style cross-fit: fit on Jan only -> judge Jul, fit on Jul only -> judge Jan, each variant vs the shared
   weights fitted the same way.
3. LOMO fit (11 months -> 12th), variant vs shared-LOMO.
Variants: U (route x seq-pool-sd tercile NNLS), UT (U + lambda*(seq10-seqnc10)), D (0.5 shared + 0.5 per-airport, A/B/C).
"""
import sys
ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import block_bootstrap
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
SCREEN = [3, 6, 9, 12]
V = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, month, P, D, f24, corr = V['y'], V['month'], V['P'], V['D'], V['f'], V['corr']
def oof(folder, fam, seeds, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o
M = {f: oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2]) for f in FAMS}
s1 = np.zeros(len(y)); s2 = np.zeros(len(y)); n = 0
for v in POOL:
    for s in (0, 1, 2):
        x = oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], '.bulk.npy'); s1 += x; s2 += x * x; n += 1
sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)); del s1, s2
M['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2]) for v in POOL], axis=0)
dl = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2]) - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2])
X = np.column_stack([M[f] for f in FAMS + ['seq']]); del M
rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc'])
apt = rw.apt.astype(str).to_numpy()
block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
A, DnP = ~D & ~P, D & ~P
R = lambda g, k: float(np.sqrt(np.mean((y[k] - g[k]) ** 2)))
APTS = sorted(set(apt))

def build(fitmask, kind):
    """Return predictions (P rows = f24) for a weight design fitted on fitmask (P excluded)."""
    fm = fitmask & ~P
    out = f24.copy()
    q = np.quantile(sd[fm], [1 / 3, 2 / 3]); b = np.digitize(sd, q)
    for g, gD, isD in ((~D, A, False), (D, DnP, True)):
        add = (0.5 * corr) if isD else 0.0
        w0, _ = nnls(X[fm & g], y[fm & g])
        if kind == 'shared':
            out[gD] = np.clip(X[gD] @ w0 + (add[gD] if isD else 0), 0, None)
        elif kind in ('U', 'UT'):
            for c in (0, 1, 2):
                gg = g & (b == c); w, _ = nnls(X[fm & gg], y[fm & gg]); base = X @ w; sel = gD & (b == c)
                lam = 0.0
                if kind == 'UT':
                    k = fm & gg; lam = float(np.dot(dl[k], (y - base)[k]) / np.dot(dl[k], dl[k]))
                out[sel] = np.clip(base[sel] + lam * dl[sel] + (add[sel] if isD else 0), 0, None)
        elif kind == 'D':
            if isD:
                out[gD] = np.clip(X[gD] @ w0 + add[gD], 0, None)
            else:
                for a_ in APTS:
                    m_ = gD & (apt == a_); w, _ = nnls(X[fm & g & (apt == a_)], y[fm & g & (apt == a_)])
                    out[m_] = np.clip(X[m_] @ (0.5 * w0 + 0.5 * w), 0, None)
    return out


fit = np.isin(month, SCREEN); held = ~fit; S = np.isin(month, [1, 7])
# calibration: arm M (v23 = ep30 only -> v24 = 27-fit pool), outside P only, same weight-fitting protocol
seq30 = oof(CACHE / 'cv' / 'seqsw_ep30', 'seq', [0, 1, 2])
X23 = X.copy(); X23[:, -1] = seq30
def shared(Xm):
    out = f24.copy(); fm = fit & ~P
    for g, gD, isD in ((~D, A, False), (D, DnP, True)):
        w, _ = nnls(Xm[fm & g], y[fm & g]); out[gD] = np.clip(Xm[gD] @ w + (0.5 * corr[gD] if isD else 0), 0, None)
    return out
f23, fM = shared(X23), shared(X)
for nm, k in (('held', held), ('scored', S)):
    gain = (y - f23) ** 2 - (y - fM) ** 2
    srt = np.sort(gain[k])[::-1]; tot = gain[k].sum()
    idx = np.where(k)[0]; top = idx[np.argsort(gain[k])[::-1][:100]]; keep = np.ones(len(y), bool); keep[top] = False
    print(f'ARM M calibration {nm}: delta {R(fM, k) - R(f23, k):+.3f}  w/o own top100 {R(fM, k & keep) - R(f23, k & keep):+.3f}  share in top 100 / 1000 rows {srt[:100].sum()/tot:.2f} / {srt[:1000].sum()/tot:.2f}')
