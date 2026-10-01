"""Skeptic re-check (read-only, no training) of today's post-hoc blend screens.
1. R5 done correctly: drop the 100 most-improved HELD-OUT rows (blend_ut.py dropped the global top-100, some of
   which sit in the screening months where the weights were fitted).
2. GREKI-style cross-fit: fit on Jan only -> judge Jul, fit on Jul only -> judge Jan, each variant vs the shared
   weights fitted the same way.
3. LOMO fit (11 months -> 12th), variant vs shared-LOMO.
Variants: U (route x seq-pool-sd tercile NNLS), UT (U + lambda*(seq10-seqnc10)), D (0.5 shared + 0.5 per-airport, A/B/C).
"""
import sys
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
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
base = build(fit, 'shared')
print(f'shared refit vs v24 held {R(base, held) - R(f24, held):+.4f} (should be ~0)')
for kind in ('U', 'UT', 'D'):
    g = build(fit, kind)
    gain = (y - f24) ** 2 - (y - g) ** 2
    top_glob = np.argsort(gain)[::-1][:100]
    n_in_fit = int(fit[top_glob].sum())
    hidx = np.where(held)[0]; top_h = hidx[np.argsort(gain[held])[::-1][:100]]
    keep = np.ones(len(y), bool); keep[top_h] = False
    sidx = np.where(S)[0]; top_s = sidx[np.argsort(gain[S])[::-1][:100]]
    keep_s = np.ones(len(y), bool); keep_s[top_s] = False
    keep_g = np.ones(len(y), bool); keep_g[top_glob] = False
    print(f'[{kind}] held {R(g, held) - R(f24, held):+.3f}  scored {R(g, S) - R(f24, S):+.3f}  | global top-100 rows in fit months: {n_in_fit}'
          f'  | w/o top100 (global, as screened) held {R(g, held & keep_g) - R(f24, held & keep_g):+.3f}'
          f'  | w/o top100 HELD-OUT rows held {R(g, held & keep) - R(f24, held & keep):+.3f}'
          f'  | w/o top100 SCORED rows scored {R(g, S & keep_s) - R(f24, S & keep_s):+.3f}')
    pa = ' '.join(f'{a_} {R(g, held & (apt == a_)) - R(f24, held & (apt == a_)):+.2f}' for a_ in APTS)
    print('    per airport held:', pa)

print('--- Jan<->Jul cross-fit (variant vs shared fitted on the same single month) ---')
for fm_, ev in ((1, 7), (7, 1)):
    sh = build(month == fm_, 'shared'); k = month == ev
    row = [f'fit {fm_} judge {ev}: shared-vs-v24 {R(sh, k) - R(f24, k):+.3f}']
    for kind in ('U', 'UT', 'D'):
        g = build(month == fm_, kind)
        row.append(f'{kind} {R(g, k) - R(sh, k):+.3f}')
    print('  ' + '  '.join(row))

print('--- LOMO fit (11 months -> 12th), variant vs shared-LOMO ---')
lo = {k: f24.copy() for k in ('shared', 'U', 'UT', 'D')}
for m in range(1, 13):
    te = month == m
    for kind in lo:
        g = build(month != m, kind); lo[kind][te] = g[te]
allm = np.ones(len(y), bool)
for kind in ('U', 'UT', 'D'):
    g, sh = lo[kind], lo['shared']
    gain = (y - sh) ** 2 - (y - g) ** 2
    top = np.argsort(gain)[::-1][:100]; keep = np.ones(len(y), bool); keep[top] = False
    b = block_bootstrap(y, g, sh, block, n_boot=500)
    mb = sum(R(g, month == m) < R(sh, month == m) for m in range(1, 13))
    print(f'  [{kind}] 12m {b["point"]:+.3f} [{b["lo"]:+.3f},{b["hi"]:+.3f}]  Jan {R(g, month == 1) - R(sh, month == 1):+.3f}'
          f'  Jul {R(g, month == 7) - R(sh, month == 7):+.3f}  months better {mb}/12  w/o top100 {R(g, keep) - R(sh, keep):+.3f}')
