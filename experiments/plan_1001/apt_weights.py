"""Screen (post hoc, read-only): airport-specific NNLS blend weights on A/B/C outside P, v24 structure.
Seen first: corr(seq - tree consensus, y - f) = +0.116 at LTFM (held-out), ~0 elsewhere -> seq under-weighted at LTFM?
"""
import os, sys
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import block_bootstrap

SCREEN = [3, 6, 9, 12]
z = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, f, month, P, D = z['y'], z['f'], z['month'], z['P'], z['D']
r = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc'])
apt = r.apt.astype(str).to_numpy().astype(str)
block = np.char.add(apt, r.date_loc.astype(str).to_numpy().astype(str))
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
def oof(folder, fam, seeds, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o
M = {fm: oof(CACHE / 'cv' / 'adsb2', fm, [0, 1, 2]) for fm in FAMS}
M['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2]) for v in POOL], axis=0)
F8 = FAMS + ['seq']; X = np.column_stack([M[m] for m in F8])
A = ~D & ~P; fit = np.isin(month, SCREEN)
held = ~fit; sc = np.isin(month, [1, 7])
w0, _ = nnls(X[fit & A], y[fit & A])
chk = np.clip(X[A] @ w0, 0, None)
print('shared A/B/C weights', dict(zip(F8, np.round(w0, 3))), ' max |v24 - refit| on A rows', np.abs(chk - f[A]).max().round(3))

def evaluate(name, g):
    out = []
    for nm, k in (('held', held), ('scored', sc), ('Jan', month == 1), ('Jul', month == 7)):
        b = block_bootstrap(y[k], g[k], f[k], block[k], n_boot=500)
        out.append(f"{nm} {b['point']:+.3f} [{b['lo']:+.3f},{b['hi']:+.3f}]")
    d = (y - g) ** 2 - (y - f) ** 2
    top = np.argsort(d[held])[:100]
    kk = np.where(held)[0]; mask = np.ones(len(y), bool); mask[kk[top]] = False
    w100 = np.sqrt(((y - g) ** 2)[held & mask].mean()) - np.sqrt(((y - f) ** 2)[held & mask].mean())
    mb = sum(np.sqrt(((y - g) ** 2)[month == m].mean()) < np.sqrt(((y - f) ** 2)[month == m].mean()) for m in range(1, 13))
    print(f'{name:30s} ' + '  '.join(out) + f'  w/o top100 held {w100:+.3f}  months better {mb}/12')
    pa = {a_: np.sqrt(((y - g) ** 2)[held & (apt == a_)].mean()) - np.sqrt(((y - f) ** 2)[held & (apt == a_)].mean()) for a_ in sorted(set(apt))}
    print('   per airport held:', ' '.join(f'{a_} {v:+.2f}' for a_, v in pa.items()))

# (a) every airport its own weights
g = f.copy(); W = {}
for a_ in sorted(set(apt)):
    m_ = A & (apt == a_)
    w, _ = nnls(X[fit & m_], y[fit & m_]); W[a_] = w
    g[m_] = np.clip(X[m_] @ w, 0, None)
evaluate('(a) per-airport NNLS', g)
print('   weights:'); print(pd.DataFrame(W, index=F8).T.round(3).to_string())
# (b) LTFM only
g = f.copy(); m_ = A & (apt == 'LTFM'); g[m_] = np.clip(X[m_] @ W['LTFM'], 0, None)
evaluate('(b) LTFM own weights only', g)
# (c) clock vs NM-blind clusters
clock = np.isin(apt, ['EHAM', 'EGLL', 'LFPG', 'EDDF'])
g = f.copy()
for cl in (clock, ~clock):
    m_ = A & cl; w, _ = nnls(X[fit & m_], y[fit & m_]); g[m_] = np.clip(X[m_] @ w, 0, None)
    print('   cluster weights', dict(zip(F8, np.round(w, 3))))
evaluate('(c) clock / NM-blind weights', g)
# (d) shrunk per-airport: 0.5 shared + 0.5 own
g = f.copy()
for a_ in sorted(set(apt)):
    m_ = A & (apt == a_); g[m_] = np.clip(X[m_] @ (0.5 * w0 + 0.5 * W[a_]), 0, None)
evaluate('(d) 0.5 shared + 0.5 per-airport', g)
# (e) Jan<->Jul style check: weights fitted on ALL held-out-free months? (fit on 11 months, apply to the 12th), shared
g = f.copy()
for m in range(1, 13):
    tr = A & (month != m); te = A & (month == m)
    w, _ = nnls(X[tr], y[tr]); g[te] = np.clip(X[te] @ w, 0, None)
evaluate('(e) shared, LOMO-fitted (11 months)', g)
g = f.copy()
for m in range(1, 13):
    for a_ in sorted(set(apt)):
        tr = A & (month != m) & (apt == a_); te = A & (month == m) & (apt == a_)
        w, _ = nnls(X[tr], y[tr]); g[te] = np.clip(X[te] @ w, 0, None)
evaluate('(f) per-airport, LOMO-fitted', g)
