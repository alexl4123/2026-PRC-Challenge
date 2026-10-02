"""Arm U+T: disagreement-tercile NNLS weights (U) plus lambda_(route,tercile) * (seq10 - seqnc10) fitted on the U residual
on screening months. Info for bundling; no training."""
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
def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o
V = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, month, P, D, f24, corr = V['y'], V['month'], V['P'], V['D'], V['f'], V['corr']
M = {f: oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month) for f in FAMS}
B = np.array([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], month, '.bulk.npy') for v in POOL for s in (0, 1, 2)])
sd = B.std(0); del B
M['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2], month) for v in POOL], axis=0)
dl = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2], month) - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2], month)
X = np.column_stack([M[f] for f in FAMS + ['seq']])
rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc'])
block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
A, DnP = ~D & ~P, D & ~P
fit = np.isin(month, SCREEN) & ~P
held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
q = np.quantile(sd[fit], [1/3, 2/3]); b = np.digitize(sd, q)
fn = f24.copy()
for g, gD, isD in ((~D, A, False), (D, DnP, True)):
    for c in (0, 1, 2):
        gg = g & (b == c)
        w, _ = nnls(X[fit & gg], y[fit & gg])
        base = X @ w
        k = fit & gg
        lam = float(np.dot(dl[k], (y - base)[k]) / np.dot(dl[k], dl[k]))
        sel = gD & (b == c)
        fn[sel] = np.clip(base[sel] + lam * dl[sel] + (0.5 * corr[sel] if isD else 0), 0, None)
        print(f'  {"D" if isD else "A"} tercile {c}: lambda {lam:.3f}')
res = {sc: block_bootstrap(y[k], fn[k], f24[k], block[k], n_boot=1000) for sc, k in (('held', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
gain = (y - f24) ** 2 - (y - fn) ** 2
keep = np.ones(len(y), bool); keep[np.argsort(gain)[::-1][:100]] = False
print('ARM U+T: ' + '  '.join(f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items())
      + f'  months better {sum(r(fn, month == m) < r(f24, month == m) for m in range(1, 13))}/12  w/o top100 held {r(fn, held & keep) - r(f24, held & keep):+.3f} scored {r(fn, S & keep) - r(f24, S & keep):+.3f}')
