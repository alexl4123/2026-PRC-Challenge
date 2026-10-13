"""Window-conditioned mixture, 2025 OOF. Exact consequence of the LOBT window: a row with LOBT has y <= off_lobt + 3606,
so where off_lobt + 3606 <= 7200 (= TAIL_CAP) the tail class (y > 7200) is IMPOSSIBLE and the mixture's tail term
p * (et - bulk) is pure hedge.  Variant: p := 0 on those rows (members' bulk predictions instead of their mixtures,
UT term with (1 - p) = 1), then the LOBT window as in v28.  Rebuilds f27 from the fold files exactly as v28_oof.py."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.config import CACHE
H = CACHE / 'hunt'
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']


def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o


V = np.load(CACHE / 'w2' / 'v28s_oof.npz')
y, month, P, D, f28, f27, b, ol = V['y'], V['month'], V['P'].astype(bool), V['D'].astype(bool), V['f'], V['f27'], V['bin'], V['off_lobt']
A = ~D & ~P
J = json.load(open(CACHE / 'seq' / 'arm_u.json'))
M = {f: oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month) for f in FAMS}
Mb = {f: (oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month, '.bulk.npy') if f != 'routed' else M[f]) for f in FAMS}
M['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2], month) for v in POOL], axis=0)
Mb['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2], month, '.bulk.npy') for v in POOL], axis=0)
dl = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2], month) - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2], month)
dl0 = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2], month, '.bulk.npy') - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2], month, '.bulk.npy')
chk = f27.copy(); g0 = f27.copy()
for c in (0, 1, 2):
    W = J['UT']['weights'][str(c)]; sel = A & (b == c)
    chk[sel] = np.clip(sum(W[k] * M[k] for k in W)[sel] + J['UT']['lambdas'][str(c)] * dl[sel], 0, None)
    g0[sel] = np.clip(sum(W[k] * Mb[k] for k in W)[sel] + J['UT']['lambdas'][str(c)] * dl0[sel], 0, None)
print('rebuild check max|f27 - rebuilt| on A:', np.abs(chk - f27)[A].max())
m = np.isfinite(ol)
lo, hi = np.maximum(0, ol - 3606), ol + 3606
win = lambda f: np.where(m, np.minimum(np.maximum(f, lo), hi), f)
q0 = win(g0)
tail_term = f27 - g0
np.savez(H / 'stand_s11_oof.npz', g0=g0, q0=q0, tail_term=tail_term)
held, S = ~np.isin(month, [3, 6, 9, 12]), np.isin(month, [1, 7])
r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
print('tail impossible rows (A, hi<=7200):', int((A & m & (hi <= 7200)).sum()), '| tails among them (y>7200):', int((A & m & (hi <= 7200) & (y > 7200)).sum()))
print('tail term on A quantiles', np.percentile(tail_term[A], [0, 50, 90, 99, 99.9, 100]).round(1))
for thr_nm, dom in (('hi<=7200 (exact)', A & m & (hi <= 7200)), ('all A (not exact)', A & m)):
    q = np.where(dom, q0, f28)
    ch = q != f28; worse = ch & (np.abs(q - y) > np.abs(f28 - y))
    dsse = (q - y) ** 2 - (f28 - y) ** 2
    pm = pd.Series(dsse).groupby(month).sum()
    print(f'[p:=0 on {thr_nm}] domain {dom.sum():,} changed {ch.sum():,} worse {worse.sum():,} | held {r(q, held) - r(f28, held):+.4f} '
          f'JanJul {r(q, S) - r(f28, S):+.4f} all {r(q, np.ones_like(held)) - r(f28, np.ones_like(held)):+.4f} | months better {(pm < 0).sum()}/12 '
          f'| top-10 rows share of gain {np.sort(dsse)[:10].sum() / dsse.sum():.2f}')
    # by size of the tail term
    for lo_t, hi_t in ((0, 10), (10, 50), (50, 200), (200, 1e9)):
        s = dom & (np.abs(tail_term) >= lo_t) & (np.abs(tail_term) < hi_t)
        print(f'    |tail term| in [{lo_t},{hi_t}): rows {s.sum():,} dSSE {dsse[s].sum():+.3e} mean err before {np.mean((f28 - y)[s]) if s.any() else 0:+.1f} after {np.mean((q - y)[s]) if s.any() else 0:+.1f}')
    print('    per month dSSE', pm.round(-3).to_dict())
