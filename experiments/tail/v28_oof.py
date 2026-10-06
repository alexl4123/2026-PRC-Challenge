"""v28-structure out-of-fold predictions on 2025 (base for the 2026-10-06 bets).

v28 = v27 (U+T: disagreement-tercile weights on routes A/B/C + lambda * (seq10 - seqnc10)), route D and zone P as v24,
then the LOBT window: predictions of rows with LOBT are projected into [MVT - LOBT - 3606, MVT - LOBT + 3606].
Built from cache/w2/v24s_oof.npz (v24 structure) and cache/seq/arm_u.json exactly as experiments/seq/arm_u_eval.py
builds the UT arm; saves cache/w2/v28s_oof.npz with the parts the bets need.

    python experiments/tail/v28_oof.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np, pandas as pd
from prc.config import CACHE, ID
from prc.io import load

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']


def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o


def main():
    V = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
    y, month, P, D, f24 = V['y'], V['month'], V['P'], V['D'], V['f']
    A = ~D & ~P
    J = json.load(open(CACHE / 'seq' / 'arm_u.json'))
    s1 = np.zeros(len(y)); s2 = np.zeros(len(y))
    for v in POOL:
        for s in (0, 1, 2):
            x = oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], month, '.bulk.npy'); s1 += x; s2 += x * x
    sd = np.sqrt(np.maximum(s2 / 27 - (s1 / 27) ** 2, 0)); del s1, s2
    b = np.digitize(sd, np.array(J['cuts']))
    M = {f: oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month) for f in FAMS}
    M['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2], month) for v in POOL], axis=0)
    dl = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2], month) - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2], month)
    f27 = f24.copy()
    for c in (0, 1, 2):
        W = J['UT']['weights'][str(c)]; sel = A & (b == c)
        base = sum(W[k] * M[k] for k in W)
        f27[sel] = np.clip(base[sel] + J['UT']['lambdas'][str(c)] * dl[sel], 0, None)
    d = load('training', phase='DEP', columns=[ID, 'MVT_TIME_UTC_mvt', 'LOBT_flt'])
    ids = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    if not np.array_equal(d[ID].to_numpy(), ids):
        sys.exit('FATAL: training row order differs from dep_training')
    ol = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds().to_numpy()
    m = np.isfinite(ol)
    lo, hi = np.maximum(0, ol - 3606), ol + 3606
    f28 = f27.copy(); f28[m] = np.minimum(np.maximum(f27[m], lo[m]), hi[m])
    np.savez(CACHE / 'w2' / 'v28s_oof.npz', y=y, month=month, P=P, D=D, f=f28, f27=f27, sd=sd, bin=b, off_lobt=ol, dl=dl)
    held, S = ~np.isin(month, [3, 6, 9, 12]), np.isin(month, [1, 7])
    r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    print(f'v24 OOF held {r(f24, held):.3f} scored {r(f24, S):.3f} | v27 held {r(f27, held):.3f} scored {r(f27, S):.3f} | '
          f'v28 held {r(f28, held):.3f} scored {r(f28, S):.3f} | window moved {int((f28 != f27).sum())} rows, worse '
          f'{int(((y - f28) ** 2 > (y - f27) ** 2).sum())}')


if __name__ == '__main__':
    main()
