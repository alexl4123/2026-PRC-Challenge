"""v24-structure out-of-fold predictions on 2025 and their parts (for residual / corrector / zone-P experiments).

v24 = 7 adsb2 members + seq (POOL of 27 fits: ep30 + 8 grid variants x seeds 0-2) outside P, weights from the pool's
swap json (refitted here and checked), route-D corrector x0.5 on route D outside P, zone P = v11 non-branch part +
mix(TabICL p, 0.5 W2 mix bulk + 0.5 pooled seq bulk) - mix(p_cur, pb_old)  (as experiments/seq/seq_swap_eval.py).

    from v24_oof import build;  B = build()      # dict of arrays, also saved to cache/w2/v24s_oof.npz
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
SCREEN = [3, 6, 9, 12]


def _oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o


def build(save=True):
    out = CACHE / 'w2' / 'v24s_oof.npz'
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return {k: z[k] for k in z.files}
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    M = {f: _oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month) for f in FAMS}
    M['seq'] = np.mean([_oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2], month) for v in POOL], axis=0)
    seq_bulk = np.mean([_oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2], month, '.bulk.npy') for v in POOL], axis=0)
    fams8 = FAMS + ['seq']
    fit = np.isin(month, SCREEN) & ~P
    W = {}
    for key, g in (('routes A/B/C', ~D), ('route D', D)):
        w, _ = nnls(np.column_stack([M[f] for f in fams8])[fit & g], y[fit & g])
        W[key] = dict(zip(fams8, np.round(w, 4).tolist()))
    name = '__'.join(f'seqsw_{v}' for v in POOL)
    ref = json.load(open(CACHE / 'seq' / f'swap_{name}.json'))['weights_new']
    dw = max(abs(W[k][f] - ref[k].get(f, 0.0)) for k in W for f in fams8)
    if dw > 1e-3:
        sys.exit(f'FATAL: refitted weights differ from the shipped v24 weights by {dw}')
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    pbP = 0.5 * _oof(CACHE / 'cv' / 'w2', 'mix', [0, 1, 2], month, '.bulk.npy')[Pidx] + 0.5 * seq_bulk[Pidx]
    mixP = lambda p, pb: np.clip(p * so[Pidx] + (1 - p) * pb, 0, None)
    blend = {k: sum(W[k][m] * M[m] for m in fams8) for k in W}
    corr = v10 - v9
    f = v11.copy()
    A, DnP = ~D & ~P, D & ~P
    f[A] = np.clip(blend['routes A/B/C'][A], 0, None)
    f[DnP] = np.clip(blend['route D'][DnP] + 0.5 * corr[DnP], 0, None)
    P_rest = v11[Pidx] - mixP(p_cur, pb_old)                 # the non-branch part of the P rows
    f[Pidx] = P_rest + mixP(p14, pbP)
    res = dict(y=y, month=month, P=P, D=D, f=f, blendD=blend['route D'], blendA=blend['routes A/B/C'], corr=corr,
               Pidx=Pidx, rome=rome, p14=p14, pbP=pbP, P_rest=P_rest, so=so)
    if save:
        np.savez(out, **res)
    return res


if __name__ == '__main__':
    B = build()
    y, f, month = B['y'], B['f'], B['month']
    held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
    r = lambda k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
    print(f'v24 OOF: held-out {r(held):.3f}  scored {r(S):.3f}  12m {r(np.ones(len(y), bool)):.3f}')
