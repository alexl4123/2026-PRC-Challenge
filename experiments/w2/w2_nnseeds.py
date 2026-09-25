"""W2: does averaging the network over 10 seeds instead of 3 help, inside the shipped structure?

Everything is the v15 structure on 2025 OOF (W2 blend on A/B/C and on route D outside P with the
route-D corrector, W2 bulk as the real-taxi branch inside P, v13/v14 TabICL p), with the blend
weights FIXED at the 3-seed fit (cache/w2/w2_weights.json) -- only the network member's seed
average changes (seeds 0-2 -> 0-9), so the difference is pure variance reduction.
Reference: the strategy agent measured -0.29 s for 3 -> 10 network seeds in the old blend (s14).

PRE-REGISTERED RULE: ship 10 seeds if better on both 12 months and scored months, 12-month
paired-bootstrap interval below 0.

    python experiments/w2/w2_nnseeds.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.validation import paired_bootstrap
from w2_eval import oof, NEW, FAMS
from w2_routeD_P import bulk_oof


def main():
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    W = json.load(open(CACHE / 'w2' / 'w2_weights.json'))
    assert W['seeds'] == [0, 1, 2], 'weights must come from the 3-seed w2_eval run'
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    A, DnP = ~D & ~P, D & ~P

    others = {f: oof(NEW, f, [0, 1, 2], month) for f in FAMS if f != 'nn'}
    mix_bulk = bulk_oof('mix', [0, 1, 2], month)[Pidx]

    def structure(nn_seeds):
        mem = dict(others, nn=oof(NEW, 'nn', nn_seeds, month))
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            blend = sum(W['weights'][key][m] * mem[m] for m in FAMS)
            f[g] = np.clip(blend[g] + ((v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        pb_new = 0.5 * mix_bulk + 0.5 * bulk_oof('nn', nn_seeds, month)[Pidx]
        f[Pidx] = v11[Pidx] + (mixP(p14, pb_new) - mixP(p_cur, pb_old))
        return f

    f3, f10 = structure([0, 1, 2]), structure(list(range(10)))
    S = np.isin(month, [1, 7])
    r = lambda f, k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
    print(f'v15 structure, network 3 seeds: 12m {r(f3, slice(None)):.2f}  scored {r(f3, S):.2f}')
    out = []
    for scope, k in (('12m', np.ones(len(y), bool)), ('scored', S)):
        bb = paired_bootstrap(y[k], f10[k], f3[k])
        out.append(f'{scope} {bb["diff"]:+.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
    per = [r(f10, month == m) - r(f3, month == m) for m in range(1, 13)]
    print('network 10 seeds - 3 seeds: ' + '   '.join(out) + f'   months better {sum(v < 0 for v in per)}/12, worst {max(per):+.3f}')


if __name__ == '__main__':
    main()
