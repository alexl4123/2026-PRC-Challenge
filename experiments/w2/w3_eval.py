"""W3 evaluation: does the second network architecture (prc/models/w3_net.py) add to the W2 blend?

Members: the six W2 families (3 seeds each, cache/cv/w2/) and w3 (cache/cv/w2/w3__seedNN__mMM.npy).
  blend6 = the shipped 3-seed W2 weights (cache/w2/w2_weights.json)
  blend7 = per-route NNLS with w3 added, fitted on the screening months (3, 6, 9, 12) outside P
           exactly as the W2 weights were
Everything else is the v15 structure on 2025 OOF (route-D corrector kept on route D outside P, W2
bulk inside P with the v13/v14 TabICL p), so the comparison is on ALL flights.

PRE-REGISTERED RULE (written 2026-09-25 before any W3 fold existed): ship blend7 if it beats blend6
on all flights on BOTH the 8 held-out months and the scored months, with the held-out paired-
bootstrap interval below 0. Also reported: w3 alone vs the W2 network, residual correlations.

    python experiments/w2/w3_eval.py              # seeds complete for w3 (all 3 expected)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import paired_bootstrap
from w2_eval import oof, complete, NEW, FAMS, SCREEN
from w2_routeD_P import bulk_oof


def main():
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    seeds = [s for s in range(3) if complete(NEW, 'w3', s)]
    if not seeds:
        sys.exit('FATAL: no complete w3 seed in cache/cv/w2 (need w3__seedNN__m01..m12.npy)')
    print(f'w3 seeds complete: {seeds}; W2 members: seeds 0-2')
    W = json.load(open(CACHE / 'w2' / 'w2_weights.json')); assert W['seeds'] == [0, 1, 2]
    mem = {f: oof(NEW, f, [0, 1, 2], month) for f in FAMS}
    mem['w3'] = oof(NEW, 'w3', seeds, month)
    F7 = FAMS + ['w3']
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    pb_new = 0.5 * bulk_oof('mix', [0, 1, 2], month)[Pidx] + 0.5 * bulk_oof('nn', [0, 1, 2], month)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    A, DnP = ~D & ~P, D & ~P
    fit = np.isin(month, SCREEN) & ~P

    def structure(weights):
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            blend = sum(weights[key].get(m, 0.0) * mem[m] for m in F7)
            f[g] = np.clip(blend[g] + ((v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        f[Pidx] = v11[Pidx] + (mixP(p14, pb_new) - mixP(p_cur, pb_old))
        return f

    w7 = {}
    for key, g in (('routes A/B/C', ~D), ('route D', D)):
        X = np.column_stack([mem[m] for m in F7])
        w, _ = nnls(X[fit & g], y[fit & g])
        w7[key] = dict(zip(F7, np.round(w, 4).tolist()))
        print(f'  blend7 weights {key:12s}: ' + '  '.join(f'{m} {v:.3f}' for m, v in w7[key].items()))
    f6, f7 = structure(W['weights']), structure(w7)

    held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
    r = lambda f, k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
    k = A & held
    res = lambda p: (y - p)[k]
    print(f'\nw3 alone vs W2 network (A/B/C outside P, held-out months): RMSE {r(mem["w3"], k):.2f} vs {r(mem["nn"], k):.2f}')
    print('residual correlation of w3 with: ' + '  '.join(f'{m} {np.corrcoef(res(mem["w3"]), res(mem[m]))[0, 1]:.3f}' for m in FAMS))
    print(f'\nv15 structure, all flights: blend6 held-out {r(f6, held):.2f} scored {r(f6, S):.2f}')
    out = []
    for scope, kk in (('8 held-out', held), ('scored', S)):
        bb = paired_bootstrap(y[kk], f7[kk], f6[kk])
        out.append(f'{scope} {bb["diff"]:+.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
    per = [r(f7, month == m) - r(f6, month == m) for m in range(1, 13)]
    print('blend7 - blend6, all flights: ' + '   '.join(out) + f'   months better {sum(v < 0 for v in per)}/12, worst {max(per):+.3f}')
    json.dump({'members': F7, 'w3_seeds': seeds, 'weights': w7}, open(CACHE / 'w2' / 'w3_weights.json', 'w'), indent=1)
    print('saved cache/w2/w3_weights.json')


if __name__ == '__main__':
    main()
