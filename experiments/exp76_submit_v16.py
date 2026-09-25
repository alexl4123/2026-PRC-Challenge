"""EXP76 -- v16 = v15 with the second network architecture (W3) as a seventh blend member.

WHAT CHANGES: routes A/B/C outside P and route D outside P. There v15 serves the 3-seed W2 blend
(cache/w2/w2_weights.json, production tail branch per LGB seed; route D also carries the Amsterdam
corrector, which is additive and stays). v16 serves the 7-member blend of experiments/w2/w3_eval.py
(cache/w2/w3_weights.json):
        delta = blend7 - blend6      (same members, same tail branch; w3 = cache/w2_final/w3__seed0[0-2])
P rows and everything else are v15 unchanged.

Pre-condition: experiments/w2/w3_eval.py PASSED its rule.

    python experiments/exp76_submit_v16.py --check        # blend7 := blend6 -> delta exactly 0
    python experiments/exp76_submit_v16.py --version 16
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.submit import build, TEAM

FIN = CACHE / 'w2_final'
BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3')


def seeds012(fam, ids):
    if not np.array_equal(np.load(FIN / 'ranking_ids.npy', allow_pickle=True), ids):
        sys.exit('FATAL: cache/w2_final is in a different ranking order')
    files = [FIN / f'{fam}__seed{s:02d}.npy' for s in range(3)]
    miss = [f.name for f in files if not f.exists()]
    if miss:
        sys.exit(f'FATAL: missing finals {miss} -- rsync cache/w2_final/ back')
    return np.mean([np.load(f).astype(float) for f in files], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if not a.check and a.version is None:
        sys.exit('pass --version N (explicit) or --check')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pt, et, in_P = c['ids'], c['pt'], c['et'], c['in_P'].astype(bool)
    from prc.io import load
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        sys.exit('FATAL: ranking order differs from the components')
    D = rk['AOBT_3_flt'].isna().to_numpy()
    W6 = json.load(open(CACHE / 'w2' / 'w2_weights.json')); assert W6['seeds'] == [0, 1, 2]
    W7 = W6 if a.check else json.load(open(CACHE / 'w2' / 'w3_weights.json'))
    cache = {}

    def mem(f):
        if f not in cache:
            cache[f] = seeds012(f, ids)
        return cache[f]

    def blend(W):
        out = []
        for s in range(pt.shape[0]):
            p = pt[s]; Wb = sum(W.get(f, 0.0) for f in BULK)
            pbm = sum(W[f] * mem(f) for f in BULK if W.get(f, 0.0) > 0)
            o = Wb * p * et + (1 - p) * pbm
            if W.get('routed', 0.0) > 0:
                o = o + W['routed'] * mem('routed')
            out.append(np.clip(o, 0, None))
        return np.mean(out, axis=0)

    delta = np.zeros(len(ids))
    for key, g in (('routes A/B/C', ~D & ~in_P), ('route D', D & ~in_P)):
        d = blend(W7['weights'][key]) - blend(W6['weights'][key])
        delta[g] = d[g]
        if not a.check:
            print(f'{key:12s} {g.sum():7,} rows: weights ' + '  '.join(f'{f} {w:.3f}' for f, w in W7['weights'][key].items() if w)
                  + f' | mean shift {d[g].mean():+.2f} s, |shift| median {np.median(np.abs(d[g])):.0f}, max {d[g].max():+.0f} / {d[g].min():+.0f}')
    if a.check:
        print(f'CHECK (blend7 := blend6): max |delta| {np.abs(delta).max():.2e} s -> {"IDENTITY" if np.abs(delta).max() < 1e-6 else "NOT ZERO"}')
        return
    v15 = pd.read_parquet(OUTPUTS / f'{TEAM}_v15.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    path = build(ids, v15.astype(float) + delta, version=a.version, clip=(0.0, 200000.0))
    v16 = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    ch = v16 != v15
    print(f'v{a.version} vs v15: rows changed {int(ch.sum()):,} (none inside P: {bool(not np.any(ch & in_P))})')


if __name__ == '__main__':
    main()
