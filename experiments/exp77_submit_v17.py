"""EXP77 -- v17 = v16 with the adsb.lol columns inside all seven members (A3).

WHAT CHANGES: routes A/B/C outside P and route D outside P. There v16 serves the 7-member blend of the W2 + W3
finals (cache/w2_final/, weights cache/w2/w3_weights.json; the route-D corrector is additive and stays). v17
serves the same seven families refitted WITH the adsb.lol columns (cache/adsb_final/, weights
cache/w2/adsb_weights.json from experiments/adsb/a3_eval.py), same production tail branch per LGB seed:
        delta = blend_adsb - blend_v16
Optionally (--with-p-adsb, only if a3_eval's P arm passed on its own) the real-taxi branch inside P becomes
0.5 * ADS-B mix + 0.5 * ADS-B nn bulk instead of the W2 bulk, under the v13/v14 TabICL p.
Everything else is v16 unchanged.

Pre-condition: experiments/adsb/a3_eval.py PASSED its rule (all flights, held-out and scored, held-out CI < 0,
January and July each better).

    python experiments/exp77_submit_v17.py --check          # new := old members and weights -> delta exactly 0
    python experiments/exp77_submit_v17.py --version 17 [--with-p-adsb]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.submit import build, TEAM

BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3')


def seeds012(folder, fam, ids):
    if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
        sys.exit(f'FATAL: {folder} is in a different ranking order')
    files = [folder / f'{fam}__seed{s:02d}.npy' for s in range(3)]
    miss = [f.name for f in files if not f.exists()]
    if miss:
        sys.exit(f'FATAL: missing finals in {folder.name}: {miss} -- rsync them back first')
    return np.mean([np.load(f).astype(float) for f in files], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--with-p-adsb', action='store_true')
    ap.add_argument('--arm', default='adsb', choices=['adsb', 'adsb2'],
                    help='adsb = arm 1 (ADS-B columns); adsb2 = arm 2 (+ wake / backlog / gap cap), judged against arm 1')
    a = ap.parse_args()
    if not a.check and a.version is None:
        sys.exit('pass --version N (explicit) or --check')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)
    from prc.io import load
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        sys.exit('FATAL: ranking order differs from the components')
    D = rk['AOBT_3_flt'].isna().to_numpy()
    OLDF, NEWF = CACHE / 'w2_final', (CACHE / 'w2_final' if a.check else CACHE / f'{a.arm}_final')
    Wold = json.load(open(CACHE / 'w2' / 'w3_weights.json'))
    Wnew = Wold if a.check else json.load(open(CACHE / 'w2' / f'{a.arm}_weights.json'))
    if not a.check:
        print(f'arm {a.arm}: finals {NEWF.name}, weights {a.arm}_weights.json (seeds {Wnew["seeds"]})')
    cache = {}

    def mem(folder, f):
        k = (folder, f)
        if k not in cache:
            cache[k] = seeds012(folder, f, ids)
        return cache[k]

    def blend(folder, W):
        out = []
        for s in range(pt.shape[0]):
            p = pt[s]; Wb = sum(W.get(f, 0.0) for f in BULK)
            pbm = sum(W[f] * mem(folder, f) for f in BULK if W.get(f, 0.0) > 0)
            o = Wb * p * et + (1 - p) * pbm
            if W.get('routed', 0.0) > 0:
                o = o + W['routed'] * mem(folder, 'routed')
            out.append(np.clip(o, 0, None))
        return np.mean(out, axis=0)

    delta = np.zeros(len(ids))
    for key, g in (('routes A/B/C', ~D & ~in_P), ('route D', D & ~in_P)):
        d = blend(NEWF, Wnew['weights'][key]) - blend(OLDF, Wold['weights'][key])
        delta[g] = d[g]
        if not a.check:
            print(f'{key:12s} {g.sum():7,} rows: weights ' + '  '.join(f'{f} {w:.3f}' for f, w in Wnew['weights'][key].items() if w)
                  + f' | mean shift {d[g].mean():+.2f} s, |shift| median {np.median(np.abs(d[g])):.0f}, max {d[g].max():+.0f} / {d[g].min():+.0f}')
    if a.with_p_adsb and not a.check:
        t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
        t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
        rows = np.r_[t1['pos'], t2['pos']]
        p14 = np.r_[t1['p_tabicl'].astype(float), t2['p_pool'].astype(float)]
        assert np.array_equal(np.sort(rows), np.flatnonzero(in_P)), 'TabICL p does not cover P exactly'
        pb_w2 = 0.5 * mem(OLDF, 'mix')[rows] + 0.5 * mem(OLDF, 'nn')[rows]
        pb_ad = 0.5 * mem(NEWF, 'mix')[rows] + 0.5 * mem(NEWF, 'nn')[rows]
        mix = lambda pbm: np.clip(p14 * et[rows] + (1 - p14) * pbm, 0, None)
        delta[rows] = mix(pb_ad) - mix(pb_w2)
        print(f'P {len(rows):,} rows: mean shift {delta[rows].mean():+.2f} s, max {delta[rows].max():+.0f} / {delta[rows].min():+.0f}')
    if a.check:
        print(f'CHECK (old members and weights as "new"): max |delta| {np.abs(delta).max():.2e} s -> '
              f'{"IDENTITY" if np.abs(delta).max() < 1e-6 else "NOT ZERO"}')
        return
    v16 = pd.read_parquet(OUTPUTS / f'{TEAM}_v16.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    path = build(ids, v16.astype(float) + delta, version=a.version, clip=(0.0, 200000.0))
    v17 = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    ch = v17 != v16
    print(f'v{a.version} vs v16: rows changed {int(ch.sum()):,} (inside P: {int((ch & in_P).sum()):,})')


if __name__ == '__main__':
    main()
