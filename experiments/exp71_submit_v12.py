"""EXP71 -- v12 = v11 with the multi-family blend on flights WITH an NM record.

WHAT CHANGES
------------
Only rows outside P that HAVE an NM off-block record (routes A/B/C). There, v11's
bulk structure (LightGBM + network at 0.35 / 0.65) is replaced by the exp66 family
blend fitted for those routes:

    network 0.369   per-airport 0.277   lintree 0.216   xgboost 0.138   LightGBM 0

Everything else -- P rows, and route-D rows with the Amsterdam corrector -- is
v11 unchanged, and the feature-stack correction stays on every outside-P row.

    v12 = v11 + delta,   delta = blend_ABC - v9_structure_ABC   on routes A/B/C outside P

Both terms are computed with identical code from the same production LightGBM
seeds (the shared tail classifier and anchor), so with v9's weights delta is
exactly 0 -- that is the regression check.

EVIDENCE (2025 OOF, vs the live v11 structure, outside P): -1.38 [-1.67,-1.13]
over 12 months, -1.75 [-2.58,-1.17] on the scored months, 11/12 months. Chosen over
blending route D as well because the route-D corrector's value is a 2026 event that
2025 CV cannot see (measured -0.6, delivered -5.0 live).

    python experiments/exp71_submit_v12.py --check-v11     # must reproduce v11 exactly
    python experiments/exp71_submit_v12.py --version 12
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import ID, CACHE, OUTPUTS
from prc.submit import TEAM

COMP = CACHE / 'final' / 'lgb_components.npz'
NET, FAM = CACHE / 'nn_final', CACHE / 'family_final'
WEIGHTS = CACHE / 'exp66_weights_perroute_all.json'
V9W = 0.65                                          # v9's network weight on routes A/B/C


def seeds_mean(folder, pattern, ids):
    files = sorted(folder.glob(pattern))
    if files and not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
        sys.exit(f'FATAL: {folder} is in a different ranking order')
    return (np.mean([np.load(f).astype(float) for f in files], axis=0) if files else None), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check-v11', action='store_true')
    a = ap.parse_args()
    if not a.check_v11 and a.version is None:
        sys.exit('pass --version N (explicit) or --check-v11')

    c = np.load(COMP, allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)
    from prc.io import load
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        sys.exit('FATAL: ranking order differs from the components')
    ABC = ~rk['AOBT_3_flt'].isna().to_numpy() & ~in_P      # rows that change
    net, k = seeds_mean(NET, 'nn_final_seed*.npy', ids)
    if k != 3:
        sys.exit(f'FATAL: expected 3 network seeds, found {k}')

    # v11 as shipped, rebuilt at full precision exactly as exp70 did
    z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
    st = np.load(CACHE / 'final' / 'stack_c_ranking.npz', allow_pickle=True)
    assert np.array_equal(z['ids'], ids) and np.array_equal(st['ids'], ids)
    v11f = np.clip(z['trailD_full'], 0, None) + np.where(st['in_P'].astype(bool), 0.0, st['corr'])
    v11 = pd.read_parquet(OUTPUTS / f'{TEAM}_v11.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt']
    assert (np.rint(np.clip(v11f, 0, 2e5)).astype(int) == v11.reindex(ids).to_numpy()).all(), \
        'rebuilt v11 != shipped v11'

    if a.check_v11:
        w = {'mix': 1 - V9W, 'nn': V9W}
        members = {'mix': None, 'nn': net}
    else:
        w = json.load(open(WEIGHTS))['weights']['routes A/B/C']
        members = {'mix': None, 'nn': net}
        for f in ('perapt', 'lintree', 'xgboost'):
            if w.get(f, 0) > 0:
                arr, k = seeds_mean(FAM, f'{f}__seed*.npy', ids)
                if k != 3:
                    sys.exit(f'FATAL: {f} has {k} of 3 seeds in {FAM} -- rsync cache/family_final/ back')
                members[f] = arr
        print('weights (routes A/B/C): ' + '  '.join(f'{f} {w.get(f, 0):.3f}' for f in members))

    blend, v9s = [], []
    for s in range(pb.shape[0]):
        p = pt[s]
        mem = {f: (pb[s] if f == 'mix' else v) for f, v in members.items()}
        W = sum(w.get(f, 0.0) for f in mem)
        pbm = sum(w.get(f, 0.0) * mem[f] for f in mem)
        blend.append(np.clip(W * p * et + (1 - p) * pbm, 0, None))
        pbm9 = (1 - V9W) * pb[s] + V9W * net                        # exactly exp65's v9 form
        v9s.append(np.clip(p * et + (1 - p) * pbm9, 0, None))
    delta = np.where(ABC, np.mean(blend, axis=0) - np.mean(v9s, axis=0), 0.0)
    pred = v11f + delta

    ints = np.rint(np.clip(pred, 0, 2e5)).astype(int)
    diff = ints != v11.reindex(ids).to_numpy()
    if a.check_v11:
        print(f"CHECK with v9's weights: {int(diff.sum()):,} of {len(ids):,} rows differ from v11"
              + (' -- IDENTICAL' if not diff.any() else ''))
        return
    d = ints - v11.reindex(ids).to_numpy()
    print(f'v12 vs v11: rows changed {int(diff.sum()):,} (all on routes A/B/C outside P: '
          f'{bool(np.all(ABC[diff]))}) | mean shift {d.mean():+.2f}s | |shift| median '
          f'{np.median(np.abs(d[diff])) if diff.any() else 0:.0f}s, p99 {np.percentile(np.abs(d), 99):.0f}s, '
          f'max +{d.max()} / {d.min()}')
    from prc.submit import build
    print(f'built {build(ids, pred, version=a.version, clip=(0.0, 200000.0))}')


if __name__ == '__main__':
    main()
