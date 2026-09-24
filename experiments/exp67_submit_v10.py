"""EXP67 -- v10 = v9 with the multi-family blend (exp66) OUTSIDE P.

WHAT CHANGES
------------
Outside P, each row's prediction becomes the exp66 blend of model families, with
separate non-negative weights for flights with an NM record (routes A/B/C) and
without (route D):

    pred = sum_f w_f * (p * et + (1 - p) * pb_f)   +   w_routed * routed_pred

where p is the production global tail classifier (per LightGBM seed), et the
tail anchor, pb_f each family's bulk prediction, and routed_pred the routed
model's own full prediction. This is exactly the object exp66 validated: a
weighted sum of full-mixture family predictions sharing one tail classifier.

INSIDE P nothing changes: the rows keep v9's prediction (specialist X; bulk =
0.5 LightGBM + 0.5 network). exp66 never evaluated P rows, so they are left alone,
and the official delta measures the family blend alone.

THE EVIDENCE (exp66, weights fitted on months 3/6/9/12 outside P, 3-seed families)
    vs production (LightGBM + network, v9 weights), rows outside P:
      8 held-out months   -1.556 +- 0.311   CI [-2.18, -1.00]
      scored months 1+7   -2.165 +- 0.470   CI [-3.12, -1.37]
    weights, routes A/B/C: nn 0.369  perapt 0.277  lintree 0.216  xgboost 0.138
    weights, route D:      routed 0.636  perapt 0.253  xgboost 0.115  nn 0.005
    (the global LightGBM gets 0 in both groups)

INPUTS
    cache/final/lgb_components.npz      production LightGBM seeds (pb, pt, et, in_P)
    cache/final/spec_x_ranking.npz      specialist X inside P
    cache/nn_final/                     3 network seeds
    cache/family_final/                 perapt / lintree / xgboost / routed, 3 seeds each
    cache/exp66_weights_perroute_all.json

REGRESSION CHECK -- run before building:
    python experiments/exp67_submit_v10.py --check-v9
builds the same blend with weights {LightGBM 0.35, network 0.65} on routes A/B/C and
{0.5, 0.5} on route D -- v9's -- and must reproduce outputs/..._v9.parquet exactly.
It needs no family files.

    python experiments/exp67_submit_v10.py --version N      (explicit; v10 = route-D corrector, v11 = feature stack)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import ID, CACHE, OUTPUTS
from prc.submit import TEAM
from prc.models.tail_specialist import apply_to_mixture

COMP = CACHE / 'final' / 'lgb_components.npz'
SPEC_X = CACHE / 'final' / 'spec_x_ranking.npz'
NET = CACHE / 'nn_final'
FAM = CACHE / 'family_final'
WEIGHTS = CACHE / 'exp66_weights_perroute_all.json'


def to_int32(pred):
    return np.rint(np.clip(np.asarray(pred, float), 0.0, 200000.0)).astype(np.int32)


def load_seeds(folder, prefix, ids):
    files = sorted(folder.glob(f'{prefix}__seed*.npy')) if prefix else sorted(folder.glob('nn_final_seed*.npy'))
    if not files:
        return None, 0
    if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
        sys.exit(f'FATAL: {folder} is in a different ranking order')
    return np.mean([np.load(f).astype(float) for f in files], axis=0), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, required=True)   # always explicit: v10 = route-D corrector, v11 = feature stack
    ap.add_argument('--check-v9', action='store_true',
                    help="use v9's weights and compare with the v9 parquet instead of building")
    a = ap.parse_args()

    c = np.load(COMP, allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)
    s = np.load(SPEC_X, allow_pickle=True)
    if not (np.array_equal(s['ids'], ids) and np.array_equal(s['in_P'].astype(bool), in_P)):
        sys.exit('FATAL: specialist file not aligned with the components')
    p_spec = s['p_spec']

    from prc.io import load
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        sys.exit('FATAL: ranking order differs from the cached components')
    D = rk['AOBT_3_flt'].isna().to_numpy()

    net, k = load_seeds(NET, None, ids)
    if k != 3:
        sys.exit(f'FATAL: expected 3 network seeds in {NET}, found {k}')

    if a.check_v9:
        # v9's weights expressed as a family blend; w_mix is computed exactly as exp65 does
        W = {'routes A/B/C': {'mix': 1 - 0.65, 'nn': 0.65}, 'route D': {'mix': 1 - 0.5, 'nn': 0.5}}
        fam = {}
    else:
        W = json.load(open(WEIGHTS))['weights']
        fam = {}
        for f in ('perapt', 'lintree', 'xgboost', 'routed'):
            if any(W[g].get(f, 0) > 0 for g in W):
                arr, k = load_seeds(FAM, f, ids)
                if k != 3:
                    sys.exit(f'FATAL: {f} has {k} of 3 seeds in {FAM} -- run cluster/family_final.sbatch '
                             'and rsync cache/family_final/ back')
                fam[f] = arr
        print('families loaded: ' + ', '.join(f'{f} (mean {v.mean():.1f}s)' for f, v in fam.items()))

    preds = []
    for seed in range(pb.shape[0]):
        out = np.empty(len(ids))
        # ---- inside P: v9 exactly -------------------------------------------
        p9 = apply_to_mixture(pt[seed], p_spec, in_P)
        pbm9 = (1 - np.where(D, 0.5, 0.65)) * pb[seed] + np.where(D, 0.5, 0.65) * net
        v9row = np.clip(p9 * et + (1 - p9) * pbm9, 0, None)
        out[in_P] = v9row[in_P]
        # ---- outside P: the family blend ------------------------------------
        for gname, g in (('route D', D & ~in_P), ('routes A/B/C', ~D & ~in_P)):
            w = W[gname]
            member_pb = {'mix': pb[seed], 'nn': net, **{f: v for f, v in fam.items() if f != 'routed'}}
            Wmix = sum(w.get(f, 0.0) for f in member_pb)
            pbm = sum(w.get(f, 0.0) * member_pb[f] for f in member_pb)
            pr = pt[seed]
            val = Wmix * pr * et + (1 - pr) * pbm
            if w.get('routed', 0.0) > 0:
                val = val + w['routed'] * fam['routed']
            out[g] = np.clip(val, 0, None)[g]
        preds.append(out)
    pred = np.mean(preds, axis=0)

    if a.check_v9:
        v9 = pd.read_parquet(OUTPUTS / f'{TEAM}_v9.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt']
        mine = pd.Series(to_int32(pred), index=pd.Index(ids, name='MVT_ID_mvt'))
        diff = mine.to_numpy() != v9.reindex(mine.index).to_numpy()
        print(f"CHECK vs v9 with v9's weights: {int(diff.sum()):,} of {len(mine):,} rows differ"
              + (f' (max |diff| {int(np.abs(mine.to_numpy() - v9.reindex(mine.index).to_numpy()).max())} s)'
                 if diff.any() else ' -- IDENTICAL'))
        return

    from prc.submit import build as build_submission
    v9 = pd.read_parquet(OUTPUTS / f'{TEAM}_v9.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt']
    d = pred - v9.reindex(pd.Index(ids)).to_numpy()
    print(f'v10 vs v9: rows moved >1s {int((np.abs(d) > 1).sum()):,} | inside P identical: '
          f'{bool(np.all(np.abs(d[in_P]) <= 0.5))} | mean shift {d.mean():+.2f}s | |shift| median '
          f'{np.median(np.abs(d)):.1f}s p99 {np.percentile(np.abs(d), 99):.0f}s max {np.abs(d).max():.0f}s')
    path = build_submission(ids, pred, version=a.version, clip=(0.0, 200000.0))
    print(f'built {path}')


if __name__ == '__main__':
    main()
