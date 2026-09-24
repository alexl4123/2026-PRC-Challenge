"""EXP75 -- v15 = v14 with the W2 members on route D outside P and as the real-taxi branch inside P.

WHAT CHANGES (the Amsterdam route-D corrector is kept everywhere; it is additive in v10..v14)
  route D outside P (3,570 ranking rows): v14 serves v9's LGB + network mixture (0.5 / 0.5) + the
      corrector + the stack arm (c). v15 serves the W2 route-D blend (weights from
      cache/w2/w2_weights.json, production tail branch per LGB seed) + the corrector:
          delta = blend_W2_D - v9_D - stack_corr
  P (1,720 rows): v14 serves p * et + (1 - p) * (0.5 pb_LGB + 0.5 network) + the corrector, with
      v13/v14's TabICL p. v15 swaps the real-taxi branch for 0.5 W2-LightGBM + 0.5 W2-network bulk:
          delta = mix(p, pbm_W2) - mix(p, pbm_old)     (per LGB seed, p unchanged)
Every other row is v14 unchanged. W2 finals: seeds 0-2 only (the CV weights were fitted on those).

EVIDENCE (experiments/w2/w2_routeD_P.py --seeds 0, all flights, baseline = v14 structure on 2025 OOF):
    D1 route D + corrector   12m -0.67 [-1.11,-0.26]  scored -0.52 [-1.23,+0.14]  10/12
    P1 W2 bulk inside P      12m -0.46 [-0.65,-0.30]  scored -0.36 [-0.75,+0.02]  10/12

    python experiments/exp75_submit_v15.py --check      # identity: old members in place of W2 -> zero delta
    python experiments/exp75_submit_v15.py --version 15
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.submit import build, TEAM

FIN = CACHE / 'w2_final'
BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost')


def seeds012(fam, ids):
    if not np.array_equal(np.load(FIN / 'ranking_ids.npy', allow_pickle=True), ids):
        sys.exit('FATAL: cache/w2_final is in a different ranking order')
    files = [FIN / f'{fam}__seed{s:02d}.npy' for s in range(3)]
    miss = [f.name for f in files if not f.exists()]
    if miss:
        sys.exit(f'FATAL: missing W2 finals {miss}')
    return np.mean([np.load(f).astype(float) for f in files], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if not a.check and a.version is None:
        sys.exit('pass --version N (explicit) or --check')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)
    from prc.io import load
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt', 'APT'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        sys.exit('FATAL: ranking order differs from the components')
    D = rk['AOBT_3_flt'].isna().to_numpy()
    DnP = D & ~in_P
    st = np.load(CACHE / 'final' / 'stack_c_ranking.npz', allow_pickle=True)
    assert np.array_equal(st['ids'], ids)
    corr = np.where(st['in_P'].astype(bool), 0.0, st['corr'].astype(float))
    net = np.mean([np.load(CACHE / 'nn_final' / f'nn_final_seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)
    S = range(pb.shape[0])

    # --- route D outside P
    v9D = np.mean([np.clip(pt[s] * et + (1 - pt[s]) * (0.5 * pb[s] + 0.5 * net), 0, None) for s in S], axis=0)
    if a.check:
        newD = v9D + corr                    # identity: what v14 serves minus the corrector
    else:
        W = json.load(open(CACHE / 'w2' / 'w2_weights.json'))['weights']['route D']
        mem = {f: seeds012(f, ids) for f, w in W.items() if w > 0}
        out = []
        for s in S:
            p = pt[s]; Wb = sum(W.get(f, 0.0) for f in BULK)
            pbm = sum(W[f] * mem[f] for f in BULK if W.get(f, 0.0) > 0)
            o = Wb * p * et + (1 - p) * pbm
            if W.get('routed', 0.0) > 0:
                o = o + W['routed'] * mem['routed']
            out.append(np.clip(o, 0, None))
        newD = np.mean(out, axis=0)
        print('W2 route-D weights: ' + '  '.join(f'{f} {w:.3f}' for f, w in W.items() if w > 0))
    delta = np.zeros(len(ids))
    delta[DnP] = (newD - v9D - corr)[DnP]

    # --- inside P: v13/v14 probabilities, real-taxi branch swapped
    t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
    rows = np.r_[t1['pos'], t2['pos']]
    p14 = np.r_[t1['p_tabicl'].astype(float), t2['p_pool'].astype(float)]
    if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), np.flatnonzero(in_P)):
        sys.exit('FATAL: the TabICL probabilities do not cover the P rows exactly once')
    pbm_old = [0.5 * pb[s][rows] + 0.5 * net[rows] for s in S]
    pbm_new = pbm_old if a.check else [0.5 * seeds012('mix', ids)[rows] + 0.5 * seeds012('nn', ids)[rows]] * len(S)
    mix = lambda pbm: np.mean([np.clip(p14 * et[rows] + (1 - p14) * pbm[i], 0, None) for i, s in enumerate(S)], axis=0)
    delta[rows] = mix(pbm_new) - mix(pbm_old)

    v14 = pd.read_parquet(OUTPUTS / f'{TEAM}_v14.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    if a.check:
        print(f'CHECK (old members in place of W2): max |delta| {np.abs(delta).max():.2e} s -> '
              f'{"IDENTITY" if np.abs(delta).max() < 1e-6 else "NOT ZERO"}')
        return
    for nm, k in (('route D outside P', DnP), ('P', in_P)):
        print(f'{nm:18s} {k.sum():5,} rows: mean shift {delta[k].mean():+7.1f} s, |shift| median '
              f'{np.median(np.abs(delta[k])):5.0f}, max {delta[k].max():+.0f} / {delta[k].min():+.0f}')
    pred = v14.astype(float) + delta
    path = build(ids, pred, version=a.version, clip=(0.0, 200000.0))
    v15 = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    ch = v15 != v14
    print(f'v{a.version} vs v14: rows changed {int(ch.sum()):,} (all on route D: {bool(np.all(D[ch]))})')


if __name__ == '__main__':
    main()
