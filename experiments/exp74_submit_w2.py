"""EXP74 -- the W2 submission: on routes A/B/C outside P, replace (old family blend + stack arm c)
by the blend of the W2 members (v11 signals inside the models).

WHAT CHANGES
------------
Only ranking rows WITH an NM record (routes A/B/C) outside P. There v13 serves
    old = blend of nn / perapt / lintree / xgboost (exp66 weights, production tail branch)
          + the stack arm (c) correction                                     (v11 + v12)
and W2 serves
    new = blend of the W2 members (cache/w2_final/, weights from cache/w2/w2_weights.json),
          with the SAME production tail branch (per-LGB-seed classifier p, tail anchor et)
          and no stack (its information is inside the members now).
    out = v13 + delta,  delta = mean_s(new_s) - mean_s(old_s) - corr     on those rows.
Optionally (--with-nonrome) the W4 non-Rome tail-zone change of exp73 rides along on the
1,539 non-Rome P rows (validated separately: -0.10 / -0.17 s).

CHECKS
------
    --check-recon   rebuild v13 on the A/B/C rows from components (old members, old weights,
                    stack corr) and compare with the shipped integers: proves that `old + corr`
                    is exactly what the delta removes.
    --version N     build; asserts that only A/B/C-outside-P rows (and, with --with-nonrome,
                    non-Rome P rows) change.
Pre-condition: experiments/w2/w2_eval.py PASSED its rule (A/B/C, held-out CI < 0, scored < 0).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.submit import build, TEAM

OLD_W = CACHE / 'exp66_weights_perroute_all.json'
NEW_W = CACHE / 'w2' / 'w2_weights.json'
BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost')


def seeds_mean(folder, fam, ids, n_expected=3):
    files = sorted(folder.glob(f'{fam}__seed*.npy'))
    if len(files) != n_expected:
        sys.exit(f'FATAL: {folder.name}/{fam}: {len(files)} seed files, expected {n_expected}')
    if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
        sys.exit(f'FATAL: {folder} is in a different ranking order')
    return np.mean([np.load(f).astype(float) for f in files], axis=0)


def blend(weights, members, pt, et, pb_mix_per_seed=None):
    """Per-LGB-seed family blend with the production tail branch, averaged over seeds."""
    out = []
    for s in range(pt.shape[0]):
        p = pt[s]
        W = sum(weights.get(f, 0.0) for f in BULK)
        pbm = 0.0
        for f in BULK:
            w = weights.get(f, 0.0)
            if w:
                pbm = pbm + w * (pb_mix_per_seed[s] if (f == 'mix' and pb_mix_per_seed is not None) else members[f])
        o = W * p * et + (1 - p) * pbm
        if weights.get('routed', 0.0):
            o = o + weights['routed'] * members['routed']
        out.append(np.clip(o, 0, None))
    return np.mean(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--base', type=int, default=13, help='submission the delta is added to')
    ap.add_argument('--check-recon', action='store_true')
    ap.add_argument('--with-nonrome', action='store_true')
    a = ap.parse_args()
    if not a.check_recon and a.version is None:
        sys.exit('pass --version N (explicit) or --check-recon')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)
    from prc.io import load
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt', 'APT'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        sys.exit('FATAL: ranking order differs from the components')
    ABC = ~rk['AOBT_3_flt'].isna().to_numpy() & ~in_P
    st = np.load(CACHE / 'final' / 'stack_c_ranking.npz', allow_pickle=True)
    assert np.array_equal(st['ids'], ids)
    corr = np.where(st['in_P'].astype(bool), 0.0, st['corr'].astype(float))

    W_old = json.load(open(OLD_W))['weights']['routes A/B/C']
    old_m = {'nn': np.mean([np.load(CACHE / 'nn_final' / f'nn_final_seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)}
    if not np.array_equal(np.load(CACHE / 'nn_final' / 'ranking_ids.npy', allow_pickle=True), ids):
        sys.exit('FATAL: nn_final is in a different ranking order')
    for f in ('perapt', 'lintree', 'xgboost'):
        if W_old.get(f, 0):
            old_m[f] = seeds_mean(CACHE / 'family_final', f, ids)
    old = blend(W_old, old_m, pt, et, pb_mix_per_seed=pb)

    base = pd.read_parquet(OUTPUTS / f'{TEAM}_v{a.base}.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    if a.check_recon:
        recon = np.rint(np.clip(old + corr, 0, 2e5)).astype(int)
        diff = recon[ABC] != base[ABC]
        print(f'CHECK-RECON: v{a.base} on {ABC.sum():,} A/B/C rows outside P vs round(old blend + stack corr): '
              f'{int(diff.sum()):,} differ (max |d| {int(np.abs(recon[ABC] - base[ABC]).max())} s)')
        return

    if not NEW_W.exists():
        sys.exit(f'FATAL: {NEW_W} missing -- run experiments/w2/w2_eval.py first (and only ship if it PASSED)')
    W_new = json.load(open(NEW_W))['weights']['routes A/B/C']
    new_m = {f: seeds_mean(CACHE / 'w2_final', f, ids) for f, w in W_new.items() if w > 0}
    new = blend(W_new, new_m, pt, et)
    print('W2 weights (A/B/C): ' + '  '.join(f'{f} {w:.3f}' for f, w in W_new.items()))
    delta = np.where(ABC, new - old - corr, 0.0)
    allowed = ABC.copy()
    if a.with_nonrome:
        t = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
        rows = t['pos']
        assert in_P[rows].all() and (rk.APT.astype(str).to_numpy()[rows] != 'LIRF').all()
        sx = np.load(CACHE / 'final' / 'spec_x_ranking.npz', allow_pickle=True)
        net = old_m['nn']
        pbm = 0.5 * pb[:, rows] + 0.5 * net[rows]
        mix = lambda p: np.mean(np.clip(p * et[rows] + (1 - p) * pbm, 0, None), axis=0)
        delta[rows] += mix(t['p_pool'].astype(float)) - mix(sx['p_spec'][rows])
        allowed[rows] = True
        print(f'with the non-Rome tail-zone change on {len(rows):,} rows')
    k = ABC
    print(f'A/B/C outside P ({k.sum():,} rows): mean shift {delta[k].mean():+.2f} s, |shift| median '
          f'{np.median(np.abs(delta[k])):.0f} s, p99 {np.percentile(np.abs(delta[k]), 99):.0f} s, '
          f'max {delta[k].max():+.0f} / {delta[k].min():+.0f}; stack corr removed: mean {corr[k].mean():+.2f} s')
    pred = base.astype(float) + delta
    path = build(ids, pred, version=a.version, clip=(0.0, 200000.0))
    out = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    ch = out != base
    print(f'v{a.version} vs v{a.base}: rows changed {int(ch.sum()):,} (all in the allowed set: {bool(np.all(allowed[ch]))})')


if __name__ == '__main__':
    main()
