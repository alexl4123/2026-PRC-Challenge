"""EXP73 -- v14 = v13 with TabICL (pooled context) deciding the tail zone at the other nine airports.

WHAT CHANGES
------------
Only the 1,539 ranking flights in P (no NM record, > 2 h behind schedule) OUTSIDE Rome. There, the
probability that the block time was stamped at the schedule comes from TabICL with all 4,944 P rows
of 2025 as context (airport one-hot added) instead of the v7 specialist. Rome's 181 flights keep
v13's TabICL; everything else is v13 unchanged.

    v14 = v13 + delta,  delta = mixture(p_new) - mixture(p_old)  on those rows,
computed per production LightGBM seed from the same components as exp72, so with p_new = p_old
delta is exactly 0 (the regression check, --check-v13).

EVIDENCE (W4, experiments/tfm/w4_tail_zone.py, 2025 leave-one-month-out, rule pre-registered):
    pool on non-Rome P vs v7:  12m -0.10 [-0.17,-0.04]   scored -0.17 [-0.39,+0.02]   8/12 months
Ceiling for these rows (perfect classifier): -0.93 / -2.38. Only 37 of 4,317 rows are tails.
Probabilities: cache/tfm_w4/pool_nonrome_ranking.npz (experiments/tfm/w4b_final.py, 3 seeds).

    python experiments/exp73_submit_v14.py --check-v13      # must be exactly zero
    python experiments/exp73_submit_v14.py --version 14
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
from prc.submit import build, TEAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check-v13', action='store_true')
    a = ap.parse_args()
    if not a.check_v13 and a.version is None:
        sys.exit('pass --version N (explicit) or --check-v13')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pb, et, in_P = c['ids'], c['pb'], c['et'], c['in_P'].astype(bool)
    sx = np.load(CACHE / 'final' / 'spec_x_ranking.npz', allow_pickle=True)
    assert np.array_equal(sx['ids'], ids) and np.array_equal(sx['in_P'].astype(bool), in_P)
    net = np.mean([np.load(CACHE / 'nn_final' / f'nn_final_seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)

    t = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
    rows = t['pos']                                           # direct row index into the ranking departures
    from prc.io import load
    rk = load('ranking', phase='DEP', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt'])
    assert np.array_equal(rk.MVT_ID_mvt.to_numpy(), ids)
    apt = rk.APT.astype(str).to_numpy()
    assert in_P[rows].all(), 'a row is outside P'
    assert (apt[rows] == t['apt']).all() and (apt[rows] != 'LIRF').all(), 'airport mismatch -- wrong row mapping'
    assert (rk.MVT_TIME_UTC_mvt.dt.month.to_numpy()[rows] == t['month']).all(), 'month mismatch -- wrong row mapping'
    assert len(rows) == int((in_P & (apt != 'LIRF')).sum()), 'not all non-Rome P rows are covered'

    p_old = sx['p_spec'][rows]
    p_new = p_old.copy() if a.check_v13 else t['p_pool'].astype(float)
    pbm = 0.5 * pb[:, rows] + 0.5 * net[rows]                  # v9 structure inside P (route D weight 0.5), as exp72
    mix = lambda p: np.mean(np.clip(p * et[rows] + (1 - p) * pbm, 0, None), axis=0)
    delta = mix(p_new) - mix(p_old)

    v13 = pd.read_parquet(OUTPUTS / f'{TEAM}_v13.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids)
    if a.check_v13:
        print(f'CHECK: delta with the old probabilities is exactly zero: {bool(np.all(delta == 0))}')
        return
    print(f'non-Rome P rows {len(rows)}: mean p {p_old.mean():.4f} -> {p_new.mean():.4f} | '
          f'mean shift {delta.mean():+.1f} s, |shift| max {np.abs(delta).max():.0f} s')
    for x in sorted(set(apt[rows])):
        k = apt[rows] == x
        print(f'  {x}: n {k.sum():4d}  p {p_old[k].mean():.4f} -> {p_new[k].mean():.4f}  mean shift {delta[k].mean():+7.1f} s')
    pred = v13.to_numpy().astype(float); pred[rows] += delta
    path = build(ids, pred, version=a.version, clip=(0.0, 200000.0))
    v14 = pd.read_parquet(path).set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    ch = v14 != v13.to_numpy()
    print(f'v{a.version} vs v13: rows changed {int(ch.sum())} (all among the {len(rows)}: '
          f'{bool(np.isin(np.flatnonzero(ch), rows).all())})')


if __name__ == '__main__':
    main()
