"""EXP72 -- v13 = v12 with a tabular foundation model deciding Rome's tail zone.

WHAT CHANGES
------------
Only the 181 ranking flights at LIRF inside P (no NM record, > 2 h behind schedule).
There, the probability that the block time was stamped at the schedule comes from a
tabular foundation model instead of the v7 specialist (TailSpecialistX):
    --variant ens     0.5 * (TabPFN v2 + TabICL)    (TabPFN v2: Apache-2.0 + attribution clause)
    --variant tabicl  TabICL alone                  (BSD-3, code and weights)
Everything else is v12 unchanged.

    v13 = v12 + delta,  delta = mixture(p_new) - mixture(p_old)  on those 181 rows,
both computed per production LightGBM seed from the same components, so with p_new =
p_old delta is exactly 0 (the regression check, --check-v12).

EVIDENCE (strategy agent, 12-fold leave-one-month-out on the 627 LIRF P rows of 2025;
experiments/strategy/s17*, s19, s21):
    specialist (v7)   Brier 0.160  AUC 0.854
    TabPFN, 3 seeds   Brier 0.132  AUC 0.880   -2.74 [-4.80,-1.34]  scored -4.53  10/12 months
    TabICL            Brier 0.130  AUC 0.881   -2.34               scored -4.73  10/12
    50/50 ensemble    Brier 0.129  AUC 0.884   -2.63 [-4.39,-1.38]  scored -4.76  10/12
    without the 20 most-affected rows: -1.48 (12-mo) / -3.59 (scored)
Probabilities come from cache/strategy/tfm_P_lirf_ranking.npz (fitted on all 627 rows).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
from prc.submit import build, TEAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['ens', 'tabicl'], default=None)
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check-v12', action='store_true')
    a = ap.parse_args()
    if not a.check_v12 and (a.variant is None or a.version is None):
        sys.exit('pass --variant and --version, or --check-v12')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)
    sx = np.load(CACHE / 'final' / 'spec_x_ranking.npz', allow_pickle=True)
    assert np.array_equal(sx['ids'], ids) and np.array_equal(sx['in_P'].astype(bool), in_P)
    p_old_all = sx['p_spec']
    net = np.mean([np.load(CACHE / 'nn_final' / f'nn_final_seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)

    t = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    rows = t['pos']                                           # direct row index into the ranking departures (verified)
    assert in_P[rows].all(), 'a row is outside P'
    from prc.io import load
    rk = load('ranking', phase='DEP', columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt'])
    assert np.array_equal(rk.MVT_ID_mvt.to_numpy(), ids)
    assert (rk.APT.astype(str).to_numpy()[rows] == 'LIRF').all() and len(rows) == 181, 'rows are not the 181 LIRF P rows'
    assert (rk.MVT_TIME_UTC_mvt.dt.month.to_numpy()[rows] == t['month']).all(), 'month mismatch -- wrong row mapping'

    p_old = p_old_all[rows]
    if a.check_v12:
        p_new = p_old.copy()
    else:
        p_new = 0.5 * (t['p_tabpfn'] + t['p_tabicl']) if a.variant == 'ens' else t['p_tabicl'].astype(float)
    pbm = 0.5 * pb[:, rows] + 0.5 * net[rows]                  # v9 structure inside P (route D weight 0.5)
    mix = lambda p: np.mean(np.clip(p * et[rows] + (1 - p) * pbm, 0, None), axis=0)
    delta = mix(p_new) - mix(p_old)

    v12 = pd.read_parquet(OUTPUTS / f'{TEAM}_v12.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids)
    if a.check_v12:
        print(f'CHECK: delta with the old probabilities is exactly zero: {bool(np.all(delta == 0))}')
        return
    # v12's float values are not stored; add delta to the integer v12 (rounding error <= 0.5 s on 181 rows)
    pred = v12.to_numpy().astype(float); pred[rows] += delta
    print(f'variant {a.variant}: p on the 181 rows  old mean {p_old.mean():.3f} -> new mean {p_new.mean():.3f} | '
          f'corr {np.corrcoef(p_old, p_new)[0,1]:.3f} | |dp| > 0.2 on {int((np.abs(p_new - p_old) > 0.2).sum())} rows')
    for m in (1, 7):
        k = t['month'] == m
        print(f'  month {m}: n {int(k.sum())}  mean p {p_old[k].mean():.3f} -> {p_new[k].mean():.3f}  '
              f'mean shift {delta[k].mean():+.0f}s  max |shift| {np.abs(delta[k]).max():.0f}s')
    path = build(ids, pred, version=a.version, clip=(0.0, 200000.0))
    v13 = pd.read_parquet(path).set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids).to_numpy()
    ch = v13 != v12.to_numpy()
    print(f'v{a.version} vs v12: rows changed {int(ch.sum())} (all among the 181: {bool(np.isin(np.flatnonzero(ch), rows).all())})')


if __name__ == '__main__':
    main()
