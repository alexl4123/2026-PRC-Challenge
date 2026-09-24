"""s15 -- probability of reaching each target by the deadline, by Monte Carlo over the
workstream ranges of reports/2026-09-24_strategy_40s.md section 2 (all ranges are [I] estimates).

Each workstream's live gain ~ triangular(lo, mode, hi) (seconds, negative = better).
Sum x overlap factor U(0.65, 1.0) + year-ahead noise N(0, 1.5).  Start: v11 = 289.6993.
Deadline top-25% cut ~ N(276.5, 1.5); top-10% cut ~ N(260.5, 2.0).
Sensitivity: the same with W2/W3 failing entirely (no GPU/cluster time).
"""
import numpy as np

rng = np.random.default_rng(0)
N = 200_000
START = 289.6993
WS = {  # (lo, mode, hi) live seconds -- updated after s05c / s17 / s17b
    'W0 TabPFN/TabICL as the LIRF-P classifier (s17)': (-4.5, -2.7, 0.0),
    'W1 family blend re-based': (-2.5, -1.3, -0.3),
    'W2 retrain, features inside members': (-5.0, -2.0, 0.0),
    'W3 second network architecture': (-1.5, -0.5, 0.0),
    'W4 TFM for the other small populations (non-LIRF P, extend-P, route D)': (-1.5, -0.5, 0.3),
    'W5 band mixture re-based': (-1.0, -0.3, 0.2),
    'W6 season weights (inside W2)': (-0.4, -0.1, 0.2),
    'W8 event pockets': (-1.0, 0.0, 0.5),
}
DISC = float(__import__('os').environ.get('DISC', '0'))   # expected number of NEW levers found before the deadline


def run(ws, label):
    tot = np.zeros(N)
    for lo, mo, hi in ws.values():
        tot += rng.triangular(lo, mo, hi, N)
    tot = tot * rng.uniform(0.7, 1.0, N) + rng.normal(0, 1.5, N)
    if DISC > 0:   # discovery: Poisson number of new levers, each U(-3, -0.5) s live
        k = rng.poisson(DISC, N); tot += np.array([rng.uniform(-3, -0.5, kk).sum() for kk in k])
    final = START + tot
    cut25 = rng.normal(277.0, 1.5, N); cut10 = rng.normal(260.5, 2.0, N)
    q = np.percentile(final, [10, 50, 90])
    print(f'\n== {label}')
    print(f'   final score p10/p50/p90: {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}   (total gain median {np.median(tot):+.1f} s)')
    print(f'   P(<= 280.33, today\'s top-25% cut)  {np.mean(final <= 280.33):.2f}')
    print(f'   P(top 25% at the deadline)         {np.mean(final <= cut25):.2f}')
    print(f'   P(top 20% today 277.25)            {np.mean(final <= 277.25):.2f}')
    print(f'   P(top 10% at the deadline)         {np.mean(final <= cut10):.2f}')
    print(f'   P(-40 s from v9, <= 257.34)        {np.mean(final <= 257.34):.3f}')


run(WS, 'all workstreams')
run({k: v for k, v in WS.items() if not k.startswith(('W2', 'W3'))}, 'no retrain / no GPU (W2, W3 fail)')
run({**WS, 'W2 retrain, features inside members': (-4.5, -2.5, -0.5)}, 'optimistic W2 (network transfers 1.65x like v8)')
