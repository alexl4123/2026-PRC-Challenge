"""r04 -- deadline cut-offs and odds (public leaderboard API + stated assumptions; nothing is fitted to labels).

1. Fixed-rank score series (rank 19, 23, 26, 38, 47, 53) at the end of each UTC day since 09-14, with
   linear slopes over the last 7 days -> projected score at each rank on 2026-10-11 22:59Z.
2. Deadline team count scenarios (n = 215 / 240 / 265) -> cut-off ranks for 10 / 20 / 25 %.
3. Monte Carlo of OUR final score vs the projected cut-offs. ALL inputs are assumptions (marked in the
   report as estimates): known levers as triangular(lo, mode, hi) live gains, an overlap discount,
   year-ahead noise, and a Poisson number of not-yet-known levers.

    python experiments/research_0925/r04_cutoff_odds.py [--json rows.json]
"""
import sys, os, json, math
from datetime import datetime, timedelta, timezone
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
import leaderboard as lb

DEADLINE = datetime(2026, 10, 11, 22, 59, tzinfo=timezone.utc)
rows = json.load(open(sys.argv[sys.argv.index('--json') + 1])) if '--json' in sys.argv else lb.fetch()
now = datetime.fromisoformat(max(r['processedAt'] for r in rows).replace('Z', '+00:00')) + timedelta(seconds=1)
days_left = (DEADLINE - now).total_seconds() / 86400

RANKS = (19, 23, 26, 38, 47, 53, 60)
series = {k: [] for k in RANKS}; xs = []
day = datetime(2026, 9, 14, 23, 59, 59, tzinfo=timezone.utc)
while day < now + timedelta(days=1):
    c = min(day, now)
    rk = sorted(lb.best_by_team(rows, until=c.strftime('%Y-%m-%dT%H:%M:%S')).values())
    xs.append((c - now).total_seconds() / 86400)
    for k in RANKS:
        series[k].append(rk[k - 1])
    day += timedelta(days=1)
xs = np.array(xs)
proj = {}
print(f'fixed-rank projection to the deadline ({days_left:.1f} days)')
for k in RANKS:
    s = np.array(series[k]); m7 = xs >= -7; m4 = xs >= -4
    b7, a7 = np.polyfit(xs[m7], s[m7], 1); b4, a4 = np.polyfit(xs[m4], s[m4], 1)
    proj[k] = (a7 + b7 * days_left, a4 + b4 * days_left)
    print(f'  rank {k:3d}: now {s[-1]:.2f} | 7-d slope {b7:+.3f}/d -> {proj[k][0]:.2f} | 4-d slope {b4:+.3f}/d -> {proj[k][1]:.2f}')
for n in (215, 240, 265):
    print(f'  n={n}: top10% rank {math.floor(.1 * n)}, top20% rank {math.floor(.2 * n)}, top25% rank {math.floor(.25 * n)}')

# --- Monte Carlo (assumptions) ---
rng = np.random.default_rng(0); N = 200_000
ours = 275.3106
levers = {  # live seconds, (lo, mode, hi); negative = better. Estimates, see the report's ranking table.
    'W3 second network architecture': (-1.5, -0.6, 0.2),
    'extend-P / route-D tail anchor at LIRF': (-1.2, -0.4, 0.3),
    'band mixture (exp63 step 2, re-based)': (-1.0, -0.3, 0.2),
    'final seeds / refit hygiene': (-0.5, -0.2, 0.0),
}
tot = np.zeros(N)
for lo, mo, hi in levers.values():
    tot += rng.triangular(lo, mo, hi, N)
tot *= rng.uniform(0.7, 1.0, N)                       # overlap between levers
tot += rng.normal(0, 1.5, N)                          # year-ahead noise of live vs CV
for lam in (0.5, 1.0, 2.0, 3.0):
    k = rng.poisson(lam, N)
    disc = np.array([rng.uniform(-6, -0.5, kk).sum() for kk in k[:20000]])
    disc = np.resize(disc, N)
    final = ours + tot + disc
    for name, cut_mu, cut_sd in (('top 10%', 256.0, 2.5), ('top 20%', 268.0, 3.5), ('top 25%', 271.0, 3.5)):
        cut = rng.normal(cut_mu, cut_sd, N)
        print(f'  new levers ~Poisson({lam}): median final {np.median(final):.1f}; P({name} at ~{cut_mu}) = '
              f'{np.mean(final <= cut):.3f}')
