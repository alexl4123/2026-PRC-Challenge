"""s08 -- how fast do the leaderboard's percentile cut-offs move?  (public API, read-only)

For each day in the last ~3 weeks, rebuild the board as it stood at 23:59 UTC (each team's
best score up to then) and report the team count, the top-10% / 20% / 25% cut-offs and the
leader.  The slope over the last 7 days is what the 17-day plan has to beat.
Saves the raw submissions to cache/strategy/s08_submissions.json.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tools'))
import numpy as np
from datetime import datetime, timedelta, timezone
import leaderboard as lb

rows = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else lb.fetch()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache', 'strategy', 's08_submissions.json')
json.dump(rows, open(out, 'w'))
print('submissions', len(rows), 'teams', len({r['teamName'] for r in rows}))
now = datetime.now(timezone.utc)
res = []
for back in list(range(21, 0, -1)) + [0]:
    t = (now - timedelta(days=back)).strftime('%Y-%m-%dT23:59:59') if back else now.isoformat()
    best = lb.best_by_team(rows, until=t)
    s = np.sort(np.array(list(best.values())))
    if len(s) == 0:
        continue
    n = len(s)
    q = lambda f: s[max(int(np.ceil(f * n)) - 1, 0)]
    me = best.get(lb.ME)
    rank = int((s < me).sum()) + 1 if me else None
    res.append((t[:10], n, s[0], q(0.10), q(0.20), q(0.25), np.median(s), me, rank))
    print(f'{t[:16]}  teams {n:4d}  leader {s[0]:7.2f}  top10% {q(0.10):7.2f}  top20% {q(0.20):7.2f}  top25% {q(0.25):7.2f}  '
          f'median {np.median(s):7.2f}  us {me}  rank {rank}')
r = np.array([(x[3], x[4], x[5]) for x in res[-8:]])
d7 = r[-1] - r[0]
print(f'\n7-day change of the cut-offs (s): top10% {d7[0]:+.2f}  top20% {d7[1]:+.2f}  top25% {d7[2]:+.2f}')
print('linear projection to the deadline (17 days):', ', '.join(f'{nm} {r[-1][i] + d7[i]*17/7:.1f}' for i, nm in enumerate(['top10%', 'top20%', 'top25%'])))
# rank needed to be in top 25% as more teams join
for nt in [186, 195, 205]:
    print(f'  with {nt} teams, top 25% = rank <= {int(np.floor(0.25*nt))}')
