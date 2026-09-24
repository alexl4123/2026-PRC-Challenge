"""s16 -- outside view: how much do teams at our level improve over the next 7-17 days?
From the public submissions (cache/strategy/s08_submissions.json): for each team whose best score
at date D was in a band, the change of its best score by D + h days.  Teams that stopped submitting
count with their (unchanged) best, i.e. this is the unconditional outcome for a team at that level."""
import json, numpy as np, sys
from datetime import datetime, timedelta
rows = json.load(open('cache/strategy/s08_submissions.json'))
def best_at(t):
    b = {}
    for r in rows:
        if r['processedAt'] <= t:
            s = r['score']; b[r['teamName']] = min(b.get(r['teamName'], 1e9), s)
    return b
for D, h in [('2026-09-10', 7), ('2026-09-10', 14), ('2026-09-14', 7), ('2026-09-14', 10), ('2026-09-17', 7)]:
    t0 = D + 'T23:59:59'; t1 = (datetime.fromisoformat(D) + timedelta(days=h)).strftime('%Y-%m-%d') + 'T23:59:59'
    b0, b1 = best_at(t0), best_at(t1)
    for lo, hi in [(275, 300), (280, 295), (300, 330)]:
        d = np.array([b1[k] - v for k, v in b0.items() if lo <= v < hi])
        active = np.array([b1[k] - v < -0.01 for k, v in b0.items() if lo <= v < hi])
        if len(d):
            print(f'{D} +{h:2d}d  band {lo}-{hi}: n={len(d):3d}  improved {active.mean():.2f}  median {np.median(d):+6.2f}  '
                  f'p25 {np.percentile(d,25):+6.2f}  p10 {np.percentile(d,10):+6.2f}  best {d.min():+7.2f}  share <= -10 s {np.mean(d <= -10):.2f}  <= -14 s {np.mean(d <= -14):.2f}')
