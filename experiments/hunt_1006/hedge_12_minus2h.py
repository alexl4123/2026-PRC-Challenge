"""Route D outside zone P: the 'BLOCK = SCHED - k h' rows (y - off_sched near +3600 / +7200) and other tails.
Do they cluster on dates (DST switch, recorder outage) -> label-free fingerprint?  Also bias of A/B/C LIRF rows where the
LOBT window excludes the stamp hypothesis."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P, D, f, ol = V['y'], V['month'], V['P'], V['D'], V['f'], V['off_lobt']
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'off_sched'])
d['y'] = y; d['f'] = f; d['grp'] = np.where(P, 'P', np.where(D, 'D', 'A'))
d['r'] = d.y - d.off_sched
d['date'] = d.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.date
tot = ((y - f) ** 2).sum()
x = d[(d.grp == 'D') & (d.y > 3000)]
print('route D outside P, y > 3000:', len(x), ' SSE share', round(float(((x.y - x.f) ** 2).sum() / tot), 4))
x = x.assign(kind=np.select([np.abs(x.r) <= 120, np.abs(x.r - 3600) <= 300, np.abs(x.r - 7200) <= 300, x.y > 43200], ['stamp', '+1h', '+2h', 'day'], 'other'))
print(pd.crosstab(x.APT, x.kind))
print(x[x.kind.isin(['+1h', '+2h'])][['APT', 'FLIGHT_mvt', 'date', 'off_sched', 'y', 'f']].sort_values('date').to_string())
# share of LIRF route-D rows on those dates that are '+2h'
dd = x[x.kind == '+2h'].date.unique()
allD = d[(d.grp == 'D') & (d.APT == 'LIRF') & d.date.isin(dd)]
print('LIRF route-D (outside P) rows on the +2h dates:', len(allD), ' of which +2h:', int((np.abs(allD.r - 7200) <= 300).sum()))
print(allD.groupby('date').apply(lambda g: f'{int((np.abs(g.r - 7200) <= 300).sum())}/{len(g)}').to_string())
# A/B/C LIRF rows: stamp excluded by the window (SCHED outside [LOBT - 3606, LOBT + 3606]) vs possible; residual bias
A = d[(d.grp == 'A') & (d.APT == 'LIRF')].copy(); A['ol'] = ol[A.index]
A['stamp_possible'] = np.abs(A.off_sched - A.ol) <= 3606
for k, g in A[A.off_sched > 1800].groupby('stamp_possible'):
    print(f'A/B/C LIRF off_sched>1800, stamp possible={k}: n {len(g)} mean(y-f) {np.mean(g.y - g.f):+.1f} SSE share {((g.y - g.f) ** 2).sum() / tot:.4f}'
          f' stamped share {(np.abs(g.y - g.off_sched) <= 60).mean():.3f}')
