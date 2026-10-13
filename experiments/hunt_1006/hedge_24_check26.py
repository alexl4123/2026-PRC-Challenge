"""Label-free 2025 vs 2026 consistency for rules B / T at LIRF: NM-matched vs route-D counts of delayed date-crossing and
very long-delay departures, and the on-time route-D share."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.io import load
for which in ('training', 'ranking'):
    d = load(which, phase='DEP', columns=['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt'])
    d = d[d.APT == 'LIRF']
    os_ = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    du = ((d.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
    nm = d.LOBT_flt.notna(); ot = (os_ > -600) & (os_ < 1800)
    s = f'{which}: LIRF deps {len(d)}, on-time route-D share {1 - nm[ot].mean():.4f}'
    for nm_, m in (('crossing&delay>2h', (du >= 1) & (os_ > 7200)), ('delay>8h', os_ > 28800), ('no crossing, 2-4h', (du < 1) & (os_ > 7200) & (os_ <= 14400))):
        s += f' | {nm_}: NM {int((m & nm).sum())} / route D {int((m & ~nm).sum())}'
    print(s)
