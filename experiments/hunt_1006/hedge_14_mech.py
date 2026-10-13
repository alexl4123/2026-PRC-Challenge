"""LIRF zone P: which date convention (UTC vs Rome local) does each record follow?  Cross-tab class by (dd_utc, dd_loc, dbl)."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250)
d = pd.read_parquet(CACHE / 'hunt' / 'hedge_lirf_tw25.parquet')
for nm, x in (('zone P', d[d.P]), ('all LIRF departures', d)):
    print(nm); print(pd.crosstab([x.dd_utc, x.dd_loc, x.dbl], x.cls, margins=True))
x = d[d.P & (d.dd_loc == 1) & (d.dd_utc == 0)]
print(x[['FLIGHT_mvt', 'dbl', 'twin', 'cls', 'y', 'off_sched']].sort_values(['dbl', 'cls']).to_string())
