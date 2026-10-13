"""LIRF zone P, rows whose (MVT - 900 s) local date is after the SCHED local date: class by delay size."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
L = pd.read_parquet(CACHE / 'hunt' / 'hedge_lirf.parquet')
W = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
p = np.full(len(d), np.nan); p[W['Pidx']] = W['p14']; pb = np.full(len(d), np.nan); pb[W['Pidx']] = W['pbP']
L['p'] = p[L.index]; L['pb'] = pb[L.index]
L['mL'] = d.MVT_TIME_UTC_mvt[L.index].dt.tz_convert('Europe/Rome').dt.tz_localize(None)
L['sL'] = d.SCHED_TIME_UTC_mvt[L.index].dt.tz_convert('Europe/Rome').dt.tz_localize(None)
P_ = L[L.grp == 'P'].copy()
P_['delay_h'] = (P_.off_sched / 3600).round(1)
P_['bin'] = pd.cut(P_.off_sched, [7200, 10800, 14400, 21600, 43200, 86400, 1e7])
print(pd.crosstab([P_.ddate, P_.bin], P_.cls, margins=True))
x = P_[P_.ddate == 1].sort_values('off_sched')
print(x[['FLIGHT_mvt', 'dbl', 'mL', 'sL', 'cls', 'y', 'off_sched', 'p', 'pb', 'f', 'month']].to_string())
# SSE of the classes in LIRF P (12m) and what a 3-hypothesis oracle would give
tot = 2.344e10 / 0.2155
for c in ('stamp', 'dayshift', 'normal', 'other'):
    m = P_.cls == c
    print(c, int(m.sum()), 'SSE share of 12m', round(float(((P_.y - P_.f) ** 2)[m].sum() / tot), 4))
