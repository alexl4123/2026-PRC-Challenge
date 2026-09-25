"""r08 -- the LIRF 'day-shift' labels (y = 86,400 + a normal taxi): is the mechanism observable?
Hypothesis (from r07's table, i.e. POST HOC): BLOCK written with the SCHEDULE's calendar date and the
actual time of day, so y = taxi + 86,400 whenever the takeoff falls on a later calendar date than the schedule.
Test: among LIRF departures, cross-tab day-shift labels (y in [86,400 + 60, 86,400 + 3,600]) with
'takeoff calendar date > schedule calendar date' (UTC and Europe/Rome), by off_sched band; and what 2026
contains in the same cells (no labels). Read-only.
    python experiments/research_0925/r08_day_shift_mechanism.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
cols = ['APT', 'month', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'off_sched', 'off_aobt3']
for name in ('training', 'ranking'):
    d = pd.read_parquet(CACHE / 'major' / f'dep_{name}.parquet', columns=cols + (['y'] if name == 'training' else []))
    d = d[d.APT.astype(str) == 'LIRF'].copy()
    mv = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True); sc = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)
    d['dchg_utc'] = (mv.dt.normalize() - sc.dt.normalize()).dt.days
    d['dchg_loc'] = (mv.dt.tz_convert('Europe/Rome').dt.normalize() - sc.dt.tz_convert('Europe/Rome').dt.normalize()).dt.days
    d['nm'] = d.off_aobt3.notna()
    d['band'] = pd.cut(d.off_sched, [-1e9, 3600, 7200, 20000, 40000, 50000, 80000, 1e9])
    print(f'\n===== {name}: LIRF departures {len(d):,}')
    if name == 'training':
        d['dayshift'] = (d.y >= 86400 + 60) & (d.y <= 86400 + 3600)
        d['stamped'] = (d.y - d.off_sched).abs() <= 120
        print(f'day-shift labels: {d.dayshift.sum()} (NM record {int((d.dayshift & d.nm).sum())})')
        g = d.groupby(['band', 'dchg_loc'], observed=True).agg(n=('y', 'size'), dayshift=('dayshift', 'sum'),
                                                               stamped=('stamped', 'sum'), noNM=('nm', lambda s: int((~s).sum())))
        print(g.to_string())
        print('day-shift rows, off_sched and date change (local / utc):')
        print(d[d.dayshift][['month', 'nm', 'off_sched', 'dchg_loc', 'dchg_utc', 'y']].sort_values('off_sched').round(0).to_string(index=False))
    else:
        g = d.groupby(['band', 'dchg_loc'], observed=True).agg(n=('off_sched', 'size'), noNM=('nm', lambda s: int((~s).sum())))
        print(g.to_string())
