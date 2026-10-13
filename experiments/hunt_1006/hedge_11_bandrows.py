"""LIRF zone P rows that crossed the local date with delay > 12 h (stamp vs day-shift): every raw column, 2025 + 2026,
looking for a mechanistic (not fitted) discriminator."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
cols = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'FLIGHT_RULE_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
        'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'APT']
for which, f in (('training', 'hedge_lirf_tw25.parquet'), ('ranking', 'hedge_lirf_tw26.parquet')):
    t = pd.read_parquet(CACHE / 'hunt' / f)
    t = t[t.P & (t.dd_loc >= 1) & (t.off_sched > 43200)]
    d = load(which, phase='DEP', columns=cols + (['BLOCK_TIME_UTC_mvt'] if which == 'training' else []))
    d = d.iloc[t.pos.to_numpy()].copy()
    d['dd_utc'] = t.dd_utc.to_numpy(); d['off_sched'] = t.off_sched.to_numpy()
    d['mL'] = d.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.strftime('%m-%d %a %H:%M:%S')
    d['sL'] = d.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.strftime('%m-%d %a %H:%M')
    if which == 'training':
        y = (d.MVT_TIME_UTC_mvt - d.BLOCK_TIME_UTC_mvt).dt.total_seconds()
        d['cls'] = np.where(np.abs(y - d.off_sched) <= 120, 'stamp', np.where(y > 43200, 'DAY', 'normal'))
        d['bL'] = d.BLOCK_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.strftime('%m-%d %H:%M:%S')
    print(which); print(d.drop(columns=['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'APT'] + (['BLOCK_TIME_UTC_mvt'] if which == 'training' else [])).sort_values('off_sched').to_string())
