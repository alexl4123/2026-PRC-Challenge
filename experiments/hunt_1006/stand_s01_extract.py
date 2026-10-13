"""Extract compact event tables (seconds since epoch) for the stand / airframe sequence hunt.
Writes cache/hunt/stand_ev_{training,ranking}.parquet. Read-only on everything else."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb

OUT = Path('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt')
EP = pd.Timestamp('1970-01-01')
COLS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'PHASE_mvt', 'APT', 'STAND_mvt', 'ADEP_mvt', 'ADES_mvt',
        'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt',
        'AIRCRAFT_OPERATOR_flt', 'CALLSIGN_flt', 'LOBT_flt', 'AOBT_3_flt', 'RUNWAY_mvt']


def sec(s):
    if getattr(s.dt, 'tz', None) is not None:
        s = s.dt.tz_convert(None)
    return (s - EP).dt.total_seconds().to_numpy()


for which in ('training', 'ranking'):
    d = load(which, columns=COLS)
    o = pd.DataFrame({'id': d.MVT_ID_mvt.to_numpy()})
    for c in ['FLIGHT_ID_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt']:
        o[c] = d[c].astype('string')
    for c in ['PHASE_mvt', 'APT', 'STAND_mvt', 'ADEP_mvt', 'ADES_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'RUNWAY_mvt']:
        o[c] = d[c].astype('category')
    for c, n in [('MVT_TIME_UTC_mvt', 'mvt'), ('BLOCK_TIME_UTC_mvt', 'blk'), ('SCHED_TIME_UTC_mvt', 'sch'),
                 ('LOBT_flt', 'lobt'), ('AOBT_3_flt', 'aobt3')]:
        o[n] = sec(d[c])
    print(which, len(o), d.MVT_TIME_UTC_mvt.dtype, 'blk nan by phase:', o.groupby('PHASE_mvt', observed=True).blk.apply(lambda x: x.isna().mean()).to_dict())
    o.to_parquet(OUT / f'stand_ev_{which}.parquet')
    del d, o
print('peak', peak_rss_gb())
