"""m00 -- compact DEP frames for the major-levers study (training + ranking).

Writes cache/major/dep_{training,ranking}.parquet with the columns every later
m-script needs, in load(..., phase='DEP') row order (so OOF arrays align), plus
derived seconds columns. Also ARR frames with a few columns for transductive work.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.io import load, peak_rss_gb
from prc.config import TARGET, CACHE

OUT = CACHE / 'major'
OUT.mkdir(exist_ok=True)
COLS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'APT', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt',
        'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'CALLSIGN_flt',
        'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
        'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt', 'ARVT_1_flt', 'ARVT_3_flt']

for which in ['training', 'ranking']:
    for ph in ['DEP', 'ARR']:
        cols = COLS + ([TARGET] if True else [])
        d = load(which, phase=ph, columns=cols)
        t = d.MVT_TIME_UTC_mvt
        for src, nm in [('SCHED_TIME_UTC_mvt', 'off_sched'), ('AOBT_3_flt', 'off_aobt3'),
                        ('EOBT_1_flt', 'off_eobt1'), ('LOBT_flt', 'off_lobt'),
                        ('IOBT_flt', 'off_iobt'), ('BLOCK_TIME_UTC_mvt', 'off_block')]:
            d[nm] = (t - d[src]).dt.total_seconds().astype('float64')
        d['month'] = t.dt.month.astype('int8')
        d['y'] = d[TARGET].astype('float64')
        d = d.drop(columns=[TARGET])
        d.to_parquet(OUT / f'{ph.lower()}_{which}.parquet')
        print(which, ph, len(d), 'peak', round(peak_rss_gb(), 2), 'GB', flush=True)
