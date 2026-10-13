"""Compact caches for the forensics hunt: times as float seconds since epoch (NaN = missing)."""
import sys; sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.io import load, peak_rss_gb
from prc.config import CACHE
H = CACHE / 'hunt'
T = ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt']
S = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'FLIGHT_RULE_mvt', 'ADEP_mvt', 'ADES_mvt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt',
     'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt', 'ADES_FILED_flt', 'MARKET_SEGMENT_flt', 'FLIGHT_RULE_flt', 'FLIGHT_TYPE_flt',
     'AIRCRAFT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'APT']
short = {'MVT_TIME_UTC_mvt': 'MVT', 'BLOCK_TIME_UTC_mvt': 'BLOCK', 'SCHED_TIME_UTC_mvt': 'SCHED', 'LOBT_flt': 'LOBT', 'IOBT_flt': 'IOBT',
         'EOBT_1_flt': 'EOBT1', 'ARVT_1_flt': 'ARVT1', 'AOBT_3_flt': 'AOBT3', 'ARVT_3_flt': 'ARVT3'}
for which in ('training', 'ranking'):
    for ph in ('DEP', 'ARR'):
        cols = S + T + (['TAXITIME_SEC_mvt'] if which == 'training' else [])
        d = load(which, phase=ph, columns=cols)
        for c in T:
            if c in d.columns:
                v = d[c]
                sub = (v.dt.microsecond != 0) & v.notna()
                if sub.any(): print(which, ph, c, 'sub-second values:', int(sub.sum()))
                d[short[c]] = (v - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
                d = d.drop(columns=c)
        for c in ['FLIGHT_mvt', 'CALLSIGN_flt']:
            d[c] = d[c].astype('category')
        if 'TAXITIME_SEC_mvt' in d.columns:
            d['TAXI'] = d.pop('TAXITIME_SEC_mvt').astype(float)
        out = H / f'{which[:2]}_{ph.lower()}.parquet'
        d.to_parquet(out)
        print(which, ph, len(d), 'rows ->', out.name, 'BLOCK present:', d['BLOCK'].notna().mean() if 'BLOCK' in d else None, f'rss {peak_rss_gb():.2f}')
        del d
dt = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt'])
d = pd.read_parquet(H / 'tr_dep.parquet', columns=['MVT_ID_mvt'])
print('tr_dep order == dep_training order:', np.array_equal(dt.MVT_ID_mvt.to_numpy(), d.MVT_ID_mvt.to_numpy()))
