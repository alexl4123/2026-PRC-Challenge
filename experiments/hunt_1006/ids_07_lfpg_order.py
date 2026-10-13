"""ids_07 -- what is the MVT_ID sort key at LFPG / EHAM / LSZH / LTFM? Print stretches, test candidate keys."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load
import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
C = ['MVT_ID_mvt', 'APT', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'FLIGHT_mvt',
     'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'ADES_mvt', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'CALLSIGN_flt', 'AIRCRAFT_OPERATOR_flt']
d = load('training', columns=C)
d['APT'] = d.APT.astype(str); d['PHASE_mvt'] = d.PHASE_mvt.astype(str)
d = d[d.MVT_TIME_UTC_mvt.dt.month.eq(7) & d.APT.isin(['LFPG', 'EHAM', 'LSZH', 'LTFM'])].sort_values('MVT_ID_mvt')
for a in ['LFPG', 'EHAM', 'LSZH', 'LTFM']:
    for ph in ['DEP']:
        g = d[(d.APT == a) & (d.PHASE_mvt == ph)].reset_index(drop=True)
        print(f'\n===== {a} {ph} July 2025: n={len(g)}')
        for k in ['SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']:
            v = g[k].values
            print(f'  adjacent nondecreasing {k}: {np.mean(v[1:] >= v[:-1]):.4f}')
        for k in ['FLIGHT_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'ADES_mvt', 'RUNWAY_mvt']:
            v = g[k].astype(str).values
            print(f'  adjacent equal {k}: {np.mean(v[1:] == v[:-1]):.4f}   adjacent nondecreasing: {np.mean(v[1:] >= v[:-1]):.4f}')
        print(g.iloc[1000:1060][['MVT_ID_mvt', 'FLIGHT_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt',
                                  'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']].to_string())
# LFPG: the equal-SCHED adjacent pairs
g = d[(d.APT == 'LFPG') & (d.PHASE_mvt == 'DEP')].reset_index(drop=True)
s = g.SCHED_TIME_UTC_mvt.values
idx = np.flatnonzero(s[1:] == s[:-1])
print('\nLFPG equal-SCHED adjacent pairs (July):', len(idx))
rows = sorted(set(np.r_[idx, idx + 1]))
print(g.iloc[rows][['MVT_ID_mvt', 'FLIGHT_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt',
                     'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']].head(60).to_string())
