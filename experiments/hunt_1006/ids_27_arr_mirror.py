"""ids_27 -- 2026 label-free validation of the mechanism 'NM join failed although the NM flight exists and is unused
=> the join window is violated', on the ARRIVAL side (window |MVT_arr - ARVT_3| <= ~3606, MVT is public in both years).
ARR rows without NM, linked to an ORIGIN departure row (same callsign/ADEP/ADES, take-off 0-24 h before landing) that
carries the NM record; NM flight id on no ARR row; ADEP != ADES. Violation = |MVT_arr - ARVT_3(origin)| <= 3606."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'ARVT_3_flt']
for w in ['training', 'ranking']:
    d = load(w, columns=C)
    for c in ['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt']:
        d[c] = d[c].astype(str)
    arr = d[d.PHASE_mvt.eq('ARR')]; dep = d[d.PHASE_mvt.eq('DEP') & d.LOBT_flt.notna()]
    used = set(arr.FLIGHT_ID_mvt.dropna())
    a0 = arr[arr.LOBT_flt.isna()].reset_index()
    o = dep[['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'FLIGHT_ID_mvt', 'ARVT_3_flt']].rename(
        columns={'MVT_TIME_UTC_mvt': 'dm', 'FLIGHT_ID_mvt': 'ofid', 'ARVT_3_flt': 'oarvt'})
    m = a0.merge(o, on=['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt'])
    m['ft'] = (m.MVT_TIME_UTC_mvt - m.dm).dt.total_seconds()
    m = m[(m.ft > 0) & (m.ft < 86400)].sort_values('ft').drop_duplicates('index')
    m['used'] = m.ofid.isin(used)
    m['circ'] = m.ADEP_mvt == m.ADES_mvt
    m['gap'] = (m.MVT_TIME_UTC_mvt - m.oarvt).dt.total_seconds()
    m['viol'] = m.gap.abs() <= 3606
    print('=====', w, ' ARR rows w/o NM linked to an origin NM record:', len(m))
    print(pd.crosstab([m.used, m.circ], m.viol).to_string())
    r = m[~m.used & ~m.circ]
    print(f'  refined rule rows {len(r)}, violations {r.viol.sum()}')
    if r.viol.sum():
        print(r[r.viol][['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'oarvt', 'gap', 'ft']].head(20).to_string())
