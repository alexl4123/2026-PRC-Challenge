"""ids_19 -- refine the destination-NM exclusion for route-D departures: WHY does the DEP-side NM join fail when the
destination arrival row carries the flight's NM record? Characterise the 'inside window' exceptions: NM flight already
matched to another DEP row (one-to-one join), twin rows, wrong link (AOBT_3/flight-time inconsistent), garbled callsign."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load
import warnings; warnings.filterwarnings('ignore')
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 280); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
     'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt', 'CALLSIGN_flt', 'APT']
def build(which):
    d = load(which, columns=C)
    for c in ['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'APT', 'CALLSIGN_flt']:
        d[c] = d[c].astype(str)
    dep = d[d.PHASE_mvt.eq('DEP')].copy()
    arr = d[d.PHASE_mvt.eq('ARR') & d.LOBT_flt.notna()].copy()
    used_ids = set(dep.FLIGHT_ID_mvt.dropna().values)
    dD = dep[dep.LOBT_flt.isna()].copy()
    a = arr[['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt', 'FLIGHT_ID_mvt', 'CALLSIGN_flt']].rename(
        columns={'MVT_TIME_UTC_mvt': 'am', 'LOBT_flt': 'dlobt', 'AOBT_3_flt': 'daobt', 'FLIGHT_ID_mvt': 'dfid', 'CALLSIGN_flt': 'dcs'})
    m = dD.reset_index().merge(a, on=['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt'])
    m['ft'] = (m.am - m.MVT_TIME_UTC_mvt).dt.total_seconds()
    m = m[(m.ft > 0) & (m.ft < 86400)].sort_values('ft').drop_duplicates('index')
    m['used'] = m.dfid.isin(used_ids)
    m['mvt_m_aobt'] = (m.MVT_TIME_UTC_mvt - m.daobt).dt.total_seconds()
    m['mvt_m_lobt'] = (m.MVT_TIME_UTC_mvt - m.dlobt).dt.total_seconds()
    m['off_sched'] = (m.MVT_TIME_UTC_mvt - m.SCHED_TIME_UTC_mvt).dt.total_seconds()
    m['cs_eq'] = m.FLIGHT_mvt == m.dcs
    # twin: another DEP row with same airport, callsign, SCHED
    key = dep.groupby(['APT', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt']).MVT_ID_mvt.transform('size')
    tw = dep.loc[key >= 2, 'MVT_ID_mvt']
    m['twin'] = m.MVT_ID_mvt.isin(set(tw.values))
    if which == 'training':
        m['blk_m_lobt'] = (m.BLOCK_TIME_UTC_mvt - m.dlobt).dt.total_seconds()
        m['inside'] = m.blk_m_lobt.abs() <= 3606
        m['y'] = (m.MVT_TIME_UTC_mvt - m.BLOCK_TIME_UTC_mvt).dt.total_seconds()
    return m
m = build('training')
print('2025 route-D deps linked to a destination NM arrival:', len(m), ' inside window (exceptions to exclusion):', m.inside.sum(),
      f'({m.inside.mean():.4f})')
print('\nexceptions by used / twin / cs_eq:')
print(pd.crosstab([m.used, m.twin], m.inside).to_string())
print(pd.crosstab(m.cs_eq, m.inside).to_string())
m['aobt_ok'] = (m.mvt_m_aobt > -60) & (m.mvt_m_aobt < 7200)
print(pd.crosstab([m.used, m.aobt_ok], m.inside).to_string())
print('\nby airport (not used):'); mm = m[~m.used]
print(pd.crosstab(mm.APT, mm.inside).to_string())
print('\nexceptions among NOT-used, aobt_ok rows:')
ex = mm[mm.inside & mm.aobt_ok]
print(len(ex))
print(ex[['APT', 'FLIGHT_mvt', 'dcs', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'dlobt', 'daobt', 'y', 'blk_m_lobt', 'mvt_m_aobt', 'twin']].head(60).to_string())
# rule candidate: not used & aobt_ok -> outside the window ; how exact?
r = m[~m.used & m.aobt_ok]
print('\nRULE candidate (not used, AOBT consistent): n', len(r), ' exceptions', r.inside.sum())
r2 = m[~m.used]
print('RULE (not used): n', len(r2), ' exceptions', r2.inside.sum())
m.to_parquet(H + 'cache/hunt/ids_19_dstlink_2025.parquet')
k = build('ranking')
print('\n2026 route-D deps linked:', len(k), ' not used:', (~k.used).sum(), ' not used & aobt_ok:', ((~k.used) & (k.mvt_m_aobt > -60) & (k.mvt_m_aobt < 7200)).sum())
k.to_parquet(H + 'cache/hunt/ids_19_dstlink_2026.parquet')
