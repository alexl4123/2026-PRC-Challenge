"""How do FLIGHT_mvt strings of the same flight (shared FLIGHT_ID_mvt) differ between the departure airport's and the
arrival airport's records?  Needed to link route-D departures (no FLIGHT_ID) to their arrival rows."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from prc.config import CACHE
pd.set_option('display.width', 250)
APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
cols = ['PHASE_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt']
t = pq.read_table(CACHE / 'training.parquet', columns=cols)
t = t.filter(pc.and_(pc.is_in(t['ADEP_mvt'].cast('string'), pa_arr := __import__('pyarrow').array(APTS)),
                     pc.is_in(t['ADES_mvt'].cast('string'), pa_arr)))
d = t.to_pandas(); del t
for c in ('PHASE_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'ADES_mvt'):
    d[c] = d[c].astype(str)
dep = d[d.PHASE_mvt == 'DEP']; arr = d[d.PHASE_mvt == 'ARR']
print('intra-network DEP', len(dep), 'ARR', len(arr), '; DEP no LOBT', int(dep.LOBT_flt.isna().sum()), '; ARR no LOBT', int(arr.LOBT_flt.isna().sum()))
m = dep.dropna(subset=['FLIGHT_ID_mvt']).merge(arr.dropna(subset=['FLIGHT_ID_mvt']), on='FLIGHT_ID_mvt', suffixes=('_d', '_a'))
print('pairs by FLIGHT_ID', len(m), '; FLIGHT_mvt equal', float((m.FLIGHT_mvt_d == m.FLIGHT_mvt_a).mean()),
      '; CALLSIGN equal', float((m.CALLSIGN_flt_d == m.CALLSIGN_flt_a).mean()),
      '; dep FLIGHT_mvt == arr CALLSIGN', float((m.FLIGHT_mvt_d == m.CALLSIGN_flt_a).mean()),
      '; arr FLIGHT_mvt == dep CALLSIGN', float((m.FLIGHT_mvt_a == m.CALLSIGN_flt_d).mean()))
for a in APTS:
    g = m[m.ADEP_mvt_d == a].sample(6, random_state=0)
    print(a, 'dep fmt -> arr fmt:', list(zip(g.FLIGHT_mvt_d, g.CALLSIGN_flt_d, g.FLIGHT_mvt_a, g.ADES_mvt_d)))
    print('   share dep FLIGHT_mvt == dep CALLSIGN', float((m.FLIGHT_mvt_d == m.CALLSIGN_flt_d)[m.ADEP_mvt_d == a].mean()),
          ' arr FLIGHT_mvt == arr CALLSIGN (arrivals AT', a, ')', float((m.FLIGHT_mvt_a == m.CALLSIGN_flt_a)[m.ADES_mvt_d == a].mean()))
tt = (m.MVT_TIME_UTC_mvt_a - m.MVT_TIME_UTC_mvt_d).dt.total_seconds()
print('flight time (arr MVT - dep MVT) quantiles', np.quantile(tt, [0, .001, .5, .999, 1]).round().tolist())
# route-D departures (no LOBT): examples of their FLIGHT_mvt
nd = dep[dep.LOBT_flt.isna()]
print('route-D intra-network deps by airport', nd.ADEP_mvt.value_counts().to_dict())
print(nd.groupby('ADEP_mvt').FLIGHT_mvt.apply(lambda s: s.sample(min(8, len(s)), random_state=1).tolist()).to_string())
print('route-D deps with a FLIGHT_ID:', int(nd.FLIGHT_ID_mvt.notna().sum()))
