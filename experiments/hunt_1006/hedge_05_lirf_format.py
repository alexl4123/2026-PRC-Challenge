"""LIRF: does the FLIGHT_mvt string format (ICAO prefix + letter, e.g. 'ITYTY672', 'RYRR90BN') mark a record source with
schedule-date stamping?  Classes on zone P (stamp / day-shift / normal) and the date-of-schedule mechanism:
day-shift = actual off-block time of day written on the SCHED (local) date."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load
pd.set_option('display.width', 250)
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P, D, f = V['y'], V['month'], V['P'], V['D'], V['f']
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet',
                    columns=['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'off_sched', 'y'])
d['f'] = f; d['grp'] = np.where(P, 'P', np.where(D, 'D', 'A')); d['month'] = month
L = d[d.APT == 'LIRF'].copy()
s = L.FLIGHT_mvt.astype(object).fillna('').astype(str)
L['dbl'] = s.str.match(r'^[A-Z]{3}[A-Z]') & (s.str.len() >= 5)
L['dbl2'] = [(len(x) >= 5 and x[:3].isalpha() and x[3].isalpha() and x[3] in x[1:3]) for x in s]
tz = 'Europe/Rome'
for c, n in (('MVT_TIME_UTC_mvt', 'mL'), ('SCHED_TIME_UTC_mvt', 'sL'), ('BLOCK_TIME_UTC_mvt', 'bL')):
    L[n] = L[c].dt.tz_convert(tz).dt.tz_localize(None)
L['stamp'] = np.abs((L.BLOCK_TIME_UTC_mvt - L.SCHED_TIME_UTC_mvt).dt.total_seconds()) <= 60
L['cls'] = np.where(L.stamp, 'stamp', np.where(L.y > 43200, 'dayshift', np.where(L.y <= 7200, 'normal', 'other')))
# truth-side check of the mechanism: for day-shift rows, BLOCK date == SCHED date?
L['b_eq_s_date'] = L.bL.dt.normalize() == L.sL.dt.normalize()
L['ddate'] = ((L.mL - pd.to_timedelta(900, 's')).dt.normalize() - L.sL.dt.normalize()).dt.days
print('LIRF departures', len(L), ' doubled-format share', round(L.dbl.mean(), 4), ' dbl2', round(L.dbl2.mean(), 4))
print(pd.crosstab([L.grp, L.dbl], L.cls, margins=True))
print('\nzone P LIRF: class x (doubled, ddate)'); P_ = L[L.grp == 'P']
print(pd.crosstab([P_.dbl, P_.ddate], P_.cls, margins=True))
print('\nzone P LIRF day-shift/normal rows: BLOCK local date == SCHED local date?')
print(pd.crosstab(P_.cls, P_.b_eq_s_date))
print('\nall LIRF: y>43200 by grp/dbl:'); print(pd.crosstab([L.grp, L.dbl], L.y > 43200))
print('\nnormal-class zone P rows with ddate=1 (sample):')
print(P_[(P_.cls == 'normal') & (P_.ddate == 1)][['FLIGHT_mvt', 'dbl', 'mL', 'sL', 'bL', 'y', 'off_sched', 'f']].head(25).to_string())
print('\ndoubled-format examples by class:')
for c in ('stamp', 'dayshift', 'normal'):
    print(c, P_[P_.cls == c].FLIGHT_mvt.astype(str).head(30).tolist())
# month profile of doubled format share (all LIRF departures) -- does the record source change over the year?
print('\ndoubled share by month (all LIRF deps):', L.groupby('month').dbl.mean().round(3).to_dict())
r = load('ranking', phase='DEP', columns=['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt'])
r = r[r.APT == 'LIRF']; rs = r.FLIGHT_mvt.astype(object).fillna('').astype(str)
print('2026 LIRF deps', len(r), ' doubled share by month', (rs.str.match(r'^[A-Z]{3}[A-Z]') & (rs.str.len() >= 5)).groupby(r.MVT_TIME_UTC_mvt.dt.month.values).mean().round(3).to_dict())
L[['FLIGHT_mvt', 'grp', 'dbl', 'cls', 'ddate', 'y', 'off_sched', 'f', 'month']].to_parquet(CACHE / 'hunt' / 'hedge_lirf.parquet')
