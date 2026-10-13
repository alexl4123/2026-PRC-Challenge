"""Hedge-row audit step 2: anatomy of the day-scale rows and the three zone-P classes (stamp / day-shift / normal).
Fingerprint candidates: local-date difference between SCHED and (MVT - taxi), UTC date difference, hour of day."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250)
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P, D, f = V['y'], V['month'], V['P'], V['D'], V['f']
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet',
                    columns=['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                             'BLOCK_TIME_UTC_mvt', 'off_sched', 'y'])
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam',
      'LEBL': 'Europe/Madrid', 'LEMD': 'Europe/Madrid', 'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome',
      'LSZH': 'Europe/Zurich', 'LTFM': 'Europe/Istanbul'}
d['f'] = f; d['grp'] = np.where(P, 'P', np.where(D, 'D', 'A'))
def loc(s, a):
    return s.dt.tz_localize('UTC').dt.tz_convert(TZ[a]).dt.tz_localize(None) if s.dt.tz is None else s.dt.tz_convert(TZ[a]).dt.tz_localize(None)
print('dtype', d.MVT_TIME_UTC_mvt.dtype)
x = d[d.y > 43200].copy()
rows = []
for a, g in x.groupby('APT', observed=True):
    g = g.copy()
    for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'):
        g[c[:5] + '_L'] = loc(g[c], a)
    rows.append(g)
x = pd.concat(rows)
x['y_minus_day'] = x.y - 86400
x['blk_minus_sched'] = (x.BLOCK_TIME_UTC_mvt - x.SCHED_TIME_UTC_mvt).dt.total_seconds()
print(x[['APT', 'grp', 'FLIGHT_mvt', 'STAND_mvt', 'MVT_T_L', 'SCHED_L', 'BLOCK_L', 'y', 'y_minus_day', 'off_sched', 'f']].sort_values(['APT', 'MVT_T_L']).to_string())

# LIRF zone P: three classes, and the local-date fingerprint
g = d[P & (d.APT == 'LIRF')].copy()
g['MVT_L'] = loc(g.MVT_TIME_UTC_mvt, 'LIRF'); g['SCH_L'] = loc(g.SCHED_TIME_UTC_mvt, 'LIRF')
cls = np.where(np.abs(g.y - g.off_sched) <= 120, 'stamp', np.where(g.y > 43200, 'dayshift', np.where(g.y <= 7200, 'normal', 'other')))
g['cls'] = cls
est_ob = g.MVT_L - pd.to_timedelta(900, 's')
g['ddate_loc'] = (est_ob.dt.normalize() - g.SCH_L.dt.normalize()).dt.days
est_ob_u = g.MVT_TIME_UTC_mvt - pd.to_timedelta(900, 's')
g['ddate_utc'] = (est_ob_u.dt.normalize() - g.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
print('\nLIRF zone P rows', len(g)); print(pd.crosstab(g.cls, g.ddate_loc, margins=True)); print(pd.crosstab(g.cls, g.ddate_utc, margins=True))
print(g[g.cls == 'other'][['FLIGHT_mvt', 'MVT_L', 'SCH_L', 'y', 'off_sched', 'f']].to_string())
# all airports zone P: date change vs class
h = d[P].copy(); h['cls'] = np.where(np.abs(h.y - h.off_sched) <= 120, 'stamp', np.where(h.y > 43200, 'dayshift', np.where(h.y <= 7200, 'normal', 'other')))
h['ddate_utc'] = ((h.MVT_TIME_UTC_mvt - pd.to_timedelta(900, 's')).dt.normalize() - h.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
print('\nall zone P, non-LIRF:'); print(pd.crosstab([h.APT[h.APT != 'LIRF'].astype(str)], h.cls[h.APT != 'LIRF']))
print(pd.crosstab(h.cls[h.APT != 'LIRF'], h.ddate_utc[h.APT != 'LIRF']))
g.to_parquet(CACHE / 'hunt' / 'hedge_lirfP.parquet')
