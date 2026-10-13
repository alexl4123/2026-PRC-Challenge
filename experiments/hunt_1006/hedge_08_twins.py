"""LIRF zone P: label-free fingerprints of stamp / day-shift / normal.
 (1) UTC vs local date crossing between SCHED and (MVT - 900 s);
 (2) 'twin': another LIRF departure with the same normalised flight string whose SCHED falls on the MVT's local date
     (a delayed flight colliding with its next-day namesake);  the doubled string format."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)


def norm(s):
    s = '' if s is None or s != s else str(s)
    if len(s) >= 5 and s[:3].isalpha() and s[3].isalpha():
        for k in (3, 2, 1):
            if s[3:3 + k] == s[3 - k:3]:
                return s[:3] + s[3 + k:]
    return s


def prep(which):
    cols = ['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt']
    if which == 'training':
        cols.append('BLOCK_TIME_UTC_mvt')
    d = load(which, phase='DEP', columns=cols)
    d['pos'] = np.arange(len(d))
    d = d[d.APT == 'LIRF'].copy()
    raw = d.FLIGHT_mvt.astype(object)
    d['fl'] = [norm(x) for x in raw]
    d['dbl'] = [(isinstance(x, str) and norm(x) != x) for x in raw]
    d['mL'] = d.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None)
    d['sL'] = d.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None)
    e = d.MVT_TIME_UTC_mvt - pd.to_timedelta(900, 's')
    d['dd_loc'] = ((d.mL - pd.to_timedelta(900, 's')).dt.normalize() - d.sL.dt.normalize()).dt.days
    d['dd_utc'] = (e.dt.normalize() - d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
    d['off_sched'] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    # twin: same normalised flight + same ADES, SCHED on the local date of the MVT (and not the row itself)
    d['key'] = (d.fl.astype(str) + '|' + d.ADES_mvt.astype(object).fillna('').astype(str)).to_numpy()
    sd = d[['key', 'sL', 'pos']].assign(sdate=d.sL.dt.normalize())
    t = d[['key', 'pos', 'mL', 'sL']].assign(mdate=(d.mL - pd.to_timedelta(900, 's')).dt.normalize())
    j = t.merge(sd, left_on=['key', 'mdate'], right_on=['key', 'sdate'], suffixes=('', '_tw'))
    j = j[j.pos != j.pos_tw]
    d['twin'] = d.pos.isin(j.pos)
    # any same-key departure within +-36 h of this row's MVT (other than itself)
    return d


V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, P, f = V['y'], V['P'], V['f']
d = prep('training')
d['y'] = y[d.pos]; d['f'] = f[d.pos]; d['P'] = P[d.pos]
d['cls'] = np.where(np.abs(d.y - d.off_sched) <= 120, 'stamp', np.where(d.y > 43200, 'dayshift', np.where(d.y <= 7200, 'normal', 'other')))
x = d[d.P]
print('2025 LIRF zone P', len(x))
print(pd.crosstab([x.dd_utc, x.dd_loc], x.cls, margins=True))
print(pd.crosstab([x.dd_loc, x.twin, x.dbl], x.cls, margins=True))
print('all LIRF deps: twin share', round(d.twin.mean(), 4), ' dbl share', round(d.dbl.mean(), 4))
print(pd.crosstab(d.twin, d.dbl))
z = x[(x.dd_loc == 1)].sort_values('off_sched')
print(z[['FLIGHT_mvt', 'fl', 'dbl', 'twin', 'dd_utc', 'mL', 'sL', 'cls', 'y', 'off_sched', 'f']].to_string())
# 2026 rows of the same kind
r = prep('ranking')
c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
inP = c['in_P'].astype(bool)
v = pd.read_parquet(CACHE.parent / 'outputs' / 'understanding-accordion_v28.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
r['P'] = inP[r.pos]; r['v28'] = v.reindex(r.MVT_ID_mvt).to_numpy()
rp = r[r.P]
print('\n2026 LIRF zone P', len(rp)); print(pd.crosstab([rp.dd_utc, rp.dd_loc], [rp.twin, rp.dbl], margins=True))
print(rp[rp.dd_loc >= 1].sort_values('off_sched')[['FLIGHT_mvt', 'dbl', 'twin', 'dd_utc', 'mL', 'sL', 'off_sched', 'v28']].to_string())
d.drop(columns=['mL', 'sL']).to_parquet(CACHE / 'hunt' / 'hedge_lirf_tw25.parquet')
r.drop(columns=['mL', 'sL']).to_parquet(CACHE / 'hunt' / 'hedge_lirf_tw26.parquet')
