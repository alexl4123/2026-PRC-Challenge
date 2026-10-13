"""ids_03 -- LOBT window via FLIGHT_ID bracket for DEP rows that have FLIGHT_ID_mvt but no _flt block.
Exact test on 2025: BLOCK in [lo_t - 3606, hi_t + 3606] where lo_t/hi_t are the LOBTs of the neighbouring
FLIGHT_IDs that do carry a LOBT (FLIGHT_ID is strictly LOBT-ordered: 0 inversions on 4.4M ids)."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load

H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'PHASE_mvt', 'LOBT_flt', 'MVT_TIME_UTC_mvt']
fr = []
for which in ['training', 'ranking']:
    d = load(which, columns=C); d['src'] = which; fr.append(d)
d = pd.concat(fr, ignore_index=True); del fr
a = d[d.FLIGHT_ID_mvt.notna() & d.LOBT_flt.notna()].drop_duplicates('FLIGHT_ID_mvt').sort_values('FLIGHT_ID_mvt')
fid = a.FLIGHT_ID_mvt.values
lob = a.LOBT_flt.values.astype('datetime64[s]').astype('int64')

q = pd.read_parquet(H + 'cache/hunt/ids_02_noflt.parquet')
q = q[q.PHASE_mvt.eq('DEP')].copy()
pos = np.searchsorted(fid, q.FLIGHT_ID_mvt.values)
q['lo_t'] = np.where(pos > 0, lob[np.clip(pos - 1, 0, None)], np.nan)
q['hi_t'] = lob[np.clip(pos, 0, len(fid) - 1)]
q['lo_id'] = fid[np.clip(pos - 1, 0, None)]
q['hi_id'] = fid[np.clip(pos, 0, len(fid) - 1)]
# y bounds implied: BLOCK >= lo_t - 3606  ->  y <= MVT - lo_t + 3606 ; BLOCK <= hi_t + 3606 -> y >= MVT - hi_t - 3606
q['y_max'] = q.MVT_T - q.lo_t + 3606
q['y_min'] = q.MVT_T - q.hi_t - 3606
tr = q[q.src.eq('training')]
ok = (tr.y <= tr.y_max) & (tr.y >= tr.y_min)
print('TRAINING DEP rows with FLIGHT_ID & no LOBT:', len(tr), ' consistent with bracket window:', ok.sum(),
      ' (y<=y_max:', (tr.y <= tr.y_max).sum(), ')')
print(tr[['APT', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'y', 'y_min', 'y_max']].assign(
    lo_gap=(tr.y_max - tr.y)).sort_values('lo_gap').head(10).to_string())

# 2026 rows + v28 prediction
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet')
rk = q[q.src.eq('ranking')].merge(v, on='MVT_ID_mvt', how='left')
rk['off_sched'] = rk.MVT_T - rk.SCHED
print('\nRANKING DEP rows with FLIGHT_ID & no LOBT (v28 pred = TAXITIME_SEC_mvt):')
print(rk[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'off_sched',
          'TAXITIME_SEC_mvt', 'y_min', 'y_max', 'lo_id', 'hi_id', 'FLIGHT_ID_mvt']].to_string())

# --- for the July-1 rows (no lower anchor in the data), how much LOBT time do N FLIGHT_IDs span just before
# the first post-midnight anchor? Calibrate on every night available (2025 all months, 2026 Jan/Jul):
# for each anchor pair around 23:00-00:30 UTC, seconds per id.
t = pd.to_datetime(lob, unit='s')
hod = t.hour + t.minute / 60
sec_per_id = np.diff(lob) / np.diff(fid)
mid = (hod[:-1] >= 22.5) & (hod[:-1] < 24)
print('\nsec per FLIGHT_ID between consecutive anchors with LOBT 22:30-24:00 UTC: quantiles',
      np.quantile(sec_per_id[mid], [.1, .25, .5, .75, .9]).round(3))
# empirical: for anchors with LOBT in [23:00, 24:00), the number of ids between them and the first anchor at/after
# the next midnight, vs the time difference -> regression time_to_midnight ~ ids_to_first_post_midnight_anchor
df = pd.DataFrame({'fid': fid, 'lob': lob})
df['day'] = df.lob // 86400
first_after = df.groupby('day').agg(f0=('fid', 'min'), t0=('lob', 'min'))  # first anchor of each UTC day
df = df.join(first_after.rename(lambda x: x - 1), on='day')  # first anchor of the NEXT day
late = df[(df.lob % 86400) >= 20 * 3600].dropna()
late['ids_to_next'] = late.f0 - late.fid
late['sec_to_next'] = late.t0 - late.lob
late['month'] = pd.to_datetime(late.lob, unit='s').dt.month
late['year'] = pd.to_datetime(late.lob, unit='s').dt.year
for (yy, mm), g in late.groupby(['year', 'month']):
    if mm not in (6, 7, 8, 12, 1):
        continue
    for lo_, hi_ in [(0, 200), (200, 600), (600, 1200), (1200, 2500), (2500, 5000)]:
        s = g[(g.ids_to_next >= lo_) & (g.ids_to_next < hi_)].sec_to_next
        if len(s):
            print(f'  {yy}-{mm:02d} ids_to_next [{lo_},{hi_}): n={len(s)} sec_to_next q05/med/q95 '
                  f'{np.quantile(s, .05):.0f} / {np.median(s):.0f} / {np.quantile(s, .95):.0f}')
rk.to_parquet(H + 'cache/hunt/ids_03_rank_noflt.parquet')
