"""Day-scale label rows in 2025: what are they? Local-date hypothesis tests."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam', 'LEBL': 'Europe/Madrid',
      'LEMD': 'Europe/Madrid', 'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome', 'LSZH': 'Europe/Zurich', 'LTFM': 'Europe/Istanbul'}
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'FLIGHT_RULE_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT', 'BLOCK', 'SCHED', 'LOBT', 'AOBT3', 'EOBT1'])
d['APT'] = d.APT.astype(str)
y = d.MVT - d.BLOCK; d['y'] = y
off = np.zeros(len(d))
for a, tz in TZ.items():
    m = (d.APT == a).to_numpy()
    for c in ['MVT', 'BLOCK', 'SCHED']:
        t = pd.to_datetime(d.loc[m, c], unit='s', utc=True).dt.tz_convert(tz)
        d.loc[m, c + '_ld'] = (t.dt.tz_localize(None).dt.normalize() - pd.Timestamp('2025-01-01')).dt.days.to_numpy()
        d.loc[m, c + '_tod'] = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy()
    t = pd.to_datetime(d.loc[m, 'MVT'] - 600, unit='s', utc=True).dt.tz_convert(tz)
    d.loc[m, 'P_ld'] = (t.dt.tz_localize(None).dt.normalize() - pd.Timestamp('2025-01-01')).dt.days.to_numpy()
for c in ['MVT', 'BLOCK', 'SCHED']:
    d[c + '_ud'] = np.floor(d[c] / 86400) - (pd.Timestamp('2025-01-01').value // 10**9) // 86400
d['route'] = np.where(d.LOBT.notna(), 'ABC', 'D')
d['off_sched'] = d.MVT - d.SCHED
big = d[d.y > 20000].copy()
big['dl_SB'] = big.BLOCK_ld - big.SCHED_ld; big['dl_MB'] = big.MVT_ld - big.BLOCK_ld; big['dl_MS'] = big.MVT_ld - big.SCHED_ld
big['todB-todM'] = big.BLOCK_tod - big.MVT_tod; big['B-S'] = big.BLOCK - big.SCHED
print('2025 departures y > 20000:', len(big))
print(big[['APT', 'route', 'FLIGHT_mvt', 'y', 'off_sched', 'B-S', 'dl_SB', 'dl_MB', 'dl_MS', 'todB-todM', 'BLOCK_tod', 'SCHED_tod', 'MVT_tod']].sort_values(['APT', 'y']).to_string())
# Hypothesis H1: BLOCK local date = SCHED local date, time of day of BLOCK = real.
# Population: departures where SCHED local date < local date of (MVT - 600)  (crossed local midnight between sched and takeoff)
cr = d.P_ld > d.SCHED_ld
print('\nrows crossing local midnight between SCHED and MVT-600:', cr.sum())
for (a, r), g in d[cr].groupby(['APT', 'route']):
    print(f'  {a} {r}: n={len(g):5d}  y>80000: {(g.y > 80000).sum():3d}   y in (20000,80000]: {((g.y > 20000) & (g.y <= 80000)).sum():3d}   BLOCK==SCHED: {(g.BLOCK == g.SCHED).sum():4d}  BLOCK ld == SCHED ld: {(g.BLOCK_ld == g.SCHED_ld).sum():5d}')
