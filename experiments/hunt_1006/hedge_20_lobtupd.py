"""Mechanism check for rules B / T: in NM, is the LOBT of a long-delayed or UTC-date-crossing flight always updated?
Uses ARRIVAL rows (their NM block describes the flight's departure: IOBT = initial, LOBT = last filed, AOBT_3 = actual).
'Not updated' = |AOBT_3 - LOBT| > 3606 s (a correct BLOCK would then fail the departure join).
Also the LIRF departure base rates: share of route D among on-time departures, and counts of NM-matched delayed rows."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.io import load
for which in ('training', 'ranking'):
    a = load(which, phase='ARR', columns=['ADEP_mvt', 'IOBT_flt', 'LOBT_flt', 'AOBT_3_flt'])
    a = a[a.LOBT_flt.notna() & a.IOBT_flt.notna()]
    delay = (a.AOBT_3_flt - a.IOBT_flt).dt.total_seconds()
    nu = (a.AOBT_3_flt - a.LOBT_flt).dt.total_seconds().abs() > 3606
    cross = (a.AOBT_3_flt - pd.to_timedelta(0, 's')).dt.normalize() > a.IOBT_flt.dt.normalize()
    rome = a.ADEP_mvt.astype(str) == 'LIRF'
    print(which, 'arrivals with NM', len(a))
    for lo, hi in ((7200, 14400), (14400, 28800), (28800, 1e7)):
        for nm, m in (('all origins', np.ones(len(a), bool)), ('from LIRF', rome.to_numpy())):
            for cn, c in (('no UTC cross', ~cross.to_numpy()), ('UTC cross', cross.to_numpy())):
                k = m & c & (delay > lo).to_numpy() & (delay <= hi).to_numpy()
                print(f'   delay ({lo},{hi:.0f}] {nm:11s} {cn:12s}: n {int(k.sum()):6d}  LOBT not updated {nu.to_numpy()[k].mean() if k.any() else np.nan:.3f}')
d = load('training', phase='DEP', columns=['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt'])
d = d[d.APT == 'LIRF']
os_ = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
du = ((d.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
nm = d.LOBT_flt.notna()
ot = (os_ > -600) & (os_ < 1800)
print(f'\nLIRF 2025 departures: route-D share among on-time rows {1 - nm[ot].mean():.4f} (n {int(ot.sum())})')
for nm_, m in (('UTC crossing & off_sched>7200', (du >= 1) & (os_ > 7200)), ('off_sched>28800', os_ > 28800),
               ('no crossing, off_sched 7200-14400', (du < 1) & (os_ > 7200) & (os_ <= 14400))):
    print(f'  {nm_}: NM-matched {int((m & nm).sum())}, route D {int((m & ~nm).sum())}')
