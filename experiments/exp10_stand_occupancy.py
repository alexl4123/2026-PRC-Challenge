"""EXP10 -- stand occupancy gives a HARD UPPER BOUND on taxi-out.

An aircraft cannot push back before it parked. The previous ARRIVAL at the same
stand went in-block at time A; the departure we are predicting pushed back at
BLOCK >= A; so

    y = MVT_TIME - BLOCK  <=  MVT_TIME - A

That is a bound, not a correlation. And every ingredient is present in
ranking.parquet: arrivals there are fully labelled, including BLOCK_TIME.

This matters most exactly where we are losing. A claimed 20-hour taxi-out is
only possible if the stand really was occupied by that aircraft for 20 hours. If
something else arrived at the stand two hours ago, the tail value is impossible
and we should not predict one. Conversely a genuinely long occupancy makes a
tail value plausible.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.io import load, peak_rss_gb
pd.set_option('display.width', 210)

t0 = time.time()
cols = ['APT', 'STAND_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
        'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'AOBT_3_flt']
full = load('training', columns=cols)          # both phases: arrivals are the input
# Epoch seconds, unit- and tz-safe. NOTE: these columns are timestamp[us], so
# `.astype('int64')` yields MICROseconds -- dividing by 1e9 (the nanosecond
# assumption) silently makes every duration 1000x too small. Subtracting a
# Timestamp and using .dt.total_seconds() avoids the whole trap.
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
full['t_mvt'] = (full.MVT_TIME_UTC_mvt - EPOCH).dt.total_seconds()
full['t_blk'] = (full.BLOCK_TIME_UTC_mvt - EPOCH).dt.total_seconds()
full['key'] = full.APT.astype(str) + '|' + full.STAND_mvt.astype(str)

arr = full[full.PHASE_mvt.eq('ARR')][['key', 't_blk']].dropna().sort_values('t_blk')
dep = full[full.PHASE_mvt.eq('DEP')].copy().sort_values('t_mvt').reset_index(drop=True)
print(f'departures {len(dep):,}   arrivals {len(arr):,}   ({time.time()-t0:.0f}s)')

# most recent arrival in-block at the same stand, strictly before this takeoff
prev = pd.merge_asof(dep[['key', 't_mvt']].assign(_i=np.arange(len(dep))),
                     arr.rename(columns={'t_blk': 'prev_arr'}),
                     left_on='t_mvt', right_on='prev_arr', by='key', direction='backward',
                     allow_exact_matches=False)
dep['prev_arr'] = prev.sort_values('_i')['prev_arr'].to_numpy()
dep['max_taxi'] = dep.t_mvt - dep.prev_arr          # hard upper bound on y
y = dep.TAXITIME_SEC_mvt.astype(float).to_numpy()
mt = dep.max_taxi.to_numpy()
off_sched = (dep.MVT_TIME_UTC_mvt - dep.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()

have = np.isfinite(mt)
print(f'\nbound available for {have.mean()*100:.1f}% of departures')
viol = have & (y > mt + 60)
print(f'bound violated (y > max_taxi + 60s) for {viol.sum():,} rows '
      f'({viol.mean()*100:.3f}%) -- these are stand-id reuse or data noise')

print('\n--- how TIGHT is the bound? ratio y / max_taxi ---')
ok = have & (mt > 0) & ~viol
r = y[ok] / mt[ok]
print(pd.Series(r).describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())

print('\n--- the bound on TAIL vs BULK rows ---')
tail = y > 7200
for nm, sel in [('bulk y<=7200', have & ~tail), ('tail y>7200', have & tail)]:
    s = pd.Series(mt[sel])
    print(f'  {nm:14s} n={int(sel.sum()):8,}  max_taxi median {s.median():9.0f}s  '
          f'p05 {s.quantile(.05):8.0f}  p95 {s.quantile(.95):9.0f}')

print('\n--- DOES THE BOUND SEPARATE TAIL FROM BULK? ---')
for thr in [7200, 14400, 43200]:
    big = have & (mt > thr)
    print(f'  max_taxi > {thr:6d}s : n={int(big.sum()):8,} ({big.mean()*100:5.2f}%)  '
          f'P(tail|.) = {tail[big].mean()*100:6.3f}%   captures '
          f'{int((tail&big).sum()):3d}/{int((tail&have).sum())} of the tail')
base = tail[have].mean() * 100
print(f'  base rate P(tail) = {base:.4f}%')

print('\n--- combining the bound with off_sched ---')
for thr in [7200, 14400]:
    sel = have & (mt > thr) & (off_sched > thr)
    print(f'  max_taxi>{thr} AND off_sched>{thr}: n={int(sel.sum()):7,}  '
          f'P(tail)={tail[sel].mean()*100:6.2f}%  captures {int((tail&sel).sum()):3d}')

print('\n--- and CLIPPING predictions at the bound: how many bulk rows would it hurt? ---')
for margin in [0, 300, 900]:
    would_clip = have & (y > 0) & ((mt + margin) < np.maximum(y, 1))
    print(f'  margin {margin:4d}s: clipping at max_taxi+margin would cut '
          f'{int(would_clip.sum()):,} true values ({would_clip.mean()*100:.3f}%)')
np.save('cache/exp10_maxtaxi.npy', np.vstack([mt, y, off_sched]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
