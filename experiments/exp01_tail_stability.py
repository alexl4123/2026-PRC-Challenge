"""EXP01 -- Is the extreme-tail rate of the taxi-time label stable year over year?

WHY THIS EXPERIMENT EXISTS
--------------------------
The claim under test is: "the scored 2026 DEPARTURE labels cannot have a 2025-like
extreme tail, because 149 such rows in Jan+Jul 2025 would impose an oracle RMSE
floor of ~391s and the leaderboard leader scores 239.13s."

That argument assumes the 2026 tail *should* have looked like the 2025 tail. We
cannot check that on departures -- their labels are blanked. But ARRIVALS are
fully labelled in BOTH training (2025) and ranking (2026). So arrivals give us a
free, ground-truth test of the assumption itself:

  * If the arrival tail rate is STABLE 2025 -> 2026, then tails are a persistent
    property of the data feed. The departure tail should have persisted too, the
    oracle floor should bind, and 239.13 would be impossible => something is wrong
    with my reasoning (or with my model of how scoring works).

  * If the arrival tail rate is VOLATILE 2025 -> 2026, then tails are event-driven
    and a 2026 departure set with a much lighter tail is entirely ordinary.

Usage: /home/thinklex/prc2026-venv/bin/python src/exp01_tail_stability.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.io import load
from prc.validation import load_training, load_ranking, oracle_floor, AIRPORTS

pd.set_option('display.width', 200)

tr, rk = load_training(), load_ranking()
tr['month'] = tr.MVT_TIME_UTC_mvt.dt.month
rk['month'] = rk.MVT_TIME_UTC_mvt.dt.month

# Comparable slices: Jan+Jul only, so seasonality is held fixed.
a25 = tr[(tr.PHASE_mvt.eq('ARR')) & (tr.month.isin([1, 7]))]
a26 = rk[rk.PHASE_mvt.eq('ARR')]
d25 = tr[(tr.PHASE_mvt.eq('DEP')) & (tr.month.isin([1, 7]))]

print('=' * 78)
print('A.  ARRIVAL tail, Jan+Jul 2025  vs  Jan+Jul 2026   (labels known for BOTH)')
print('=' * 78)
rows = []
for name, s in [('ARR 2025 Jan+Jul', a25.TAXITIME_SEC_mvt), ('ARR 2026 Jan+Jul', a26.TAXITIME_SEC_mvt)]:
    y = s.astype(float).values
    rows.append(dict(slice=name, n=len(y), mean=y.mean(), std=y.std(),
                     n_gt_3600=(y > 3600).sum(), n_gt_7200=(y > 7200).sum(),
                     max=y.max(),
                     floor_3600=oracle_floor(y, 3600), floor_7200=oracle_floor(y, 7200)))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print('\n  -> ratio of oracle floors (2026 / 2025):',
      round(rows[1]['floor_7200'] / max(rows[0]['floor_7200'], 1e-9), 3))

print('\n' + '=' * 78)
print('B.  Same, per airport (oracle floor at 7200s, seconds)')
print('=' * 78)
per = pd.DataFrame({
    'n25': a25.groupby('APT').size(),
    'n26': a26.groupby('APT').size(),
    'floor25': a25.groupby('APT').TAXITIME_SEC_mvt.apply(lambda s: oracle_floor(s, 7200)),
    'floor26': a26.groupby('APT').TAXITIME_SEC_mvt.apply(lambda s: oracle_floor(s, 7200)),
    'ngt7200_25': a25.groupby('APT').TAXITIME_SEC_mvt.apply(lambda s: (s > 7200).sum()),
    'ngt7200_26': a26.groupby('APT').TAXITIME_SEC_mvt.apply(lambda s: (s > 7200).sum()),
})
print(per.round(1).to_string())

print('\n' + '=' * 78)
print('C.  How volatile is the tail MONTH TO MONTH within 2025?')
print('    (if a single year swings wildly, a 2026 slice being mild is unremarkable)')
print('=' * 78)
for phase in ['ARR', 'DEP']:
    g = tr[tr.PHASE_mvt.eq(phase)].groupby('month').TAXITIME_SEC_mvt
    f = g.apply(lambda s: oracle_floor(s, 7200))
    n = g.apply(lambda s: (s > 7200).sum())
    print(f'\n  {phase}: oracle floor @7200 by month of 2025 (s)')
    print('   ', {int(k): round(v, 1) for k, v in f.items()})
    print(f'  {phase}: count of rows > 7200s by month')
    print('   ', {int(k): int(v) for k, v in n.items()})
    print(f'  {phase}: min={f.min():.1f}  max={f.max():.1f}  ratio max/min={f.max()/max(f.min(),1e-9):.1f}x')

print('\n' + '=' * 78)
print('D.  THE ACTUAL ARGUMENT, restated with numbers')
print('=' * 78)
y = d25.TAXITIME_SEC_mvt.astype(float).values
print(f'  DEP Jan+Jul 2025: n={len(y)}, rows>7200s = {(y>7200).sum()}, max = {y.max():.0f}s')
print(f'  oracle floor @7200 = {oracle_floor(y,7200):.1f}s      <- no model can beat this')
print(f'  oracle floor @3600 = {oracle_floor(y,3600):.1f}s')
print(f'  leaderboard #1     = 239.13s')
print(f'  => a 2026 DEP set with a 2025-like tail is excluded by a factor of '
      f'{oracle_floor(y,7200)/239.13:.2f}x')
