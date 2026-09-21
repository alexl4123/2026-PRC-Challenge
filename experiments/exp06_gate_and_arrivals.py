"""EXP06 -- two questions.

(1) HOW NOISY IS OUR LOCAL GATE?
    v1: local uncapped 563.89 -> official 543.67   (local 20s PESSIMISTIC)
    v2: local uncapped 466.37 -> official 507.55   (local 41s OPTIMISTIC)
    The two calibration points disagree in direction, which is a warning sign.
    If the holdout's ~150 tail rows dominate its RMSE, the gate has huge
    sampling variance and small local differences mean nothing. Bootstrap it.

(2) CAN ARRIVALS SUPPLY MORE TAIL EXAMPLES?
    We cannot get per-flight taxi times for earlier years -- that data is not
    public. But 2.08M ARRIVALS in training and 344,693 in the RANKING set are
    fully labelled and completely unused. If the BLOCK==SCHED stamping artifact
    is a property of the airport's reporting system rather than of departures,
    arrivals carry the same signal and multiply our tail sample.
    The ranking arrivals are especially valuable: they are the ACTUAL SCORING
    MONTHS, with labels.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.io import load, peak_rss_gb
from prc.validation import rmse
pd.set_option('display.width', 200)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- (1) gate noise
print('=' * 78)
print('(1) BOOTSTRAP NOISE OF THE LOCAL HOLDOUT GATE')
print('=' * 78)
tr = load('training', phase='DEP',
          columns=['APT', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                   'AOBT_3_flt', 'BLOCK_TIME_UTC_mvt'])
m = tr.MVT_TIME_UTC_mvt.dt.month
val = m.isin([1, 7]).to_numpy()
y = tr.TAXITIME_SEC_mvt.astype(float).to_numpy()
yv = y[val]
# a cheap stand-in predictor so the bootstrap measures GATE noise, not model noise
pv = np.full(len(yv), y[~val].mean())
n = len(yv)
boot = np.array([rmse(yv[i], pv[i]) for i in
                 (rng.integers(0, n, n) for _ in range(200))])
print(f'  holdout n={n:,}, tail rows (y>7200) = {int((yv>7200).sum())}')
print(f'  uncapped RMSE point estimate = {rmse(yv,pv):.1f}')
print(f'  bootstrap mean {boot.mean():.1f}  sd {boot.std():.1f}  '
      f'95% CI [{np.percentile(boot,2.5):.0f}, {np.percentile(boot,97.5):.0f}]')
print(f'  => width of the 95% CI is {np.percentile(boot,97.5)-np.percentile(boot,2.5):.0f}s. '
      'Local differences smaller than that are NOISE.')
capped = yv <= 7200
bootc = np.array([rmse(yv[capped][i], pv[capped][i]) for i in
                  (rng.integers(0, capped.sum(), capped.sum()) for _ in range(200))])
print(f'  for contrast, CAPPED RMSE: {rmse(yv[capped],pv[capped]):.1f}  sd {bootc.std():.2f}  '
      '<- stable, but measures the wrong quantity')

# ---------------------------------------------------------- (2) arrivals as data
print('\n' + '=' * 78)
print('(2) DO ARRIVALS CARRY THE SAME BLOCK==SCHED ARTIFACT?')
print('=' * 78)
del tr
cols = ['APT', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
        'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt']
for label, (which, phase) in [('train DEP', ('training', 'DEP')),
                              ('train ARR', ('training', 'ARR')),
                              ('RANK  ARR', ('ranking', 'ARR'))]:
    d = load(which, phase=phase, columns=cols)
    yy = d.TAXITIME_SEC_mvt.astype(float)
    b = (d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    art = b.abs() <= 1
    miss = d.AOBT_3_flt.isna()
    tail = yy > 7200
    print(f'\n  {label}: n={len(d):,}')
    print(f'    |BLOCK-SCHED|<=1s        : {art.sum():7,} ({art.mean()*100:5.2f}%)')
    print(f'    AOBT_3 missing           : {miss.sum():7,} ({miss.mean()*100:5.2f}%)')
    print(f'    tail rows y>7200         : {int(tail.sum()):7,} ({tail.mean()*100:.4f}%)')
    if tail.sum():
        print(f'    tail AND AOBT_3 missing  : {int((tail&miss).sum()):7,} '
              f'({(tail&miss).sum()/max(tail.sum(),1)*100:.0f}% of tail)')
    del d

print('\n' + '=' * 78)
print('EXTRA TAIL EXAMPLES AVAILABLE FROM ARRIVALS')
print('=' * 78)
for which, phase in [('training', 'ARR'), ('ranking', 'ARR')]:
    d = load(which, phase=phase, columns=['TAXITIME_SEC_mvt'])
    print(f'  {which:9s} {phase}: {int((d.TAXITIME_SEC_mvt>7200).sum()):5,} tail rows, '
          f'{len(d):,} labelled rows total')
    del d
print(f'\npeak RSS = {peak_rss_gb():.2f} GB')
