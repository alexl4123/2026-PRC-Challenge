"""EXP02 -- What exactly is AOBT_3_flt, and how does it relate to the hidden target?

Three timestamps matter:
  MVT_TIME_UTC_mvt    = wheels-off (takeoff).  GIVEN in the ranking set.
  BLOCK_TIME_UTC_mvt  = off-block (pushback), as recorded BY THE AIRPORT.  HIDDEN.
  AOBT_3_flt          = off-block (pushback), as recorded by EUROCONTROL Network
                        Manager on the flown (M3) trajectory.  GIVEN, ~98.5%.

BLOCK_TIME and AOBT_3 are the SAME PHYSICAL EVENT measured by two different
systems. So MVT_TIME - AOBT_3 is an independent estimate of the target that the
organizers left in the file.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.io import load
from prc.validation import load_training, load_ranking, rmse
pd.set_option('display.width', 200)

tr, rk = load_training(), load_ranking()
d = tr[tr.PHASE_mvt.eq('DEP')].copy()
rd = rk[rk.PHASE_mvt.eq('DEP')].copy()

print('=== 1. target identity check ===')
chk = (d.MVT_TIME_UTC_mvt - d.BLOCK_TIME_UTC_mvt).dt.total_seconds() - d.TAXITIME_SEC_mvt
print(f'  max |(MVT_TIME - BLOCK_TIME) - TAXITIME_SEC| = {chk.abs().max():.6f} s over {len(d)} rows')
print('  -> TAXITIME_SEC_mvt is EXACTLY (takeoff - offblock), in seconds. Hence RMSE is in seconds.')

print('\n=== 2. the two pushback measurements disagree by how much? ===')
m = d.dropna(subset=['AOBT_3_flt']).copy()
m['disagree'] = (m.BLOCK_TIME_UTC_mvt - m.AOBT_3_flt).dt.total_seconds()
print(f'  coverage: AOBT_3 present for {len(m)/len(d)*100:.2f}% of training DEP, '
      f'{rd.AOBT_3_flt.notna().mean()*100:.2f}% of ranking DEP')
print('  BLOCK_TIME - AOBT_3 (seconds):')
print(m.disagree.describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())
print(f'  within +-60s: {(m.disagree.abs()<=60).mean()*100:.1f}%   '
      f'+-180s: {(m.disagree.abs()<=180).mean()*100:.1f}%   '
      f'+-300s: {(m.disagree.abs()<=300).mean()*100:.1f}%')

print('\n=== 3. so how good is the free estimate MVT_TIME - AOBT_3 on its own? ===')
est = (m.MVT_TIME_UTC_mvt - m.AOBT_3_flt).dt.total_seconds()
y = m.TAXITIME_SEC_mvt.astype(float)
keep = y <= 7200          # the regime the 2026 test set is in (see EXP01)
print(f'  RMSE(MVT-AOBT3)  all rows      : {rmse(y, est):8.2f} s')
print(f'  RMSE(MVT-AOBT3)  y<=7200 rows  : {rmse(y[keep], est[keep]):8.2f} s')
print(f'  RMSE(global mean) y<=7200 rows : {rmse(y[keep], y[keep].mean()):8.2f} s')
print(f'  correlation with target (y<=7200): {np.corrcoef(y[keep], est[keep])[0,1]:.4f}')

print('\n=== 4. per-airport bias of the free estimate (BLOCK - AOBT_3, seconds) ===')
print(m[keep].groupby('APT').disagree.agg(['mean', 'std', 'median']).round(1).to_string())
print('\n  -> the bias is airport-specific and large (LTFM +236s vs LIRF -214s),')
print('     which is exactly what a tree with APT + off_aobt3 can absorb.')
