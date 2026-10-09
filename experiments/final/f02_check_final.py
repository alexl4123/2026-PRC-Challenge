"""f02 -- pre-upload checks of outputs_final/understanding-accordion_final.parquet (no labels involved).

1. VALIDATION: prc.submit.validate (the official checks: 670,790 departures, ids, dtype, no NaN).
2. REPLAY on Jan + Jul: the final file against the scored v29 on the 344,841 departures both ranking files share
   (same MVT_ID). Only the +-6 s re-dither of the movement times separates the two inputs, so the files must agree
   closely: RMS difference, share of rows moved > 60 s, per route / zone P; any large block of disagreement = build error.
3. NEW MONTHS: per month mean prediction, route-D share, zone-P count per day and the share of rows the LOBT window
   moved, next to the same month of 2025 (labels y and the 2025 out-of-fold prediction of the v28 structure).
   Feb and Jun 2026 must look like Feb/Jun 2025 does relative to its own Jan/Jul.

    PRC_RANKING=final python experiments/final/f02_check_final.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import FINAL, CACHE, CACHE_BASE, OUTPUTS, REPO, ID
from prc.submit import validate

if not FINAL:
    sys.exit('run with PRC_RANKING=final')
TEAM = 'understanding-accordion'
path = OUTPUTS / f'{TEAM}_final.parquet'
print('== 1. validation');  validate(path)
f = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].astype(float)
d = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=[ID, 'APT', 'month', 'off_aobt3', 'off_sched', 'off_lobt'])
d['pred'] = f.reindex(d[ID]).to_numpy()
if d.pred.isna().any():
    sys.exit('FATAL: final file misses ranking departures')
d['D'] = ~np.isfinite(d.off_aobt3)
d['P'] = d.D & (d.off_sched > 7200)

print('\n== 2. replay vs v29 on the shared Jan/Jul departures')
v = pd.read_parquet(REPO / 'outputs' / f'{TEAM}_v29.parquet').set_index(ID)['TAXITIME_SEC_mvt'].astype(float)
s = d[d[ID].isin(v.index)].copy(); s['v29'] = v.reindex(s[ID]).to_numpy(); s['dd'] = s.pred - s.v29
print(f'shared {len(s):,} of v29 {len(v):,}')
for nm, k in [('all', np.ones(len(s), bool)), ('route A/B/C', ~s.D), ('route D not P', s.D & ~s.P), ('zone P', s.P)]:
    x = s.dd[k]
    print(f'  {nm:14s} n {k.sum():7,}  RMS diff {np.sqrt((x**2).mean()):7.2f} s  mean {x.mean():+6.2f}  '
          f'|d|>60s {np.mean(np.abs(x) > 60):.4f}  max {np.abs(x).max():7.0f}')
print('  per airport RMS diff:', ' '.join(f'{a}:{np.sqrt((g.dd**2).mean()):.1f}' for a, g in s.groupby('APT')))

print('\n== 3. per month: 2026 final file vs the same month of 2025')
z = np.load(CACHE_BASE / 'w2' / 'v28s_oof.npz')
t = pd.DataFrame({'month': z['month'], 'y': z['y'], 'oof': z['f'], 'D': z['D'], 'P': z['P']})
days = {1: 31, 2: 28, 6: 30, 7: 31}
rows = []
for m in [1, 2, 6, 7]:
    a, b = d[d.month == m], t[t.month == m]
    nd25 = 28 if m == 2 else days[m]
    rows.append(dict(month=m, n26=len(a), pred26=a.pred.mean(), D26=a.D.mean(), Pday26=a.P.sum() / days[m],
                     n25=len(b), oof25=b.oof.mean(), y25=b.y.mean(), D25=b.D.mean(), Pday25=b.P.sum() / nd25))
print(pd.DataFrame(rows).round(3).to_string(index=False))
lob = ((d.off_lobt - d.pred).abs() >= 3600) & ~d.D     # implied |BLOCK - LOBT| at the window edge (diagnostic only)
print('\nper month: share of A/B/C rows whose prediction sits at the LOBT window edge:',
      ' '.join(f'{m}:{lob[d.month == m].mean():.4f}' for m in [1, 2, 6, 7]))
print('prediction range', d.pred.min(), d.pred.max(), '| rows > 7200 s:', int((d.pred > 7200).sum()))
