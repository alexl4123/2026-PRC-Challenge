"""r09 -- LOMO price of a third anchor (day-shift: y = 86,400 + taxi) in the LIRF date-change zone. Read-only.
CAVEAT: the cell was chosen AFTER seeing r07/r08 (post hoc); this is a pricing screen, not a shipping test.
CELL Q = LIRF, takeoff local date > schedule local date, off_sched > 50,000 s (any route).
RULE written before running: prediction in Q for month m := q_m * (86,400 + 1,000) + (1 - q_m) * v15~,
  q_m = share of day-shift labels (y in [86,460, 90,000]) among Q rows of the OTHER 11 months.
  'Promising' if the 12-month paired-bootstrap CI is below 0 and the scored months improve.
Also: drop-top-k rows sensitivity and the 2026 rows it would change (no labels).
    python experiments/research_0925/r09_day_shift_lomo.py --v15s <dir>/v15s_oof.npy
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.validation import paired_bootstrap
ap = argparse.ArgumentParser(); ap.add_argument('--v15s', required=True); a = ap.parse_args()
cols = ['MVT_ID_mvt', 'APT', 'month', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'off_sched', 'off_aobt3']


def cell(d):
    mv = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert('Europe/Rome').dt.normalize()
    sc = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.tz_convert('Europe/Rome').dt.normalize()
    return ((d.APT.astype(str) == 'LIRF') & ((mv - sc).dt.days >= 1) & (d.off_sched > 50000)).to_numpy()


tr = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=cols + ['y'])
y = tr.y.to_numpy().astype(float); p = np.load(a.v15s).astype(float); mon = tr.month.to_numpy()
Q = cell(tr); ds = (y >= 86460) & (y <= 90000)
c = p.copy(); ANCH = 86400 + 1000
for m in range(1, 13):
    k = Q & (mon == m); o = Q & (mon != m)
    if k.sum():
        q = ds[o].mean(); c[k] = q * ANCH + (1 - q) * p[k]
print(f'Q rows 2025: {Q.sum()}, day-shift {int((Q & ds).sum())}; LOMO q range {min(ds[Q & (mon != m)].mean() for m in range(1, 13)):.2f}'
      f'-{max(ds[Q & (mon != m)].mean() for m in range(1, 13)):.2f}')
S = np.isin(mon, [1, 7]); al = np.ones(len(y), bool)
r = lambda f, k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
for scope, k in (('12m', al), ('scored', S)):
    bb = paired_bootstrap(y[k], c[k], p[k], n_boot=2000)
    print(f'  {scope}: {r(p, k):.2f} -> {r(c, k):.2f}  {bb["diff"]:+.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
per = [r(c, mon == m) - r(p, mon == m) for m in range(1, 13)]
print('  per month: ' + ' '.join(f'm{m}:{v:+.2f}' for m, v in zip(range(1, 13), per)))
dsse = (y - c) ** 2 - (y - p) ** 2
order = np.argsort(dsse)
for kdrop in (1, 3, 5):
    keep = np.ones(len(y), bool); keep[order[:kdrop]] = False
    print(f'  drop the {kdrop} most-helped rows: 12m {r(c, keep) - r(p, keep):+.3f}')
t = pd.DataFrame(dict(month=mon[Q], off_sched=tr.off_sched.to_numpy()[Q], y=y[Q], v15s=p[Q], new=c[Q]))
print(t.sort_values('off_sched').round(0).to_string(index=False))
rk = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=cols)
sub = pd.read_parquet(CACHE.parent / 'outputs' / 'understanding-accordion_v15.parquet')
j = rk.merge(sub, on='MVT_ID_mvt'); Qr = cell(j); q = ds[Q].mean()
u = j[Qr][['month', 'off_sched', 'off_aobt3', 'TAXITIME_SEC_mvt']].copy(); u['with_q_all'] = q * ANCH + (1 - q) * u.TAXITIME_SEC_mvt
print(f'\n2026 rows in Q: {Qr.sum()} (q from all 2025 = {q:.3f})'); print(u.round(0).to_string(index=False))
