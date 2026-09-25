"""r07 -- LIRF extreme zone: are day-scale rows served too low? LOMO test on the v15 stand-in. Read-only.

ZONE Z = LIRF departures with off_sched > 40,000 s (any route). 2025 fact (IDEAS s4): with AOBT_3 missing,
36/36 such rows were tails; r06: all 6 LIRF rows with off_sched > 80,000 had y > 80,000.
PRE-REGISTERED (written before running, one primary variant):
  PRIMARY  C1: inside Z, prediction := r_m * off_sched, r_m = mean(y / off_sched) over Z rows of the OTHER
           11 months (leave-one-month-out, so no row sees its own label).
  Adopt-worthy if, vs v15~, the 12-month paired-bootstrap CI is below 0 AND the scored months improve.
  Reported only (not selectable): C2 = max(v15~, r_m * off_sched) (only raises), and the per-row table.
Also: what v15 serves on the 2026 Z rows (ratio pred / off_sched), no labels.
    python experiments/research_0925/r07_lirf_extreme_zone.py --v15s <dir>/v15s_oof.npy
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.validation import paired_bootstrap
ap = argparse.ArgumentParser(); ap.add_argument('--v15s', required=True); a = ap.parse_args()
cols = ['MVT_ID_mvt', 'APT', 'month', 'off_sched', 'off_aobt3']
tr = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=cols + ['y'])
y = tr.y.to_numpy().astype(float); p = np.load(a.v15s).astype(float); mon = tr.month.to_numpy()
so = tr.off_sched.to_numpy(); Z = (tr.APT.astype(str).to_numpy() == 'LIRF') & (so > 40000)
print(f'2025 zone rows: {Z.sum()} (NM record: {int((Z & tr.off_aobt3.notna().to_numpy()).sum())}); '
      f'y > 7200: {(y[Z] > 7200).mean():.3f}; mean y/off_sched {np.mean(y[Z] / so[Z]):.3f}; '
      f'mean v15~/off_sched {np.mean(p[Z] / so[Z]):.3f}')
c1 = p.copy(); c2 = p.copy()
for m in range(1, 13):
    k = Z & (mon == m); o = Z & (mon != m)
    if k.sum():
        rm = np.mean(y[o] / so[o]); c1[k] = rm * so[k]; c2[k] = np.maximum(p[k], rm * so[k])
S = np.isin(mon, [1, 7]); al = np.ones(len(y), bool)
r = lambda f, k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
for nm, f in (('C1 primary', c1), ('C2 info', c2)):
    out = []
    for scope, k in (('12m', al), ('scored', S)):
        bb = paired_bootstrap(y[k], f[k], p[k], n_boot=2000)
        out.append(f'{scope} {bb["diff"]:+.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
    per = [r(f, mon == m) - r(p, mon == m) for m in range(1, 13)]
    print(f'{nm}: ' + '  '.join(out) + f'  months better {sum(v < -1e-9 for v in per)}, worse {sum(v > 1e-9 for v in per)}')
d = pd.DataFrame(dict(month=mon[Z], off_sched=so[Z], nm=tr.off_aobt3.notna().to_numpy()[Z], y=y[Z], v15s=p[Z], c1=c1[Z]))
d['err_v15s'] = d.y - d.v15s; d['err_c1'] = d.y - d.c1
print(d.sort_values('off_sched').round(0).to_string(index=False))
rk = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=cols)
sub = pd.read_parquet(CACHE.parent / 'outputs' / 'understanding-accordion_v15.parquet')
j = rk.merge(sub, on='MVT_ID_mvt'); u = j[(j.APT.astype(str) == 'LIRF') & (j.off_sched > 40000)].copy()
rm_all = np.mean(y[Z] / so[Z]); u['ratio_served'] = u.TAXITIME_SEC_mvt / u.off_sched; u['c1_all_months'] = rm_all * u.off_sched
print(f'\n2026 zone rows: {len(u)}; served mean ratio {u.ratio_served.mean():.3f} (2025 labels: {rm_all:.3f}); '
      f'sum of |C1 - served| {np.abs(u.c1_all_months - u.TAXITIME_SEC_mvt).sum():.0f} s')
print(u[['month', 'off_sched', 'off_aobt3', 'TAXITIME_SEC_mvt', 'ratio_served', 'c1_all_months']].sort_values('off_sched').round(3).to_string(index=False))
