"""r03 -- the high-stakes rows: calibration of LARGE predictions (2025 OOF, v15 stand-in) and what v15 serves
on 2026. Read-only; no 2026 label is used (there are none for departures).

Why: one row with a 30,000 s error adds ~9e8 SSE, i.e. ~4.7 s at our live RMSE. The board moves in such
steps. The MSE-optimal prediction for a row is its conditional mean, so the question is purely whether
our LARGE predictions are calibrated in 2025 (mean y vs mean prediction per prediction band and per
anchor type), and how many such rows 2026 contains.

Needs the v15 stand-in written by r01 (R01_OUT=<dir> ... r01_v15_anatomy.py) -> pass --v15s <dir>/v15s_oof.npy
    systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 nice -n 5 \
        python experiments/research_0925/r03_extreme_rows.py --v15s <dir>/v15s_oof.npy
"""
import os, sys, argparse, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE

ap = argparse.ArgumentParser(); ap.add_argument('--v15s', required=True); a = ap.parse_args()
M = CACHE / 'major'
cols = ['MVT_ID_mvt', 'APT', 'month', 'y', 'off_sched', 'off_aobt3', 'off_eobt1', 'off_lobt']
tr = pd.read_parquet(M / 'dep_training.parquet', columns=cols)
b = np.load(CACHE / 'v11' / 'base.npz'); assert np.allclose(b['y'], tr.y.to_numpy())
p = np.load(a.v15s).astype(float); y = tr.y.to_numpy().astype(float)
S = np.isin(tr.month.to_numpy(), [1, 7])
tot = ((y - p) ** 2).sum(); totS = ((y - p)[S] ** 2).sum()
oa = tr.off_aobt3.to_numpy(); so = tr.off_sched.to_numpy()

print('=== 2025 OOF (v15~): calibration by PREDICTION band (all 12 months)')
bands = [(3600, 7200), (7200, 20000), (20000, 40000), (40000, 70000), (70000, 1e9)]
for lo, hi in bands:
    k = (p > lo) & (p <= hi)
    if k.sum():
        e = (y - p)[k]
        print(f'  pred ({lo:>6.0f},{hi:>9.0f}]  n {k.sum():5d}  mean pred {p[k].mean():8.0f}  mean y {y[k].mean():8.0f}  '
              f'median y {np.median(y[k]):8.0f}  SSE share {100 * (e ** 2).sum() / tot:5.2f}%  (scored months n {int((k & S).sum())})')
print('=== 2025: rows with LARGE labels the model did NOT predict large (y > 20,000, pred <= 7,200)')
k = (y > 20000) & (p <= 7200)
print(f'  n {k.sum()}  SSE share {100 * ((y - p)[k] ** 2).sum() / tot:.2f}% of all 12-month SSE; scored n {int((k & S).sum())}, '
      f'share of scored SSE {100 * ((y - p)[k & S] ** 2).sum() / totS:.2f}%')

print('\n=== 2025: NM-record rows with a DAY-SCALE NM proxy (off_aobt3 > 20,000 s)')
k = np.isfinite(oa) & (oa > 20000)
d = pd.DataFrame(dict(apt=tr.APT.astype(str)[k], month=tr.month[k], off_aobt3=oa[k], off_sched=so[k], y=y[k], pred=p[k]))
d['y_near_aobt3'] = (d.y - d.off_aobt3).abs() <= 900
d['y_near_sched'] = (d.y - d.off_sched).abs() <= 900
print(f'  n {len(d)}; y within 15 min of off_aobt3: {d.y_near_aobt3.mean():.2f}; of off_sched: {d.y_near_sched.mean():.2f}; '
      f'y <= 3,600: {(d.y <= 3600).mean():.2f}')
print(f'  mean pred {d.pred.mean():.0f} vs mean y {d.y.mean():.0f}; SSE share {100 * ((d.y - d.pred) ** 2).sum() / tot:.2f}%')
print(d.sort_values('off_aobt3').round(0).to_string(index=False, max_rows=60))

print('\n=== 2026 (v15 as uploaded): large predictions, joined to ranking features (no labels)')
rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=[c for c in cols if c != 'y'])
sub = pd.read_parquet(CACHE.parent / 'outputs' / 'understanding-accordion_v15.parquet')
j = rk.merge(sub, on='MVT_ID_mvt', how='left'); assert j.TAXITIME_SEC_mvt.notna().all() and len(j) == 344841
q = j.TAXITIME_SEC_mvt.to_numpy()
for lo, hi in bands:
    k = (q > lo) & (q <= hi)
    print(f'  pred ({lo:>6.0f},{hi:>9.0f}]  n {k.sum():5d}  by apt {dict(j.APT.astype(str)[k].value_counts().head(5))}')
k = np.isfinite(j.off_aobt3) & (j.off_aobt3 > 20000)
print(f'  NM-record rows with off_aobt3 > 20,000: {k.sum()}')
print(j.loc[k, ['APT', 'month', 'off_aobt3', 'off_sched', 'TAXITIME_SEC_mvt']].round(0).to_string(index=False))
print('peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
