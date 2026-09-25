"""r06 -- flights that took off about a DAY after schedule (off_sched > 80,000 s): what were their labels in 2025,
and what does v15 serve for such rows in 2026? Read-only. (--v15s from r01 for the 2025 OOF column)
    python experiments/research_0925/r06_day_shift.py --v15s <dir>/v15s_oof.npy
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
ap = argparse.ArgumentParser(); ap.add_argument('--v15s', required=True); a = ap.parse_args()
cols = ['MVT_ID_mvt', 'APT', 'month', 'off_sched', 'off_aobt3', 'off_eobt1', 'off_lobt']
tr = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=cols + ['y'])
tr['pred_v15s'] = np.load(a.v15s)
k = tr.off_sched > 80000
t = tr[k].copy(); t['route'] = np.where(t.off_aobt3.notna(), 'NM', 'noNM')
t['y_minus_offsched'] = t.y - t.off_sched
print(f'2025 rows with off_sched > 80,000: {len(t)}; by apt/route:')
print(t.groupby(['APT', 'route']).size().to_string())
print(f'  y > 7,200: {(t.y > 7200).mean():.3f};  |y - off_sched| <= 900: {(t.y_minus_offsched.abs() <= 900).mean():.3f}; '
      f'y within [off_sched - 9000, off_sched]: {((t.y_minus_offsched <= 0) & (t.y_minus_offsched >= -9000)).mean():.3f}')
print(f'  RMSE of v15~ on them {np.sqrt(np.mean((t.y - t.pred_v15s) ** 2)):.0f}; of predicting off_sched '
      f'{np.sqrt(np.mean((t.y - t.off_sched) ** 2)):.0f}; mean y {t.y.mean():.0f} mean pred {t.pred_v15s.mean():.0f}')
print(t.sort_values('off_sched')[['APT', 'month', 'route', 'off_sched', 'off_aobt3', 'y', 'pred_v15s']].round(0).to_string(index=False))
rk = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=cols)
sub = pd.read_parquet(CACHE.parent / 'outputs' / 'understanding-accordion_v15.parquet')
j = rk.merge(sub, on='MVT_ID_mvt'); u = j[j.off_sched > 80000]
print(f'\n2026 ranking rows with off_sched > 80,000: {len(u)}')
print(u[['APT', 'month', 'off_sched', 'off_aobt3', 'TAXITIME_SEC_mvt']].sort_values('off_sched').round(0).to_string(index=False))
