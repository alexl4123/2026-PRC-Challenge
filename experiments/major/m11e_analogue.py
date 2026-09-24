"""m11e -- calibrate the 2026 expectation on the 2025 analogue: disruption days = airport-days
whose route-D share is >= 8% (as on EHAM 2-7 Jan 2026).  v9~ bias on route-D rows there and
the fraction of it the route-D state corrector removed (LOMO)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'y'])
d['v9'] = np.load(M / 'v9s_oof.npy'); d['tD'] = np.load(M / 'm11d_oof_trailD.npy')
d['D'] = d.off_aobt3.isna(); d['day'] = d.MVT_TIME_UTC_mvt.dt.floor('D')
share = d.groupby(['APT', 'day'], observed=True).D.transform('mean')
for thr in [0.05, 0.08, 0.12]:
    k = d.D & (share >= thr) & (d.y <= 7200) & ((d.y - d.off_sched).abs() > 120)
    days = d[k].groupby(['APT', 'day'], observed=True).size()
    print(f'route-D share >= {thr:.2f}: {len(days)} airport-days, {k.sum()} route-D non-stamped rows; '
          f'mean y {d.y[k].mean():.0f}  mean v9~ {d.v9[k].mean():.0f} (bias {d.v9[k].mean()-d.y[k].mean():+.0f})  '
          f'mean trailD {d.tD[k].mean():.0f}  -> fraction of bias removed {(d.tD[k].mean()-d.v9[k].mean())/(d.y[k].mean()-d.v9[k].mean()):.2f}; '
          f'RMSE v9~ {np.sqrt(((d.y-d.v9)[k]**2).mean()):.0f} trailD {np.sqrt(((d.y-d.tD)[k]**2).mean()):.0f}')
    print('    airport-days:', ', '.join(f'{a} {str(t)[:10]}({n})' for (a, t), n in days.sort_values(ascending=False).head(8).items()))
