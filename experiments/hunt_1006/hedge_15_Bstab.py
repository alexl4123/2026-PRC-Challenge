"""Rule B stability: LIRF zone P rows whose (MVT - 900 s) UTC date is after SCHED's UTC date -- class by month, sensitivity
to the assumed taxi (600 / 900 / 1800 s), and the same split for the other airports (mechanism check)."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250)
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P = V['y'], V['month'], V['P']
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'off_sched'])
d['y'] = y; d['month'] = month
x = d[P].copy()
x['cls'] = np.where(np.abs(x.y - x.off_sched) <= 120, 'stamp', np.where(x.y > 43200, 'day', np.where(x.y <= 7200, 'normal', 'other')))
for tx in (600, 900, 1800):
    x[f'du{tx}'] = ((x.MVT_TIME_UTC_mvt - pd.to_timedelta(tx, 's')).dt.normalize() - x.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
L = x[x.APT == 'LIRF']
for tx in (600, 900, 1800):
    print(f'LIRF zone P, taxi assumption {tx}s: dd_utc>=1 classes', L[L[f'du{tx}'] >= 1].cls.value_counts().to_dict())
print('by month (dd_utc>=1, 900 s):'); print(pd.crosstab(L[L.du900 >= 1].month, L[L.du900 >= 1].cls))
print('\nother airports zone P, dd_utc>=1:'); o = x[(x.APT != 'LIRF') & (x.du900 >= 1)]
print(pd.crosstab(o.APT, o.cls))
# true BLOCK date check for LIRF normal zone-P rows: does any normal row have its true off-block on a later UTC date than SCHED?
