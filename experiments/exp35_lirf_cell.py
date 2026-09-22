"""EXP35 -- inside the cell where the coin flip lives.

627 rows (full year) are LIRF, AOBT_3 missing, off_sched > 7200. 413 of them
are tail. That single cell holds 71% of all tail rows and, with its 214
non-tail rows, most of the false-positive damage too. Everything else is
already near-deterministic. So: does anything inside this cell discriminate?
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import NEEDED
from prc.io import load, peak_rss_gb
pd.set_option('display.width', 230)

t0 = time.time()
tr = load('training', phase='DEP',
          columns=sorted(set(NEEDED + [TARGET, 'BLOCK_TIME_UTC_mvt', 'FLIGHT_mvt'])))
y = tr[TARGET].astype(float).to_numpy()
M = tr.MVT_TIME_UTC_mvt
os_ = (M - tr.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
miss = tr.AOBT_3_flt.isna().to_numpy()
apt = tr.APT.astype(str).to_numpy()
tail = y > TAIL_CAP
cell = (apt == 'LIRF') & miss & (os_ > TAIL_CAP)
print(f'cell n={int(cell.sum())}  tail={int(tail[cell].sum())} '
      f'({tail[cell].mean()*100:.1f}%)   ({time.time()-t0:.0f}s)')

d = pd.DataFrame({
    'y': y[cell], 'istail': tail[cell], 'off_sched': os_[cell],
    'hour': M.dt.hour.to_numpy()[cell], 'dow': M.dt.dayofweek.to_numpy()[cell],
    'month': M.dt.month.to_numpy()[cell], 'date': M.dt.floor('D').to_numpy()[cell],
    'stand': tr.STAND_mvt.astype(str).to_numpy()[cell],
    'rwy': tr.RUNWAY_mvt.astype(str).to_numpy()[cell],
    'op': tr.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy()[cell],
    'ac': tr.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()[cell],
    'seg': tr.MARKET_SEGMENT_flt.astype(str).to_numpy()[cell],
    'wk': tr.WK_TBL_CAT_flt.astype(str).to_numpy()[cell],
    'ftype': tr.FLIGHT_TYPE_flt.astype(str).to_numpy()[cell],

})
d['pier'] = d.stand.str.extract(r'^([A-Za-z]+|\d)', expand=False).fillna('NA')


def split(col, minn=15, top=12):
    g = d.groupby(col).agg(n=('istail', 'size'), p=('istail', 'mean'))
    g = g[g.n >= minn].sort_values('p')
    if g.empty:
        print(f'  {col}: no level with n>={minn}')
        return
    print(f'  --- {col} (levels with n>={minn}; cell base rate '
          f'{d.istail.mean()*100:.1f}%) ---')
    print(pd.concat([g.head(top // 2), g.tail(top // 2)]).assign(
        p=lambda x: (x.p * 100).round(1)).to_string())


for c in ['pier', 'op', 'ac', 'seg', 'wk', 'ftype', 'rwy', 'month', 'dow']:
    split(c)

print('\n  --- hour of day ---')
g = d.groupby('hour').agg(n=('istail', 'size'), p=('istail', 'mean'))
print(g.assign(p=lambda x: (x.p * 100).round(1)).to_string())

print('\n  --- off_sched, finer ---')
d['osb'] = pd.cut(d.off_sched, [7200, 8000, 9000, 10000, 12000, 15000, 20000, 30000, 1e9])
print(d.groupby('osb', observed=True).agg(n=('istail', 'size'), p=('istail', 'mean'),
                                          med_y=('y', 'median')).assign(
    p=lambda x: (x.p * 100).round(1)).to_string())

print('\n' + '=' * 100)
print('CLUSTERING WITHIN THE CELL: do artifact rows share a day?')
print('=' * 100)
g = d.groupby('date').agg(n=('istail', 'size'), k=('istail', 'sum'))
print(f'  {len(g)} distinct LIRF days carry the cell; cell rows per day: '
      f'mean {g.n.mean():.1f} max {g.n.max()}')
print(f'  days where ALL cell rows are tail: {int((g.k == g.n).sum())}  '
      f'| none are: {int((g.k == 0).sum())}  | mixed: {int(((g.k > 0) & (g.k < g.n)).sum())}')
# permutation test on within-day homogeneity
obs = float((g.k * (g.k - 1)).sum() + ((g.n - g.k) * (g.n - g.k - 1)).sum())
rng = np.random.default_rng(0)
lab = d.istail.to_numpy().copy()
sizes = g.n.to_numpy()
null = []
for _ in range(2000):
    p = rng.permutation(lab)
    i = 0
    tot = 0.0
    for s in sizes:
        kk = p[i:i + s].sum(); i += s
        tot += kk * (kk - 1) + (s - kk) * (s - kk - 1)
    null.append(tot)
null = np.array(null)
print(f'  same-day concordant pairs: observed {obs:.0f}, permuted mean {null.mean():.0f} '
      f'sd {null.std():.0f}  z={(obs-null.mean())/null.std():+.1f}  '
      f'P(perm >= obs)={np.mean(null >= obs):.4f}')

print('\n  --- the 15 busiest cell-days ---')
print(g.assign(p=lambda x: (x.k / x.n * 100).round(0)).sort_values('n', ascending=False)
      .head(15).to_string())

print('\n' + '=' * 100)
print('A ROUGH CEILING: best achievable in-cell RMSE given a perfect p per subgroup')
print('=' * 100)
for cols in [['osb'], ['osb', 'pier'], ['osb', 'op'], ['osb', 'hour']]:
    gg = d.groupby(cols, observed=True)['istail'].transform('mean').to_numpy()
    anchor = d.off_sched.to_numpy()
    bulkp = 1100.0
    pred = gg * anchor + (1 - gg) * bulkp
    print(f'  group by {str(cols):22s} in-cell RMSE {np.sqrt(np.mean((d.y-pred)**2)):9.1f}'
          f'   (constant p: '
          f'{np.sqrt(np.mean((d.y-(d.istail.mean()*anchor+(1-d.istail.mean())*bulkp))**2)):9.1f}'
          f', oracle p: {np.sqrt(np.mean((d.y-np.where(d.istail, anchor, bulkp))**2)):7.1f})')

print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
