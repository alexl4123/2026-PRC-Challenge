"""ids_24 -- timezone artefacts: BLOCK = SCHED - k h (k = 1,2,3) and BLOCK = true - k h; counts by airport x month,
compared with the airport's UTC offset in that month (local-time stamping would follow DST)."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['APT', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'y', 'off_sched', 'month'])
tr['APT'] = tr.APT.astype(str)
tr['D'] = tr.LOBT_flt.isna()
tr['bs'] = tr.off_sched - tr.y            # BLOCK - SCHED
z = np.load(H + 'cache/w2/v28s_oof.npz'); tr['f'] = z['f']
rows = []
for k in [-3, -2, -1, 0, 1, 2, 3]:
    m = (tr.bs - k * 3600).abs() <= 6
    t = tr[m].groupby(['APT', 'month']).size().rename(f'k{k:+d}')
    rows.append(t)
t = pd.concat(rows, axis=1).fillna(0).astype(int)
print('rows with BLOCK = SCHED + k h (+-6 s), by airport x month:')
print(t.to_string())
# Rome SCHED-k h rows: v28 OOF error
for k in [-1, -2]:
    m = ((tr.bs - k * 3600).abs() <= 6) & tr.APT.eq('LIRF')
    g = tr[m]
    print(f'LIRF BLOCK=SCHED{k:+d}h: n={len(g)}, route D {g.D.sum()}, mean y {g.y.mean():.0f}, mean f {g.f.mean():.0f}, SSE {((g.y - g.f) ** 2).sum():.3e}, months', g.month.value_counts().sort_index().to_dict())
