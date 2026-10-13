"""Skeptic re-check of the LOBT projection on the v24 OOF (2025 only). Read-only."""
import glob, numpy as np, pandas as pd
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
z = np.load(ROOT + 'prc2026/cache/w2/v24s_oof.npz', allow_pickle=True)
y, f, month, P, D = z['y'], z['f'], z['month'], z['P'], z['D']
fs = sorted(glob.glob(ROOT + 'training_2025-*.parquet'))
tr = pd.concat([pd.read_parquet(x, columns=['PHASE_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'TAXITIME_SEC_mvt', 'ADEP_mvt'])
                for x in fs], ignore_index=True)
tr = tr[tr.PHASE_mvt == 'DEP'].reset_index(drop=True)
assert len(tr) == len(y) and np.array_equal(tr.TAXITIME_SEC_mvt.to_numpy().astype(float), y), 'row order mismatch'
ml = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds().to_numpy()
has = ~np.isnan(ml)
print('has LOBT & P:', (has & P).sum(), ' has LOBT & D:', (has & D).sum())
lo = np.where(has, np.maximum(ml - 3606, 0), -np.inf); hi = np.where(has, ml + 3606, np.inf)
q = np.clip(f, lo, hi)
mv = q != f
print('moved', mv.sum(), ' worse', (np.abs(y - q) > np.abs(y - f) + 1e-9).sum())
rm = lambda e, m: np.sqrt(np.mean(e[m] ** 2))
held = ~np.isin(month, [3, 6, 9, 12]); sc = np.isin(month, [1, 7]); al = np.ones_like(sc)
for name, m in [('12m', al), ('held', held), ('Jan+Jul', sc), ('Jan', month == 1), ('Jul', month == 7)]:
    print(f'{name:8s} {rm(y - f, m):.3f} -> {rm(y - q, m):.3f}  d {rm(y - q, m) - rm(y - f, m):+.3f}')
g = (y - f) ** 2 - (y - q) ** 2
o = np.sort(g[sc])[::-1]
print('Jan+Jul gain share top-1/10 rows:', o[0] / o.sum(), o[:10].sum() / o.sum())
apt = tr.ADEP_mvt.to_numpy()
i = np.argsort(-g)[:8]
print(pd.DataFrame({'apt': apt[i], 'm': month[i], 'y': y[i], 'f': f[i], 'q': q[i], 'ml': ml[i], 'gain': g[i]}).to_string())
# how many months better
print('months better:', sum(rm(y - q, month == k) < rm(y - f, month == k) - 1e-9 for k in range(1, 13)), '/12')
