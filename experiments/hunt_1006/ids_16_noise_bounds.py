"""ids_16 -- timestamp noise (+-6 s?) and hard range rules: y >= -12, BLOCK-SCHED, MVT-SCHED extremes; v28 preds vs them."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
                                                                    'LOBT_flt', 'y', 'off_sched', 'off_lobt'])
tr['bs'] = (tr.BLOCK_TIME_UTC_mvt - tr.SCHED_TIME_UTC_mvt).dt.total_seconds()
tr['bl'] = (tr.BLOCK_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
x = tr.bs[(tr.bs.abs() <= 30)]
print('BLOCK - SCHED in [-30,30]: histogram')
print(x.value_counts().sort_index().to_string())
x = tr.y[tr.y <= 20]
print('y <= 20 histogram'); print(x.value_counts().sort_index().to_string())
print('BLOCK - LOBT near +-3600: counts of |bl| in (3594, 3606]:', ((tr.bl.abs() > 3594) & (tr.bl.abs() <= 3606)).sum())
print(tr.bl[(tr.bl.abs() > 3590)].round().value_counts().sort_index().to_string())
print('\nextremes: y', tr.y.min(), tr.y.max(), '| BLOCK-SCHED', tr.bs.min(), tr.bs.max(), '| MVT-SCHED', tr.off_sched.min(), tr.off_sched.max())
print('BLOCK-SCHED lowest 10', np.sort(tr.bs.values)[:10], ' highest 10', np.sort(tr.bs.values)[-10:])
print('MVT-SCHED lowest 10', np.sort(tr.off_sched.values)[:10], ' highest 10', np.sort(tr.off_sched.values)[-10:])
rk = pd.read_parquet(H + 'cache/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'off_sched', 'off_lobt'])
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet')
rk = rk.merge(v, on='MVT_ID_mvt')
f = rk.TAXITIME_SEC_mvt
print('\n2026 MVT-SCHED extremes', rk.off_sched.min(), rk.off_sched.max())
print('v28 preds: min', f.min(), ' max', f.max(), ' n < -12:', (f < -12).sum(), ' n > 131167:', (f > 131167).sum())
print('v28: rows with pred > off_sched + 86406 (BLOCK < SCHED - 1 day):', (f > rk.off_sched + 86406).sum(),
      ' rows with pred < off_sched - 151090:', (f < rk.off_sched - 151090).sum())
print('2025: rows with y > off_sched + 86406:', (tr.y > tr.off_sched + 86406).sum())
