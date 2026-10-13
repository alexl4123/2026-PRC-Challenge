"""SKEPTIC: package A+B+C without RYR515 (B row crossing local midnight at LIRF -> day-shift exposure). Scratch file only."""
import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
c = pd.read_parquet(H + 'cand_v28_ABC.parquet'); v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet')
m = c.merge(v, on='MVT_ID_mvt', suffixes=('', '_v28')); assert len(m) == len(c) == len(v) == 344841
rid = 202682249  # RYR515 LIRF 2026-01-05
k = m.MVT_ID_mvt.astype(np.int64) == rid; assert k.sum() == 1
print('RYR515 cand / v28:', int(m.TAXITIME_SEC_mvt[k].iloc[0]), int(m.TAXITIME_SEC_mvt_v28[k].iloc[0]))
m.loc[k, 'TAXITIME_SEC_mvt'] = m.loc[k, 'TAXITIME_SEC_mvt_v28']
out = m[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']].copy(); out['TAXITIME_SEC_mvt'] = out.TAXITIME_SEC_mvt.astype(np.int64)
assert (out.MVT_ID_mvt.to_numpy() == c.MVT_ID_mvt.to_numpy()).all() and out.TAXITIME_SEC_mvt.notna().all() and out.TAXITIME_SEC_mvt.min() >= 0
out.to_parquet(H + 'cand_v28_ABC_noRYR515.parquet', index=False)
s = (out.TAXITIME_SEC_mvt - m.TAXITIME_SEC_mvt_v28).to_numpy().astype(float)
N = len(out); b = np.sqrt(235.3996 ** 2 - np.sum(s ** 2) / N)
print('rows differing from v28', int((s != 0).sum()), 'sum sq', f'{np.sum(s ** 2):.4g}', 'bound RMSE', round(b, 4), f'({b - 235.3996:+.4f})')
# RYR515 decision arithmetic: normal y~900 vs day-shift y~86400+661
for nm, yy in (('normal 900', 900.), ('day-shift 87061', 87061.)):
    d = ((yy - 2275) ** 2 - (yy - 5335) ** 2) / N
    print(f'  RYR515 move 5335->2275 if {nm}: dRMSE {np.sqrt(235.3996 ** 2 + d) - 235.3996:+.3f}')
