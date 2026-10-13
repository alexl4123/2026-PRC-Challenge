import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet')
for nm in ['cand_v28_ABC.parquet', 'cand_v28_ABC_E1.parquet']:
    c = pd.read_parquet(R + 'cache/hunt/' + nm)
    m = v.merge(c, on='MVT_ID_mvt', suffixes=('_v28', '_c'))
    print(nm, 'rows', len(c), 'ids match', len(m) == len(v) == len(c), 'dtype', c.TAXITIME_SEC_mvt.dtype, 'nan', c.TAXITIME_SEC_mvt.isna().sum(), 'min', c.TAXITIME_SEC_mvt.min(),
          'changed', (m.TAXITIME_SEC_mvt_v28 != m.TAXITIME_SEC_mvt_c).sum(), 'max |move|', (m.TAXITIME_SEC_mvt_v28 - m.TAXITIME_SEC_mvt_c).abs().max())
