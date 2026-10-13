import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
v27 = pd.read_parquet(R + 'outputs/understanding-accordion_v27.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
v28 = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
proj = set(v28.index[(v28 - v27.reindex(v28.index)) != 0])
print('rows projected by the LOBT window (v28 != v27):', len(proj))
for nm in ['cand_v28_ABC.parquet', 'cand_v28_ABC_E1.parquet']:
    c = pd.read_parquet(R + 'cache/hunt/' + nm).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(v28.index)
    ch = set(v28.index[c != v28])
    print(nm, 'rows changed vs v28:', len(ch), ' overlap with projected rows:', len(ch & proj), sorted(ch & proj))
