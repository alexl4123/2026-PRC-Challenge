"""ids_29 -- overlap of candidate rows with the 63 rows v28 projected (pre-commitment: never change those)."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
v27 = pd.read_parquet(H + 'outputs/understanding-accordion_v27.parquet')
v28 = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet')
x = v27.merge(v28, on='MVT_ID_mvt', suffixes=('_27', '_28'))
proj = set(x[x.TAXITIME_SEC_mvt_27 != x.TAXITIME_SEC_mvt_28].MVT_ID_mvt)
print('projected rows', len(proj))
q = pd.read_parquet(H + 'cache/hunt/ids_23_dst_2026.parquet')
mv = q[q.move != 0]
print('DST moved rows overlapping projected:', len(set(mv.MVT_ID_mvt) & proj))
rt = pd.read_parquet(H + 'cache/hunt/ids_08_twins2026.parquet')
t = rt[(rt.gsize == 2) & (rt.dmvt <= 600)]
one = t.groupby('gid').nm.transform('sum') == 1
a = t[one & t.nm]
print('twin NM rows 2026', len(a), ' overlapping projected:', len(set(a.MVT_ID_mvt) & proj))
