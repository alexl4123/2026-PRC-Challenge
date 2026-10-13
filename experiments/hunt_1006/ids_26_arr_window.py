"""ids_26 -- the ARR-side NM join window: which clock pair is bounded exactly? (2025 + 2026 ARR rows, labels public)."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load
C = ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'AOBT_3_flt', 'ARVT_1_flt', 'ARVT_3_flt']
for w in ['training', 'ranking']:
    d = load(w, phase='ARR', columns=C)
    d = d[d.LOBT_flt.notna()]
    print('=====', w, 'ARR rows with NM', len(d))
    for a in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']:
        for b in ['ARVT_1_flt', 'ARVT_3_flt', 'LOBT_flt', 'AOBT_3_flt', 'EOBT_1_flt']:
            x = (d[a] - d[b]).dt.total_seconds().dropna()
            print(f'  {a[:5]} - {b[:6]}: n={len(x)} min {x.min():.0f} max {x.max():.0f}  |.|>3606: {(x.abs() > 3606).sum()}  q001/q999 {x.quantile(.001):.0f}/{x.quantile(.999):.0f}')
