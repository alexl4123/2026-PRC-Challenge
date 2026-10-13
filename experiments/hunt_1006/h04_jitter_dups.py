"""(a) seconds patterns / jitter hypothesis; (b) duplicate departure rows of the same flight (shared BLOCK?)."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT', 'BLOCK', 'SCHED', 'LOBT', 'AOBT3', 'EOBT1', 'IOBT', 'CALLSIGN_flt'])
d['APT'] = d.APT.astype(str); d['y'] = d.MVT - d.BLOCK
sm = lambda x: ((x + 30) % 60) - 30   # signed seconds offset from the nearest minute
print('share of times with |sec offset from minute| <= 6, per airport:')
rows = []
for a, g in d.groupby('APT'):
    r = {'APT': a}
    for c in ['BLOCK', 'MVT', 'SCHED', 'LOBT', 'AOBT3', 'EOBT1']:
        x = g[c].dropna(); r[c] = round(float((sm(x).abs() <= 6).mean()), 3); r[c + '_s0'] = round(float((x % 60 == 0).mean()), 3)
    rows.append(r)
print(pd.DataFrame(rows).to_string())
# stamped rows: distribution of BLOCK - SCHED in [-10, 10]
bs = d.BLOCK - d.SCHED
print('\nBLOCK-SCHED value counts in [-10,10] (all airports):'); print(bs[(bs.abs() <= 10)].value_counts().sort_index().to_string())
print('\ny value counts in [-15, 3]:'); print(d.y[d.y <= 3].value_counts().sort_index().to_string())
# (b) duplicates: same APT, FLIGHT_mvt, SCHED
k = d.dropna(subset=['FLIGHT_mvt']).groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True)
sz = k.MVT.transform('size')
dup = d.loc[sz[sz > 1].index].copy()
print('\nrows in (APT, FLIGHT_mvt, SCHED) groups of size > 1:', len(dup))
if len(dup):
    g = dup.groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True)
    st = g.agg(n=('MVT', 'size'), block_nuniq=('BLOCK', 'nunique'), mvt_nuniq=('MVT', 'nunique'), blk_rng=('BLOCK', lambda x: x.max() - x.min()), mvt_rng=('MVT', lambda x: x.max() - x.min()))
    print(st.describe().to_string())
    print('groups with identical BLOCK:', (st.block_nuniq == 1).sum(), ' of', len(st), ' identical MVT:', (st.mvt_nuniq == 1).sum())
    print(st.head(20).to_string())
# same FLIGHT_ID_mvt shared by >1 departure rows
f = d.dropna(subset=['FLIGHT_ID_mvt'])
c = f.FLIGHT_ID_mvt.value_counts()
print('\nFLIGHT_ID_mvt present on', len(f), 'deps; ids used by >1 dep rows:', (c > 1).sum())
