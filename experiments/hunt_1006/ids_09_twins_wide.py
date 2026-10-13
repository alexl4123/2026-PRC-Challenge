"""ids_09 -- twin groups (same APT, callsign, SCHED) whose take-offs are far apart: what are they? labels, v28 OOF."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 40)
tt = pd.read_parquet(H + 'cache/hunt/ids_08_twins2025.parquet')
w = tt[tt.dmvt > 600].copy()
for c in ['SCHED', 'MVT_T', 'BLOCK', 'LOBT_']:
    w[c] = pd.to_datetime(w[c], unit='s')
w['e'] = w.y - w.f
print('2025 wide twin groups:', w.gid.nunique(), 'rows', len(w))
print('SSE of v28 OOF on these rows', (w.e ** 2).sum(), ' scored months', (w[w.month.isin([1, 7])].e ** 2).sum())
print(w[['gid', 'APT', 'FLIGHT_mvt', 'MVT_ID_mvt', 'SCHED', 'MVT_T', 'BLOCK', 'LOBT_', 'y', 'f', 'e', 'P', 'D', 'nm', 'idrank']].to_string())
r = pd.read_parquet(H + 'cache/hunt/ids_08_twins2026.parquet')
rw = r[r.dmvt > 600].copy()
for c in ['SCHED', 'MVT_T', 'LOBT_']:
    rw[c] = pd.to_datetime(rw[c], unit='s')
print('\n2026 wide twin groups:', rw.gid.nunique())
print(rw[['gid', 'APT', 'FLIGHT_mvt', 'MVT_ID_mvt', 'SCHED', 'MVT_T', 'LOBT_', 'off_sched', 'f', 'P', 'nm', 'idrank']].to_string())
