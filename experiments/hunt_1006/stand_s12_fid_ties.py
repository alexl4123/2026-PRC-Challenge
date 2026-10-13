"""FLIGHT_ID is LOBT-ordered. Within equal-LOBT groups, what breaks the tie? If it were an actual time, FID would leak
BLOCK order. Pairwise concordance of FID order with candidate keys, within (LOBT) ties (all phases, 2025)."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
t = pd.read_parquet(H + 'stand_ev_training.parquet', columns=['FLIGHT_ID_mvt', 'PHASE_mvt', 'APT', 'ADEP_mvt', 'ADES_mvt', 'mvt', 'blk', 'sch', 'lobt', 'aobt3', 'CALLSIGN_flt', 'FLIGHT_mvt'])
t['fid'] = pd.to_numeric(t.FLIGHT_ID_mvt, errors='coerce')
t = t[np.isfinite(t.fid) & np.isfinite(t.lobt)]
# one row per flight id: prefer the DEP row (has BLOCK = off-block)
dep = t[t.PHASE_mvt == 'DEP'].drop_duplicates('fid')
arr = t[(t.PHASE_mvt == 'ARR') & ~t.fid.isin(dep.fid)].drop_duplicates('fid')
print('flights with DEP row', len(dep), 'ARR-only', len(arr))
a = pd.concat([dep.assign(k='DEP'), arr.assign(k='ARR')]).sort_values('fid').reset_index(drop=True)
same = a.lobt.to_numpy()[1:] == a.lobt.to_numpy()[:-1]
print('consecutive fid pairs with equal LOBT:', same.sum(), 'of', len(a) - 1)
def conc(col, pairs_mask):
    v = a[col].to_numpy(); d = v[1:] - v[:-1]; ok = pairs_mask & np.isfinite(d)
    return ok.sum(), np.mean(d[ok] > 0), np.mean(d[ok] == 0)
bothdep = same & (a.k.to_numpy()[1:] == 'DEP') & (a.k.to_numpy()[:-1] == 'DEP')
for col in ('blk', 'mvt', 'sch', 'aobt3'):
    print(f'within equal-LOBT, both DEP rows: {col}: n, share increasing, share tie =', conc(col, bothdep))
for col in ('ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt'):
    v = a[col].astype(str).to_numpy(); inc = v[1:] > v[:-1]; eq = v[1:] == v[:-1]
    print(f'within equal-LOBT: {col} share increasing {inc[same].mean():.4f} equal {eq[same].mean():.4f}')
# ties sorted by ADEP then? check lexicographic (ADEP, ADES, CALLSIGN)
key = (a.ADEP_mvt.astype(str) + '|' + a.ADES_mvt.astype(str) + '|' + a.CALLSIGN_flt.astype(str)).to_numpy()
print('within equal-LOBT: (ADEP,ADES,CALLSIGN) increasing', np.mean((key[1:] > key[:-1])[same]))
