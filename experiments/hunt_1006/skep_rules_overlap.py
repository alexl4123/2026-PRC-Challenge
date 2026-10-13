"""Rules skeptic: pre-commitment overlap (63 v28-projected rows) and row-set overlap between candidates. Read-only."""
import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
rd = lambda p: pd.read_parquet(p).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
v27 = rd(R + 'outputs/understanding-accordion_v27.parquet'); v28 = rd(R + 'outputs/understanding-accordion_v28.parquet')
v27 = v27.reindex(v28.index)
proj = set(v28.index[(v28 != v27).to_numpy()]); print('projected rows (v28 != v27):', len(proj))
c = {}
for nm, f in (('ABC', 'cand_v28_ABC.parquet'), ('ABC_E1', 'cand_v28_ABC_E1.parquet'), ('DST', 'ids_cand_v28_dst.parquet'), ('TWIN', 'ids_cand_v28_twin.parquet')):
    x = rd(H + f).reindex(v28.index); assert x.notna().all()
    c[nm] = set(v28.index[(x != v28).to_numpy()])
z = np.load(H + 'hedge_candidates_2026.npz', allow_pickle=True); ids = z['ids']; zv = z['v28']
s28 = v28.reindex(ids).to_numpy(); print('hedge npz v28 == shipped v28:', np.array_equal(zv, s28))
for k in ('B', 'A1', 'BT', 'A1BC'):
    c['H_' + k] = set(ids[z[k] != zv])
for k, s in c.items():
    print(f'{k:8s} rows {len(s):4d}  overlap with projected {len(s & proj)}')
print('ABC & DST', len(c['ABC'] & c['DST']), '| DST - ABC', len(c['DST'] - c['ABC']), '| ABC - DST', len(c['ABC'] - c['DST']))
print('ABC & H_A1', len(c['ABC'] & c['H_A1']), '| DST & H_A1', len(c['DST'] & c['H_A1']), '| ABC & H_BT', len(c['ABC'] & c['H_BT']), '| DST & H_BT', len(c['DST'] & c['H_BT']))
print('E1-only rows', len(c['ABC_E1'] - c['ABC']), '| TWIN & ABC', len(c['TWIN'] & c['ABC']), '| TWIN & DST', len(c['TWIN'] & c['DST']))
dep = pd.read_parquet(H + 'ra_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'MVT', 'SCHED']).set_index('MVT_ID_mvt')
dst = pd.DataFrame({'v28': v28, 'DST': rd(H + 'ids_cand_v28_dst.parquet').reindex(v28.index), 'ABC': rd(H + 'cand_v28_ABC.parquet').reindex(v28.index)})
dd = dst.loc[sorted(c['DST'] - c['ABC'])].join(dep[['APT', 'FLIGHT_mvt']]); dd['off_sched'] = (dep.MVT - dep.SCHED).reindex(dd.index)
print('DST rows not in ABC:'); print(dd.to_string())
dd = dst.loc[sorted(c['ABC'] - c['DST'])].join(dep[['APT', 'FLIGHT_mvt']]); print('ABC rows not in DST:'); print(dd.to_string())
both = sorted(c['ABC'] & c['DST']); x = dst.loc[both]; print('rows in both with different values:', int((x.DST != x.ABC).sum()))
# concentration of the package's move^2
m = (dst.ABC - dst.v28) ** 2; m = m[m > 0].sort_values(ascending=False); print('ABC top-1 share of sum move^2 %.2f, top-3 %.2f' % (m.iloc[0] / m.sum(), m.iloc[:3].sum() / m.sum()))
m = (dst.DST - dst.v28) ** 2; m = m[m > 0].sort_values(ascending=False); print('DST top-1 share of sum move^2 %.2f, top-3 %.2f' % (m.iloc[0] / m.sum(), m.iloc[:3].sum() / m.sum()))
