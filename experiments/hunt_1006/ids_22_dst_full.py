"""ids_22 -- the refined destination exclusion (not used, ADEP != ADES) as a full rule: BLOCK outside
[LOBT_dst - 3606, LOBT_dst + 3606]  <=>  y outside (L, U), L = MVT-LOBT_dst-3606, U = MVT-LOBT_dst+3606.
Exactness on 2025 (all linked rows), and pricing of the non-convex part (L > -12) with two projections:
nearest edge, and 'side of off_sched' (stamped hypothesis side). 2026 rows moved."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
m = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2025.parquet')
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
tr['f'] = z['f']; tr['P'] = z['P']; tr['month'] = z['month']
m = m.merge(tr, on='MVT_ID_mvt', how='left')
ok = (~m.used) & (m.ADEP_mvt != m.ADES_mvt)
r = m[ok].copy()
r['L'] = r.mvt_m_lobt - 3606; r['U'] = r.mvt_m_lobt + 3606
inside_y = (r.y > r.L) & (r.y < r.U)
print('2025 refined rule rows', len(r), ' violations (y strictly inside (L,U))', inside_y.sum(), '  months', r.month.nunique(),
      ' Jan+Jul rows', r.month.isin([1, 7]).sum())
print('  by airport', r.APT.value_counts().to_dict())
nc = r[r.L > -12].copy()          # non-convex part
f = nc.f.values; L = nc.L.values; U = nc.U.values; y = nc.y.values; os_ = nc.off_sched.values
ins = (f > L) & (f < U)
print('non-convex rows', len(nc), ' v28 OOF inside band', ins.sum())
near = np.where(f - L < U - f, L, U)
side = np.where(os_ >= U, U, np.where(os_ <= L, L, near))
for name, tgt in [('nearest edge', near), ('off_sched side', side)]:
    g = np.where(ins, tgt, f)
    d = (y - g) ** 2 - (y - f) ** 2
    jj = nc.month.isin([1, 7]).values
    print(f'  {name}: moved {ins.sum()}, worse {np.sum(d > 0)}, dSSE 12m {d.sum():.3e}, Jan+Jul {d[jj].sum():.3e}')
print(nc[ins][['APT', 'FLIGHT_mvt', 'month', 'y', 'f', 'L', 'U', 'off_sched']].round(0).to_string())
k = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2026.parquet')
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
k = k.merge(v, on='MVT_ID_mvt', how='left')
q = k[(~k.used) & (k.ADEP_mvt != k.ADES_mvt)].copy()
q['L'] = q.mvt_m_lobt - 3606; q['U'] = q.mvt_m_lobt + 3606
ins2 = (q.f > q.L) & (q.f < q.U)
print('\n2026 refined rule rows', len(q), ' v28 inside the excluded band:', ins2.sum(), ' (convex part L<=-12:', (ins2 & (q.L <= -12)).sum(), ')')
print(q[ins2][['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'off_sched', 'mvt_m_lobt', 'mvt_m_aobt', 'L', 'U', 'f']].round(0).to_string())
