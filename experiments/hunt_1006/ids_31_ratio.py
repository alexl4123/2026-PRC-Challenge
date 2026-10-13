"""ids_31 -- realised dSSE / sum(move^2) on 2025 OOF for the DST projection (to scale the 2026 [E]); per-month 2026 counts."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
m = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2025.parquet')
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt']); tr['i'] = np.arange(len(tr))
z = np.load(H + 'cache/w2/v28s_oof.npz'); y, f, month = z['y'], z['f'], z['month']
m = m.merge(tr, on='MVT_ID_mvt'); r = m[(~m.used) & (m.ADEP_mvt != m.ADES_mvt)]
L = (r.mvt_m_lobt - 3606).values; U = (r.mvt_m_lobt + 3606).values; fi = f[r.i.values]; yi = y[r.i.values]
conv = L <= -12
g = np.where(conv, np.maximum(fi, U), np.where((fi > L) & (fi < U), np.where(fi - L < U - fi, L, U), fi))
mv2 = (g - fi) ** 2; d = (yi - g) ** 2 - (yi - fi) ** 2
for nm, s in [('convex', conv), ('non-convex', ~conv), ('all', np.ones_like(conv))]:
    print(f'{nm}: sum move^2 {mv2[s].sum():.3e}  realised dSSE {d[s].sum():.3e}  ratio {(-d[s].sum()) / mv2[s].sum():.2f}')
q = pd.read_parquet(H + 'cache/hunt/ids_23_dst_2026.parquet')
print('2026 refined rows by month', q.SCHED_TIME_UTC_mvt.dt.month.value_counts().to_dict(), ' convex share', np.mean(q.L <= -12).round(3))
print('2025 refined rows: convex share', conv.mean().round(3), ' Jan', np.sum(r.month == 1), ' Jul', np.sum(r.month == 7))
print('MVT - AOBT_3(dst) quantiles 2025', np.nanquantile(r.mvt_m_aobt, [.1, .5, .9]).round(0), ' 2026', np.nanquantile(q.mvt_m_aobt, [.1, .5, .9]).round(0))
