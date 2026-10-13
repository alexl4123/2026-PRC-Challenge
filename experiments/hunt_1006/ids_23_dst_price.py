"""ids_23 -- final pricing of the refined destination-NM exclusion ('DST window') on top of v28.
Rule rows: route-D departure (no NM record) linked to an ARR row of the same callsign/ADEP/ADES landing 0-24 h after
the take-off that carries an NM record, whose NM FLIGHT_ID is on no departure row (training + ranking), ADEP != ADES.
Rule (2025: 830 rows, 0 violations): y not in (L, U), L = MVT - LOBT_dst - 3606, U = MVT - LOBT_dst + 3606.
Projection: convex part (L <= -12): f -> max(f, U) [guaranteed]; non-convex part: f inside (L,U) -> nearest edge [E]."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
fid_tr = set(pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['FLIGHT_ID_mvt']).FLIGHT_ID_mvt.dropna())
fid_rk = set(pd.read_parquet(H + 'cache/major/dep_ranking.parquet', columns=['FLIGHT_ID_mvt']).FLIGHT_ID_mvt.dropna())
allfid = fid_tr | fid_rk
def project(f, L, U):
    g = f.copy()
    conv = L <= -12
    g = np.where(conv, np.maximum(f, U), g)
    ins = (~conv) & (f > L) & (f < U)
    g = np.where(ins, np.where(f - L < U - f, L, U), g)
    return g, conv
# ---- 2025 OOF
m = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2025.parquet')
m['used2'] = m.dfid.isin(allfid)
print('2025: used (own file) vs used (train+rank):', m.used.sum(), m.used2.sum())
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
y_all, f_all, month = z['y'], z['f'], z['month']
tr['i'] = np.arange(len(tr))
m = m.merge(tr, on='MVT_ID_mvt')
r = m[(~m.used2) & (m.ADEP_mvt != m.ADES_mvt)].copy()
L = (r.mvt_m_lobt - 3606).values; U = (r.mvt_m_lobt + 3606).values
viol = (r.y.values > L) & (r.y.values < U)
print('2025 rule rows', len(r), ' violations', viol.sum())
g_all = f_all.copy()
g, conv = project(f_all[r.i.values], L, U)
g_all[r.i.values] = g
ch = g_all != f_all
d = (y_all - g_all) ** 2 - (y_all - f_all) ** 2
rm = lambda h, msk: np.sqrt(np.mean((y_all[msk] - h[msk]) ** 2))
held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7]); A = np.ones_like(held)
print(f'2025 OOF: rows changed {ch.sum()} (convex {np.sum(ch[r.i.values] & conv)}), rows worse {np.sum(d > 0)}, '
      f'dSSE {d.sum():.3e};  dRMSE 12m {rm(g_all, A) - rm(f_all, A):+.4f}  held {rm(g_all, held) - rm(f_all, held):+.4f}  '
      f'Jan+Jul {rm(g_all, S) - rm(f_all, S):+.4f}')
mo = pd.Series(d[ch]).groupby(month[ch]).sum()
print('  dSSE by month (changed rows):', mo.round(0).to_dict())
# ---- 2026
k = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2026.parquet')
k['used2'] = k.dfid.isin(allfid)
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
k = k.merge(v, on='MVT_ID_mvt', how='left')
q = k[(~k.used2) & (k.ADEP_mvt != k.ADES_mvt)].copy()
L2 = (q.mvt_m_lobt - 3606).values; U2 = (q.mvt_m_lobt + 3606).values
g2, conv2 = project(q.f.values.astype(float), L2, U2)
q['g'] = g2; q['move'] = g2 - q.f.values; q['convex'] = conv2; q['L'] = L2; q['U'] = U2
mv = q[q.move != 0]
N = 344841; R = 235.3996
sc = (mv[mv.convex].move ** 2).sum(); snc = (mv[~mv.convex].move ** 2).sum()
print(f'\n2026: rule rows {len(q)}, rows moved {len(mv)} (convex {mv.convex.sum()}, non-convex {(~mv.convex).sum()}); '
      f'sum move^2 convex {sc:.3e} non-convex {snc:.3e}')
print(f'  guaranteed (convex part, if rule holds): RMSE <= {np.sqrt(R ** 2 - sc / N):.4f} ({np.sqrt(R ** 2 - sc / N) - R:+.4f} s)')
print(f'  if the non-convex truths also lie beyond the near edge: RMSE <= {np.sqrt(R ** 2 - (sc + snc) / N):.4f} '
      f'({np.sqrt(R ** 2 - (sc + snc) / N) - R:+.4f} s)')
print(mv[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt', 'off_sched', 'mvt_m_lobt', 'mvt_m_aobt', 'ft', 'f', 'L', 'U', 'g',
          'move', 'convex']].round(0).to_string())
q.to_parquet(H + 'cache/hunt/ids_23_dst_2026.parquet')
