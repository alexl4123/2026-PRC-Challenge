"""ids_21 -- Rule A detail: label anatomy of the 61 rows (stamped share, slack y - U), per-month counts, variant
A+ = max(f, U, off_sched) when off_sched >= U (stamped hypothesis), priced LOMO-free (no constants) on 2025 OOF; 2026 moves."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
m = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2025.parquet')
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
tr['f'] = z['f']; tr['P'] = z['P']; tr['month'] = z['month']
m = m.merge(tr, on='MVT_ID_mvt', how='left')
sel = (~m.used) & (m.ADEP_mvt != m.ADES_mvt) & (m.mvt_m_lobt <= 3594)
r = m[sel].copy()
r['U'] = r.mvt_m_lobt + 3606
print('n', len(r), ' by month', r.month.value_counts().sort_index().to_dict())
print('slack y - U quantiles', np.quantile(r.y - r.U, [0, .1, .25, .5, .75, .9, 1]).round(0))
print('y - off_sched quantiles', np.quantile(r.y - r.off_sched, [0, .1, .25, .5, .75, .9, 1]).round(0))
print('stamped share |y - off_sched| <= 60:', np.mean(np.abs(r.y - r.off_sched) <= 60), '  <= 600:', np.mean(np.abs(r.y - r.off_sched) <= 600))
print('off_sched >= U share', np.mean(r.off_sched >= r.U))
N = 2085047
for name, g in [('A (clip to U)', np.maximum(r.f, r.U)),
                ('A+ (max(f,U,off_sched) if off_sched>=U)', np.where(r.off_sched >= r.U, np.maximum(np.maximum(r.f, r.U), r.off_sched), np.maximum(r.f, r.U)))]:
    d = (r.y - g) ** 2 - (r.y - r.f) ** 2
    jj = r.month.isin([1, 7])
    print(f'{name}: moved {np.sum(g != r.f)}, worse {np.sum(d > 0)}, dSSE 12m {d.sum():.3e}, Jan+Jul {d[jj].sum():.3e}, '
          f'dRMSE12m ~{d.sum() / N / (2 * 234.37):+.3f}; months better {sum(d[r.month == mm].sum() < 0 for mm in r.month.unique())}/{r.month.nunique()}'
          f' (months with change {sum(d[r.month == mm].sum() != 0 for mm in r.month.unique())})')
k = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2026.parquet')
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
k = k.merge(v, on='MVT_ID_mvt', how='left')
q = k[(~k.used) & (k.ADEP_mvt != k.ADES_mvt) & (k.mvt_m_lobt <= 3594)].copy()
q['U'] = q.mvt_m_lobt + 3606
gA = np.maximum(q.f, q.U)
gAp = np.where(q.off_sched >= q.U, np.maximum(gA, q.off_sched), gA)
for name, g in [('A', gA), ('A+', gAp)]:
    mv = g - q.f
    print(f'2026 {name}: rows moved {np.sum(mv != 0)}, sum move^2 {np.sum(mv ** 2):.3e}, '
          f'bound/scale {235.3996 - np.sqrt(235.3996 ** 2 - np.sum(mv ** 2) / 344841):.3f} s')
print(q.assign(A=gA, Ap=gAp)[['APT', 'FLIGHT_mvt', 'off_sched', 'U', 'f', 'A', 'Ap']].round(0).to_string())
# 2026 consistency checks (label-free): link flight-time and AOBT_3 consistency, ADES match
print('\n2026 rule rows: flight time h', (q.ft / 3600).round(2).tolist())
print('2026 rule rows: MVT - AOBT_3(dst) s', q.mvt_m_aobt.round(0).tolist())
print('2025 rule rows: MVT - AOBT_3(dst) quantiles', np.quantile(r.mvt_m_aobt, [0, .1, .5, .9, 1]).round(0))
