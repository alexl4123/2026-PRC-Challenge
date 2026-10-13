"""ids_10 -- where does v28 hedge in 2026? Two-point proxy: a = off_sched (stamped), b = 1000 (normal);
expected SSE ~ (f-b)(a-f) for b<f<a. Same proxy on 2025 OOF to compare against realised SSE."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 40)
rk = pd.read_parquet(H + 'cache/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt',
                                                                   'MVT_TIME_UTC_mvt', 'off_sched', 'off_lobt', 'off_aobt3', 'month', 'STAND_mvt'])
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
rk = rk.merge(v, on='MVT_ID_mvt', how='left')
rk['D'] = rk.off_lobt.isna()
rk['P'] = rk.D & (rk.off_sched > 7200)
b = 1000.0
def hedge(f, a):
    return np.where((f > b) & (f < a), (f - b) * (a - f), 0.0)
rk['h'] = hedge(rk.f.values, rk.off_sched.values)
N = len(rk)
print('2026: total proxy hedge SSE', rk.h.sum(), ' -> RMSE^2 share', rk.h.sum() / N, ' (v28 RMSE^2 =', 235.3996 ** 2, ')')
print(rk.groupby(['P', 'D']).h.agg(['sum', 'count']).assign(rmse2=lambda x: x['sum'] / N).to_string())
print(rk[rk.P].groupby(['APT', 'month']).h.agg(['sum', 'count']).assign(rmse2=lambda x: x['sum'] / N).round(1).to_string())
top = rk.sort_values('h', ascending=False).head(40)
print(top[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'off_sched', 'f', 'h', 'P', 'D', 'off_lobt']].round(0).to_string())
print('top-10 share', top.h.head(10).sum() / rk.h.sum(), ' top-40 share', top.h.sum() / rk.h.sum())
# 2025 OOF comparison
z = np.load(H + 'cache/w2/v28s_oof.npz')
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['APT', 'off_sched', 'month'])
tr['f'] = z['f']; tr['y'] = z['y']; tr['P'] = z['P']; tr['D'] = z['D']
tr['h'] = hedge(tr.f.values, tr.off_sched.values)
tr['se'] = (tr.y - tr.f) ** 2
g = tr.groupby(['P', 'D'])[['h', 'se']].sum()
print('\n2025 OOF, proxy hedge vs realised SSE (all months):\n', g.to_string())
jj = tr[tr.month.isin([1, 7])]
print('2025 Jan+Jul:\n', jj.groupby(['P', 'D'])[['h', 'se']].sum().to_string())
print('2025 top realised-SSE rows share: top 20 rows', np.sort(tr.se.values)[-20:].sum() / tr.se.sum())
