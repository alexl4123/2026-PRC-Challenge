import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_outputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
rk = pd.read_parquet(O + 'rk.parquet')
rk = rk.merge(pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'off_aobt3']), on='MVT_ID_mvt')
tr = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'y', 'RUNWAY_mvt'])
tr['f'] = np.load(R + 'cache/w2/v28s_oof.npz')['f']; tr['day'] = tr.MVT_TIME_UTC_mvt.dt.floor('D'); tr['month'] = tr.MVT_TIME_UTC_mvt.dt.month
# --- per-row gap model: within ABC rows, residual r = v - clip(off_aobt3); daily mean residual. Expected from 2025 labels, compare OOF & 2026
def dl(d, v):
    x = d[d.off_aobt3.notna()].assign(px=lambda t: t.off_aobt3.clip(0, 7200))
    x['r'] = x[v] - x.px
    g = x.groupby(['APT', 'day']); return pd.DataFrame({'px': g.px.mean(), 'r': g.r.mean(), 'n': g.size()}).reset_index()
y25 = dl(tr, 'y'); f25 = dl(tr, 'f'); p26 = dl(rk, 'TAXITIME_SEC_mvt')
# fit per APT: r_y ~ a + b*px (daily, 2025)
co = {}
for apt, g in y25.groupby('APT'):
    b, a = np.polyfit(g.px, g.r, 1); co[apt] = (a, b)
for d in (y25, f25, p26):
    d['exp'] = [co[a][0] + co[a][1] * p for a, p in zip(d.APT, d.px)]; d['adj'] = d.r - d.exp
f25['err'] = f25.r - y25.r   # = daily mean OOF error on ABC rows
print('slope of daily (y-px) on px per APT 2025:', {k: round(v[1], 3) for k, v in co.items()})
print('\nadj = daily (v - px) minus 2025-label expectation given px. Per APT: 2025 OOF adj [0.5,50,99.5,max|.|], 2026 pred adj')
A = f25.groupby('APT').adj.agg(q005=lambda s: s.quantile(.005), med='median', q995=lambda s: s.quantile(.995), mx=lambda s: s.abs().max())
B = p26.groupby('APT').adj.agg(q005=lambda s: s.quantile(.005), med='median', q995=lambda s: s.quantile(.995), mx=lambda s: s.abs().max())
print(pd.concat([A, B], axis=1, keys=['2025 OOF', '2026']).round(1).to_string())
p26['month'] = p26.day.dt.month; f25['month'] = f25.day.dt.month
print('\nmonthly mean adj (row-weighted by n): 2025 OOF vs 2026')
mm = lambda d: d.groupby(['APT', 'month']).apply(lambda g: np.average(g.adj, weights=g.n)).unstack()
print(pd.concat([mm(f25[f25.month.isin([1, 2, 6, 7])]), mm(p26)], axis=1, keys=['2025', '2026']).round(1).to_string())
lim = f25.groupby('APT').adj.agg(['min', 'max'])
x = p26.merge(lim, left_on='APT', right_index=True); x = x[(x.adj < x['min']) | (x.adj > x['max'])]
print('\n2026 days outside 2025 OOF adj range:'); print(x[['APT', 'day', 'n', 'px', 'r', 'exp', 'adj', 'min', 'max']].round(1).to_string())
# --- LFPG runway breakdown
for apt in ['LFPG']:
    a = tr[(tr.APT == apt) & tr.month.isin([6, 7])]; b = rk[(rk.APT == apt) & rk.month.isin([6, 7])]
    ga = a.groupby('RUNWAY_mvt').agg(n25=('y', 'size'), y25=('y', 'mean'), f25=('f', 'mean'), px25=('off_aobt3', lambda s: s.clip(0, 7200).mean()))
    gb = b.groupby('RUNWAY_mvt').agg(n26=('TAXITIME_SEC_mvt', 'size'), p26=('TAXITIME_SEC_mvt', 'mean'), px26=('off_aobt3', lambda s: s.clip(0, 7200).mean()))
    print(f'\n{apt} Jun+Jul by runway'); print(ga.join(gb, how='outer').round(0).to_string())
    # daily trend LFPG px and pred
    print(p26[p26.APT == apt].assign(m=lambda t: t.day.dt.month).groupby('m')[['px', 'r', 'adj']].mean().round(1))
    print(f25[f25.APT == apt].groupby('month')[['px', 'r', 'adj']].mean().round(1).T.to_string())
