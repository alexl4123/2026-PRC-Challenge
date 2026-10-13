"""ids_17 -- date artefact: recorded BLOCK local date = SCHED local date (seen at LIRF). For every airport, rows whose
take-off local date differs from the SCHED local date: label classes (shifted by k days / normal / stamped), by route,
v28 OOF on them; 2026 counts and v28 predictions."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam', 'LEBL': 'Europe/Madrid',
      'LEMD': 'Europe/Madrid', 'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome', 'LSZH': 'Europe/Zurich', 'LTFM': 'Europe/Istanbul'}
cols = ['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'LOBT_flt', 'off_sched', 'off_lobt', 'y', 'month']
def prep(which):
    d = pd.read_parquet(H + f'cache/major/dep_{which}.parquet', columns=cols)
    d['APT'] = d.APT.astype(str)
    out = []
    for a, g in d.groupby('APT'):
        g = g.copy()
        tz = TZ[a]
        g['m_loc'] = g.MVT_TIME_UTC_mvt.dt.tz_convert(tz)
        g['s_loc'] = g.SCHED_TIME_UTC_mvt.dt.tz_convert(tz)
        g['k'] = (g.m_loc.dt.normalize().dt.tz_localize(None) - g.s_loc.dt.normalize().dt.tz_localize(None)).dt.days
        if which == 'training':
            g['b_loc'] = g.BLOCK_TIME_UTC_mvt.dt.tz_convert(tz)
            g['kb'] = (g.b_loc.dt.normalize().dt.tz_localize(None) - g.s_loc.dt.normalize().dt.tz_localize(None)).dt.days
        out.append(g)
    d = pd.concat(out).sort_index()
    d['D'] = d.LOBT_flt.isna()
    return d
tr = prep('training')
z = np.load(H + 'cache/w2/v28s_oof.npz')
tr['f'] = z['f']; tr['P'] = z['P']
assert np.allclose(tr.y, z['y'])
tr['res'] = tr.y - tr.f
# classes for k >= 1
x = tr[tr.k >= 1].copy()
x['ynorm'] = x.y - 86400 * x.k
x['cls'] = np.select([(x.kb == 0) & (x.ynorm > -600) & (x.ynorm < 7200),     # block date = sched date, time-of-day plausible
                      (x.y - x.off_sched).abs() <= 60,                         # stamped at SCHED
                      (x.y > -20) & (x.y < 7200)],                             # normal
                     ['dayshift', 'stamped', 'normal'], 'other')
print('2025 departures with take-off local date > SCHED local date:', len(x))
print(pd.crosstab([x.APT, x.D], x.cls).to_string())
print('\nby k (all airports):', pd.crosstab(x.k, x.cls).to_string())
# LIRF detail by month and route
l = x[x.APT == 'LIRF']
print('\nLIRF by month x cls:\n', pd.crosstab(l.month, l.cls).to_string())
print('\nLIRF dayshift rows: ynorm quantiles', np.quantile(l[l.cls == 'dayshift'].ynorm, [.1, .5, .9]).round(0))
# v28 OOF on these rows: SSE by class
g = x.groupby(['APT', 'cls']).agg(n=('res', 'size'), sse=('res', lambda r: (r ** 2).sum()), mean_f=('f', 'mean'), mean_y=('y', 'mean'))
print('\nv28 OOF on k>=1 rows (12 months):\n', g[g.n > 0].round(0).to_string())
jj = x[x.month.isin([1, 7])]
print('Jan+Jul SSE by apt/cls:\n', jj.groupby(['APT', 'cls']).res.apply(lambda r: (r ** 2).sum()).round(0).to_string())
# also k == 0 rows with day-shift (BLOCK on previous local day while take-off on SCHED date)?
y0 = tr[(tr.k == 0) & (tr.y > 80000)]
print('\nk==0 rows with y > 80000:', len(y0))
print(y0[['APT', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'y', 'f', 'D']].to_string())
x.to_parquet(H + 'cache/hunt/ids_17_k_train.parquet')

rk = prep('ranking')
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
rk = rk.merge(v, on='MVT_ID_mvt', how='left')
r = rk[rk.k >= 1]
print('\n2026 rows with k>=1:', len(r), pd.crosstab(r.APT, r.D).to_string())
print(r[r.APT == 'LIRF'][['MVT_ID_mvt', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'off_sched', 'off_lobt', 'k', 'f', 'D']].to_string())
rk[rk.k >= 1].to_parquet(H + 'cache/hunt/ids_17_k_rank.parquet')
