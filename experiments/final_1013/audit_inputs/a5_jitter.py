import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
cols = ['MVT_ID_mvt','PHASE_mvt','APT','RUNWAY_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','FLIGHT_ID_mvt','CALLSIGN_flt','AIRCRAFT_TYPE_mvt','FLIGHT_mvt']
a = pd.read_parquet(R+'cache/ranking.parquet', columns=cols).set_index('MVT_ID_mvt')
b = pd.read_parquet(R+'cache_final/ranking.parquet', columns=cols).set_index('MVT_ID_mvt')
bb = b.loc[a.index]
dm = (bb.MVT_TIME_UTC_mvt - a.MVT_TIME_UTC_mvt).dt.total_seconds()
db = (bb.BLOCK_TIME_UTC_mvt - a.BLOCK_TIME_UTC_mvt).dt.total_seconds()
arr = a.PHASE_mvt.eq('ARR')
print('MVT shift value counts (all):'); print(dm.value_counts().sort_index().to_string())
print('ARR: share MVT shift == BLOCK shift', (dm[arr] == db[arr]).mean(), ' corr', np.corrcoef(dm[arr], db[arr])[0,1])
ti_a = (a.BLOCK_TIME_UTC_mvt - a.MVT_TIME_UTC_mvt).dt.total_seconds()[arr]
ti_b = (bb.BLOCK_TIME_UTC_mvt - bb.MVT_TIME_UTC_mvt).dt.total_seconds()[arr]
print('ARR taxi-in identical share', (ti_a == ti_b).mean(), 'diff q', (ti_b - ti_a).quantile([.01,.5,.99]).tolist())
for c in ['FLIGHT_ID_mvt','CALLSIGN_flt','AIRCRAFT_TYPE_mvt','FLIGHT_mvt']:
    x, y = a[c].astype(str), bb[c].astype(str)
    ne = x != y
    print(c, 'differ', ne.mean().round(4), 'examples', list(zip(x[ne].head(5), y[ne].head(5))), 'na init->final', (a[c].isna() & ne).sum(), (bb[c].isna() & ne).sum())
# runway takeoff gap test: which file looks jittered relative to 2025 training?
def gaps(df):
    d = df[df.PHASE_mvt.eq('DEP')]
    k = d.APT.astype(str) + '|' + d.RUNWAY_mvt.astype(str)
    t = (d.MVT_TIME_UTC_mvt - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    s = pd.DataFrame({'k': k.to_numpy(), 't': t.to_numpy()}).sort_values(['k', 't'])
    g = s.groupby('k').t.diff().dropna()
    return g
tr = pd.read_parquet(R+'cache/training.parquet', columns=['PHASE_mvt','APT','RUNWAY_mvt','MVT_TIME_UTC_mvt'], filters=[('PHASE_mvt','==','DEP')])
tr = tr[tr.MVT_TIME_UTC_mvt.dt.month.isin([1,7])]
for name, df in [('train25 Jan+Jul', tr), ('initial26', a.reset_index()), ('final26 Jan+Jul', bb.reset_index())]:
    g = gaps(df)
    print(f'{name:16s} n={len(g):,} gap<=20s {np.mean(g<=20):.5f} gap<=40s {np.mean(g<=40):.5f} gap<=60s {np.mean(g<=60):.5f} p1 {np.percentile(g,1):.0f} p5 {np.percentile(g,5):.0f}')
# seconds-of-minute uniformity
for name, df in [('train25', tr), ('initial26', a), ('final26', bb)]:
    s = df[df.PHASE_mvt.eq('DEP')].MVT_TIME_UTC_mvt.dt.second if 'PHASE_mvt' in df else None
    print(name, 'sec==0 share', round((s == 0).mean(), 4))
