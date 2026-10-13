"""Raw file audit: final_ranking vs training (months 1,2,6,7), DEP and ARR."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd, pyarrow.parquet as pq
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt',
        'RUNWAY_mvt', 'STAND_mvt', 'LOBT_flt', 'CALLSIGN_flt', 'ADES_mvt', 'ADEP_mvt', 'MARKET_SEGMENT_flt', 'IOBT_flt',
        'FLIGHT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt', 'FLIGHT_RULE_mvt']
tr = pd.read_parquet(R + 'cache/training.parquet', columns=cols)
tr = tr[tr.MVT_TIME_UTC_mvt.dt.month.isin([1, 2, 6, 7])]
rk = pd.read_parquet(R + 'cache_final/ranking.parquet', columns=cols)
ri = pd.read_parquet(R + 'cache/ranking.parquet', columns=['MVT_ID_mvt'])
print('final rows', len(rk), 'initial ids in final:', ri.MVT_ID_mvt.isin(rk.MVT_ID_mvt).mean())
df = pd.concat([tr.assign(yr=2025), rk.assign(yr=2026)], ignore_index=True)
del tr
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category')
df['mo'] = df.MVT_TIME_UTC_mvt.dt.month
df['ym'] = df.yr.astype(str).str[2:] + '-' + df.mo.astype(str).str.zfill(2)
print(df.groupby(['ym', 'PHASE_mvt'], observed=True).size().unstack())
# missing share per column
for ph in ['DEP', 'ARR']:
    d = df[df.PHASE_mvt == ph]
    miss = d[[c for c in cols if c not in ('MVT_ID_mvt', 'PHASE_mvt', 'APT')]].isna().groupby(d.ym).mean().T.round(4)
    print('\nMISSING share', ph); print(miss)
# DEP per airport missing of key cols
d = df[df.PHASE_mvt == 'DEP']
for c in ['AOBT_3_flt', 'LOBT_flt', 'STAND_mvt', 'RUNWAY_mvt', 'WK_TBL_CAT_flt', 'SCHED_TIME_UTC_mvt', 'CALLSIGN_flt', 'ARVT_3_flt', 'IOBT_flt']:
    print('\nDEP missing by apt:', c)
    print(d[c].isna().groupby([d.APT, d.ym], observed=True).mean().unstack().round(3))
a = df[df.PHASE_mvt == 'ARR']
for c in ['BLOCK_TIME_UTC_mvt', 'STAND_mvt', 'AOBT_3_flt', 'SCHED_TIME_UTC_mvt']:
    print('\nARR missing by apt:', c)
    print(a[c].isna().groupby([a.APT, a.ym], observed=True).mean().unstack().round(3))
# unseen categories (vs all of training) per airport/month
trall = pd.read_parquet(R + 'cache/training.parquet', columns=['PHASE_mvt', 'APT', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'RUNWAY_mvt', 'ADES_mvt'])
trall = trall[trall.PHASE_mvt == 'DEP']
for c in ['STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'RUNWAY_mvt', 'ADES_mvt']:
    seen = set(zip(trall.APT.astype(str), trall[c].astype(str)))
    k = pd.Series([(x, y) in seen for x, y in zip(d.APT.astype(str), d[c].astype(str))], index=d.index)
    k = k | d[c].isna()
    print('\nDEP share UNSEEN in 2025 (apt-wise):', c)
    t = (~k).groupby([d.APT, d.ym], observed=True).mean().unstack().round(4)
    t.loc['ALL'] = (~k).groupby(d.ym).mean().round(4)
    print(t)
    if c in ('STAND_mvt', 'RUNWAY_mvt'):
        u = d[~k & (d.yr == 2026)]
        print(u.groupby(['APT', c], observed=True).size().sort_values(ascending=False).head(15))
# per-day counts
d['day'] = d.MVT_TIME_UTC_mvt.dt.date
cnt = d.groupby(['APT', 'ym', 'day'], observed=True).size().rename('n').reset_index()
cnt['med'] = cnt.groupby(['APT', 'ym']).n.transform('median')
print('\nDEP days with count < 60% of airport-month median:')
print(cnt[cnt.n < 0.6 * cnt.med].to_string())
print('days per ym-apt'); print(cnt.groupby(['APT', 'ym']).size().unstack())
a = df[df.PHASE_mvt == 'ARR'].copy(); a['day'] = a.MVT_TIME_UTC_mvt.dt.date
cnt = a.groupby(['APT', 'ym', 'day'], observed=True).size().rename('n').reset_index()
cnt['med'] = cnt.groupby(['APT', 'ym']).n.transform('median')
print('\nARR days with count < 60% of airport-month median:')
print(cnt[cnt.n < 0.6 * cnt.med].to_string())
# per-day AOBT_3 missing share in 2026 flagged
d['a3m'] = d.AOBT_3_flt.isna()
g = d.groupby(['APT', 'ym', 'day'], observed=True).a3m.mean().reset_index()
print('\nDEP days with AOBT_3 missing share > 10%:'); print(g[g.a3m > 0.10].to_string())
# offsets distribution
d['off_a3'] = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
d['off_sched'] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
d['off_lobt'] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
for c in ['off_a3', 'off_sched', 'off_lobt']:
    print('\n', c, 'quantiles by ym')
    print(d.groupby('ym')[c].quantile([.1, .5, .9]).unstack().round(0))
    print(d.groupby(['APT', 'ym'], observed=True)[c].median().unstack().round(0))
# hour of day (UTC) distribution check for timezone shifts
print('\nmedian UTC hour of DEP by apt/ym')
print(d.groupby(['APT', 'ym'], observed=True).MVT_TIME_UTC_mvt.apply(lambda s: (s.dt.hour + s.dt.minute / 60).median()).unstack().round(2))
print('\nmedian UTC hour of SCHED by apt/ym')
print(d.groupby(['APT', 'ym'], observed=True).SCHED_TIME_UTC_mvt.apply(lambda s: (s.dt.hour + s.dt.minute / 60).median()).unstack().round(2))
