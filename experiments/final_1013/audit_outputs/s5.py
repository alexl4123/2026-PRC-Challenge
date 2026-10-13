import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_outputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
rk = pd.read_parquet(O + 'rk.parquet')
r2 = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'off_aobt3', 'FLIGHT_mvt', 'CALLSIGN_flt'])
rk = rk.merge(r2, on='MVT_ID_mvt')
tr = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'off_aobt3', 'off_sched', 'y',
                     'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt'])
tr['f'] = np.load(R + 'cache/w2/v28s_oof.npz')['f']; tr['month'] = tr.MVT_TIME_UTC_mvt.dt.month; tr['D'] = tr.AOBT_3_flt.isna()
tr['P'] = tr.D & (tr.off_sched > 7200)
# month-level proxy
def pm(d, v):
    x = d[d.off_aobt3.notna()]
    g = x.groupby(['APT', 'month'])
    return pd.DataFrame({'px': g.off_aobt3.apply(lambda s: s.clip(0, 7200).mean()), 'v': g[v].mean()})
a = pm(tr[tr.month.isin([1, 2, 6, 7])], 'y'); b = pm(tr[tr.month.isin([1, 2, 6, 7])], 'f'); c = pm(rk, 'TAXITIME_SEC_mvt')
T = pd.DataFrame({'px25': a.px, 'y25': a.v, 'gap_y25': a.v - a.px, 'gap_oof25': b.v - b.px, 'px26': c.px, 'pred26': c.v, 'gap26': c.v - c.px})
T['dpx'] = T.px26 - T.px25; T['dpred_vs_y25'] = T.pred26 - T.y25
print('A/B/C rows: month-level proxy clip(MVT-AOBT3) and gaps'); print(T.round(1).to_string())
# LEMD 2026-02-05 detail
x = rk[(rk.APT == 'LEMD')]; x = x.assign(h=x.MVT_TIME_UTC_mvt.dt.floor('h'))
for dd in ['2026-02-04', '2026-02-05', '2026-02-06']:
    y = x[x.day.dt.strftime('%Y-%m-%d') == dd]
    print(dd, 'n', len(y), 'Dsh %.3f' % y.D.mean(), 'Psh %.3f' % y.P.mean(), 'off_sched med %.0f mean %.0f' % (y.off_sched.median(), y.off_sched.mean()),
          'pred %.0f' % y.TAXITIME_SEC_mvt.mean(), 'px %.0f' % y.off_aobt3.clip(0, 7200).mean(), 'pred ABC %.0f D %.0f' % (y[~y.D].TAXITIME_SEC_mvt.mean(), y[y.D].TAXITIME_SEC_mvt.mean()))
y = x[x.day.dt.strftime('%Y-%m-%d') == '2026-02-05']
print(y.groupby(y.MVT_TIME_UTC_mvt.dt.hour).agg(n=('D', 'size'), Dsh=('D', 'mean'), offs=('off_sched', 'median'), pred=('TAXITIME_SEC_mvt', 'mean'), px=('off_aobt3', 'median')).round(2).T.to_string())
# (3) hourly gaps: per APT, hours with 0 departures between 06 and 21 local-ish (use UTC 6-20) where same-APT/hour median count >= 8
rk['h'] = rk.MVT_TIME_UTC_mvt.dt.floor('h')
cnt = rk.groupby(['APT', 'h']).size()
out = []
for apt in sorted(rk.APT.unique()):
    s = cnt.loc[apt]; full = pd.date_range(s.index.min().floor('D'), s.index.max().ceil('D'), freq='h', inclusive='left')
    # restrict to days present in the months
    full = full[full.month.isin([1, 2, 6, 7])]
    s = s.reindex(full, fill_value=0)
    df = pd.DataFrame({'c': s.values, 'h': s.index.hour, 'm': s.index.month}, index=s.index)
    med = df.groupby(['m', 'h']).c.transform('median')
    low = df[(med >= 8) & (df.c <= 0.25 * med)]
    for t, r in low.iterrows(): out.append((apt, str(t), int(r.c), float(med[t])))
H = pd.DataFrame(out, columns=['APT', 'hour', 'n', 'median_same_hour'])
print('\nhours with <=25% of the month median count for that hour (median>=8):', len(H)); print(H.to_string())
# 2025 equivalent count for reference
tr['h'] = tr.MVT_TIME_UTC_mvt.dt.floor('h'); c25 = tr.groupby(['APT', 'h']).size(); n25 = 0
for apt in sorted(tr.APT.unique()):
    s = c25.loc[apt]; full = pd.date_range(s.index.min().floor('D'), s.index.max().ceil('D'), freq='h', inclusive='left'); s = s.reindex(full, fill_value=0)
    df = pd.DataFrame({'c': s.values, 'h': s.index.hour, 'm': s.index.month}, index=s.index); med = df.groupby(['m', 'h']).c.transform('median')
    n25 += int(((med >= 8) & (df.c <= 0.25 * med)).sum())
print('2025 (12 months) count of such hours:', n25)
# days coverage per APT
dc = rk.groupby(['APT', 'month']).day.nunique().unstack(); print('\ndays present per APT x month'); print(dc.to_string())
# daily count drop: n < 60% of +-3-day median
d26 = pd.read_parquet(O + 'd26.parquet'); d25 = pd.read_parquet(O + 'd25.parquet')
for nm, d in (('2026', d26), ('2025', d25)):
    rr = []
    for apt, g in d.groupby('APT'):
        g = g.sort_values('day').reset_index(drop=True)
        for i in range(len(g)):
            nb = g.n.iloc[[j for j in range(max(0, i - 3), min(len(g), i + 4)) if j != i]].median()
            if g.n[i] < 0.7 * nb: rr.append((apt, str(g.day[i].date()), int(g.n[i]), nb))
    print(f'\n{nm} airport-days with n < 70% of neighbour median: {len(rr)}'); print(rr if nm == '2026' else rr[:40])
# duplicate keys beyond MVT id: same APT, FLIGHT, MVT_TIME
dup = rk.duplicated(['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt'], keep=False); print('\n2026 duplicate (APT,FLIGHT,MVT_TIME) rows', int(dup.sum()))
dup2 = rk.duplicated(['APT', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt'], keep=False); print('2026 duplicate (APT,MVT_TIME,RUNWAY) rows', int(dup2.sum()))
# (4) unseen segments
def unseen(col):
    seen = set(zip(tr.APT, tr[col].astype(str)))
    k = list(zip(rk.APT, rk[col].astype(str))); m = np.array([t not in seen for t in k])
    u = rk[m].groupby(['APT', rk[m][col].astype(str)]).agg(n=('TAXITIME_SEC_mvt', 'size'), pred=('TAXITIME_SEC_mvt', 'mean')).reset_index()
    return m, u.sort_values('n', ascending=False)
apm = rk.groupby('APT').TAXITIME_SEC_mvt.mean()
for col in ['RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']:
    m, u = unseen(col)
    print(f'\n== unseen {col}: rows {int(m.sum())} ({100*m.mean():.2f}%), groups {len(u)}; by APT rows', rk[m].APT.value_counts().to_dict())
    u['apt_mean'] = u.APT.map(apm).round(0); print(u.head(15).round(0).to_string())
# stand family = alnum prefix
import re
fam = lambda s: re.sub(r'\d.*$', '', str(s)) if s is not None else 'None'
tr['sf'] = [fam(s) for s in tr.STAND_mvt]; rk['sf'] = [fam(s) for s in rk.STAND_mvt]
seen = set(zip(tr.APT, tr.sf)); m = np.array([t not in seen for t in zip(rk.APT, rk.sf)])
print('\nunseen stand-family rows', int(m.sum())); print(rk[m].groupby(['APT', 'sf']).TAXITIME_SEC_mvt.agg(['size', 'mean']).sort_values('size', ascending=False).head(15).round(0).to_string())
# missing stand/runway share
for col in ['RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']:
    a = tr[tr.month.isin([1, 2, 6, 7])].groupby(['APT', 'month'])[col].apply(lambda s: s.isna().mean()); b = rk.groupby(['APT', 'month'])[col].apply(lambda s: s.isna().mean())
    dd = (b - a).abs(); print(f'{col} missing share: max |2026-2025| per APT-month {dd.max():.4f} at {dd.idxmax()} (2026 {b[dd.idxmax()]:.4f} vs 2025 {a[dd.idxmax()]:.4f})')
