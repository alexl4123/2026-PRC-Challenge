"""SKEPTIC re-checks (scratch). Independent re-implementation of the clean destination link (A/B), exception counts,
day-shift exposure of B rows (local-midnight crossing), LFPG/LIRF one-sided analogs, EHAM-surge links 2026, C cap, file check."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 40)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
W = 3606
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT', 'SCHED', 'LOBT', 'CALLSIGN_flt',
     'FLIGHT_RULE_mvt', 'ADEP_flt', 'AOBT3', 'AIRCRAFT_OPERATOR_flt']
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam', 'LEBL': 'Europe/Madrid',
      'LEMD': 'Europe/Madrid', 'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome', 'LSZH': 'Europe/Zurich', 'LTFM': 'Europe/Istanbul'}


def link(yr):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=C); a = pd.read_parquet(H + f'{yr}_arr.parquet', columns=C)
    for x in (d, a):
        for c in ['APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt', 'FLIGHT_RULE_mvt', 'ADEP_flt']:
            x[c] = x[c].astype(str)
    d['i'] = np.arange(len(d))
    used = set(d.FLIGHT_ID_mvt.dropna().unique())
    D = d[d.LOBT.isna() & (d.FLIGHT_RULE_mvt == 'I') & (d.APT != d.ADES_mvt)]
    A = a[a.LOBT.notna()][['FLIGHT_ID_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'APT', 'MVT', 'LOBT', 'AOBT3', 'ADEP_flt']].copy()
    A['used'] = A.FLIGHT_ID_mvt.isin(used)
    A.columns = ['a_' + c for c in A.columns]
    L = []
    for key in ('a_FLIGHT_mvt', 'a_CALLSIGN_flt'):
        m = D.merge(A, left_on=['FLIGHT_mvt', 'APT', 'ADES_mvt'], right_on=[key, 'a_ADEP_mvt', 'a_APT'])
        m['t'] = m.a_MVT - m.MVT; m['key'] = key
        L.append(m[(m.t > 0) & (m.t < 86400)])
    m = pd.concat(L).sort_values('t').drop_duplicates('i').reset_index(drop=True)
    m['off'] = m.MVT - m.a_LOBT; m['lo'] = m.off - W; m['hi'] = m.off + W; m['osd'] = m.MVT - m.SCHED
    m['one'] = m.lo < -12; m['sx'] = (m.osd >= m.lo) & (m.osd <= m.hi)
    tz = m.APT.map(TZ)
    m['dloc'] = 0
    for apt, z in TZ.items():
        k = (m.APT == apt).to_numpy()
        if k.any():
            mv = pd.to_datetime(m.MVT[k] - 900, unit='s', utc=True).dt.tz_convert(z).dt.tz_localize(None).dt.normalize()
            sc = pd.to_datetime(m.SCHED[k], unit='s', utc=True).dt.tz_convert(z).dt.tz_localize(None).dt.normalize()
            m.loc[k, 'dloc'] = (mv - sc).dt.days.to_numpy()
    return m, d


V = np.load(R + 'cache/w2/v28s_oof.npz'); y = V['y']; f = V['f']; month = V['month']
m, d = link('tr')
m['y'] = y[m.i]; m['f'] = f[m.i]; m['month'] = month[m.i]
clean = ~m.a_used
inside = (m.y > m.lo) & (m.y < m.hi)
print('2025 links (all)', len(m), 'inside window', int(inside.sum()))
print('2025 CLEAN links', int(clean.sum()), 'exceptions', int((clean & inside).sum()), '| by key', m[clean].key.value_counts().to_dict())
c = m[clean]
print('  clean: ADEP_flt(arr NM) != APT:', int((c.a_ADEP_flt != c.APT).sum()), 'exceptions among them', int(((c.a_ADEP_flt != c.APT) & inside[clean]).sum()))
print('  clean one-sided', int(c.one.sum()), ' B (two-sided & stamp excl)', int((~c.one & c.sx).sum()), ' B2 (two-sided, stamp possible)', int((~c.one & ~c.sx).sum()))
print('  clean exception count with 3600 instead of 3606:', int(((c.y > c.off - 3600) & (c.y < c.off + 3600)).sum()),
      ' min margin to the window edges (s):', float(np.minimum(np.abs(c.y - c.lo), np.abs(c.y - c.hi)).min()))
# B rows: local-midnight crossing exposure
B = c[~c.one & c.sx]
print('\n2025 B rows crossing LOCAL midnight (MVT-900 local date > SCHED local date):', int((B.dloc >= 1).sum()), 'by APT', B[B.dloc >= 1].APT.value_counts().to_dict())
print('   of them y > hi (day-shift etc.):', int(((B.dloc >= 1) & (B.y > B.hi)).sum()))
print(B[B.dloc >= 1][['APT', 'FLIGHT_mvt', 'month', 'y', 'f', 'lo', 'hi', 'osd']].to_string())
# LIRF zone-P route-D 2025 (all rows, not only linked): class by local/UTC crossing
dt = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'off_sched'])
P = V['P']
x = dt[P].copy(); x['y'] = y[P]; x['f'] = f[P]
x['cls'] = np.where(np.abs(x.y - x.off_sched) <= 120, 'stamp', np.where(x.y > 43200, 'day', np.where(x.y <= 7200, 'normal', 'other')))
x = x[x.APT == 'LIRF']
ml = x.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None); sl = x.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None)
x['dloc'] = ((ml - pd.to_timedelta(900, 's')).dt.normalize() - sl.dt.normalize()).dt.days
x['dutc'] = ((x.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - x.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
print('\n2025 LIRF zone P classes by (dloc>=1, dutc>=1):'); print(pd.crosstab([x.dloc >= 1, x.dutc >= 1], x.cls))
print('day-shift rows: off_sched values', sorted(x[x.cls == 'day'].off_sched.round().tolist()))
k = (x.dloc >= 1) & (x.dutc < 1)
print('LIRF P local-crossing, no UTC-crossing rows, off_sched<12000:'); print(x[k & (x.off_sched < 12000)][['off_sched', 'y', 'f', 'cls']].to_string())

# One-sided analogs
o = c[c.one].copy(); o['nm_taxi'] = o.MVT - o.a_AOBT3
print('\n2025 clean one-sided rows at LFPG / LIRF with off_sched < hi (no stamp explanation) / LTFM:')
print(o[(o.APT == 'LFPG') | ((o.APT == 'LIRF') & (o.osd < o.hi))][['APT', 'FLIGHT_mvt', 'month', 'y', 'f', 'hi', 'osd', 'off', 'nm_taxi']].to_string())
print('one-sided 2025: nm_taxi (MVT - AOBT3_arr) quantiles', o.nm_taxi.quantile([.1, .5, .9]).round().tolist(), ' y - hi quantiles', (o.y - o.hi).quantile([.1, .5, .9]).round().tolist())
print('one-sided 2025 rows with nm_taxi < 1200 and y > hi:', int(((o.nm_taxi < 1200) & (o.y > o.hi)).sum()), 'of', int((o.nm_taxi < 1200).sum()))
# 2025 Jan+Jul one-sided per airport vs 2026
m6, d6 = link('ra')
c6 = m6[~m6.a_used]
print('\nclean links 2025 vs 2026:', len(c), len(c6), ' one-sided', int(c.one.sum()), int(c6.one.sum()))
print('one-sided by APT 2025 (all months) :', c[c.one].APT.value_counts().to_dict())
print('one-sided by APT 2025 (Jan+Jul)    :', c[c.one & c.month.isin([1, 7])].APT.value_counts().to_dict(), ' n clean Jan+Jul', int(c.month.isin([1, 7]).sum()))
print('one-sided by APT 2026              :', c6[c6.one].APT.value_counts().to_dict())
# EHAM surge 2026
dd6 = pd.to_datetime(d6.MVT, unit='s')
sur = (d6.APT == 'EHAM') & (dd6 >= '2026-01-02') & (dd6 < '2026-01-07')
rd = sur & d6.LOBT.isna() & (d6.FLIGHT_RULE_mvt == 'I')
print('\nEHAM 2-6 Jan 2026: deps', int(sur.sum()), 'IFR route D', int(rd.sum()))
s6 = m6[m6.i.isin(d6.i[rd])]
print('  linked', len(s6), 'clean', int((~s6.a_used).sum()), 'clean one-sided', int((~s6.a_used & s6.one).sum()))
print('  off (MVT - LOBT_arr) quantiles of clean surge links', s6[~s6.a_used].off.quantile([.1, .25, .5, .75, .9]).round().tolist())
print('  nm_taxi (MVT - AOBT3_arr) quantiles', (s6[~s6.a_used].MVT - s6[~s6.a_used].a_AOBT3).quantile([.1, .5, .9]).round().tolist())
# 2026 candidate B rows: local-midnight crossing
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
c6 = c6.assign(v28=v.reindex(c6.MVT_ID_mvt).to_numpy())
b6 = c6[~c6.one & c6.sx & (c6.v28 > c6.lo)]
print('\n2026 B moved rows: dloc, key, ADEP_flt of the arrival NM record')
print(b6[['APT', 'FLIGHT_mvt', 'a_FLIGHT_mvt', 'a_CALLSIGN_flt', 'a_ADEP_flt', 'key', 'dloc', 'v28', 'lo', 'hi', 'osd']].to_string())
a6 = c6[c6.one & (c6.v28 < c6.hi)]
print('2026 A moved rows:'); print(a6[['APT', 'FLIGHT_mvt', 'a_FLIGHT_mvt', 'a_CALLSIGN_flt', 'a_ADEP_flt', 'key', 'dloc', 'v28', 'hi', 'osd', 'off']].to_string())
print('2026 clean links with arr ADEP_flt != APT:', int((c6.a_ADEP_flt != c6.APT).sum()))
# C cap
cap = d.APT.isin(['LEBL', 'LEMD']).to_numpy()
print('\nC: 2025 LEBL/LEMD deps', int(cap.sum()), 'y > 3606:', int((y[cap] > 3606).sum()), 'max', y[cap].max(), ' y in (3500,3606]:', int(((y[cap] > 3500) & (y[cap] <= 3606)).sum()))
for apt in ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LFPG', 'LIRF', 'LSZH', 'LTFM']:
    k = (d.APT == apt).to_numpy()
    print(f'   {apt}: y in (3000,3606] {int(((y[k] > 3000) & (y[k] <= 3606)).sum())}, y > 3606 {int((y[k] > 3606).sum())}', end=';')
print()
k = (d6.APT == 'LEMD').to_numpy(); vv = v.reindex(d6.MVT_ID_mvt).to_numpy()
cr = d6[k & (vv > 3606)]
print('2026 LEMD rows with v28 > 3606:', len(cr), 'dates', pd.to_datetime(cr.MVT, unit='s').dt.date.value_counts().to_dict())
# candidate file check
cf = pd.read_parquet(H + 'cand_v28_ABC.parquet'); v28 = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet')
assert (cf.MVT_ID_mvt.to_numpy() == d6.MVT_ID_mvt.to_numpy()).all() or set(cf.MVT_ID_mvt) == set(v28.MVT_ID_mvt)
mm = cf.merge(v28, on='MVT_ID_mvt', suffixes=('_c', '_v'))
dif = mm[mm.TAXITIME_SEC_mvt_c != mm.TAXITIME_SEC_mvt_v]
print('\ncand file rows', len(cf), 'merged', len(mm), 'rows differing', len(dif), 'sum sq', float(((dif.TAXITIME_SEC_mvt_c - dif.TAXITIME_SEC_mvt_v) ** 2).sum()),
      'dtype', cf.TAXITIME_SEC_mvt.dtype, 'NaN', int(cf.TAXITIME_SEC_mvt.isna().sum()), 'min', int(cf.TAXITIME_SEC_mvt.min()))
