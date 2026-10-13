"""Robustness of the clean exclusion rule to the link definition; per-month; 2025 vs 2026 label-free consistency; 2026 row details."""
import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 400)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT', 'SCHED', 'LOBT', 'AOBT3', 'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt', 'FLIGHT_RULE_mvt', 'FLIGHT_RULE_flt']
def links(yr, variant):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=C + (['BLOCK'] if yr == 'tr' else []))
    a = pd.read_parquet(H + f'{yr}_arr.parquet', columns=C + ['BLOCK'])
    for x in (d, a):
        for c in ['APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt']: x[c] = x[c].astype(str)
    d['i'] = np.arange(len(d)); used = set(d.FLIGHT_ID_mvt.dropna().unique())
    D = d[d.LOBT.isna() & (d.FLIGHT_RULE_mvt.astype(str) == 'I')]
    A = a[a.LOBT.notna()][['FLIGHT_mvt', 'ADEP_mvt', 'APT', 'MVT', 'BLOCK', 'LOBT', 'AOBT3', 'CALLSIGN_flt', 'ADEP_flt', 'FLIGHT_ID_mvt', 'FLIGHT_RULE_flt']]
    A = A.rename(columns={c: 'a_' + c for c in A.columns if c != 'FLIGHT_mvt'})
    if variant == 'callsign':   # link by the arrival's NM callsign instead of the airport flight number
        A2 = A.copy(); A2['FLIGHT_mvt'] = A2.a_CALLSIGN_flt; A = A2
    m = D.merge(A, left_on=['FLIGHT_mvt', 'APT', 'ADES_mvt'], right_on=['FLIGHT_mvt', 'a_ADEP_mvt', 'a_APT'])
    t = m.a_MVT - m.MVT; m = m.assign(t=t)
    tmax = {'base': 86400, 't12h': 43200, 'all24h': 86400, 'callsign': 86400, 'nmclock': 86400}[variant]
    m = m[(m.t > 0) & (m.t < tmax)]
    if variant == 'nmclock':     # NM actual off-block of the arrival record within (-1 h, +3 h) before our take-off
        m = m[((m.MVT - m.a_AOBT3) > -3600) & ((m.MVT - m.a_AOBT3) < 10800)]
    if variant != 'all24h':
        m = m.sort_values('t').drop_duplicates('i')
    m = m[~m.a_FLIGHT_ID_mvt.isin(used)].copy()
    m['off_alobt'] = m.MVT - m.a_LOBT; m['lo'] = m.off_alobt - 3606; m['hi'] = m.off_alobt + 3606; m['onesided'] = m.lo < -12
    m['month'] = pd.to_datetime(m.MVT, unit='s').dt.month
    if yr == 'tr':
        m['y'] = m.MVT - m.BLOCK; m['inside'] = (m.y >= m.lo) & (m.y <= m.hi); m['above'] = m.y > m.hi
    return d, m
for v in ['base', 't12h', 'all24h', 'callsign', 'nmclock']:
    _, m = links('tr', v)
    print(f'2025 variant {v:9s}: linked clean rows {len(m):4d} (unique deps {m.i.nunique()}), exclusion exceptions {m.inside.sum()}, two-sided above {((~m.onesided) & m.above).sum()}')
d, m = links('tr', 'base')
print('\nper month (2025, base): n, exceptions, one-sided'); print(m.groupby('month').agg(n=('i', 'size'), exc=('inside', 'sum'), one=('onesided', 'sum')).T.to_string())
d6, m6 = links('ra', 'base')
for nm, x, dd in (('2025', m, d), ('2026', m6, d6)):
    nD = (dd.LOBT.isna() & (dd.FLIGHT_RULE_mvt.astype(str) == 'I')).sum()
    print(f'{nm}: IFR route-D deps {nD}, clean-linked {len(x)} ({len(x) / nD:.3%}), one-sided {x.onesided.mean():.3f}, off_alobt q[.1,.5,.9] {x.off_alobt.quantile([.1, .5, .9]).round().tolist()}, '
          f'MVT-AOBT3_arr q[.1,.5,.9] {(x.MVT - x.a_AOBT3).quantile([.1, .5, .9]).round().tolist()}, flight time t q[.5] {x.t.median():.0f}, share Jan/Jul rows {x.month.isin([1, 7]).mean():.2f}')
print('2025 Jan+Jul only: clean-linked', m.month.isin([1, 7]).sum(), ' one-sided share', m[m.month.isin([1, 7])].onesided.mean().round(3))
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
m6 = m6.merge(v, on='MVT_ID_mvt', how='left'); m6['off_sched'] = m6.MVT - m6.SCHED
m6['new'] = np.where(m6.onesided, np.maximum(m6.f, m6.hi), np.where((m6.f > m6.lo) & (m6.f < m6.hi), np.maximum(m6.lo, 0), m6.f))
k = m6[m6.new != m6.f].copy(); k['move'] = k.new - k.f
ts = lambda s: pd.to_datetime(s, unit='s').dt.strftime('%m-%d %H:%M')
for c in ['MVT', 'SCHED', 'a_LOBT', 'a_AOBT3', 'a_MVT']: k[c + '_t'] = ts(k[c])
print('\n2026 rows that move:'); print(k[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'a_APT', 'onesided', 'SCHED_t', 'a_LOBT_t', 'a_AOBT3_t', 'MVT_t', 'a_MVT_t', 'f', 'new', 'move', 'off_sched']].sort_values('onesided').to_string())
for nm, kk in (('one-sided (guaranteed)', k.onesided), ('two-sided (0/780 empirical)', ~k.onesided)):
    s = k[kk].move
    print(f'{nm}: rows {kk.sum()}, sum sq {(s ** 2).sum():.4g}, RMSE bound {np.sqrt(235.3996 ** 2 - (s ** 2).sum() / 344841) - 235.3996:+.4f}')
s = k.move; print(f'both: rows {len(k)}, RMSE bound {np.sqrt(235.3996 ** 2 - (s ** 2).sum() / 344841) - 235.3996:+.4f}')
k[['MVT_ID_mvt', 'f', 'new', 'onesided']].to_parquet(H + 'h15_moves26.parquet')
