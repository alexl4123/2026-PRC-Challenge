"""Route-D departures linked to their own arrival row (other airport of the 10) that has an NM record.
Does the departure's BLOCK lie OUTSIDE [LOBT_arr - 3606, LOBT_arr + 3606]? Which label-free subset makes it exact?"""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT', 'SCHED', 'LOBT', 'AOBT3', 'IOBT', 'EOBT1', 'ARVT3',
     'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt', 'FLIGHT_RULE_mvt', 'AIRCRAFT_TYPE_mvt', 'FLIGHT_TYPE_flt', 'STAND_mvt']
def link(yr):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=C + (['BLOCK'] if yr == 'tr' else []))
    a = pd.read_parquet(H + f'{yr}_arr.parquet', columns=C + ['BLOCK'])
    for x in (d, a):
        for c in ['APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt']: x[c] = x[c].astype(str)
    d['i'] = np.arange(len(d))
    D = d[d.LOBT.isna()]
    A = a[a.LOBT.notna()][['FLIGHT_mvt', 'ADEP_mvt', 'APT', 'MVT', 'BLOCK', 'LOBT', 'AOBT3', 'IOBT', 'EOBT1', 'ARVT3', 'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt', 'FLIGHT_ID_mvt', 'AIRCRAFT_TYPE_mvt']]
    A = A.rename(columns={c: 'a_' + c for c in A.columns if c not in ('FLIGHT_mvt',)})
    m = D.merge(A, left_on=['FLIGHT_mvt', 'APT', 'ADES_mvt'], right_on=['FLIGHT_mvt', 'a_ADEP_mvt', 'a_APT'])
    t = m.a_MVT - m.MVT
    m = m[(t > 0) & (t < 86400)].assign(t=t[(t > 0) & (t < 86400)]).sort_values('t').drop_duplicates('i')
    # strict: NM's actual off-block of the arrival record is within 2 h before our take-off (same flight instance)
    m['mvt_aobt'] = m.MVT - m.a_AOBT3; m['off_alobt'] = m.MVT - m.a_LOBT
    m['cs_eq'] = m.FLIGHT_mvt == m.a_CALLSIGN_flt; m['adep_eq'] = m.a_ADEP_flt == m.APT; m['type_eq'] = m.AIRCRAFT_TYPE_mvt.astype(str) == m.a_AIRCRAFT_TYPE_mvt.astype(str)
    return d, m
d, m = link('tr')
m['y'] = m.MVT - m.BLOCK; m['b_al'] = m.BLOCK - m.a_LOBT; m['excl'] = m.b_al.abs() > 3606
print('2025 route-D deps:', d.LOBT.isna().sum(), ' linked to an NM arrival:', len(m), ' exclusion holds:', m.excl.mean().round(4), ' violations:', (~m.excl).sum())
for c in ['cs_eq', 'adep_eq', 'type_eq']:
    print(' ', c, m.groupby(c).excl.agg(['size', 'mean', lambda x: (~x).sum()]).to_string())
m['strict'] = (m.mvt_aobt > -600) & (m.mvt_aobt < 7200)
print(' strict (MVT - AOBT3_arr in (-600, 7200)):', m.groupby('strict').excl.agg(['size', 'mean', lambda x: (~x).sum()]).to_string())
print(' by APT:', m.groupby('APT').excl.agg(['size', 'mean', lambda x: (~x).sum()]).to_string())
print(' by FLIGHT_RULE:', m.groupby('FLIGHT_RULE_mvt', observed=True).excl.agg(['size', 'mean', lambda x: (~x).sum()]).to_string())
print('violations:'); print(m[~m.excl][['APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'ADES_mvt', 'a_ADEP_flt', 'a_ADES_flt', 'y', 'b_al', 'off_alobt', 'mvt_aobt', 't', 'type_eq']].head(60).to_string())
print('holders sample:'); print(m[m.excl][['APT', 'FLIGHT_mvt', 'a_CALLSIGN_flt', 'y', 'b_al', 'off_alobt', 'mvt_aobt', 't']].head(30).to_string())
print('holders: y quantiles', m[m.excl].y.quantile([0, .1, .25, .5, .75, .9, 1]).round().to_dict())
print('holders: sign of b_al  (BLOCK before window / after window):', (m[m.excl].b_al < 0).sum(), (m[m.excl].b_al > 0).sum())
m.to_parquet(H + 'h07_link25.parquet')
d6, m6 = link('ra')
print('\n2026 route-D deps:', d6.LOBT.isna().sum(), ' linked:', len(m6), ' cs_eq:', m6.cs_eq.sum(), ' adep_eq:', m6.adep_eq.sum())
m6.to_parquet(H + 'h07_link26.parquet')
