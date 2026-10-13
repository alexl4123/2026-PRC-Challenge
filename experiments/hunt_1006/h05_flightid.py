"""FLIGHT_ID_mvt: is it the NM flight id? Route-D rows that carry it; links to arrival rows."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT', 'BLOCK', 'SCHED', 'LOBT', 'AOBT3', 'ARVT3', 'CALLSIGN_flt', 'FLIGHT_RULE_mvt']
for yr in ('tr', 'ra'):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=C); a = pd.read_parquet(H + f'{yr}_arr.parquet', columns=C)
    for x in (d, a): x['APT'] = x.APT.astype(str); x['route'] = np.where(x.LOBT.notna(), 'NM', 'noNM')
    print(f'\n===== {yr}: deps {len(d)}  arrs {len(a)}')
    for nm, x in (('DEP', d), ('ARR', a)):
        print(nm, pd.crosstab(x.route, x.FLIGHT_ID_mvt.notna()).to_string())
    # route-D deps that carry a FLIGHT_ID
    w = d[(d.route == 'noNM') & d.FLIGHT_ID_mvt.notna()].copy()
    m = w.merge(a[['FLIGHT_ID_mvt', 'APT', 'MVT', 'BLOCK', 'LOBT', 'AOBT3', 'ARVT3', 'route']].rename(columns=lambda c: c if c == 'FLIGHT_ID_mvt' else 'a_' + c), on='FLIGHT_ID_mvt', how='left')
    m['y'] = m.MVT - m.BLOCK; m['off_sched'] = m.MVT - m.SCHED; m['off_alobt'] = m.MVT - m.a_LOBT; m['off_aaobt'] = m.MVT - m.a_AOBT3; m['B-aL'] = m.BLOCK - m.a_LOBT
    print('route-D deps with FLIGHT_ID:', len(w), ' linked to an arrival row:', m.a_APT.notna().sum(), ' with NM on that arrival:', (m.a_route == 'NM').sum())
    print(m[['APT', 'FLIGHT_mvt', 'ADES_mvt', 'FLIGHT_RULE_mvt', 'y', 'off_sched', 'a_APT', 'a_route', 'off_alobt', 'off_aaobt', 'B-aL']].to_string())
    # dep<->arr link via FLIGHT_ID in general
    both = d.dropna(subset=['FLIGHT_ID_mvt']).merge(a.dropna(subset=['FLIGHT_ID_mvt'])[['FLIGHT_ID_mvt', 'LOBT', 'MVT', 'APT']].rename(columns={'LOBT': 'a_LOBT', 'MVT': 'a_MVT', 'APT': 'a_APT'}), on='FLIGHT_ID_mvt')
    print('dep-arr pairs via FLIGHT_ID:', len(both), ' same LOBT:', (both.LOBT == both.a_LOBT).mean(), ' arr after dep:', (both.a_MVT > both.MVT).mean())
