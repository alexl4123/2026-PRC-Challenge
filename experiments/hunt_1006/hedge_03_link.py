"""Hedge-row audit step 3: can a departure be linked to its arrival row at another of the 10 airports?
Checks FLIGHT_ID_mvt sharing between DEP and ARR rows, and the (FLIGHT_mvt, ADEP, ADES, time) link."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.io import load, peak_rss_gb
pd.set_option('display.width', 250)
for which in ('training', 'ranking'):
    cols = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'CALLSIGN_flt']
    dep = load(which, phase='DEP', columns=cols); arr = load(which, phase='ARR', columns=cols)
    print(which, 'DEP', len(dep), 'ARR', len(arr))
    print(' FLIGHT_ID_mvt dtype', dep.FLIGHT_ID_mvt.dtype, 'example', dep.FLIGHT_ID_mvt.head(3).tolist(), 'null dep', dep.FLIGHT_ID_mvt.isna().mean())
    s = set(dep.FLIGHT_ID_mvt.dropna().astype(str)) & set(arr.FLIGHT_ID_mvt.dropna().astype(str))
    print(' FLIGHT_ID shared by DEP and ARR:', len(s))
    print(' unique FLIGHT_ID per DEP rows:', dep.FLIGHT_ID_mvt.nunique(), 'of', len(dep))
    apts = set(dep.ADEP_mvt.astype(str).unique())
    inn = dep.ADES_mvt.astype(str).isin(apts)
    print(' DEP rows whose ADES is one of the 10 airports:', int(inn.sum()), f'{inn.mean():.3f}', '; with no LOBT:', int((inn & dep.LOBT_flt.isna()).sum()), 'of', int(dep.LOBT_flt.isna().sum()))
    # link by FLIGHT_mvt + ADEP + ADES, arrival MVT within (0, 24h] after the departure MVT
    a = arr[['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'FLIGHT_ID_mvt']].rename(columns={'MVT_TIME_UTC_mvt': 'am', 'LOBT_flt': 'alobt', 'FLIGHT_ID_mvt': 'afid'})
    for c in ('FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt'):
        a[c] = a[c].astype(str)
    dd = dep[['MVT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'FLIGHT_ID_mvt']].copy()
    for c in ('FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt'):
        dd[c] = dd[c].astype(str)
    dd = dd[dd.ADES_mvt.isin(apts) & (dd.FLIGHT_mvt != 'nan')]
    m = dd.merge(a, on=['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt'])
    t = (m.am - m.MVT_TIME_UTC_mvt).dt.total_seconds()
    m = m[(t > 0) & (t < 86400)].assign(t=t).sort_values('t').drop_duplicates('MVT_ID_mvt')
    print(' linked deps (FLIGHT_mvt, ADEP, ADES, 0<t<24h):', len(m), 'of', len(dd), '; flight time median', m.t.median(), 'p99', m.t.quantile(.99))
    print('  same FLIGHT_ID among linked:', float((m.FLIGHT_ID_mvt.astype(str) == m.afid.astype(str)).mean()))
    both = m.LOBT_flt.notna() & m.alobt.notna()
    print('  dep LOBT == arr LOBT where both present:', float((m.LOBT_flt[both] == m.alobt[both]).mean()), int(both.sum()))
    print('  dep no LOBT, arr LOBT present:', int((m.LOBT_flt.isna() & m.alobt.notna()).sum()), '; dep no LOBT, arr none:', int((m.LOBT_flt.isna() & m.alobt.isna()).sum()))
    print(' peak GB', peak_rss_gb())
