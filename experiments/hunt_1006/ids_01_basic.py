"""ids_01 -- basic construction facts: ID presence/duplicates, label arithmetic, seconds resolution, y range."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load

C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'APT', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
     'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'LOBT_flt', 'AOBT_3_flt', 'ADEP_mvt', 'ADES_mvt']
pd.set_option('display.width', 220)
for which in ['training', 'ranking']:
    d = load(which, columns=C)
    print('=====', which, len(d), d.PHASE_mvt.value_counts().to_dict())
    for ph in ['DEP', 'ARR']:
        g = d[d.PHASE_mvt.eq(ph)]
        fid = g.FLIGHT_ID_mvt
        print(f'-- {ph}: n={len(g)} FLIGHT_ID notna={fid.notna().sum()} unique={fid.nunique()} dup rows(notna)={fid.notna().sum()-fid.nunique()}'
              f' | LOBT notna={g.LOBT_flt.notna().sum()} AOBT3 notna={g.AOBT_3_flt.notna().sum()}')
        print('   crosstab FLIGHT_ID notna x LOBT notna:\n', pd.crosstab(fid.notna(), g.LOBT_flt.notna()).to_string())
        print('   MVT_ID unique', g.MVT_ID_mvt.nunique(), 'min', g.MVT_ID_mvt.min(), 'max', g.MVT_ID_mvt.max())
        print('   BLOCK notna', g.BLOCK_TIME_UTC_mvt.notna().sum(), 'TAXITIME notna', g.TAXITIME_SEC_mvt.notna().sum(),
              'SCHED notna', g.SCHED_TIME_UTC_mvt.notna().sum(), 'MVT notna', g.MVT_TIME_UTC_mvt.notna().sum())
        # seconds component
        for c in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt']:
            s = g[c].dropna()
            if len(s) == 0:
                continue
            sec0 = (s.dt.second == 0)
            print(f'   {c}: share sec==0 overall {sec0.mean():.4f}; ns!=0 {(s.dt.nanosecond+s.dt.microsecond != 0).mean():.4f}')
        s = g[['APT']].copy()
        for c in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']:
            s[c[:5]] = (g[c].dt.second == 0).where(g[c].notna())
        print('   sec==0 share by airport:\n', s.groupby('APT', observed=True).mean().round(3).T.to_string())
        if g.BLOCK_TIME_UTC_mvt.notna().any():
            yy = (g.MVT_TIME_UTC_mvt - g.BLOCK_TIME_UTC_mvt).dt.total_seconds()
            if ph == 'ARR':
                yy = -yy
            t = g.TAXITIME_SEC_mvt
            print('   TAXITIME == (MVT-BLOCK) sign-adjusted:', (np.abs(t - yy) < 0.5).sum(), 'of', t.notna().sum(),
                  ' max abs diff', np.nanmax(np.abs(t - yy)))
            print('   y quantiles', np.nanquantile(t, [0, 1e-5, 1e-4, .001, .01, .5, .99, .999, .9999, .99999, 1]).round(0))
            print('   y top 20', np.sort(t.dropna().values)[-20:])
            print('   y bottom 20', np.sort(t.dropna().values)[:20])
    del d
