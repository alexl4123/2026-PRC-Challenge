"""m07 -- the January 2026 northern disruption: which days, how big, and how v9 serves it.
Uses only provided data: 2026 departures (no labels) + 2026 arrivals (labelled)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
M = CACHE / 'major'
rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'MVT_ID_mvt'])
v9 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v9.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
rk['pred'] = v9.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
ar = pd.read_parquet(M / 'arr_ranking.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'off_sched', 'y', 'off_aobt3'])
ar25 = pd.read_parquet(M / 'arr_training.parquet', columns=['APT', 'month', 'MVT_TIME_UTC_mvt', 'off_sched', 'y', 'off_aobt3'])
for d in (rk, ar, ar25):
    d['day'] = d.MVT_TIME_UTC_mvt.dt.day
rk['D'] = rk.off_aobt3.isna(); rk['P'] = rk.D & (rk.off_sched > 7200)
pd.set_option('display.width', 250)
for apt in ['EHAM', 'LSZH', 'LFPG', 'EDDM', 'EDDF']:
    g = rk[(rk.APT == apt) & (rk.month == 1)]
    a = ar[(ar.APT == apt) & (ar.month == 1)]
    a25 = ar25[(ar25.APT == apt) & (ar25.month == 1)]
    base_in = a25.y.median(); base_n = len(g) / 31
    t = pd.DataFrame({'ndep': g.groupby('day').size(), 'P': g.groupby('day').P.sum(),
                      'D%': (g.groupby('day').D.mean() * 100).round(1),
                      'med_proxyA': g[~g.D].groupby('day').off_aobt3.median(),
                      'p90_proxyA': g[~g.D].groupby('day').off_aobt3.quantile(0.9),
                      'med_predA': g[~g.D].groupby('day').pred.median(),
                      'med_so': g.groupby('day').off_sched.median(),
                      'arr_n': a.groupby('day').size(), 'arr_taxiin_med': a.groupby('day').y.median(),
                      'arr_taxiin_p90': a.groupby('day').y.quantile(0.9)})
    bad = t[(t.P >= 10) | (t['D%'] > 4)]
    print(f'\n=== {apt} Jan 2026: 2025-Jan arrival taxi-in median {base_in:.0f}; dep/day {base_n:.0f}.  Days with P>=10 or D%>4:')
    print(bad.round(0).to_string())
    # route A: is v9 following the NM proxy when the proxy is long?
    kA = ~g.D & (g.off_aobt3 > 2400)
    if kA.any():
        print(f'  route A rows with proxy>2400: {kA.sum()}, median proxy {g.off_aobt3[kA].median():.0f}, median v9 pred {g.pred[kA].median():.0f}')
