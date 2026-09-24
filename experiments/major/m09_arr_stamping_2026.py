"""m09 -- did the airports' recording behaviour change in 2026?  Arrival in-block
stamping at the scheduled time (|BLOCK - SCHED| <= 60 s), by airport x month, 2025 vs
2026, for arrivals with and without an NM record, and on the EHAM disruption days.
Provided data only (2026 arrivals are labelled in ranking.parquet)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
M = CACHE / 'major'
cols = ['APT', 'month', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'y']
out = []
for yr, f in [(2025, 'arr_training'), (2026, 'arr_ranking')]:
    a = pd.read_parquet(M / f'{f}.parquet', columns=cols)
    a = a[a.month.isin([1, 7])]
    a['st'] = (a.BLOCK_TIME_UTC_mvt - a.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
    a['late'] = a.off_sched > 1800          # landed >30 min after schedule: stamping would be visible
    a['noNM'] = a.off_aobt3.isna()
    for (apt, m), g in a.groupby(['APT', 'month'], observed=True):
        gl = g[g.late]
        out.append((apt, m, yr, len(g), 100 * g.st.mean(), 100 * gl.st.mean(), len(gl),
                    100 * gl[gl.noNM].st.mean() if gl.noNM.any() else np.nan, gl.noNM.sum()))
R = pd.DataFrame(out, columns=['apt', 'm', 'yr', 'n', 'stamp%', 'stamp%_late', 'n_late', 'stamp%_late_noNM', 'n_late_noNM'])
W = R.pivot_table(index=['apt', 'm'], columns='yr', values=['stamp%_late', 'stamp%_late_noNM', 'n_late_noNM'])
pd.set_option('display.width', 200)
print(W.round(2).to_string())
