"""s03 -- can a route-D departure (no NM record on its own row) recover NM's off-block clock
from its OWN ARRIVAL row at the destination, when the destination is one of the ten airports?

The arrival row carries its own _flt join.  If the join failed on the departure side but
worked on the arrival side, AOBT_3_flt there is NM's off-block at our airport -> route D
becomes route A.  Match: same FLIGHT_mvt callsign, arrival APT == our ADES_mvt, landing in
(our takeoff, our takeoff + 10 h), first such arrival.

Reports coverage (training and ranking) and, on training, how well the recovered clock
predicts y compared with v9~ on those rows.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import peak_rss_gb

M = CACHE / 'major'
for which in ['training', 'ranking']:
    dep = pd.read_parquet(M / f'dep_{which}.parquet', columns=['APT', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'off_aobt3',
                                                              'off_sched', 'y', 'month', 'FLIGHT_ID_mvt'])
    arr = pd.read_parquet(M / f'arr_{which}.parquet', columns=['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'FLIGHT_ID_mvt'])
    dep['row'] = np.arange(len(dep))
    D = ~np.isfinite(dep.off_aobt3.to_numpy())
    P = D & (dep.off_sched.to_numpy() > 7200)
    ten = set(arr.APT.astype(str).unique())
    q = dep.loc[D & dep.ADES_mvt.astype(str).isin(ten) & dep.FLIGHT_mvt.notna(),
                ['row', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt']].copy()
    q['k'] = q.ADES_mvt.astype(str) + '|' + q.FLIGHT_mvt.astype(str)
    a = arr.loc[arr.FLIGHT_mvt.notna(), ['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'FLIGHT_ID_mvt']].copy()
    a['k'] = a.APT.astype(str) + '|' + a.FLIGHT_mvt.astype(str)
    a = a.rename(columns={'MVT_TIME_UTC_mvt': 't_land'})
    q = q.sort_values('MVT_TIME_UTC_mvt'); a = a.sort_values('t_land')
    mm = pd.merge_asof(q, a[['k', 't_land', 'AOBT_3_flt', 'FLIGHT_ID_mvt']], left_on='MVT_TIME_UTC_mvt', right_on='t_land',
                       by='k', direction='forward', tolerance=pd.Timedelta(hours=10))
    got = mm.t_land.notna(); clk = mm.AOBT_3_flt.notna()
    print(f'\n=== {which}: route-D departures {D.sum():,} (P {P.sum():,}); with ADES in the ten airports {len(q):,}')
    print(f'   matched to an arrival row {got.sum():,} ({100*got.mean():.1f}%); arrival row has AOBT_3 {clk.sum():,} ({100*clk.mean():.1f}%)')
    rows = mm.row[clk].to_numpy()
    off_rec = (mm.MVT_TIME_UTC_mvt[clk] - mm.AOBT_3_flt[clk]).dt.total_seconds().to_numpy()
    inP = P[rows]
    print(f'   recovered clocks: in P {inP.sum():,}, not in P {(~inP).sum():,}; off_rec median {np.median(off_rec):.0f}, '
          f'share 0<off_rec<=7200 {np.mean((off_rec > 0) & (off_rec <= 7200)):.3f}')
    if which == 'training':
        y = dep.y.to_numpy()[rows]
        v9 = np.load(M / 'v9s_oof.npy')[rows]
        ok = (off_rec > 0) & (off_rec <= 7200)
        for nm, k in [('all recovered, 0<off<=7200', ok), ('  of which in P', ok & inP), ('  of which not in P', ok & ~inP)]:
            if k.sum() < 5:
                continue
            e_v9 = np.sqrt(np.mean((y[k] - v9[k]) ** 2)); e_rec = np.sqrt(np.mean((y[k] - off_rec[k]) ** 2))
            b = np.polyfit(off_rec[k], y[k], 1)
            print(f'   {nm:30s} n={k.sum():5d}  corr(y,off_rec) {np.corrcoef(y[k], off_rec[k])[0,1]:.3f}  '
                  f'rmse v9~ {e_v9:8.1f}  rmse off_rec {e_rec:8.1f}  y>7200 {np.mean(y[k] > 7200):.3f}  |y-off_rec|<=120 {np.mean(np.abs(y[k]-off_rec[k])<=120):.3f}')
        # SSE at stake
        e2 = (dep.y.to_numpy() - np.load(M / 'v9s_oof.npy')) ** 2
        S = np.isin(dep.month.to_numpy(), [1, 7])
        sel = np.zeros(len(dep), bool); sel[rows[ok]] = True
        print(f'   SSE share of total on recovered rows: 12m {100*e2[sel].sum()/e2.sum():.2f}%  scored {100*e2[sel & S].sum()/e2[S].sum():.2f}%')
print('peak', round(peak_rss_gb(), 2), 'GB')
