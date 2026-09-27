"""How much of v21's remaining error is SHARED by neighbouring departures (queue / airport state) vs flight-specific?
(Exploratory MEASUREMENT, 2026-09-27.)

For each departure: r = y - prediction (v21 OOF; v17 OOF for comparison); nbr = mean r of the OTHER departures at the
same airport taking off within +-W min (W = 15, 30, 60). Reported per airport and per y-band (routes outside zone P,
day-scale rows excluded): corr(r, nbr), and the share of the band's SSE that a perfect "shared state" term would remove
= 1 - SSE(r - b*nbr) / SSE(r) with b fitted by least squares (an optimistic bound: nbr uses labels, which the ranking
file does not have; it measures how much airport-time structure is LEFT, not what is reachable).

    python experiments/tail/t05_shared_residual.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'drivers_0926'))
import numpy as np, pandas as pd
from prc.config import CACHE
from d02_decompose import mechanism

b = np.load(CACHE / 'v11' / 'base.npz'); y, P = b['y'], b['P']
f21 = np.load(CACHE / 'w2' / 'v21s_oof.npy').astype(float)
r0 = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet'); f17 = r0.pred.to_numpy()
mech = mechanism(r0); day = mech == 'day'
dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt'])
apt = dd.APT.astype(str).to_numpy()
t = (dd.MVT_TIME_UTC_mvt - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()
band = np.select([y <= 900, y <= 1800, y <= 3600], ['<15m', '15-30m', '30-60m'], '>60m')
keep = ~P & ~day & np.isfinite(y)
rows = []
for W in (15, 30, 60):
    nb = {}
    for nm, f in (('v21', f21), ('v17', f17)):
        r = y - f
        out = np.full(len(y), np.nan)
        for a in np.unique(apt):
            k = np.flatnonzero((apt == a) & keep)
            o = np.argsort(t[k]); kk = k[o]; tt = t[kk]; rr = r[kk]
            cs = np.r_[0, np.cumsum(rr)]
            lo, hi = np.searchsorted(tt, tt - W * 60), np.searchsorted(tt, tt + W * 60, side='right')
            n = hi - lo - 1
            out[kk] = np.where(n > 0, (cs[hi] - cs[lo] - rr) / np.maximum(n, 1), np.nan)
        nb[nm] = (r, out)
    for a in ['LTFM', 'LFPG', 'LIRF', 'EGLL', 'EHAM', 'ALL']:
        for bd in ['<15m', '15-30m', '30-60m', '>60m', 'all']:
            k = keep & (True if a == 'ALL' else apt == a) & (True if bd == 'all' else band == bd)
            rec = dict(W=W, apt=a, band=bd, rows=int(k.sum()))
            for nm in ('v21', 'v17'):
                r, nbr = nb[nm]; kk = k & np.isfinite(nbr)
                if kk.sum() < 50:
                    continue
                bb = np.sum(r[kk] * nbr[kk]) / np.sum(nbr[kk] ** 2)
                rec[f'{nm}_rmse'] = np.sqrt(np.mean(r[kk] ** 2)); rec[f'{nm}_mean_r'] = r[kk].mean()
                rec[f'{nm}_corr'] = np.corrcoef(r[kk], nbr[kk])[0, 1]
                rec[f'{nm}_shared_pct'] = 100 * (1 - np.sum((r[kk] - bb * nbr[kk]) ** 2) / np.sum(r[kk] ** 2))
            rows.append(rec)
T = pd.DataFrame(rows); T.to_csv(CACHE / 'tail' / 't05_shared_residual.csv', index=False)
pd.set_option('display.width', 200)
print(T[T.W == 30].round(2).to_string(index=False))
print('\nALL airports, all bands, by window:'); print(T[(T.apt == 'ALL') & (T.band == 'all')].round(3).to_string(index=False))
