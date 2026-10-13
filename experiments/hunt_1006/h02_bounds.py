"""Systematic bound survey on 2025 departures: BLOCK - X for every clock X and pairwise min/max, overall and per route."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
d = pd.read_parquet(H + 'tr_dep.parquet')
y = d.MVT - d.BLOCK
print('y == TAXI:', np.allclose(y, d.TAXI), ' y min', y.min(), ' max', y.max())
nm = d.LOBT.notna()
print('route ABC y max', y[nm].max(), ' route D y max', y[~nm].max(), ' y<=0 rows', (y <= 0).sum(), ' y<0', (y < 0).sum())
CL = ['SCHED', 'LOBT', 'IOBT', 'EOBT1', 'ARVT1', 'AOBT3', 'ARVT3', 'MVT']
def edge(x, name):
    x = x.dropna().to_numpy()
    if len(x) == 0: return
    s = np.sort(x); n = len(s)
    q = np.quantile(s, [1e-5, 1e-4, 1e-3, .5, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5])
    lo, hi = s[0], s[-1]
    # edge density: rows within 6 s / 60 s of each extreme; gap between the k-th extreme values
    print(f'{name:34s} n={n:8d} min {lo:9.0f} max {hi:9.0f} | q1e-5 {q[0]:8.0f} q1e-4 {q[1]:8.0f} q.001 {q[2]:7.0f} med {q[3]:6.0f} '
          f'q.999 {q[4]:7.0f} q1-1e-4 {q[5]:8.0f} q1-1e-5 {q[6]:8.0f} | <=min+6 {(s <= lo + 6).sum():4d} <=min+60 {(s <= lo + 60).sum():4d} '
          f'>=max-6 {(s >= hi - 6).sum():4d} >=max-60 {(s >= hi - 60).sum():4d}')
for X in CL:
    edge(d.BLOCK - d[X], f'BLOCK-{X} all')
    edge((d.BLOCK - d[X])[~nm], f'BLOCK-{X} routeD')
print('\nPAIRWISE min/max combos (rows with both clocks):')
pairs = [(a, b) for i, a in enumerate(CL) for b in CL[i + 1:] if a != 'MVT' and b != 'MVT']
for a, b in pairs:
    lo = np.fmin(d[a], d[b]).where(d[a].notna() & d[b].notna()); hi = np.fmax(d[a], d[b]).where(d[a].notna() & d[b].notna())
    edge(d.BLOCK - lo, f'BLOCK-min({a},{b})'); edge(d.BLOCK - hi, f'BLOCK-max({a},{b})')
print('\nDERIVED:')
edge(d.BLOCK - (d.ARVT3 - (d.ARVT1 - d.EOBT1)), 'BLOCK-(ARVT3-(ARVT1-EOBT1))')
edge(d.BLOCK - (d.ARVT3 - (d.ARVT1 - d.LOBT)), 'BLOCK-(ARVT3-(ARVT1-LOBT))')
edge(d.BLOCK - (d.ARVT1 - (d.ARVT3 - d.AOBT3)), 'BLOCK-(ARVT1-(ARVT3-AOBT3))')
edge(d.ARVT3 - d.BLOCK, 'ARVT3-BLOCK')
edge(d.ARVT3 - d.MVT, 'ARVT3-MVT (label-free)')
edge(d.AOBT3 - d.MVT, 'AOBT3-MVT (label-free)')
edge(d.LOBT - d.SCHED, 'LOBT-SCHED (label-free)')
edge(d.IOBT - d.LOBT, 'IOBT-LOBT (label-free)')
edge(d.EOBT1 - d.LOBT, 'EOBT1-LOBT (label-free)')
