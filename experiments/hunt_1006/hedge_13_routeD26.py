"""2026 label-free: how much of v28 is a two-hypothesis hedge on route D outside P (tail anchor et vs bulk) and on A/B/C
(same production tail mixture), by airport.  Proxy: Wb^2 * pbar (1 - pbar) * (et - bulk/Wb)^2 (expected mixture SSE if pbar calibrated)."""
import sys, json
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
import assemble as A
pd.set_option('display.width', 250)
C = A.Components()
name = '__'.join(f'seqsw_{v}' for v in A.POOL)
Wt = json.load(open(CACHE / 'seq' / f'swap_{name}.json'))['weights_new']
v = A.shipped(C, 28)
pbar = C.pt.mean(0)
tot = 235.3996 ** 2 * len(v)
for key, g in (('route D', C.DnP), ('routes A/B/C', C.ABC)):
    W = Wt[key]; Wb = sum(W.get(f, 0.0) for f in A.BULK)
    bulk = sum(W[f] * C.member(f) for f in A.BULK if W.get(f, 0.0) > 0) / Wb
    h = Wb ** 2 * pbar * (1 - pbar) * (C.et - bulk) ** 2
    df = pd.DataFrame(dict(apt=C.apt[g], p=pbar[g], h=h[g], et=C.et[g], bulk=bulk[g], v=v[g], os=C.off_sched[g]))
    print(f'{key}: rows {g.sum():,}; rows with 0.05<p<0.95: {int(((df.p > .05) & (df.p < .95)).sum())}; expected hedge SSE {df.h.sum():.3e} '
          f'= {100 * df.h.sum() / tot:.2f}% of file SSE; et median {df.et.median():.0f}')
    print(df.groupby('apt').agg(n=('p', 'size'), mid=('p', lambda x: int(((x > .05) & (x < .95)).sum())), hedge=('h', 'sum')).to_string())
    print(' top rows:'); print(df.sort_values('h').tail(8).round(3).to_string())
