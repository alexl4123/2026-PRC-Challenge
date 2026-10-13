"""Hedge-row audit step 1: where does the v28-structure 2025 OOF carry its SSE, by hypothesis class.
Read-only; writes cache/hunt/hedge_base.parquet (per-row training columns we reuse)."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load, peak_rss_gb

V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P, D, f, f27, ol = V['y'], V['month'], V['P'], V['D'], V['f'], V['f27'], V['off_lobt']
W = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
Pidx, p14, pbP, so = W['Pidx'], W['p14'], W['pbP'], W['so']
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet',
                    columns=['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                             'BLOCK_TIME_UTC_mvt', 'off_sched', 'off_aobt3', 'off_iobt', 'off_eobt1', 'y'])
assert np.allclose(d.y.to_numpy(), y)
apt = d.APT.astype(str).to_numpy()
osch = d.off_sched.to_numpy()
N = len(y); tot = ((y - f) ** 2).sum()
S = np.isin(month, [1, 7]); held = ~np.isin(month, [3, 6, 9, 12])
e2 = (y - f) ** 2
print(f'N {N:,}  RMSE 12m {np.sqrt(tot / N):.3f}  Jan+Jul {np.sqrt(e2[S].mean()):.3f}  held {np.sqrt(e2[held].mean()):.3f}')

A = ~D & ~P; DnP = D & ~P
def show(name, m):
    s = e2[m].sum()
    print(f'{name:42s} rows {m.sum():9,}  SSE share 12m {s / tot:6.2%}  Jan+Jul share {e2[m & S].sum() / e2[S].sum():6.2%}'
          f'  RMSE-if-oracle(0) 12m {np.sqrt((tot - s) / N):.3f} (-{np.sqrt(tot / N) - np.sqrt((tot - s) / N):.2f})')
show('A/B/C', A); show('route D outside P', DnP); show('zone P', P)

# classes by the truth
bs = (d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
stamp = np.abs(bs) <= 60
dayscale = y > 43200
show('truth y > 7200', y > 7200)
show('truth y > 43200 (day-scale)', dayscale)
show('truth |BLOCK-SCHED|<=60 & off_sched>3600', stamp & (osch > 3600))
show('  LIRF of that', stamp & (osch > 3600) & (apt == 'LIRF'))
show('  A/B/C LIRF of that', stamp & (osch > 3600) & (apt == 'LIRF') & A)
for k in ('A', 'DnP', 'P'):
    m = {'A': A, 'DnP': DnP, 'P': P}[k]
    print(k, 'day-scale rows', int((m & dayscale).sum()), 'SSE share', f'{e2[m & dayscale].sum() / tot:.2%}',
          ' y>7200 rows', int((m & (y > 7200)).sum()), f'{e2[m & (y > 7200)].sum() / tot:.2%}')

# zone P: the two hypotheses
fP = f[Pidx]; yP = y[Pidx]; Hs = so[Pidx]; Hn = pbP
assert np.array_equal(np.sort(Pidx), np.flatnonzero(P))
dist = np.abs(Hs - Hn)
best = np.where(np.abs(yP - Hs) < np.abs(yP - Hn), Hs, Hn)
cls_s = np.abs(yP - Hs) <= 120
print(f'\nzone P 2025: {len(Pidx):,} rows ({(apt[Pidx] == "LIRF").sum():,} LIRF); truth = stamp (|y-off_sched|<=120) {cls_s.mean():.1%}; '
      f'|Hs-Hn| median {np.median(dist):.0f} p90 {np.quantile(dist, .9):.0f}')
sP = ((yP - fP) ** 2).sum(); sB = ((yP - best) ** 2).sum()
print(f'  SSE(f) {sP:.3e} = {sP / tot:.2%} of 12m; oracle pick-closer-hypothesis SSE {sB:.3e}; 12m RMSE gain '
      f'{np.sqrt(tot / N) - np.sqrt((tot - sP + sB) / N):.2f}; Jan+Jul gain '
      f'{np.sqrt(e2[S].mean()) - np.sqrt((e2[S].sum() - e2[Pidx][S[Pidx]].sum() + ((yP - best) ** 2)[S[Pidx]].sum()) / S.sum()):.2f}')
for a in np.unique(apt[Pidx]):
    m = apt[Pidx] == a
    print(f'  {a}: rows {m.sum():5d} stamp share {cls_s[m].mean():.2f}  SSE share {((yP - fP) ** 2)[m].sum() / tot:.2%}  '
          f'oracle SSE share {((yP - best) ** 2)[m].sum() / tot:.2%}  p mean {p14[m].mean():.2f}')

# route D outside P: tail
yd = y[DnP]; fd = f[DnP]
print(f'\nroute D outside P: {DnP.sum():,} rows; y>7200 {int((yd > 7200).sum())}; SSE of y>7200 rows {((yd - fd) ** 2)[yd > 7200].sum() / tot:.2%};'
      f' f>3000 rows {int((fd > 3000).sum())} SSE {((yd - fd) ** 2)[fd > 3000].sum() / tot:.2%}')
# top-30 SSE rows overall
o = np.argsort(-e2)[:40]
print('\ntop 40 SSE rows:')
print(pd.DataFrame(dict(apt=apt[o], month=month[o], grp=np.where(P[o], 'P', np.where(D[o], 'D', 'A')), y=y[o], f=f[o].round(),
                        off_sched=osch[o], off_lobt=ol[o], off_aobt3=d.off_aobt3.to_numpy()[o].round(),
                        sse_share=(e2[o] / tot).round(4))).to_string())
print('cum share top 10/40/100/1000:', [round(float(np.sort(e2)[::-1][:k].sum() / tot), 3) for k in (10, 40, 100, 1000)])
print('peak GB', peak_rss_gb())
