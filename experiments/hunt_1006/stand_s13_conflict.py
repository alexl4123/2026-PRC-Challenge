"""The physical stand rules fail exactly on stamped rows. Flip side: on route-D rows where the 'stamp at SCHED'
hypothesis is PHYSICALLY impossible (previous same-stand departure's exact LOBT window says that aircraft was still on
our stand at our SCHED), is the recorded block nevertheless a stamp?  Model-free flag:
   F = route D  &  prev same-stand departure (by take-off) has LOBT  &  SCHED_d < LOBT_prev - 3606
Stamp = |y - off_sched| <= 120.  Compare stamp rate, the model's implied weight, and the OOF effect of moving toward off_sched."""
import numpy as np, pandas as pd
from stand_common import load_2025, load_2026


def ctx(c):
    c = c.copy().sort_values(['ki', 'T', 'id'])
    sp = np.r_[False, c.ki.to_numpy()[1:] == c.ki.to_numpy()[:-1]]
    c['prev_lobt'] = np.where(sp, np.r_[np.nan, c.lobt.to_numpy()[:-1]], np.nan)
    return c.sort_index()


c = ctx(load_2025()); k = ctx(load_2026())
for d in (c, k):
    d['off_sched'] = d['T'] - d.sch
    d['Dr'] = ~np.isfinite(d.aobt3)
    d['F'] = d.Dr & np.isfinite(d.prev_lobt) & (d.sch < d.prev_lobt - 3606)
c['stamp'] = (c.y - c.off_sched).abs() <= 120
print('2025 route D rows', int(c.Dr.sum()), '| F rows', int(c.F.sum()), '| F & P', int((c.F & c.P).sum()))
for nm, s in (('P & F', c.P & c.F), ('P & ~F', c.P & ~c.F), ('D~P & F', c.Dr & ~c.P & c.F), ('D~P & ~F', c.Dr & ~c.P & ~c.F)):
    g = c[s]
    w = ((g.f - 900) / (g.off_sched - 900)).clip(0, 1)  # rough implied stamp weight of the model
    print(f'{nm:10s} n {len(g):5d} stamp rate {g.stamp.mean():.3f} | model implied weight mean {w.mean():.3f} | by APT stamp rate',
          g.groupby('APT').stamp.agg(['size', 'mean']).query('size >= 5').round(3).to_dict('index'))
# OOF effect of q = a * off_sched + (1 - a) * f on P & F rows, a in grid; per month
s = c.P & c.F
y, f, o, mth = c.y.to_numpy(), c.f.to_numpy(), c.off_sched.to_numpy(), c.month.to_numpy()
held, S = ~np.isin(mth, [3, 6, 9, 12]), np.isin(mth, [1, 7])
rm = lambda q, kk: np.sqrt(np.mean((q - y)[kk] ** 2))
for a in (0.25, 0.5, 0.75, 1.0):
    q = np.where(s, a * o + (1 - a) * f, f)
    ds = (q - y) ** 2 - (f - y) ** 2
    pm = pd.Series(ds).groupby(mth).sum()
    print(f'a={a}: rows {s.sum()} worse {int(((np.abs(q - y) > np.abs(f - y)) & s).sum())} | held {rm(q, held) - rm(f, held):+.3f} JanJul {rm(q, S) - rm(f, S):+.3f} '
          f'| months better {(pm < 0).sum()}/{(pm != 0).sum()} | per APT dSSE ' + str(pd.Series(ds[s]).groupby(c.APT[s].to_numpy()).sum().round(-5).to_dict()))
print('2026: F rows', int(k.F.sum()), '| F & zone-P-like (route D, off_sched>7200)', int((k.F & (k.off_sched > 7200)).sum()),
      '| by APT', k[k.F & (k.off_sched > 7200)].APT.value_counts().to_dict())
kk = k[k.F & (k.off_sched > 7200)]
print('2026 v28 on those: f/off_sched quantiles', np.percentile(kk.f / kk.off_sched, [0, 25, 50, 75, 100]).round(3) if len(kk) else None)
