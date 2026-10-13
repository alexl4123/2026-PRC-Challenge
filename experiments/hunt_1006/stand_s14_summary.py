"""Summary: exception anatomy of the stand rules (what kind of recorded BLOCK breaks them) + 2025 vs 2026 label-free
consistency of the stand context."""
import numpy as np, pandas as pd
from stand_common import load_2025, load_2026


def prev_lobt(c):
    c = c.copy().sort_values(['ki', 'T', 'id'])
    sp = np.r_[False, c.ki.to_numpy()[1:] == c.ki.to_numpy()[:-1]]
    c['prev_lobt'] = np.where(sp, np.r_[np.nan, c.lobt.to_numpy()[:-1]], np.nan)
    return c.sort_index()


c = prev_lobt(load_2025()); k = prev_lobt(load_2026())
c['os'] = c['T'] - c.sch
B_S = c.blk - c.sch


def anatomy(mask, nm):
    m = mask
    cat = np.select([np.abs(B_S) <= 120, np.abs(B_S + 86400) <= 600, np.abs(c.blk - c.lobt) <= 120],
                    ['BLOCK=SCHED (stamp)', 'BLOCK=SCHED-1d', 'BLOCK=LOBT'], 'other')
    s = pd.Series(cat[m]).value_counts().to_dict()
    print(f'{nm}: {int(m.sum())} exceptions | {s} | LIRF share {(c.APT[m] == "LIRF").mean():.2f}')


Tam = c['T'] - c.a_m
anatomy(np.isfinite(Tam) & (Tam > 3600) & (c.y > Tam), 'U1 (latest stand arrival, T-a_m>1h): y > T-a_m')
anatomy(np.isfinite(Tam) & (Tam > 7200) & (c.y > Tam), 'U1 (T-a_m>2h)')
hi = c['T'] - c.prev_lobt + 3606
anatomy(np.isfinite(hi) & (c.y > hi), 'O1 (prev same-stand departure LOBT-3606)')
anatomy(np.isfinite(Tam) & (Tam <= 600) & (c.y < Tam), 'L1 (T-a_m<=600): y < T-a_m')
print('\nLABEL-FREE CONSISTENCY 2025 vs 2026 (departures):')
for nm, d in (('2025', c), ('2026', k)):
    Tm = d['T'] - d.a_m
    print(f'{nm}: stand present {np.isfinite(d.a_m).mean():.4f} | link present {np.isfinite(d.a_lk).mean():.4f} | '
          f'T-a_m<=600 {np.mean(Tm <= 600):.4f} | T-a_m in (1h,2h] {np.mean((Tm > 3600) & (Tm <= 7200)):.4f} | '
          f'next stand arrival before take-off by other type/op (m_link False & T-a_m<=1200) {np.mean(~d.m_link & (Tm <= 1200)):.4f} | '
          f'prev same-stand dep has LOBT {np.isfinite(d.prev_lobt).mean():.4f} | route D {np.mean(~np.isfinite(d.aobt3)):.4f}')
