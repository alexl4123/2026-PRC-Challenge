"""Same-stand departure ORDER (2025). Departures at one stand sorted by take-off T.
 (1) how often is the recorded off-block order reversed vs take-off order (overtaking)?
 (2) rule O1: previous same-stand departure d (by T) has LOBT -> B_d >= LOBT_d - 3606 (exact) and, IF stand order holds,
     our B > B_d  =>  y <= T - LOBT_d + 3606.  Exactness + OOF effect; 2026 effect on v28.
 (3) rule O2: next same-stand departure d' (by T) has LOBT -> B_d' <= LOBT_d' + 3606, IF order holds our B < B_d'
     => y >= T - LOBT_d' - 3606 (only informative if T - LOBT_d' - 3606 > 0)."""
import numpy as np, pandas as pd
from stand_common import load_2025, load_2026, evaluate, apply_2026


def ctx(c):
    c = c.copy()
    c = c.sort_values(['ki', 'T', 'id'])
    same_prev = np.r_[False, c.ki.to_numpy()[1:] == c.ki.to_numpy()[:-1]]
    same_next = np.r_[c.ki.to_numpy()[1:] == c.ki.to_numpy()[:-1], False]
    lob = c.lobt.to_numpy(); T = c['T'].to_numpy(); B = c.blk.to_numpy()
    c['prev_lobt'] = np.where(same_prev, np.r_[np.nan, lob[:-1]], np.nan)
    c['next_lobt'] = np.where(same_next, np.r_[lob[1:], np.nan], np.nan)
    c['prev_T'] = np.where(same_prev, np.r_[np.nan, T[:-1]], np.nan)
    c['prev_B'] = np.where(same_prev, np.r_[np.nan, B[:-1]], np.nan)
    return c.sort_index()


c = ctx(load_2025()); k = ctx(load_2026())
ok = np.isfinite(c.prev_B)
print('consecutive same-stand departure pairs (by take-off):', int(ok.sum()), '| off-block order reversed (B < B_prev):',
      int((c.blk < c.prev_B)[ok].sum()), f'({(c.blk < c.prev_B)[ok].mean():.4%})')
print('by APT reversed share:'); print((c.blk < c.prev_B)[ok].groupby(c.APT[ok]).mean().round(4).to_dict())
for nm, d in (('2025', c), ('2026', k)):
    pass
NA = np.nan
for gap in (0, 1800, 3600):
    hi25 = np.where(np.isfinite(c.prev_lobt) & ((c['T'] - c.prev_lobt) > gap), c['T'] - c.prev_lobt + 3606, NA)
    hi26 = np.where(np.isfinite(k.prev_lobt) & ((k['T'] - k.prev_lobt) > gap), k['T'] - k.prev_lobt + 3606, NA)
    for sub, nm in ((np.ones(len(c), bool), 'all'), (c.D.to_numpy(), 'route D only')):
        h = np.where(sub, hi25, NA)
        r = evaluate(c, np.full(len(c), NA), h, f'O1 prev-dep LOBT bound ({nm}), T-LOBT_prev>{gap}')
        if r['exc']:
            e = r['exc_mask']; print('    exceptions APT', c.APT[e].value_counts().head(6).to_dict(), '| P', int(c.P[e].sum()))
    D26 = ~np.isfinite(k.aobt3.to_numpy())
    apply_2026(k, np.full(len(k), NA), hi26, f'O1 all, gap {gap}')
    apply_2026(k, np.full(len(k), NA), np.where(D26, hi26, NA), f'O1 route D, gap {gap}')
lo25 = np.where(np.isfinite(c.next_lobt), np.maximum(c['T'] - c.next_lobt - 3606, 0), NA)
lo26 = np.where(np.isfinite(k.next_lobt), np.maximum(k['T'] - k.next_lobt - 3606, 0), NA)
r = evaluate(c, lo25, np.full(len(c), NA), 'O2 next-dep LOBT lower bound')
if r['exc']:
    e = r['exc_mask']; print('    exceptions APT', c.APT[e].value_counts().head(6).to_dict(), '| P', int(c.P[e].sum()), 'D', int(c.D[e].sum()))
apply_2026(k, lo26, np.full(len(k), NA), 'O2')
