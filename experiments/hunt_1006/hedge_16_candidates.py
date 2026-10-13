"""Zone-P decision candidates, final form (pre-specified here; 2025 OOF effect + 2026 label-free moves and bounds).

  B   LIRF zone-P row whose (MVT - 1800 s) falls on a later UTC date than SCHED  ->  p := 1  (stamp branch only)
      2025: 0 normal among 53 such rows (42 stamp, 11 day-shift); same-delay rows without the crossing are 24-42% normal.
  A1  zone-P IFR row linked to its own arrival row (another of the 10 airports) carrying an NM record with LOBT_a, and
      off_sched inside [off_lobt_a - 3606, off_lobt_a + 3606] (stamp would have joined NM)  ->  p := 0
  C   B rows with off_sched in (50,000, 95,000]: y = q (86400 dd_loc + pb) + (1 - q) off_sched, q = 0.5   [statistical]
  T   (statistical companion of B) LIRF zone-P row with off_sched > 28,800 -> p := 1
Writes cache/hunt/hedge_candidates_2026.npz (ids, v28, B, A1, A1B, A1BC, BT, A1BT) -- scratch only, no submission file."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load
import assemble as A
pd.set_option('display.width', 250)
MARGIN = 1800

# ---------------- 2025 OOF
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P, f = V['y'], V['month'], V['P'], V['f']
W = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
Pidx, p14, pbP, Prest, so = W['Pidx'], W['p14'], W['pbP'], W['P_rest'], W['so']
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
dP = d.iloc[Pidx]
du = ((dP.MVT_TIME_UTC_mvt - pd.to_timedelta(MARGIN, 's')).dt.normalize() - dP.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()
mL = dP.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None); sL = dP.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None)
dl = ((mL - pd.to_timedelta(900, 's')).dt.normalize() - sL.dt.normalize()).dt.days.to_numpy()
B = (dP.APT.to_numpy() == 'LIRF') & (du >= 1)
Lk = pd.read_parquet(CACHE / 'hunt' / 'hedge_plink25.parquet'); Lk = Lk[Lk.P & (Lk.FLIGHT_RULE_mvt.astype(str) == 'I')]
k = pd.Series(np.arange(len(Pidx)), index=Pidx); olo = np.full(len(Pidx), np.nan); olo[k.reindex(Lk.pos).to_numpy()] = Lk.off_lobt.to_numpy()
A1 = (so[Pidx] >= olo - 3606) & (so[Pidx] <= olo + 3606)
print(f'2025: B rows {int(B.sum())}, A1 rows {int(A1.sum())}, overlap {int((A1 & B).sum())}')
N = len(y); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7]); allm = np.ones(N, bool)
rm = lambda g, m: np.sqrt(np.mean((y[m] - g[m]) ** 2))


def oof(name, p_new, cband=False):
    v = Prest + np.clip(p_new * so[Pidx] + (1 - p_new) * pbP, 0, None)
    if cband:
        band = B & (so[Pidx] > 50000) & (so[Pidx] <= 95000)
        v[band] = Prest[band] + 0.5 * (86400 * np.maximum(dl[band], 1) + pbP[band]) + 0.5 * so[Pidx][band]
    g = f.copy(); g[Pidx] = v
    ch = np.abs(g - f) > 0.5; wo = ch & ((y - g) ** 2 > (y - f) ** 2 + 1)
    per = [rm(g, month == m) - rm(f, month == m) for m in range(1, 13)]
    print(f'  {name:10s} changed {int(ch.sum()):3d} worse {int(wo.sum()):3d} | 12m {rm(g, allm) - rm(f, allm):+.3f} held {rm(g, held) - rm(f, held):+.3f} '
          f'Jan+Jul {rm(g, S) - rm(f, S):+.3f} (Jan {per[0]:+.3f} Jul {per[6]:+.3f}) months better {sum(x < -1e-9 for x in per)}/12 '
          f'worse {sum(x > 1e-9 for x in per)}/12')


print('2025 OOF (v28 structure):')
oof('B', np.where(B, 1.0, p14))
oof('A1', np.where(A1, 0.0, p14))
oof('A1+B', np.where(B, 1.0, np.where(A1, 0.0, p14)))
oof('A1+B+C', np.where(B, 1.0, np.where(A1, 0.0, p14)), cband=True)

# ---------------- 2026
C_ = A.Components()
t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
rows = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
pb = 0.5 * C_.fit('w2_final', 'mix')[rows] + 0.5 * C_.member('seq')[rows]
s6 = C_.off_sched[rows]; corr = C_.corr[rows]
v28 = A.shipped(C_, 28)
assert np.array_equal(np.rint(np.clip(p * s6 + (1 - p) * pb + corr, 0, A.HI)), v28[rows])
rk = load('ranking', phase='DEP', columns=['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).iloc[rows]
du6 = ((rk.MVT_TIME_UTC_mvt - pd.to_timedelta(MARGIN, 's')).dt.normalize() - rk.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()
mL6 = rk.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None); sL6 = rk.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.tz_localize(None)
dl6 = ((mL6 - pd.to_timedelta(900, 's')).dt.normalize() - sL6.dt.normalize()).dt.days.to_numpy()
B6 = (C_.apt[rows] == 'LIRF') & (du6 >= 1)
L6 = pd.read_parquet(CACHE / 'hunt' / 'hedge_plink26.parquet'); L6 = L6[L6.P & (L6.FLIGHT_RULE_mvt.astype(str) == 'I')]
k6 = pd.Series(np.arange(len(rows)), index=rows); ol6 = np.full(len(rows), np.nan)
ol6[k6.reindex(L6.pos).to_numpy()] = (L6.mvt - L6.lobt).to_numpy()
A16 = (s6 >= ol6 - 3606) & (s6 <= ol6 + 3606)
mon = C_.month[rows]


def build(p_new, cband=False):
    v = p_new * s6 + (1 - p_new) * pb + corr
    if cband:
        band = B6 & (s6 > 50000) & (s6 <= 95000)
        v[band] = 0.5 * (86400 * np.maximum(dl6[band], 1) + pb[band]) + 0.5 * s6[band] + corr[band]
    out = v28.copy(); out[rows] = np.rint(np.clip(v, 0, A.HI))
    return out


tot = 235.3996 ** 2 * len(v28)
cand = {}
for name, pn, cb in (('B', np.where(B6, 1.0, p), False), ('A1', np.where(A16, 0.0, p), False),
                     ('A1B', np.where(B6, 1.0, np.where(A16, 0.0, p)), False), ('A1BC', np.where(B6, 1.0, np.where(A16, 0.0, p)), True),
                     ('BT', np.where(B6 | ((C_.apt[rows] == 'LIRF') & (s6 > 28800)), 1.0, p), False),
                     ('A1BT', np.where(B6 | ((C_.apt[rows] == 'LIRF') & (s6 > 28800)), 1.0, np.where(A16, 0.0, p)), False)):
    o = build(pn, cb); cand[name] = o
    mv = o != v28; m2 = ((o - v28) ** 2)[mv]
    print(f'2026 {name:5s}: rows moved {int(mv.sum()):3d} (Jan {int((mv[rows] & (mon == 1)).sum())}, Jul {int((mv[rows] & (mon == 7)).sum())}); '
          f'sum move^2 {m2.sum():.3e} -> RMSE bound if every move is toward the truth class {np.sqrt(max(tot - m2.sum(), 0) / len(v28)) - 235.3996:+.2f} s; '
          f'max |move| {np.abs(o - v28).max():.0f}; airport x month mean shift max {max(abs((o - v28)[(C_.apt == a) & (C_.month == m)].mean()) for a in A.np.unique(C_.apt) for m in (1, 7)):.2f} s')
print(f'B rows 2026: {int(B6.sum())}; A1 rows 2026: {int(A16.sum())} (of {len(L6)} linked IFR zone-P rows)')
np.savez(CACHE / 'hunt' / 'hedge_candidates_2026.npz', ids=C_.ids, v28=v28, **cand)
