"""2025 OOF effect of the zone-P decision rules (v28 structure; zone P rows of v28 = v24's).
  A  dest-NM anti-window: zone-P row linked to its arrival's NM record (LOBT_a);  W = [off_lobt_a - 3606, off_lobt_a + 3606]
     A1: stamp hypothesis off_sched in W -> p := 0;  A2: normal hypothesis pb in W (and stamp not) -> p := 1
  B  LIRF zone P with (MVT - 900 s) on a later UTC date than SCHED -> p := 1 (normal never seen there in 2025)
  C  B rows + 3-way: y = q * (86400 * dd_loc + pb) + (1 - q) * off_sched  with q from the OTHER months (LOMO)
     restricted to off_sched in (50,000, 95,000] (the band where day-shift rows live)
f_new = P_rest + clip(p_new * off_sched + (1 - p_new) * pb, 0)."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, month, P, f = V['y'], V['month'], V['P'], V['f']
W = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
Pidx, p14, pbP, Prest, so = W['Pidx'], W['p14'], W['pbP'], W['P_rest'], W['so']
assert np.allclose(f[Pidx], Prest + np.clip(p14 * so[Pidx] + (1 - p14) * pbP, 0, None))
N = len(y); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
r = lambda g, m: np.sqrt(np.mean((y[m] - g[m]) ** 2))


def report(name, pnew, extra=None):
    g = f.copy()
    v = Prest + np.clip(pnew * so[Pidx] + (1 - pnew) * pbP, 0, None)
    if extra is not None:
        v = np.where(np.isnan(extra), v, extra)
    g[Pidx] = v
    ch = np.abs(g - f) > 0.5
    worse = ((y - g) ** 2 > (y - f) ** 2 + 1) & ch
    dsse = ((y - g) ** 2 - (y - f) ** 2).sum()
    print(f'{name:58s} rows changed {int(ch.sum()):4d} worse {int(worse.sum()):3d} | dSSE {dsse:+.3e} | 12m {r(g, np.ones(N, bool)) - r(f, np.ones(N, bool)):+.3f}'
          f' held {r(g, held) - r(f, held):+.3f} Jan+Jul {r(g, S) - r(f, S):+.3f} Jan {r(g, month == 1) - r(f, month == 1):+.3f}'
          f' Jul {r(g, month == 7) - r(f, month == 7):+.3f}')
    return g, ch


# --- A: linked rows
L = pd.read_parquet(CACHE / 'hunt' / 'hedge_plink25.parquet')
L = L[L.P & (L.FLIGHT_RULE_mvt.astype(str) == 'I')]
k = pd.Series(np.arange(len(Pidx)), index=Pidx)
ix = k.reindex(L.pos).to_numpy()
assert not np.isnan(ix.astype(float)).any()
olo = np.full(len(Pidx), np.nan); olo[ix] = L.off_lobt.to_numpy()
lo, hi = olo - 3606, olo + 3606
s_in = (so[Pidx] >= lo) & (so[Pidx] <= hi)
n_in = (pbP >= lo) & (pbP <= hi) & ~s_in
print(f'A: zone-P IFR rows linked {len(L)}; stamp excluded {int(s_in.sum())}; normal excluded {int(n_in.sum())}')
pA1 = np.where(s_in, 0.0, p14); pA = np.where(n_in, 1.0, pA1)
report('A1 stamp excluded -> p=0', pA1)
gA, chA = report('A  (A1 + normal excluded -> p=1)', pA)
# oracle on the same rows: pick the closer hypothesis
yP = y[Pidx]; best = np.where(np.abs(yP - so[Pidx]) < np.abs(yP - pbP), 1.0, 0.0)
lk = ~np.isnan(olo)
report('oracle p in {0,1} on the linked rows', np.where(lk, best, p14))

# --- B / C: LIRF zone P, UTC date crossing
d = pd.read_parquet(CACHE / 'hunt' / 'hedge_lirf_tw25.parquet')
d = d[d.P]
ixL = k.reindex(d.pos).to_numpy()
ddu = np.zeros(len(Pidx), int); ddu[ixL] = d.dd_utc.to_numpy()
ddl = np.zeros(len(Pidx), int); ddl[ixL] = d.dd_loc.to_numpy()
isL = np.zeros(len(Pidx), bool); isL[ixL] = True
B = isL & (ddu >= 1)
print(f'\nB: LIRF zone P rows with UTC date crossing: {int(B.sum())}; their p mean {p14[B].mean():.3f}, min {p14[B].min():.3f}')
report('B  LIRF dd_utc>=1 -> p=1', np.where(B, 1.0, p14))
cls = np.where(np.abs(yP - so[Pidx]) <= 120, 's', np.where(yP > 43200, 'd', 'n'))
band = B & (so[Pidx] > 50000) & (so[Pidx] <= 95000)
print(' band rows', int(band.sum()), ' classes', pd.Series(cls[band]).value_counts().to_dict(), ' months', np.bincount(month[Pidx][band], minlength=13)[1:].tolist())
mP = month[Pidx]
for qfix in (None, 0.5):
    ext = np.full(len(Pidx), np.nan)
    for i in np.flatnonzero(band):
        if qfix is None:
            o = band & (mP != mP[i])
            q = (cls[o] == 'd').mean() if o.any() else 0.0
        else:
            q = qfix
        ext[i] = Prest[i] + q * (86400 * max(ddl[i], 1) + pbP[i]) + (1 - q) * so[Pidx][i]
    report(f'C  B + 3-way in band, q={"LOMO" if qfix is None else qfix}', np.where(B, 1.0, p14), ext)
# oracle for the day-shift class: predict 86400 + pb on true day-shift rows
ext = np.where(cls == 'd', Prest + 86400 * np.maximum(ddl, 1) + pbP, np.nan)
report('oracle: true day-shift rows -> 86400*dd + pb', p14, ext)
# full 3-hypothesis oracle on all zone-P rows
ho = np.where(cls == 'd', 86400 * np.maximum(ddl, 1) + pbP, np.where(cls == 's', so[Pidx], pbP))
report('oracle: every zone-P row -> its class hypothesis', p14, Prest + ho)

# --- A2': the normal branch cannot sit inside the anti-window: push pb to the upper edge (keep p)
pbA = np.where(n_in, np.maximum(pbP, hi), pbP)
g = f.copy(); g[Pidx] = Prest + np.clip(pA1 * so[Pidx] + (1 - pA1) * pbA, 0, None)
ch = np.abs(g - f) > 0.5; worse = ((y - g) ** 2 > (y - f) ** 2 + 1) & ch
print(f"A1+A2' (normal branch -> upper edge)  changed {int(ch.sum())} worse {int(worse.sum())} 12m {r(g, np.ones(N, bool)) - r(f, np.ones(N, bool)):+.3f} "
      f"held {r(g, held) - r(f, held):+.3f} Jan+Jul {r(g, S) - r(f, S):+.3f}")
# B: which delays do the dd_utc>=1 rows have, and the normal share among dd_utc==0 rows of the same delay band (LIRF zone P)
isLd = isL
for lo_, hi_ in ((7200, 14400), (14400, 21600), (21600, 43200), (43200, 1e7)):
    m1 = isLd & (ddu >= 1) & (so[Pidx] > lo_) & (so[Pidx] <= hi_); m0 = isLd & (ddu == 0) & (so[Pidx] > lo_) & (so[Pidx] <= hi_)
    print(f'  LIRF P off_sched ({lo_},{hi_}]: dd_utc>=1 n={int(m1.sum())} normal {int((cls[m1] == "n").sum())} | dd_utc=0 n={int(m0.sum())} normal {int((cls[m0] == "n").sum())}')
