"""ids_28 -- twin rule priced in the same harness as ids_23 (RMSE deltas 12m / held / Jan+Jul on the v28 OOF).
Twin pair: exactly two DEP rows with same airport, callsign, SCHED; take-offs within 600 s; exactly one has an NM record.
Rule T3: NM row -> max(f, off_sched - c) clipped into its LOBT window, c = LOMO median of (off_sched - y) on pair NM rows.
Rule T4: NM row -> 0.5 * (f + (off_sched - c)), clipped.  ID-free (NM presence identifies the first block, 99/99)."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
tt = pd.read_parquet(H + 'cache/hunt/ids_08_twins2025.parquet')
lob = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt', 'off_lobt'])
lob['i'] = np.arange(len(lob))
tt = tt.merge(lob, on='MVT_ID_mvt')
t = tt[(tt.gsize == 2) & (tt.dmvt <= 600)]
one = t.groupby('gid').nm.transform('sum') == 1
a = t[one & t.nm].copy()
z = np.load(H + 'cache/w2/v28s_oof.npz'); y, f, month = z['y'], z['f'], z['month']
rm = lambda h, msk: np.sqrt(np.mean((y[msk] - h[msk]) ** 2))
held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7]); A = np.ones_like(held)
for name in ['T3', 'T4']:
    g = f.copy()
    for mth in a.month.unique():
        trn = a[a.month != mth]; te = a[a.month == mth]
        c = np.median(trn.off_sched - trn.y)
        r1 = te.off_sched.values - c
        fi = f[te.i.values]
        p = np.maximum(fi, r1) if name == 'T3' else 0.5 * (fi + r1)
        p = np.clip(p, te.off_lobt.values - 3606, te.off_lobt.values + 3606)
        g[te.i.values] = p
    ch = g != f; d = (y - g) ** 2 - (y - f) ** 2
    print(f'{name}: rows changed {ch.sum()}, worse {np.sum(d > 0)}, dRMSE 12m {rm(g, A) - rm(f, A):+.4f} held {rm(g, held) - rm(f, held):+.4f} '
          f'Jan+Jul {rm(g, S) - rm(f, S):+.4f}; months better {sum(d[month == m].sum() < 0 for m in np.unique(month[ch]))}/{len(np.unique(month[ch]))}')
