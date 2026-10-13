"""Synthesizer: independent re-check of the package files (read-only on inputs)."""
import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
rd = lambda p: pd.read_parquet(p).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
v27 = rd(R + 'outputs/understanding-accordion_v27.parquet'); v28 = rd(R + 'outputs/understanding-accordion_v28.parquet')
proj = set(v28.index[(v28 != v27.reindex(v28.index)).to_numpy()]); print('projected', len(proj))
N = len(v28); r0 = 235.3996
mv = pd.read_csv(H + 'cand_moves.csv').set_index('MVT_ID_mvt')
for f in ('cand_v28_ABC.parquet', 'cand_v28_ABC_noRYR515.parquet', 'cand_v28_ABC_E1.parquet'):
    raw = pd.read_parquet(H + f); x = raw.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    same_order = np.array_equal(raw.MVT_ID_mvt.to_numpy(), pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').MVT_ID_mvt.to_numpy())
    d = x.reindex(v28.index) - v28; ch = d[d != 0]
    S = float((ch.astype(float) ** 2).sum())
    print(f, 'rows', len(raw), 'dtype', raw.TAXITIME_SEC_mvt.dtype, 'nan', int(raw.TAXITIME_SEC_mvt.isna().sum()), 'min', raw.TAXITIME_SEC_mvt.min(),
          'same order', same_order, '| changed', len(ch), 'up', int((ch > 0).sum()), 'down', int((ch < 0).sum()), 'overlap proj', len(set(ch.index) & proj),
          'sum sq %.4e bound %.3f (%.3f)' % (S, np.sqrt(r0**2 - S / N), np.sqrt(r0**2 - S / N) - r0))
    top = (ch.astype(float) ** 2).sort_values(ascending=False)
    print('   top1 share %.2f top3 %.2f' % (top.iloc[0] / S, top.iloc[:3].sum() / S), list(mv.reindex(top.index[:3]).FLIGHT_mvt))
# per-rule bound pieces from cand_moves
for lab, m in (('A', mv.one == True), ('B', mv.B == True), ('C', mv.one.isna())):
    s = mv[m]; dm = (s.ABC - s.v28); dm = dm[dm != 0]; S = float((dm ** 2).sum())
    print(lab, 'rows moved', len(dm), 'sum sq %.4e bound d %.3f' % (S, np.sqrt(r0**2 - S / N) - r0))
s = mv[(mv.B == True) & (mv.FLIGHT_mvt != 'RYR515')]; dm = s.ABC - s.v28; dm = dm[dm != 0]; S = float((dm ** 2).sum())
print("B' rows", len(dm), 'sum sq %.4e bound d %.3f' % (S, np.sqrt(r0**2 - S / N) - r0))
# CFG4EF cost/gain scenarios
for y in (800, 7139, 8225):
    dS = (1013 - y) ** 2 - (7085 - y) ** 2; print('CFG4EF y', y, 'dRMSE %.3f' % (np.sqrt(r0**2 + dS / N) - r0))
