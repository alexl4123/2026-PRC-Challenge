"""ids_30 -- dry-run builder (scratch only, cache/hunt/): v28 + DST window (refined destination-NM exclusion),
and separately v28 + twin rule T4. Never touches the 63 rows v28 projected. Prints the moves; writes candidate parquets
to cache/hunt/ for inspection (NOT outputs/, NOT for upload without the main session's review)."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
v27 = pd.read_parquet(H + 'outputs/understanding-accordion_v27.parquet')
v28 = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet')
x = v27.merge(v28, on='MVT_ID_mvt', suffixes=('_27', '_28'))
proj = set(x[x.TAXITIME_SEC_mvt_27 != x.TAXITIME_SEC_mvt_28].MVT_ID_mvt)
N = len(v28); R = 235.3996
# --- DST window
q = pd.read_parquet(H + 'cache/hunt/ids_23_dst_2026.parquet')
q = q[(q.move != 0) & ~q.MVT_ID_mvt.isin(proj)]
dst = v28.copy()
mp = dict(zip(q.MVT_ID_mvt, q.g))
dst['TAXITIME_SEC_mvt'] = [mp.get(i, t) for i, t in zip(dst.MVT_ID_mvt, dst.TAXITIME_SEC_mvt)]
mv = dst.TAXITIME_SEC_mvt.values - v28.TAXITIME_SEC_mvt.values
print(f'DST: rows moved {np.sum(mv != 0)}, sum move^2 {np.sum(mv ** 2):.3e}, |max move| {np.abs(mv).max():.0f}')
dst.to_parquet(H + 'cache/hunt/ids_cand_v28_dst.parquet')
# --- twin T4 (constant from all 2025 pairs)
tt = pd.read_parquet(H + 'cache/hunt/ids_08_twins2025.parquet')
t = tt[(tt.gsize == 2) & (tt.dmvt <= 600)]
a = t[(t.groupby('gid').nm.transform('sum') == 1) & t.nm]
c = float(np.median(a.off_sched - a.y))
rt = pd.read_parquet(H + 'cache/hunt/ids_08_twins2026.parquet')
lo = pd.read_parquet(H + 'cache/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'off_lobt'])
rt = rt.merge(lo, on='MVT_ID_mvt')
t2 = rt[(rt.gsize == 2) & (rt.dmvt <= 600)]
a2 = t2[(t2.groupby('gid').nm.transform('sum') == 1) & t2.nm & ~t2.MVT_ID_mvt.isin(proj)].copy()
a2['g'] = np.clip(0.5 * (a2.f + a2.off_sched - c), a2.off_lobt - 3606, a2.off_lobt + 3606)
tw = v28.copy()
mp = dict(zip(a2.MVT_ID_mvt, a2.g))
tw['TAXITIME_SEC_mvt'] = [mp.get(i, t) for i, t in zip(tw.MVT_ID_mvt, tw.TAXITIME_SEC_mvt)]
mv2 = tw.TAXITIME_SEC_mvt.values - v28.TAXITIME_SEC_mvt.values
print(f'TWIN T4 (c={c:.0f}): rows moved {np.sum(mv2 != 0)}, sum move^2 {np.sum(mv2 ** 2):.3e}, |max move| {np.abs(mv2).max():.0f}')
print(a2[['APT', 'FLIGHT_mvt', 'off_sched', 'off_lobt', 'f', 'g']].round(0).to_string())
tw.to_parquet(H + 'cache/hunt/ids_cand_v28_twin.parquet')
