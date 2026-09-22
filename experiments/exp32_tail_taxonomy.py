"""EXP32 -- taxonomy of the 584 tail rows, and what the shipped mixture can
structurally represent.

EXP31 showed the squared error on the 180 tail rows of months 1/3/7 is 46.5% of
all SSE at RMSE 12,707, and that the classifier is ALREADY sharp on the
AOBT-missing population (P=74% on true tail rows at LIRF). So the damage must be
concentrated in tail rows the mixture cannot represent at all. Two suspects:

  * the bulk regressor is trained on y <= 7200 and therefore CANNOT emit a
    prediction above ~7200, whatever the features say;
  * e_tail is anchored on off_sched, which is the wrong anchor for the 103 tail
    rows where AOBT_3 exists.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.models.tail_harness import assemble, artifact_label
pd.set_option('display.width', 230)

t0 = time.time()
X, meta = assemble()
y = meta['y'].to_numpy(); month = meta['month'].to_numpy()
os_ = meta['off_sched'].to_numpy(); apt = meta['apt'].to_numpy()
miss = meta['aobt_missing'].to_numpy(); A = artifact_label(meta).astype(bool)
oa = X.off_aobt3.to_numpy()
tail = y > TAIL_CAP
print(f'{len(y):,} rows, {int(tail.sum())} tail  ({time.time()-t0:.0f}s)')

print('\n' + '=' * 110)
print('1. TAXONOMY OF THE 584 TAIL ROWS')
print('=' * 110)
cats = {
    'S  artifact: miss & os>cap':        miss & (os_ > TAIL_CAP),
    'Sb miss & os<=cap':                 miss & ~(os_ > TAIL_CAP),
    'L  AOBT present & off_aobt3>cap':   ~miss & (oa > TAIL_CAP),
    'X  AOBT present & off_aobt3<=cap':  ~miss & ~(oa > TAIL_CAP),
}
rows = []
for k, s in cats.items():
    st = s & tail
    rows.append(dict(cat=k, n=int(s.sum()), n_tail=int(st.sum()),
                     p_tail=st.sum() / max(s.sum(), 1) * 100,
                     med_y_tail=float(np.median(y[st])) if st.any() else np.nan,
                     max_y=float(y[st].max()) if st.any() else np.nan,
                     med_os=float(np.nanmedian(os_[st])) if st.any() else np.nan,
                     med_oa=float(np.nanmedian(oa[st])) if st.any() else np.nan))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print('\n--- category L: AOBT_3 present and off_aobt3 > 7200 ---')
L = ~miss & (oa > TAIL_CAP)
print(f'  n={int(L.sum()):,}  tail={int((L & tail).sum())}  P(tail|L)={tail[L].mean()*100:.1f}%')
print(f'  |y - off_aobt3| on the L tail rows: p10 {np.percentile(np.abs(y-oa)[L&tail],10):.0f} '
      f'p50 {np.percentile(np.abs(y-oa)[L&tail],50):.0f} '
      f'p90 {np.percentile(np.abs(y-oa)[L&tail],90):.0f}')
print(f'  |y - off_aobt3| on the L BULK rows: p50 {np.percentile(np.abs(y-oa)[L&~tail],50):.0f} '
      f'p90 {np.percentile(np.abs(y-oa)[L&~tail],90):.0f}   (median y {np.median(y[L&~tail]):.0f})')
bb = [7200, 9000, 12000, 20000, 40000, np.inf]
for lo, hi in zip(bb[:-1], bb[1:]):
    s = L & (oa >= lo) & (oa < hi)
    if s.sum() == 0:
        continue
    print(f'    off_aobt3 [{lo:7.0f},{hi:8.0f}) n={int(s.sum()):6,}  P(tail)={tail[s].mean()*100:6.2f}%  '
          f'n_tail={int((s&tail).sum()):4d}  med_y={np.median(y[s]):8.0f}  '
          f'rmse_of_oa_anchor={np.sqrt(np.mean((y[s]-oa[s])**2)):9.1f}')
print('\n  L tail rows by airport:')
print(pd.Series(apt[L & tail]).value_counts().to_string())

print('\n--- category Sb: AOBT missing, off_sched <= 7200, y > 7200 ---')
Sb = cats['Sb miss & os<=cap'] & tail
print(f'  n={int(Sb.sum())}   y: {np.sort(y[Sb])[:10]} ... max {y[Sb].max():.0f}')
print(f'  off_sched: p50 {np.median(os_[Sb]):.0f}   apt: {dict(pd.Series(apt[Sb]).value_counts())}')

print('\n--- category X: AOBT present, off_aobt3 <= 7200, y > 7200 ---')
Xc = cats['X  AOBT present & off_aobt3<=cap'] & tail
print(f'  n={int(Xc.sum())}  y p50 {np.median(y[Xc]):.0f} max {y[Xc].max():.0f}  '
      f'off_aobt3 p50 {np.nanmedian(oa[Xc]):.0f}  off_sched p50 {np.nanmedian(os_[Xc]):.0f}')
print(f'  apt: {dict(pd.Series(apt[Xc]).value_counts())}')

print('\n' + '=' * 110)
print('2. THE STRUCTURAL CEILING: a regressor trained on y<=7200 cannot exceed 7200')
print('=' * 110)
print('  SSE if every tail row in category L were predicted at exactly 7200:')
for k, s in cats.items():
    st = s & tail
    if not st.any():
        continue
    sse_cap = ((y[st] - TAIL_CAP) ** 2).sum()
    sse_bulk = ((y[st] - 1100.0) ** 2).sum()
    print(f'    {k:34s} n={int(st.sum()):4d}  SSE@1100 {sse_bulk:.3e} -> contrib '
          f'{np.sqrt(sse_bulk/len(y)):7.1f}   SSE@7200 {sse_cap:.3e} -> contrib '
          f'{np.sqrt(sse_cap/len(y)):7.1f}')

print('\n  ORACLE-ANCHOR experiment: replace the prediction on each tail category')
print('  with its natural anchor and read off the contribution to total RMSE.')
anchors = {'off_sched': os_, 'off_aobt3': oa, 'max(oa,os)': np.fmax(oa, os_)}
for k, s in cats.items():
    st = s & tail
    if not st.any():
        continue
    line = f'    {k:34s} n={int(st.sum()):4d}  '
    for an, v in anchors.items():
        vv = np.where(np.isfinite(v[st]), v[st], 1100.0)
        line += f'{an}={np.sqrt(np.mean((y[st]-vv)**2)):9.1f}  '
    print(line)

print('\n' + '=' * 110)
print('3. HOW OFTEN IS off_aobt3 > 7200 A FALSE ALARM?')
print('=' * 110)
print('  (this is the precision available for a "long genuine taxi" detector)')
for thr in [7200, 8000, 9000, 10000, 12000]:
    s = ~miss & (oa > thr)
    print(f'    off_aobt3 > {thr:6d}: n={int(s.sum()):7,}  P(y>7200)={tail[s].mean()*100:6.2f}%  '
          f'rmse(y, off_aobt3)={np.sqrt(np.nanmean((y[s]-oa[s])**2)):8.1f}  '
          f'rmse(y, 1100)={np.sqrt(np.nanmean((y[s]-1100)**2)):8.1f}')

print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
