"""Skeptic check of the LOBT +-3606 s window claim. Read-only. 2025 labels only; 2026 departures label-free
(only MVT/LOBT/AOBT_3/SCHED clocks of the provided ranking file, departure BLOCK is never read)."""
import glob, numpy as np, pandas as pd
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
cols = ['MVT_ID_mvt', 'ADEP_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
        'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt']
fs = sorted(glob.glob(ROOT + 'training_2025-*.parquet'))
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
dep = tr[tr.PHASE_mvt == 'DEP'].copy()
del tr
sec = lambda a, b: (a - b).dt.total_seconds()
dep['y'] = dep.TAXITIME_SEC_mvt.astype(float)
dep['b_l'] = sec(dep.BLOCK_TIME_UTC_mvt, dep.LOBT_flt)
dep['m_l'] = sec(dep.MVT_TIME_UTC_mvt, dep.LOBT_flt)
dep['m_a'] = sec(dep.MVT_TIME_UTC_mvt, dep.AOBT_3_flt)
dep['m_s'] = sec(dep.MVT_TIME_UTC_mvt, dep.SCHED_TIME_UTC_mvt)
dep['mon'] = dep.MVT_TIME_UTC_mvt.dt.month
has = dep.LOBT_flt.notna()
print('2025 DEP', len(dep), 'with LOBT', has.sum(), f'{has.mean():.4%}')
print('y == MVT-BLOCK exactly:', (np.abs(dep.y - sec(dep.MVT_TIME_UTC_mvt, dep.BLOCK_TIME_UTC_mvt)) < 1).mean())
bl = dep.loc[has, 'b_l']
print('BLOCK-LOBT min/max', bl.min(), bl.max(), ' |.|>3606:', (bl.abs() > 3606).sum())
# sharpness of the edge: counts per 30 s bin approaching +-3606
for lo, hi in [(3000, 3030), (3300, 3330), (3500, 3530), (3570, 3600), (3600, 3606), (3606, 3700)]:
    print(f'  BLOCK-LOBT in ({lo},{hi}]: {((bl > lo) & (bl <= hi)).sum():6d}    in [-{hi},-{lo}): {((bl < -lo) & (bl >= -hi)).sum():6d}')
print('BLOCK-LOBT quantiles', bl.quantile([.001, .01, .05, .5, .95, .99, .999]).round(0).to_dict())
# LOBT-missing rows: is the whole _flt block missing?
nh = ~has
print('no-LOBT rows', nh.sum(), 'of which AOBT_3 also missing', dep.loc[nh, 'AOBT_3_flt'].isna().mean())
print('rows with AOBT_3 but no LOBT', (dep.AOBT_3_flt.notna() & nh).sum(), '; LOBT but no AOBT_3', (has & dep.AOBT_3_flt.isna()).sum())
# lower tail of MVT - LOBT (label-free quantity) in 2025
ml = dep.loc[has, 'm_l']
print('2025 MVT-LOBT min', ml.min(), ' q.0001/.001/.01', ml.quantile([.0001, .001, .01]).round(0).to_dict())
print('2025 MVT-LOBT < -3000:', (ml < -3000).sum(), ' < -3606:', (ml < -3606).sum(), ' > 12h:', (ml > 43200).sum())
big = dep[has & (dep.m_l > 20000)]
print('2025 rows MVT-LOBT > 20000 s:')
print(big[['ADEP_mvt', 'mon', 'y', 'm_l', 'm_a', 'm_s', 'b_l']].sort_values('m_l').to_string())
# large labels in 2025 and whether they have LOBT
yy = dep[dep.y > 80000]
print('2025 rows y > 80,000:'); print(yy[['ADEP_mvt', 'mon', 'y', 'm_l', 'm_a', 'm_s']].to_string())

# ---------------- 2026 label-free ----------------
rk = pd.read_parquet(ROOT + 'ranking.parquet', columns=[c for c in cols if c not in ('BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt')])
rd = rk[rk.PHASE_mvt == 'DEP'].copy()
rd['m_l'] = sec(rd.MVT_TIME_UTC_mvt, rd.LOBT_flt)
rd['m_a'] = sec(rd.MVT_TIME_UTC_mvt, rd.AOBT_3_flt)
rd['m_s'] = sec(rd.MVT_TIME_UTC_mvt, rd.SCHED_TIME_UTC_mvt)
rd['mon'] = rd.MVT_TIME_UTC_mvt.dt.month
h6 = rd.LOBT_flt.notna()
print('\n2026 DEP', len(rd), 'with LOBT', h6.sum(), f'{h6.mean():.4%}')
m6 = rd.loc[h6, 'm_l']
print('2026 MVT-LOBT min', m6.min(), ' q.0001/.001/.01', m6.quantile([.0001, .001, .01]).round(0).to_dict())
print('2026 MVT-LOBT < -3000:', (m6 < -3000).sum(), ' < -3606 (window upper edge < 0 => rule broken):', (m6 < -3606).sum(),
      ' > 12h:', (m6 > 43200).sum())
# compare Jan+Jul 2025 vs 2026 MVT-LOBT tail shares (label-free)
jj = dep[has & dep.mon.isin([1, 7])]
for thr in [3606, 7200, 20000]:
    print(f'  share MVT-LOBT > {thr}: 2025 Jan+Jul {(jj.m_l > thr).mean():.5%}  2026 {(m6 > thr).mean():.5%}')
print('  share MVT-LOBT < -1800: 2025 Jan+Jul', f'{(jj.m_l < -1800).mean():.5%}', ' 2026', f'{(m6 < -1800).mean():.5%}')
print('  share |MVT-AOBT_3| within window of LOBT: 2025', f'{((dep.m_a - dep.m_l).abs() <= 3606)[has & dep.AOBT_3_flt.notna()].mean():.5%}',
      '2026', f'{((rd.m_a - rd.m_l).abs() <= 3606)[h6 & rd.AOBT_3_flt.notna()].mean():.5%}')
# v27 outside window
v27 = pd.read_parquet(ROOT + 'prc2026/outputs/understanding-accordion_v27.parquet')
rd = rd.merge(v27.rename(columns={'TAXITIME_SEC_mvt': 'p'}), on='MVT_ID_mvt', how='left')
assert rd.p.notna().all()
h6 = rd.LOBT_flt.notna().values
lo = np.maximum(rd.m_l - 3606, 0); hi = rd.m_l + 3606
q = rd.p.where(~h6, np.clip(rd.p, lo, hi))
mv = (q - rd.p).abs() > 0
print('v27 rows outside window', mv.sum(), ' below', ((rd.p < lo) & h6).sum(), ' above', ((rd.p > hi) & h6).sum())
N = len(rd)
ss = ((q - rd.p) ** 2).sum()
print('sum sq shift', f'{ss:.4g}', ' bound', np.sqrt(243.1478 ** 2 - ss / N))
top = rd.loc[mv].assign(sh=(q - rd.p)[mv]).sort_values('sh', key=abs, ascending=False)
print(top[['ADEP_mvt', 'mon', 'p', 'm_l', 'm_a', 'm_s', 'sh']].head(8).to_string())
print('bound without top row', np.sqrt(243.1478 ** 2 - (ss - top.sh.iloc[0] ** 2) / N))
