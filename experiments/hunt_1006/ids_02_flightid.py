"""ids_02 -- is FLIGHT_ID_mvt a global LOBT-ordered key? Recover LOBT for rows with FLIGHT_ID but no _flt block,
and test the hypothesis 'the _flt join failed because |BLOCK - LOBT| > 3600' on 2025 (label side) and 2026 ARR."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load

C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'APT', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
     'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'IOBT_flt', 'ADEP_mvt', 'ADES_mvt']
pd.set_option('display.width', 250)
fr = []
for which in ['training', 'ranking']:
    d = load(which, columns=C)
    d['src'] = which
    fr.append(d)
d = pd.concat(fr, ignore_index=True)
del fr
ts = lambda s: s.astype('int64') // 10**9 if s.notna().all() else s.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
for c in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'IOBT_flt']:
    v = d[c]
    d[c[:5]] = np.where(v.notna(), v.values.astype('datetime64[s]').astype('int64'), np.nan) if True else None
d.loc[d.BLOCK_TIME_UTC_mvt.isna(), 'BLOCK'] = np.nan
d.loc[d.LOBT_flt.isna(), 'LOBT_'] = np.nan
d.loc[d.IOBT_flt.isna(), 'IOBT_'] = np.nan

# --- anchor table: FLIGHT_ID -> LOBT (dedupe on FLIGHT_ID)
a = d[d.FLIGHT_ID_mvt.notna() & d.LOBT_.notna()][['FLIGHT_ID_mvt', 'LOBT_', 'IOBT_']].copy()
nun = a.groupby('FLIGHT_ID_mvt').LOBT_.nunique()
print('anchors', len(a), 'unique ids', a.FLIGHT_ID_mvt.nunique(), 'ids with >1 distinct LOBT', (nun > 1).sum())
a = a.drop_duplicates('FLIGHT_ID_mvt').sort_values('FLIGHT_ID_mvt')
fid = a.FLIGHT_ID_mvt.values
lob = a.LOBT_.values
dl = np.diff(lob)
print('sorted by FLIGHT_ID: LOBT diffs <0 share', (dl < 0).mean(), ' <-60', (dl < -60).mean(), ' < -3600', (dl < -3600).mean())
print('diff quantiles', np.quantile(dl, [0, .001, .01, .1, .5, .9, .99, .999, 1]))
cm = np.maximum.accumulate(lob)
print('LOBT below running max by >60 s share', ((cm - lob) > 60).mean(), ' >600', ((cm - lob) > 600).mean())
# IOBT vs LOBT
iv = a.IOBT_.values
print('IOBT==LOBT share', np.mean(iv == lob))
dli = np.diff(iv)
print('sorted by FLIGHT_ID: IOBT diffs <0 share', (dli < 0).mean())
# id gap -> seconds
idg = np.diff(fid)
print('FLIGHT_ID gaps between anchors quantiles', np.quantile(idg, [.5, .9, .99]))
slope = (lob[-1] - lob[0]) / (fid[-1] - fid[0])
print('global sec per FLIGHT_ID', slope)

# --- interpolate LOBT for rows with FLIGHT_ID but no LOBT (exclude their own id from anchors: they have none)
q = d[d.FLIGHT_ID_mvt.notna() & d.LOBT_.isna()].copy()
pos = np.searchsorted(fid, q.FLIGHT_ID_mvt.values)
pos = np.clip(pos, 1, len(fid) - 1)
lo_id, hi_id = fid[pos - 1], fid[pos]
lo_t, hi_t = lob[pos - 1], lob[pos]
q['lobt_lo'] = lo_t; q['lobt_hi'] = hi_t
q['id_gap'] = hi_id - lo_id
w = (q.FLIGHT_ID_mvt.values - lo_id) / np.maximum(hi_id - lo_id, 1)
q['lobt_est'] = lo_t + w * (hi_t - lo_t)
q['bracket_width'] = hi_t - lo_t
q['blk_m_lobt'] = q.BLOCK - q.lobt_est
q['mvt_m_lobt'] = q.MVT_T - q.lobt_est
q['sch_m_lobt'] = q.SCHED - q.lobt_est
q['y'] = np.where(q.PHASE_mvt.eq('DEP'), q.MVT_T - q.BLOCK, q.BLOCK - q.MVT_T)
for (src, ph), g in q.groupby(['src', 'PHASE_mvt']):
    print(f'\n===== {src} {ph}: rows with FLIGHT_ID but no LOBT: {len(g)}')
    print('bracket width quantiles', np.nanquantile(g.bracket_width, [.5, .9, .99, 1]))
    for c in ['blk_m_lobt', 'mvt_m_lobt', 'sch_m_lobt']:
        v = g[c].dropna()
        if len(v):
            print(f'  {c}: n={len(v)} |.|<=3606 share {np.mean(np.abs(v) <= 3606):.3f}  quantiles',
                  np.quantile(v, [0, .1, .25, .5, .75, .9, 1]).round(0))
    if ph == 'DEP':
        cols = ['APT', 'FLIGHT_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'y',
                'bracket_width', 'blk_m_lobt', 'mvt_m_lobt', 'sch_m_lobt']
        print(g[cols].sort_values('SCHED_TIME_UTC_mvt').to_string())
# control: for rows WITH LOBT, how good is leave-one-out interpolation? (DEP training, sample)
rng = np.random.default_rng(0)
k = rng.choice(np.arange(1, len(fid) - 1), 200000, replace=False)
est = lob[k - 1] + (fid[k] - fid[k - 1]) / (fid[k + 1] - fid[k - 1]) * (lob[k + 1] - lob[k - 1])
err = est - lob[k]
print('\nLOO interpolation error on anchors: quantiles of |err|', np.quantile(np.abs(err), [.5, .9, .99, .999, 1]))
q.to_parquet('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/ids_02_noflt.parquet')
