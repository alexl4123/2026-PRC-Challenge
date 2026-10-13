"""FLIGHT_ID is assigned in LOBT order (global, exact). Departures with a FLIGHT_ID but no LOBT (month-boundary rows):
infer LOBT from the FLIGHT_ID neighbours, check the LOBT window on 2025, apply to 2026."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
pd.set_option('display.width', 250)
fr = []
for which in ('training', 'ranking'):
    t = pd.read_parquet(H + f'stand_ev_{which}.parquet', columns=['id', 'FLIGHT_ID_mvt', 'PHASE_mvt', 'APT', 'mvt', 'blk', 'sch', 'lobt', 'FLIGHT_mvt'])
    t['fid'] = pd.to_numeric(t.FLIGHT_ID_mvt, errors='coerce'); t['file'] = which
    fr.append(t.drop(columns='FLIGHT_ID_mvt'))
a = pd.concat(fr, ignore_index=True)
k = a[np.isfinite(a.fid) & np.isfinite(a.lobt)].groupby('fid').lobt.agg(['min', 'max']).sort_index()
assert (k['min'] == k['max']).all()
fids = k.index.to_numpy(); lob = k['min'].to_numpy()
print('combined fid->lobt pairs', len(fids), 'monotone nondecr:', bool(np.all(np.diff(lob) >= 0)))
# do the two files interleave? (any training fid > some ranking fid with smaller lobt is covered by the monotone check)
tgt = a[(a.PHASE_mvt == 'DEP') & np.isfinite(a.fid) & ~np.isfinite(a.lobt)].copy()
j = np.searchsorted(fids, tgt.fid.to_numpy())
tgt['lobt_lo'] = np.where(j > 0, lob[np.maximum(j - 1, 0)], np.nan)
tgt['lobt_hi'] = np.where(j < len(fids), lob[np.minimum(j, len(fids) - 1)], np.nan)
tgt['fid_gap_lo'] = tgt.fid - np.where(j > 0, fids[np.maximum(j - 1, 0)], np.nan)
tgt['fid_gap_hi'] = np.where(j < len(fids), fids[np.minimum(j, len(fids) - 1)], np.nan) - tgt.fid
tgt['width'] = tgt.lobt_hi - tgt.lobt_lo
tgt['y'] = tgt.mvt - tgt.blk
tgt['B_lo'] = tgt.blk - tgt.lobt_lo; tgt['B_hi'] = tgt.blk - tgt.lobt_hi
# window: B in [lobt_lo - 3606, lobt_hi + 3606]  <=> y in [T - lobt_hi - 3606, T - lobt_lo + 3606]
tgt['y_lo'] = np.maximum(0, tgt.mvt - tgt.lobt_hi - 3606); tgt['y_hi'] = tgt.mvt - tgt.lobt_lo + 3606
tgt['off_sched'] = tgt.mvt - tgt.sch
T = lambda c: pd.to_datetime(tgt[c], unit='s')
show = tgt.assign(T=T('mvt'), SCHED=T('sch'), LOBT_lo=T('lobt_lo'), LOBT_hi=T('lobt_hi'))
cols = ['file', 'APT', 'FLIGHT_mvt', 'T', 'SCHED', 'off_sched', 'LOBT_lo', 'LOBT_hi', 'width', 'y', 'y_lo', 'y_hi', 'B_lo', 'B_hi']
tr = show[show.file == 'training']
print('2025 rows', len(tr), '| width quantiles', np.nanpercentile(tr.width, [0, 50, 100]))
inside = (tr.y >= tr.y_lo) & (tr.y <= tr.y_hi)
print('2025: y inside the inferred window:', int(inside.sum()), 'of', int(np.isfinite(tr.y_hi).sum()), '| B - LOBT_lo range', tr.B_lo.min(), tr.B_lo.max(), '| B - LOBT_hi range', tr.B_hi.min(), tr.B_hi.max())
print(tr[cols].to_string())
print(show[show.file == 'ranking'][cols].drop(columns=['y', 'B_lo', 'B_hi']).to_string())
tgt.to_parquet(H + 'stand_s08_fid_targets.parquet')
