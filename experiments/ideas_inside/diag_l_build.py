"""Build exp62 candidate features for P2 = route D & 1800 < off_sched <= 7200 (training + ranking),
by calling prc.p_features.candidates with its population() swapped IN THIS PROCESS ONLY."""
import sys, time
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from common import *
import pyarrow.parquet as pq
import prc.p_features as pf
from prc.config import CAT_COLS
def pop2(off_sched, aobt_missing, cap=None):
    o = np.asarray(off_sched, float)
    return np.asarray(aobt_missing, bool) & np.isfinite(o) & (o > 1800) & (o <= 7200)
pf.population = pop2
t0 = time.time()
for which in ['training', 'ranking']:
    cols = pf.LOADCOLS + (['TAXITIME_SEC_mvt'] if which == 'training' else [])
    strs = [c for c in cols if c in CAT_COLS or c in ('FLIGHT_mvt', 'STAND_mvt', 'APT')]
    d = pq.read_table(f'{R}/{which}.parquet', columns=cols, read_dictionary=strs).to_pandas()
    for c in d.columns:
        if str(d[c].dtype) == 'category':
            d[c] = d[c].cat.remove_unused_categories()
    dep = d.PHASE_mvt.eq('DEP').to_numpy()
    if which == 'training':
        d = d.drop(columns='TAXITIME_SEC_mvt')
    d['BLOCK_TIME_UTC_mvt'] = d['BLOCK_TIME_UTC_mvt'].where(~dep)          # leakage guard, as in load_set
    assert d.loc[dep, 'BLOCK_TIME_UTC_mvt'].isna().all()
    cand, pool, P = pf.candidates(d, dep)
    cand.to_parquet(f'{D}/p2_cand_{which}.parquet'); pool.to_parquet(f'{D}/p2_pool_{which}.parquet')
    print(which, 'P2 rows', len(cand), f'{time.time()-t0:.0f}s peak {peak():.2f}', flush=True)
    del d, cand, pool
