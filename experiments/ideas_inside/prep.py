"""Compact DEP cache for the inside-ideas tests. Row order = load('training', phase='DEP')."""
import os, resource, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache'
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
STR = ['APT', 'FLIGHT_mvt', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
       'AIRCRAFT_OPERATOR_flt', 'CALLSIGN_flt']
TS = ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt',
      'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt']
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
for which in ['training', 'ranking']:
    for phase in ['DEP', 'ARR']:
        cols = STR + TS + ['TAXITIME_SEC_mvt', 'MVT_ID_mvt', 'PHASE_mvt']
        t = pq.read_table(f'{R}/{which}.parquet', columns=cols, read_dictionary=STR + ['PHASE_mvt'])
        mask = pc.equal(pc.cast(t['PHASE_mvt'], pa.string()), phase)
        t = t.filter(mask).drop(['PHASE_mvt'])
        df = t.to_pandas()
        del t
        for c in TS:
            df[c] = (df[c] - EPOCH).dt.total_seconds().astype('float64')
        df['TAXITIME_SEC_mvt'] = df['TAXITIME_SEC_mvt'].astype('float64')
        df.to_parquet(f'{OUT}/{which}_{phase}.parquet')
        print(which, phase, len(df), f'{time.time()-t0:.0f}s', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, 'GB')
        del df
d = pd.read_parquet(f'{OUT}/training_DEP.parquet', columns=['TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy()
o = np.load(f'{R}/exp62_oof.npy', mmap_mode='r')
print('rows', len(y), 'v6 RMSE', np.sqrt(np.mean((y - o[1])**2)), '(expect 274.114)')
print('peak GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
