"""ids_14 -- what is common inside a raw-file chunk? Print chunks; test candidate grouping keys by within-chunk purity."""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
D = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
H = D + 'prc2026/'
pd.set_option('display.width', 280); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
d = pd.read_parquet(D + 'training_2025-07-01_2025-08-01.parquet')
ph = d.PHASE_mvt.astype(str).values
d['APT'] = np.where(ph == 'DEP', d.ADEP_mvt.astype(str), d.ADES_mvt.astype(str))
m = d.MVT_TIME_UTC_mvt.values.astype('datetime64[s]').astype('int64')
brk = np.r_[True, (d.APT.values[1:] != d.APT.values[:-1]) | (np.diff(m) < -60)]
d['cid'] = np.cumsum(brk)
d['csize'] = d.groupby('cid').cid.transform('size')
d['date'] = d.MVT_TIME_UTC_mvt.dt.date
d['sdate'] = d.SCHED_TIME_UTC_mvt.dt.date
d['bdate'] = d.BLOCK_TIME_UTC_mvt.dt.date
d['op'] = d.FLIGHT_mvt.astype(str).str[:3]
d['hr'] = d.MVT_TIME_UTC_mvt.dt.hour
d['bhr'] = d.BLOCK_TIME_UTC_mvt.dt.hour
cols = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
        'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']
for c in [1, 2, 3]:
    print(f'--- chunk {c}')
    print(d[d.cid == c][cols].head(25).to_string())
# purity: for each candidate key, the mean (over chunks with >= 10 rows) of the modal share
big = d[d.csize >= 10]
for k in ['PHASE_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'op', 'date', 'sdate', 'bdate', 'hr', 'bhr', 'ADES_mvt', 'ADEP_mvt',
          'FLIGHT_RULE_mvt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'FLIGHT_TYPE_flt']:
    s = big.groupby('cid')[k].agg(lambda x: x.astype(str).value_counts(normalize=True).iloc[0]).mean()
    # baseline: same chunk sizes but random rows of the same airport
    print(f'  purity {k}: {s:.3f}')
# chunk count per (APT, date): how many chunks contain a given day's rows?
cc = d.groupby(['APT', 'date']).cid.nunique()
print('chunks per airport-day: quantiles', cc.quantile([.1, .5, .9]).to_dict())
# Do chunks partition by MVT_ID modulo something? by FLIGHT_ID parity?
for k, v in [('MVT_ID mod 2', d.MVT_ID_mvt % 2), ('MVT_ID mod 3', d.MVT_ID_mvt % 3), ('FLIGHT_ID mod 2', d.FLIGHT_ID_mvt % 2)]:
    d['_k'] = v
    s = d[d.csize >= 10].groupby('cid')._k.agg(lambda x: x.astype(str).value_counts(normalize=True).iloc[0]).mean()
    print(f'  purity {k}: {s:.3f}')
