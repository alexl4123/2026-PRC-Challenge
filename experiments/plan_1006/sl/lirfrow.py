import pandas as pd, numpy as np, glob
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
T=['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt','AOBT_3_flt','ARVT_1_flt','ARVT_3_flt']
rk=pd.read_parquet(ROOT+'ranking.parquet'); rd=rk[rk.PHASE_mvt=='DEP']
off=(rd.MVT_TIME_UTC_mvt-rd.LOBT_flt).dt.total_seconds()
row=rd[off>40000]
for _,r in row.iterrows():
    print('2026 row: apt',r.ADEP_mvt,'ades',r.ADES_mvt,'type',r.AIRCRAFT_TYPE_mvt,'stand',r.STAND_mvt)
    for c in T: print('  MVT -',c, (r.MVT_TIME_UTC_mvt-r[c]).total_seconds() if pd.notna(r[c]) else None)
    print('  ARVT_3 - MVT', (r.ARVT_3_flt-r.MVT_TIME_UTC_mvt).total_seconds())
d=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(ROOT+'training_2025-07*.parquet'))]); dd=d[d.PHASE_mvt=='DEP']
off2=(dd.MVT_TIME_UTC_mvt-dd.LOBT_flt).dt.total_seconds()
for _,r in dd[off2>40000].iterrows():
    print('2025 analogue: apt',r.ADEP_mvt,'y',r.TAXITIME_SEC_mvt)
    for c in T+['BLOCK_TIME_UTC_mvt']: print('  MVT -',c, (r.MVT_TIME_UTC_mvt-r[c]).total_seconds() if pd.notna(r[c]) else None)
    print('  ARVT_3 - MVT', (r.ARVT_3_flt-r.MVT_TIME_UTC_mvt).total_seconds())
# how many 2025 deps have MVT-AOBT_3 > 40000 and their y
dall=pd.concat([pd.read_parquet(f,columns=['PHASE_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','LOBT_flt','TAXITIME_SEC_mvt']) for f in sorted(glob.glob(ROOT+'training_2025-*.parquet'))]); dall=dall[dall.PHASE_mvt=='DEP']
a3=(dall.MVT_TIME_UTC_mvt-dall.AOBT_3_flt).dt.total_seconds(); k=a3>20000
print('2025 deps with MVT-AOBT_3 > 20000:', k.sum(), ' y:', dall.TAXITIME_SEC_mvt[k].tolist(), ' off_lobt', ((dall.MVT_TIME_UTC_mvt-dall.LOBT_flt).dt.total_seconds()[k]).tolist())
