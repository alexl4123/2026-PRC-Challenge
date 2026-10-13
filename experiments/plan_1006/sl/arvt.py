import pandas as pd, numpy as np, glob
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','ARVT_3_flt','TAXITIME_SEC_mvt']
d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(ROOT+'training_2025-*.parquet'))]); d=d[d.PHASE_mvt=='DEP']
rk=pd.read_parquet(ROOT+'ranking.parquet',columns=cols); rk=rk[rk.PHASE_mvt=='DEP']
s=pd.read_parquet(ROOT+'prc2026/outputs/understanding-accordion_v27.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
rk=rk.merge(s,on='MVT_ID_mvt')
for nm,x in [('2025',d),('2026',rk)]:
    a=(x.MVT_TIME_UTC_mvt-x.ARVT_3_flt).dt.total_seconds(); b=(x.MVT_TIME_UTC_mvt-x.AOBT_3_flt).dt.total_seconds(); l=(x.MVT_TIME_UTC_mvt-x.LOBT_flt).dt.total_seconds()
    k=a>0
    print(nm,'deps',len(x),' MVT after NM arrival (ARVT_3<MVT):',k.sum(), ' MVT-AOBT_3>7200:',(b>7200).sum(), ' MVT-LOBT>7200:', (l>7200).sum())
    cols_=['ADEP_mvt']+(['TAXITIME_SEC_mvt'] if nm=='2025' else ['pred'])
    print(x[k].assign(mvt_arvt3=a[k],mvt_aobt3=b[k],off_lobt=l[k])[cols_+['mvt_arvt3','mvt_aobt3','off_lobt']].to_string())
