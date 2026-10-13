import pandas as pd, numpy as np
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
rk=pd.read_parquet(ROOT+'ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt'])
rk=rk[rk.PHASE_mvt=='DEP']
print('ranking deps', len(rk), 'LOBT present', rk.LOBT_flt.notna().mean())
for v in ['v24','v26','v27']:
    s=pd.read_parquet(ROOT+f'prc2026/outputs/understanding-accordion_{v}.parquet')
    m=rk.merge(s,on='MVT_ID_mvt',how='inner')
    p=m.TAXITIME_SEC_mvt.to_numpy(float)
    offl=(m.MVT_TIME_UTC_mvt-m.LOBT_flt).dt.total_seconds().to_numpy()
    has=~np.isnan(offl)
    lo=np.maximum(offl-3606,0); hi=offl+3606
    pc=p.copy(); pc[has]=np.clip(p[has],lo[has],hi[has])
    ch=pc!=p
    n=len(s)
    lb=np.sum((p-pc)**2)
    R=243.1478 if v=='v27' else (243.4219 if v=='v26' else 244.2213)
    newR=np.sqrt(max(R**2-lb/n,0))
    print(v, 'rows', n, 'merged', len(m), 'changed', ch.sum(), 'below', ((p<lo)&has).sum(), 'above', ((p>hi)&has).sum(),
          'sum sq shift', f'{lb:.3e}', 'guaranteed live delta <=', round(newR-R,3))
    if v=='v27':
        idx=np.where(ch)[0]; o=np.argsort(-np.abs(p[idx]-pc[idx]))
        t=pd.DataFrame({'apt':m.ADEP_mvt.to_numpy()[idx[o]],'month':m.MVT_TIME_UTC_mvt.dt.month.to_numpy()[idx[o]],'pred':p[idx[o]],'proj':pc[idx[o]].round(),'off_lobt':offl[idx[o]],
           'off_sched':(m.MVT_TIME_UTC_mvt-m.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()[idx[o]],'aobt_na':m.AOBT_3_flt.isna().to_numpy()[idx[o]]})
        pd.set_option('display.width',200); print(t.head(25)); print(t.groupby(['apt','month']).size())
        print('shift abs: median', np.median(np.abs(p[idx]-pc[idx])), 'max', np.max(np.abs(p[idx]-pc[idx])))
