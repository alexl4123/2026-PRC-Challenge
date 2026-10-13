import pandas as pd, numpy as np
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
rk=pd.read_parquet(ROOT+'ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','LOBT_flt'])
rk=rk[rk.PHASE_mvt=='DEP']
s=pd.read_parquet(ROOT+'prc2026/outputs/understanding-accordion_v27.parquet')
m=s.merge(rk,on='MVT_ID_mvt',how='left'); assert len(m)==len(s)==344841
p=m.TAXITIME_SEC_mvt.to_numpy(float); off=(m.MVT_TIME_UTC_mvt-m.LOBT_flt).dt.total_seconds().to_numpy(); has=~np.isnan(off)
R=243.1478; n=len(p)
for W in [3606,3660,3700]:
    lo=np.ceil(np.maximum(off-W,0)); hi=np.floor(off+W)
    q=p.copy(); q[has]=np.clip(p[has],lo[has],hi[has]); ch=q!=p
    lb=np.sum((q-p)**2); lb_x=np.sum(((q-p)**2)[ch & ~(np.abs(q-p)>10000)])
    print(f'W={W}: rows changed {ch.sum()}, guaranteed live <= {np.sqrt(R**2-lb/n):.2f} ({np.sqrt(R**2-lb/n)-R:+.2f}); without the LIRF 86,820-s row: {np.sqrt(R**2-lb_x/n)-R:+.3f}')
    if W==3606:
        mm=m.assign(sh=q-p, mon=m.MVT_TIME_UTC_mvt.dt.month).groupby(['ADEP_mvt','mon']).sh.mean()
        print('  airport x month mean shift range', round(mm.min(),2), round(mm.max(),2))
        # scenarios for the LIRF row
        i=np.argmax(np.abs(q-p)); others=lb-(q[i]-p[i])**2
        for yv,lab in [(lo[i],'y at lower bound'),(off[i],'y = off_lobt (BLOCK=LOBT, 2025 analogue)'),(hi[i],'y at upper bound'),(94560,'y = off_sched (stamped at SCHED, outside window)'),(1000,'y = 1000 s real taxi (window violated)')]:
            d_sse=(yv-p[i])**2-(yv-q[i])**2
            print(f'   LIRF row scenario {lab}: y={yv:.0f}, live change {np.sqrt(R**2-(d_sse+others)/n)-R:+.2f} s (others at lower bound)')
