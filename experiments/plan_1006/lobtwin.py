import pandas as pd, numpy as np, glob
D='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt','AOBT_3_flt']
# 1) 2025 rule check (training labels, 2025 only)
tot=0; inw=0; mx=0; nl=0
for f in sorted(glob.glob(D+'training_2025-*.parquet')):
    t=pd.read_parquet(f, columns=cols)
    t=t[t.PHASE_mvt=='DEP']
    m=t.LOBT_flt.notna()&t.BLOCK_TIME_UTC_mvt.notna()
    d=(t.BLOCK_TIME_UTC_mvt[m]-t.LOBT_flt[m]).dt.total_seconds().abs()
    tot+=m.sum(); inw+=(d<=3606).sum(); mx=max(mx,d.max()); nl+=(~t.LOBT_flt.notna()).sum()
print(f'2025 DEP with LOBT: {tot:,}; within 3606: {inw:,} ({inw/tot:.6%}); max |BLOCK-LOBT| = {mx:.0f}s; DEP without LOBT: {nl:,}')
# 2) v27 / v26 / v24 submissions vs the window on the ranking set (features only)
r=pd.read_parquet(D+'ranking.parquet', columns=[c for c in cols if c not in ('BLOCK_TIME_UTC_mvt',)])
r=r[r.PHASE_mvt=='DEP']
for v,S in [(27,243.1478),(26,243.42),(24,244.22)]:
    s=pd.read_parquet(D+f'prc2026/outputs/understanding-accordion_v{v}.parquet')
    x=r.merge(s,on='MVT_ID_mvt',how='inner')
    N=len(s)
    ml=(x.MVT_TIME_UTC_mvt-x.LOBT_flt).dt.total_seconds()
    lo=(ml-3606).clip(lower=0); hi=ml+3606
    p=x.TAXITIME_SEC_mvt.astype(float)
    q=p.clip(lower=lo, upper=hi).where(ml.notna(), p)
    q=q.clip(lower=0)
    dd=(p-q)
    bad=dd!=0
    gain=(dd**2).sum()/N
    print(f'v{v}: N={N:,} merged={len(x):,}; rows outside LOBT window: {bad.sum()} (above hi: {(p>hi).sum()}, below lo: {(p<lo).sum()}); sum(p-q)^2/N={gain:.1f}; score bound sqrt(S^2-gain)={np.sqrt(S**2-gain):.3f} (delta {np.sqrt(S**2-gain)-S:+.3f})')
    if v==27:
        b=x[bad].assign(p=p[bad],q=q[bad],ml=ml[bad])
        print(b.groupby('ADEP_mvt').size().to_dict())
        print(b[['ADEP_mvt','p','q','ml']].assign(d=lambda z:z.p-z.q).sort_values('d',key=abs,ascending=False).head(12).to_string())
