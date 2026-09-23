import sys, os, numpy as np, pandas as pd
os.environ['OMP_NUM_THREADS']='2'
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
cols=['APT','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','LOBT_flt','IOBT_flt','TAXITIME_SEC_mvt']
d=load('training',phase='DEP',columns=cols)
y=d.TAXITIME_SEC_mvt.astype(float).values
oof=np.load(C+'exp62_oof.npy')[1]
res=y-oof
proxy=(d.MVT_TIME_UTC_mvt-d.AOBT_3_flt).dt.total_seconds().values
A=np.isfinite(proxy)&(proxy>0)&(proxy<=7200)
bulk=y<=7200
def secs(s): return s.dt.second.values
print('seconds-of-minute==0 share (all DEP rows, non-null):')
for c in ['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','LOBT_flt','IOBT_flt','SCHED_TIME_UTC_mvt']:
    s=d[c]; ok=s.notna().values
    z=(s.dt.second.values[ok]==0)
    print(f'  {c:22s} {z.mean():.3f}')
print()
print('per airport (route A & y<=7200):')
print(f'{"apt":5s} {"n":>8s} {"blk:00":>7s} {"aob:00":>7s} {"mvt:00":>7s} {"|d|<=30":>8s} {"|d|<=60":>8s} {"|d|<=120":>8s} {"med d":>7s} {"iqr d":>12s} {"sd d":>7s} {"rmse v6":>8s} {"sd y":>7s} {"corr":>6s}')
for a in sorted(d.APT.unique()):
    m=(d.APT.values==a)&A&bulk
    dd=y[m]-proxy[m]
    bs=(d.BLOCK_TIME_UTC_mvt[m].dt.second.values==0).mean()
    as_=(d.AOBT_3_flt[m].dt.second.values==0).mean()
    ms=(d.MVT_TIME_UTC_mvt[m].dt.second.values==0).mean()
    q=np.percentile(dd,[25,50,75])
    print(f'{a:5s} {m.sum():8d} {bs:7.3f} {as_:7.3f} {ms:7.3f} {np.mean(np.abs(dd)<=30):8.3f} {np.mean(np.abs(dd)<=60):8.3f} {np.mean(np.abs(dd)<=120):8.3f} {q[1]:7.0f} [{q[0]:5.0f},{q[2]:5.0f}] {dd.std():7.0f} {np.sqrt(np.mean(res[m]**2)):8.1f} {y[m].std():7.0f} {np.corrcoef(y[m],proxy[m])[0,1]:6.3f}')
# what share of bulk SSE (v6) by route, per airport
print()
print('v6 residual SSE share by route among bulk rows (y<=7200):')
D=~np.isfinite(proxy)
for name,mm in [('A',A),('D',D),('other',~A&~D)]:
    m=mm&bulk
    print(f'  {name}: rows {m.mean():.4f} SSE share {np.sum(res[m]**2)/np.sum(res[bulk]**2):.3f} rmse {np.sqrt(np.mean(res[m]**2)):.1f}')
# how much of route-A bulk v6 error is "proxy discrepancy": regress resid on (y-proxy)
m=A&bulk
dd=y[m]-proxy[m]
print('route A bulk: rmse v6 %.1f ; rmse of raw proxy %.1f ; rmse of proxy+apt-median-shift %.1f'%(np.sqrt(np.mean(res[m]**2)), np.sqrt(np.mean(dd**2)), np.sqrt(np.mean((dd-pd.Series(dd).groupby(d.APT.values[m]).transform('median').values)**2))))
print('corr(v6 resid, y-proxy) on route A bulk: %.3f'%np.corrcoef(res[m],dd)[0,1])
print('peak rss',peak_rss_gb())
