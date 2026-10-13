# READ-ONLY: LOBT window rule (|BLOCK - LOBT| <= 3606 s) -- 2025 truth check + value on v24 2025 OOF + rows moved in v27 (label-free)
import numpy as np, pandas as pd, glob
ROOT='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
C=ROOT+'prc2026/cache/'
cols=['PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','TAXITIME_SEC_mvt','APT','MVT_ID_mvt']
t=pd.read_parquet(C+'training.parquet',columns=cols); t=t[t.PHASE_mvt=='DEP'].reset_index(drop=True)
z=np.load(C+'w2/v24s_oof.npz'); y,f,month=z['y'],z['f'],z['month']
print('rows',len(t),len(y),'y aligned:',np.allclose(t.TAXITIME_SEC_mvt.to_numpy(float),y,equal_nan=True))
SEC=lambda s: (pd.to_datetime(s,utc=True)-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds()
ts=lambda c: SEC(t[c])
mvt,blk,lobt=ts('MVT_TIME_UTC_mvt'),ts('BLOCK_TIME_UTC_mvt'),pd.to_datetime(t.LOBT_flt)
has=lobt.notna().to_numpy(); lob=SEC(t.LOBT_flt).to_numpy()
d=np.abs(blk.to_numpy()-lob)
print('has LOBT %.4f'%has.mean(),'| |BLOCK-LOBT|<=3606 among has: %.6f'%np.mean(d[has]<=3606),' max %.0f'%np.nanmax(d), ' n>3606:',int(np.sum(d[has]>3606)))
lo=mvt.to_numpy()-lob-3606; hi=mvt.to_numpy()-lob+3606
g=f.copy(); m=has
g[m]=np.clip(f[m],np.maximum(lo[m],0),hi[m])
moved=m&(g!=f)
sc=np.isin(month,[1,7]); held=~np.isin(month,[3,6,9,12]); al=np.ones(len(y),bool)
R=lambda p,mm: np.sqrt(((y-p)[mm]**2).mean())
print('rows moved: all %d, Jan+Jul %d; |move| median %.0f p90 %.0f max %.0f'%(moved.sum(),(moved&sc).sum(),np.median(np.abs(g-f)[moved]),np.percentile(np.abs(g-f)[moved],90),np.abs(g-f)[moved].max()))
for nm,mm in (('12m',al),('held',held),('Jan+Jul',sc),('Jan',month==1),('Jul',month==7)):
    print(f'  {nm:8s} v24 {R(f,mm):8.3f} -> window {R(g,mm):8.3f}  ({R(g,mm)-R(f,mm):+.3f})')
de=(y-f)**2-(y-g)**2
print('  gain share of top-10 moved rows (12m): %.0f%%'%(100*np.sort(de[moved])[::-1][:10].sum()/de[moved].sum()), ' rows worse:',int((de<0).sum()))
print('  by month SSE change:',pd.Series(de).groupby(month).sum().round(-6).to_dict())
# 2026 ranking: rows the window would move in v27 (label-free)
k=pd.read_parquet(C+'ranking.parquet',columns=['PHASE_mvt','MVT_ID_mvt','MVT_TIME_UTC_mvt','LOBT_flt','APT']); k=k[k.PHASE_mvt=='DEP']
sub=pd.read_parquet(ROOT+'prc2026/outputs/understanding-accordion_v27.parquet') if glob.glob(ROOT+'prc2026/outputs/understanding-accordion_v27.parquet') else None
if sub is None:
    print('v27 file not found at prc2026/outputs; candidates:',glob.glob(ROOT+'**/understanding-accordion_v27.parquet',recursive=True)[:3])
else:
    k=k.merge(sub,on='MVT_ID_mvt',how='left')
    kl=pd.to_datetime(k.LOBT_flt); hk=kl.notna().to_numpy()
    km=SEC(k.MVT_TIME_UTC_mvt)
    klo=km.to_numpy()-SEC(k.LOBT_flt).to_numpy()-3606; khi=klo+7212
    p=k.TAXITIME_SEC_mvt.to_numpy(float)
    q=p.copy(); q[hk]=np.clip(p[hk],np.maximum(klo[hk],0),khi[hk])
    mv=hk&(q!=p)
    print('2026 v27: has LOBT %.4f, rows moved %d, |move| median %.0f max %.0f; by apt:'%(hk.mean(),mv.sum(),np.median(np.abs(q-p)[mv]) if mv.any() else 0,np.abs(q-p)[mv].max() if mv.any() else 0), k.APT[mv].value_counts().to_dict())
    print('  implied SSE-neutral bound: sum of squared moves %.3g (if all moves were correct, RMSE gain <= %.2f s)'%(((q-p)[mv]**2).sum(), 243.1478-np.sqrt(max(243.1478**2-((q-p)[mv]**2).sum()/len(p),0))))
