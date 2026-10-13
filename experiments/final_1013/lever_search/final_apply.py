import os, sys, numpy as np, pandas as pd
os.environ['PRC_RANKING']='final'
R='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'; sys.path.insert(0,R)
import assemble as A
from sklearn.isotonic import IsotonicRegression
S=sys.argv[1]
C=A.Components()
out=A.assemble(C,'UT',lobt=False)
win=A.lobt_window(C,out)
q=A.join_complement(C,win,out)
shipped=pd.read_parquet(R+'/outputs_final/understanding-accordion_final.parquet').set_index(A.ID)['TAXITIME_SEC_mvt'].reindex(C.ids).to_numpy(float)
print('rebuild vs shipped final: rows differing', int((q!=shipped).sum()), 'max', np.abs(q-shipped).max())
proj=q!=out
ol=C.off_lobt; ok=C.ABC&np.isfinite(ol)&(ol>3606)&~proj
# 2025 fits (full year)
Z=np.load(S+'/base.npz',allow_pickle=True); y,B=Z['y'],Z['B']; ol25=Z['off_lobt']
ABC25=~Z['D']&~Z['P']; ok25=ABC25&np.isfinite(ol25)&(ol25>3606)&~Z['proj']; e=y-B
cell25=np.digitize(ol25,[4200,5000,6000,8000]); delta=np.array([e[ok25&(cell25==c)].mean() for c in range(5)])
ir=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(ol25[ok25],e[ok25])
lo=np.ceil(np.maximum(0,ol-3606)); hi=np.floor(ol+3606); cap=np.isin(C.apt,A.CAP_APTS)
def fin(g):
    g=np.rint(np.clip(g,0,A.HI)); m=np.isfinite(ol); g[m]=np.minimum(np.maximum(g[m],lo[m]),hi[m]); g[cap]=np.minimum(g[cap],3606); g[proj]=q[proj]; return g
cell=np.digitize(ol,[4200,5000,6000,8000])
g3b=q.astype(float).copy(); g3b[ok]+=delta[cell[ok]]; g3b=fin(g3b)
g3c=q.astype(float).copy(); g3c[ok]+=ir.predict(ol[ok]); g3c=fin(g3c)
print('2025 deltas K3b', delta.round(2))
mon=C.month
for nm,g in (('K3b',g3b),('K3c',g3c)):
    d=g-q; mv=d!=0
    print(f'{nm}: rows moved {mv.sum()}, RMS move over all rows {np.sqrt(np.mean(d**2)):.3f}, mean move on moved {d[mv].mean():.1f}, max +{d.max():.0f} / {d.min():.0f}')
    print('   moved per bucket', [int((mv&(cell==c)&ok).sum()) for c in range(5)], ' per month', {m:int((mv&(mon==m)).sum()) for m in np.unique(mon)})
print('final ABC rows per bucket (label-free):', [int((C.ABC&np.isfinite(ol)&(ol>3606)&(cell==c)).sum()) for c in range(5)], 'of', int(C.ABC.sum()))
print('2025 ABC rows per bucket:', [int((ABC25&np.isfinite(ol25)&(ol25>3606)&(cell25==c)).sum()) for c in range(5)], 'of', int(ABC25.sum()))
print('share ABC with off_lobt>3606: 2025 %.4f final %.4f; >8000: 2025 %.5f final %.5f' % ((ABC25&(ol25>3606)).sum()/ABC25.sum(), (C.ABC&(ol>3606)).sum()/C.ABC.sum(), (ABC25&(ol25>8000)).sum()/ABC25.sum(), (C.ABC&(ol>8000)).sum()/C.ABC.sum()))
print('final per month share >3606:', {m: round(float(((C.ABC&(ol>3606))&(mon==m)).sum()/(C.ABC&(mon==m)).sum()),4) for m in np.unique(mon)})
np.savez(S+'/final_k3.npz', ids=C.ids, q=q, g3b=g3b, g3c=g3c)
