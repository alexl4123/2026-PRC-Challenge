# READ-ONLY, 2025 OOF only: what would a perfect airport-month taxi-out aggregate be worth (capped vs uncapped)?
import numpy as np, pandas as pd
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
z=np.load(C+'w2/v24s_oof.npz',allow_pickle=True)
y,f,month,P=z['y'],z['f'],z['month'],z['P']
r=pd.read_parquet(C+'drivers_0926/rows_training.parquet',columns=['apt','month','off_sched','route','y'])
assert np.allclose(r.y.to_numpy(float),y,equal_nan=True)
apt=r.apt.astype(str).to_numpy(); so=r.off_sched.to_numpy(float)
Pidx=z['Pidx']; p=z['p14']; pb=z['pbP']; pr=z['P_rest']
recon=np.rint(np.clip(p*so[Pidx]+(1-p)*pb+pr,0,None))
print('P recon max |diff|',np.nanmax(np.abs(recon-f[Pidx])), 'median', np.nanmedian(np.abs(recon-f[Pidx])))
sc=np.isin(month,[1,7]); N=len(y)
R=lambda g,m: np.sqrt(((y-g)[m]**2).mean())
base={'Jan+Jul':R(f,sc),'Jan':R(f,month==1),'Jul':R(f,month==7),'12m':R(f,np.ones(N,bool))}
print('base',{k:round(v,3) for k,v in base.items()})
# SSE concentration
e=(y-f)**2
for scope,m in (('Jan+Jul',sc),('12m',np.ones(N,bool))):
    es=np.sort(e[m])[::-1]; tot=es.sum()
    print(scope,'SSE share of top-k rows:',{k:round(100*es[:k].sum()/tot,1) for k in (1,3,7,15,30,100,1000)})
    print(scope,'RMSE if top-k rows were perfect:',{k:round(np.sqrt((tot-es[:k].sum())/m.sum()),1) for k in (1,3,7,15,30,100,1000)})
cell=pd.Series(apt+'|'+month.astype(str))
# (1) capped level: mean of y and f over rows with y<=7200 (indicator-like), shift all non-P rows of the cell
for cap in (7200,):
    ok=y<=cap
    df=pd.DataFrame({'c':cell,'y':y,'f':f,'ok':ok})
    mu=df[df.ok].groupby('c').apply(lambda d:(d.y-d.f).mean())
    shift=cell.map(mu).fillna(0).to_numpy()
    for nm,mask in (('all rows',np.ones(N,bool)),('non-P rows',~P)):
        g=f+np.where(mask,shift,0)
        print(f'cap {cap} level shift ({nm}): Jan+Jul {R(g,sc)-base["Jan+Jul"]:+.3f} Jan {R(g,month==1)-base["Jan"]:+.3f} Jul {R(g,month==7)-base["Jul"]:+.3f} 12m {R(g,np.ones(N,bool))-base["12m"]:+.3f}')
    s=mu.loc[[c for c in mu.index if c.endswith('|1') or c.endswith('|7')]]
    print('  Jan/Jul cell level biases (s):', s.round(1).to_dict())
# (2) uncapped: cell sum known -> scale zone-P p per cell (p'=clip(a p)) so the cell sum matches; non-P rows get capped level
df=pd.DataFrame({'c':cell})
g=f+np.where(~P,shift,0).astype(float)
gP=g.copy()
cP=cell.to_numpy()[Pidx]
for c in np.unique(cell):
    m=(cell==c).to_numpy(); idx=np.where(cP==c)[0]
    if len(idx)==0: continue
    target=y[m].sum(); rest=g[m].sum()-g[Pidx[idx]].sum()
    def tot(a):
        pp=np.clip(a*p[idx],0,1); return rest+np.clip(pp*so[Pidx[idx]]+(1-pp)*pb[idx]+pr[idx],0,None).sum()
    lo,hi=0.0,1e4
    if tot(lo)>=target: a=lo
    elif tot(hi)<=target: a=hi
    else:
        for _ in range(80):
            mid=(lo+hi)/2
            if tot(mid)<target: lo=mid
            else: hi=mid
        a=(lo+hi)/2
    pp=np.clip(a*p[idx],0,1); gP[Pidx[idx]]=np.clip(pp*so[Pidx[idx]]+(1-pp)*pb[idx]+pr[idx],0,None)
print(f'uncapped cell-sum oracle (P rescale + level): Jan+Jul {R(gP,sc)-base["Jan+Jul"]:+.3f} Jan {R(gP,month==1)-base["Jan"]:+.3f} Jul {R(gP,month==7)-base["Jul"]:+.3f} 12m {R(gP,np.ones(N,bool))-base["12m"]:+.3f}')
print('P rows Jan+Jul:', int(np.isin(month[Pidx],[1,7]).sum()), 'P share of Jan+Jul SSE %.1f%%'%(100*e[Pidx][np.isin(month[Pidx],[1,7])].sum()/e[sc].sum()))
