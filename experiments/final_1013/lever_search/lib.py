import sys, numpy as np
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.validation import block_bootstrap
S='/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/lever_search'
Z=dict(np.load(S+'/base.npz', allow_pickle=True))
y,B,month=Z['y'],Z['B'],Z['month']
ABC=~Z['D']&~Z['P']
LAM=[0,1e2,1e3,1e4,1e5,np.inf]
def finish(G):
    G=np.rint(np.clip(G,0,200000.0))
    m=np.isfinite(Z['lo']); G[m]=np.minimum(np.maximum(G[m],Z['lo'][m]),Z['hi'][m])
    G=np.minimum(np.maximum(G,Z['jl']),Z['jh'])
    G[Z['proj']]=B[Z['proj']]
    return G
r=lambda g,k: float(np.sqrt(np.mean((y[k]-g[k])**2)))
def evaluate(name,G,extra=''):
    allr=np.ones(len(y),bool); JJ=np.isin(month,[1,7])
    bb=block_bootstrap(y,G,B,Z['dateblk'],1000)
    ba=block_bootstrap(y,G,B,Z['aptday'],1000)
    jj=block_bootstrap(y[JJ],G[JJ],B[JJ],Z['dateblk'][JJ],1000)
    md=[r(G,month==m)-r(B,month==m) for m in range(1,13)]
    gain=(y-B)**2-(y-G)**2; keep=np.ones(len(y),bool); keep[np.argsort(gain)[::-1][:100]]=False
    t100=r(G,keep)-r(B,keep)
    R1=bb['hi']<0; R2=jj['point']<0; R3=md[0]<0 and md[6]<0; R4=sum(d<0 for d in md)>=8
    v='PASS' if (R1 and R2 and R3 and R4) else 'FAIL'
    s=(f'== {name}: rows changed {int((G!=B).sum())}\n  12m {bb["point"]:+.4f} date-CI [{bb["lo"]:+.4f},{bb["hi"]:+.4f}] aptday-CI [{ba["lo"]:+.4f},{ba["hi"]:+.4f}]'
       f' | J+J {jj["point"]:+.4f} [{jj["lo"]:+.4f},{jj["hi"]:+.4f}] | Jan {md[0]:+.4f} Jul {md[6]:+.4f} | months better {sum(d<0 for d in md)}/12'
       f' | w/o top100 {t100:+.4f}\n  per month ' + ' '.join(f'{d:+.3f}' for d in md) +
       f'\n  R1 {R1} R2 {R2} R3 {R3} R4 {R4} => {v} {extra}')
    print(s, flush=True); open(S+'/results.txt','a').write(s+'\n')
    return v
def cellshift(cell, ncell, lamgrid=LAM):
    """cell: int per row (-1 untouched). nested LOMO additive shift of shrunk mean residual; lambda by inner LOMO (closed form, pre-rint)."""
    e=y-B; k=cell>=0
    Sm=np.zeros((ncell,13)); Nm=np.zeros((ncell,13))
    np.add.at(Sm,(cell[k],month[k]),e[k]); np.add.at(Nm,(cell[k],month[k]),1)
    G=B.astype(float).copy(); chosen={}
    for m in range(1,13):
        T=[t for t in range(1,13) if t!=m]
        best=None
        for lam in lamgrid:
            sse=0.0
            for kk in T:
                tr=[t for t in T if t!=kk]
                d=np.zeros(ncell) if np.isinf(lam) else Sm[:,tr].sum(1)/(Nm[:,tr].sum(1)+lam+1e-12)
                sse+=(-2*d*Sm[:,kk]+d*d*Nm[:,kk]).sum()
            if best is None or sse<best[0]: best=(sse,lam)
        lam=best[1]; chosen[m]=lam
        d=np.zeros(ncell) if np.isinf(lam) else Sm[:,T].sum(1)/(Nm[:,T].sum(1)+lam+1e-12)
        sel=k&(month==m); G[sel]=B[sel]+d[cell[sel]]
    return finish(G), chosen
