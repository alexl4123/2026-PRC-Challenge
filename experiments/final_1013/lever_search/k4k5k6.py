from lib import *
import pandas as pd
from scipy.special import expit, logit
# K4: clip relative to off_aobt3 on ABC (non-projected); lower and upper sides separable (disjoint rows for c>0)
oa=Z['off_aobt3']; ok=ABC&np.isfinite(oa)&~Z['proj']
C=[600,900,1200,1800,2400,3600,np.inf]
def sse_month(G, sel):
    return np.bincount(month[sel], (y[sel]-G[sel])**2, minlength=13)
base_sm=sse_month(B.astype(float), ok)
lo_sm={c: sse_month(np.where(ok, np.maximum(B, oa-c), B), ok)-base_sm for c in C}
hi_sm={c: sse_month(np.where(ok, np.minimum(B, oa+c), B), ok)-base_sm for c in C}
G=B.astype(float).copy(); chosen={}
for m in range(1,13):
    T=[t for t in range(1,13) if t!=m]
    # inner LOMO: params chosen on T\{k}, scored on k ; but the clip has no fitted parameter besides c, so inner LOMO = pick c by SSE on each inner fold's training part, score on k
    def pick(D):
        tot=0.0
        for k in T:
            tr=[t for t in T if t!=k]; c=min(C,key=lambda c: D[c][tr].sum()); tot+=D[c][k]
        return tot
    # selection of c for outer month m: choose c minimising training-month SSE (T); inner-LOMO estimate reported
    clo=min(C,key=lambda c: lo_sm[c][T].sum()); chi=min(C,key=lambda c: hi_sm[c][T].sum())
    chosen[m]=(clo,chi, round(pick(lo_sm),0), round(pick(hi_sm),0))
    s=ok&(month==m); G[s]=np.minimum(np.maximum(B[s],oa[s]-clo),oa[s]+chi)
evaluate('K4 off_aobt3 soft clip',finish(G),f'(c_lo,c_hi,innerLOMO dSSE lo,hi) {chosen}')
# K5 zone-P p recalibration
V=np.load('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/w2/v24s_oof.npz',allow_pickle=True)
Pidx,rome,p,pbP,Prest,so=V['Pidx'],V['rome'],V['p14'].astype(float),V['pbP'],V['P_rest'],V['so'][V['Pidx']]
chk=Prest+np.clip(p*so+(1-p)*pbP,0,None); z28=np.load('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/w2/v28s_oof.npz')
print('K5 check |f27[P]-rebuild| max', np.abs(z28['f27'][Pidx]-chk).max())
pc=np.clip(p,1e-6,1-1e-6); lp=logit(pc); yP=y[Pidx]; mP=month[Pidx]
AG=np.round(np.arange(0.5,2.01,0.1),2); BG=np.arange(-2,2.01,0.25)
SM={}
for a in AG:
    for b in BG:
        q=expit(a*lp+b); pr=np.rint(np.clip(Prest+np.clip(q*so+(1-q)*pbP,0,None),0,None))
        SM[(a,b)]=(np.bincount(mP[rome],(yP[rome]-pr[rome])**2,minlength=13), np.bincount(mP[~rome],(yP[~rome]-pr[~rome])**2,minlength=13), pr)
G=B.astype(float).copy(); chosen={}
for m in range(1,13):
    T=[t for t in range(1,13) if t!=m]
    ar=min(SM,key=lambda k: SM[k][0][T].sum()); an=min(SM,key=lambda k: SM[k][1][T].sum())
    chosen[m]=(ar,an)
    s=(mP==m); idx=Pidx[s]
    G[idx]=np.where(rome[s], SM[ar][2][s], SM[an][2][s])
evaluate('K5 zone-P logit recalibration',finish(G),f'(rome (a,b), nonrome (a,b)) {chosen}')
# K6 per-airport floor
apt=Z['apt']; AL=[0,0.0005,0.001,0.005,0.01]; ok6=~Z['proj']
Q={}
aps=np.unique(apt)
idx_a={a: np.where(apt==a)[0] for a in aps}
def floors(months, al):
    return {a: (np.quantile(y[idx_a[a]][np.isin(month[idx_a[a]],months)], al) if al>0 else -np.inf) for a in aps}
G=B.astype(float).copy(); chosen={}
for m in range(1,13):
    T=[t for t in range(1,13) if t!=m]; best=None
    for al in AL:
        sse=0.0
        for k in T:
            tr=[t for t in T if t!=k]; F=floors(tr,al); s=(month==k)&ok6
            fl=np.array([F[a] for a in apt[s]]) if al>0 else np.full(s.sum(),-np.inf)
            sse+=((y[s]-np.maximum(B[s],fl))**2).sum()
        if best is None or sse<best[0]: best=(sse,al)
    al=best[1]; chosen[m]=al
    if al>0:
        F=floors(T,al); s=(month==m)&ok6; G[s]=np.maximum(B[s],np.array([F[a] for a in apt[s]]))
evaluate('K6 per-airport floor',finish(G),f'alpha {chosen}')
