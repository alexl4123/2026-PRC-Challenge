from lib import *
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache'
V=np.load(C+'/w2/v24s_oof.npz',allow_pickle=True); z=np.load(C+'/w2/v28s_oof.npz')
corr=V['corr']; Pidx,p,pbP,so=V['Pidx'],V['p14'],V['pbP'],V['so'][V['Pidx']]
blendD=V['blendD']; P,D=Z['P'],Z['D']; DnP=D&~P
br=np.clip(p*so+(1-p)*pbP,0,None)
def run(name, cand, rows, build):
    Gs={k: finish_pre(build(k)) for k in cand}
    sm={k: np.bincount(month[rows],(y[rows]-Gs[k][rows])**2,minlength=13) for k in cand}
    G=B.astype(float).copy(); ch={}
    for m in range(1,13):
        T=[t for t in range(1,13) if t!=m]; k=min(cand,key=lambda c: sm[c][T].sum()); ch[m]=k
        s=rows&(month==m); G[s]=Gs[k][s]
    evaluate(name, finish(G), f'chosen {ch}')
def finish_pre(G): return finish(G)
def bP(k):
    G=B.astype(float).copy(); G[Pidx]=k*corr[Pidx]+br; return G
run('K10 zone-P corrector multiplier',[0,0.25,0.5,0.75,1.0,1.25],P,bP)
def bD(k):
    G=B.astype(float).copy(); G[DnP]=np.clip(blendD[DnP]+k*corr[DnP],0,None); return G
print('check D rebuild at 0.5:', np.abs(finish(bD(0.5))-B)[DnP].max(), ' P rebuild at 1:', np.abs(finish(bP(1.0))-B)[P].max())
run('K11 route-D corrector multiplier',[0,0.25,0.5,0.75,1.0],DnP,bD)
ol=Z['off_lobt']; ok=ABC&np.isfinite(ol)&(ol>3606)&~Z['proj']
cell=np.where(ok, np.digitize(ol,[4200,5000,6000,8000]), -1)
print('K3b cell n', np.bincount(cell[cell>=0]))
G,ch=cellshift(cell,5); evaluate('K3b off_lobt>3606 bucket shift',G,f'lambda {ch}')
