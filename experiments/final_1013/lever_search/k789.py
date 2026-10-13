from lib import *
import json
from scipy.optimize import nnls
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache'
X=np.load(S+'/members.npz')['X']; names=list(np.load(S+'/members.npz')['names'])
z=np.load(C+'/w2/v28s_oof.npz'); f27,b,dl=z['f27'],z['bin'],z['dl']
V=np.load(C+'/w2/v24s_oof.npz',allow_pickle=True); corr=V['corr']
P,D=Z['P'],Z['D']; A=~D&~P; DnP=D&~P
J=json.load(open(C+'/seq/arm_u.json'))
# reproduce checks
chk=f27.copy()
for c in (0,1,2):
    w=np.array([J['UT']['weights'][str(c)].get(n,0) for n in names]); sel=A&(b==c)
    chk[sel]=np.clip(X[sel]@w+J['UT']['lambdas'][str(c)]*dl[sel],0,None)
print('A/B/C reproduce max |diff|', np.abs(chk-f27)[A].max())
name='__'.join(f'seqsw_{v}' for v in ['ep30','ep40','ep50','ep60','ep80','ep40_d128','ep50_d128','ep40_l3','ep40_lr3'])
WD=json.load(open(C+f'/seq/swap_{name}.json'))['weights_new']['route D']; wd=np.array([WD.get(n,0) for n in names])
print('route D reproduce max |diff|', np.abs(np.clip(X[DnP]@wd+0.5*corr[DnP],0,None)-f27[DnP]).max())
# K8
G=B.astype(float).copy(); WW={}
for m in range(1,13):
    for c in (0,1,2):
        k=(month!=m)&A&(b==c); w,_=nnls(X[k],y[k]); base=X@w if False else None
        bk=X[k]@w; lam=float(np.dot(dl[k],y[k]-bk)/np.dot(dl[k],dl[k]))
        sel=A&(b==c)&(month==m); G[sel]=np.clip(X[sel]@w+lam*dl[sel],0,None); WW[(m,c)]=(np.round(w,3),round(lam,3))
G8=finish(G); 
held=~np.isin(month,[3,6,9,12])
print('K8 held-out(8m) diff %.4f' % (r(G8,held)-r(B,held)))
evaluate('K8 A/B/C U+T refit on 11 months',G8,f'weights m1: {WW[(1,0)]} {WW[(1,1)]} {WW[(1,2)]}')
# K7/K9
x9=np.clip(Z['off_sched'],0,7200)
for nm,XX in (('K9 route-D NNLS refit 11 months',X),('K7 route-D refit + schedule-anchor column',np.column_stack([X,x9]))):
    G=B.astype(float).copy(); ws={}
    for m in range(1,13):
        k=(month!=m)&DnP; w,_=nnls(XX[k],y[k]); sel=DnP&(month==m)
        G[sel]=np.clip(XX[sel]@w+0.5*corr[sel],0,None); ws[m]=np.round(w,3)
    Gf=finish(G); print(nm,'held-out(8m) diff %.4f' % (r(Gf,held)-r(B,held)))
    evaluate(nm,Gf,f'weights m1 {ws[1]} m7 {ws[7]}')
