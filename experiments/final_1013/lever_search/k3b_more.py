from lib import *
G=np.load(S+'/G_k3b.npy'); ol=Z['off_lobt']
held=~np.isin(month,[3,6,9,12]); gain=(y-B)**2-(y-G)**2
hi_=np.where(held)[0]; keep=np.ones(len(y),bool); keep[hi_[np.argsort(gain[held])[::-1][:100]]]=False
print('team-R5 (held-out w/o top-100 held-out rows): %+.4f ; held-out %+.4f' % (r(G,held&keep)-r(B,held&keep), r(G,held)-r(B,held)))
for k in (10,30,50): 
    kk=np.ones(len(y),bool); kk[np.argsort(gain)[::-1][:k]]=False; print(f'12m w/o top-{k} rows: {r(G,kk)-r(B,kk):+.4f}')
ok=ABC&np.isfinite(ol)&(ol>3606)&~Z['proj']; c=np.digitize(ol,[4200,5000,6000,8000]); e=y-B
for a,b_ in ((1,7),(7,1)):
    d=np.array([e[ok&(c==k)&(month==a)].mean() if (ok&(c==k)&(month==a)).any() else 0 for k in range(5)])
    g=B.astype(float).copy(); s=ok&(month==b_); g[s]+=d[c[s]]; g=finish(g)
    print(f'cross-fit {a}->{b_}: deltas {d.round(1)} diff {r(g,month==b_)-r(B,month==b_):+.4f}')
# drop-one-airport stability: 12m diff excluding each airport's rows
for a in np.unique(Z['apt']):
    k=Z['apt']!=a; print(a, f'{r(G,k)-r(B,k):+.4f}', end=' | ')
print()
