from lib import *
import pandas as pd
ol=Z['off_lobt']; ok=ABC&np.isfinite(ol)&(ol>3606)&~Z['proj']
cell=np.where(ok, np.digitize(ol,[4200,5000,6000,8000]), -1)
G,ch=cellshift(cell,5)
e=y-B
df=pd.DataFrame(dict(c=cell,m=month,e=e,apt=Z['apt'],gain=(y-B)**2-(y-G)**2,d=G-B))[ok]
pd.set_option('display.width',220)
print('per-bucket per-month mean residual (B):'); print(df.pivot_table(index='c',columns='m',values='e',aggfunc='mean').round(0))
print(df.groupby('c').agg(n=('e','size'),mean=('e','mean'),median=('e','median'),shift_mean=('d','mean'),gainsum=('gain','sum')).round(1))
print('all-12-month deltas (recipe):', [round(e[cell==c].mean(),1) for c in range(5)])
T=((y-B)**2).sum()
print('gain by airport (SSE share of total x1e4):'); print((df.groupby('apt').gain.sum()/T*1e4).round(2))
print('rows improved / worsened', (df.gain>0).sum(), (df.gain<0).sum())
# month-block bootstrap (12 blocks)
bb=block_bootstrap(y,G,B,month,2000); print('month-block CI', round(bb['lo'],4), round(bb['hi'],4))
# held-out 8 months only
held=~np.isin(month,[3,6,9,12]); print('held-out 8m diff', r(G,held)-r(B,held))
# route4 / bin
print('gain by route4:', df.assign(r4=Z['route4'][ok]).groupby('r4').gain.sum().round(0).to_dict())
yy=y[ok]; print('y quantiles in bucket rows', np.quantile(yy,[.5,.9,.99,.999]).round(0), ' tail(>7200) rows', (yy>7200).sum())
# how many rows have B at the lower bound edge (projected excluded) vs lo distribution
print('mean (B-lo) by bucket', [round(float(np.mean((B-Z['lo'])[cell==c])),0) for c in range(5)])
np.save(S+'/G_k3b.npy',G)
