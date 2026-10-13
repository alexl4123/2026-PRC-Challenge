import numpy as np, pandas as pd
S='/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/lever_search'
Z=np.load(S+'/base.npz',allow_pickle=True)
y,B=Z['y'],Z['B']; e=y-B
DnP=Z['D']&~Z['P']
df=pd.DataFrame(dict(apt=Z['apt'],m=Z['month'],y=y,B=B,e=e,osch=Z['off_sched'],oe1=Z['off_eobt1'],oi=Z['off_iobt'],hr=Z['hour'],r4=Z['route4'],adsb=Z['adsb']))
d=df[DnP]; T=(e**2).sum()
big=d[np.abs(d.e)>3600].sort_values('e')
pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
print(big.round(0).to_string())
print('share of total SSE:', (big.e**2).sum()/T)
print(d.osch.describe())
# tail rows in D\P by off_sched bucket
d2=d.assign(ob=pd.cut(d.osch,[-1e9,-3600,0,600,1800,3600,7200,1e9]))
print(d2.groupby('ob').agg(n=('y','size'),tail=('y',lambda v:(v>7200).sum()),sse=('e',lambda v:(v**2).sum()/T),mean_e=('e','mean')))
