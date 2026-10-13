from lib import *
import pandas as pd
apt=Z['apt']; grp=np.where(ABC,0,np.where(Z['D']&~Z['P'],1,-1))
ai=pd.factorize(apt)[0]
cell=np.where(grp>=0, ai*2+grp, -1); G,ch=cellshift(cell,20); evaluate('K1 airport x route shift',G,f'lambda {ch}')
hb=np.clip(Z['hour'],0,23)//4
cell=np.where(grp>=0,(ai*2+grp)*6+hb,-1); G,ch=cellshift(cell,120); evaluate('K2 airport x route x hour-band shift',G,f'lambda {ch}')
# K3
lo,hi=Z['lo'],Z['hi']; m=np.isfinite(lo)&ABC&~Z['proj']
edges=[0,60,180,400,800,1600]
dlo=np.where(m&(lo>0),B-lo,np.inf); dhi=np.where(m,hi-B,np.inf)
cl=np.digitize(dlo,edges)-1; ch_=np.digitize(dhi,edges)-1
cell=np.full(len(y),-1)
use_lo=(cl>=0)&(cl<5); use_hi=(ch_>=0)&(ch_<5)
both=use_lo&use_hi
cell[use_hi]=5+ch_[use_hi]; cell[use_lo]=cl[use_lo]   # lower side takes precedence if both (window width 7212 s: rare)
print('K3 cells n:', np.bincount(cell[cell>=0],minlength=10), 'both', both.sum())
G,chh=cellshift(cell,10); evaluate('K3 LOBT-window margin shift',G,f'lambda {chh}')
