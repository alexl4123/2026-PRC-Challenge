import sys, numpy as np, pandas as pd
R='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
sys.path.insert(0, R)
import assemble as A
from prc.config import CACHE
S=sys.argv[1]
z=np.load(CACHE/'w2'/'v28s_oof.npz')
y, month, P, D, f27, ol = z['y'], z['month'], z['P'].astype(bool), z['D'].astype(bool), z['f27'], z['off_lobt']
dt=pd.read_parquet(CACHE/'major'/'dep_training.parquet', columns=['APT','MVT_TIME_UTC_mvt','off_aobt3','off_sched','off_eobt1','off_iobt','y'])
assert np.allclose(dt.y.to_numpy(float), y)
rw=pd.read_parquet(CACHE/'drivers_0926'/'rows_training.parquet', columns=['apt','hour_loc','date_loc','route4','adsb_src'])
apt=dt.APT.astype(str).to_numpy(); assert (rw.apt.astype(str).to_numpy()==apt).all()
r0=np.rint(np.clip(f27,0,A.HI))
m=np.isfinite(ol); assert not (m & (D|P)).any()
lo=np.full(len(y),np.nan); hi=lo.copy()
lo[m]=np.ceil(np.maximum(0, ol[m]-3606)); hi[m]=np.floor(ol[m]+3606)
w=r0.copy(); w[m]=np.minimum(np.maximum(r0[m],lo[m]),hi[m])
L=A.clean_links('training')
i,jlo,jhi,osd,dloc=L['i'],L['lo'],L['hi'],L['off_sched'],L['dloc']
q=w.copy(); one=jlo< -12
q[i[one]]=np.maximum(q[i[one]],np.ceil(jhi[one]))
b=~one&(osd>=jlo)&(osd<=jhi)&(dloc<=0)
q[i[b]]=np.minimum(q[i[b]],np.floor(np.maximum(jlo[b],0)))
cap=np.isin(apt,A.CAP_APTS); q[cap]=np.minimum(q[cap],3606)
# jc bounds per row for re-projection
jl=np.full(len(y),-np.inf); jh=np.full(len(y),np.inf)
jl[i[one]]=np.ceil(jhi[one]); jh[i[b]]=np.floor(np.maximum(jlo[b],0)); jh[cap]=np.minimum(jh[cap],3606)
proj=q!=r0
r=lambda g,k: np.sqrt(np.mean((y[k]-g[k])**2))
print('rows', len(y), 'projected', proj.sum(), 'win moved', (w!=r0).sum(), 'jc moved', (q!=w).sum())
print('RMSE all: f27 %.3f  rint %.3f  v28 %.3f  v29 %.3f' % (r(f27,np.ones(len(y),bool)), r(r0,np.ones(len(y),bool)), r(w,np.ones(len(y),bool)), r(q,np.ones(len(y),bool))))
print('v28s f vs our window: max diff on non-rint', np.abs(z['f']-np.where(m, np.minimum(np.maximum(f27, np.maximum(0,ol-3606)), ol+3606), f27)).max())
date=dt.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d').to_numpy()
np.savez(S+'/base.npz', y=y, month=month, P=P, D=D, B=q, r0=r0, proj=proj, lo=lo, hi=hi, jl=jl, jh=jh, apt=apt,
         hour=rw.hour_loc.to_numpy(), dateblk=pd.factorize(date)[0], aptday=pd.factorize(rw.apt.astype(str)+'_'+rw.date_loc.astype(str))[0],
         off_lobt=ol, off_aobt3=dt.off_aobt3.to_numpy(), off_sched=dt.off_sched.to_numpy(), off_eobt1=dt.off_eobt1.to_numpy(),
         off_iobt=dt.off_iobt.to_numpy(), route4=rw.route4.astype(str).to_numpy(), adsb=rw.adsb_src.astype(str).to_numpy(), sd=z['sd'])
