"""LIRF: does the month's ARRIVAL stamping rate predict the month-level calibration error of the tail specialist inside P?
LOMO over 12 months: regress needed logit shift on arr_stamp (other 11 months), apply to held-out month (moderate zone only
and all P), score total RMSE change. Uses arm b (v7/v8 specialist) and v6."""
import sys, numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
cols=['APT','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','TAXITIME_SEC_mvt']
f=load('training',columns=cols)
arr=f[f.PHASE_mvt.eq('ARR')]
arr_stamp=arr.assign(mon=arr.MVT_TIME_UTC_mvt.dt.month,s=(arr.BLOCK_TIME_UTC_mvt-arr.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=5).groupby(['APT','mon']).s.mean()
dep=f[f.PHASE_mvt.eq('DEP')].reset_index(drop=True); del f
y=dep.TAXITIME_SEC_mvt.astype(float).values; mon=dep.MVT_TIME_UTC_mvt.dt.month.values
os_=(dep.MVT_TIME_UTC_mvt-dep.SCHED_TIME_UTC_mvt).dt.total_seconds().values
P=dep.AOBT_3_flt.isna().values&(os_>7200)
Pidx=np.where(P)[0]
pb=np.load(C+'exp62_pb.npy').astype(float); ptg=np.load(C+'exp62_ptg.npy').astype(float)
spec=np.load(C+'exp62_spec_P.npy'); oof=np.load(C+'exp62_oof.npy')
# et per exp62: off_sched if >7200 else tail mean of other months (only P rows matter here -> off_sched)
lirf=dep.APT.astype(str).values=='LIRF'
logit=lambda p: np.log(p/(1-p)); sig=lambda z: 1/(1+np.exp(-z))
for arm,row in [('v6',0),('b (v7/v8)',1)]:
    base=oof[1+row].copy()
    p=np.clip(spec[row],1e-6,1-1e-6)
    pm=np.zeros(len(y)); pm[Pidx]=p
    mods={'all P':np.ones(len(y),bool),'moderate (<=20000)':os_<=20000}
    for zname,zone in mods.items():
        sel_all=P&lirf&zone
        # needed shift per month
        d=[]; xs=[]
        for m in range(1,13):
            s=sel_all&(mon==m); n_t=(y[s]>7200).sum()
            if s.sum()<5 or n_t==0 or n_t==s.sum(): d.append(np.nan); xs.append(arr_stamp['LIRF'][m]); continue
            fn=lambda dd: sig(logit(pm[s])+dd).sum()-n_t
            d.append(brentq(fn,-10,10)); xs.append(arr_stamp['LIRF'][m])
        d=np.array(d); xs=np.array(xs)
        ok=np.isfinite(d)
        print(f'{arm} | {zname}: corr(needed shift, arr_stamp) = {np.corrcoef(d[ok],xs[ok])[0,1]:+.2f}; shifts',np.round(d,2))
        new=base.copy()
        for m in range(1,13):
            tr=ok&(np.arange(1,13)!=m)
            A=np.c_[np.ones(tr.sum()),xs[tr]]; c=np.linalg.lstsq(A,d[tr],rcond=None)[0]
            dm=c[0]+c[1]*xs[m-1]
            s=sel_all&(mon==m)
            pn=sig(logit(pm[s])+dm)
            new[s]=np.clip(pn*os_[s]+(1-pn)*pb[s],0,None)
        r0=np.sqrt(np.mean((y-base)**2)); r1=np.sqrt(np.mean((y-new)**2))
        s17=np.isin(mon,[1,7])
        print(f'    LOMO arrival-rate recalibration: 12-month {r1-r0:+.3f} s | months 1+7 {np.sqrt(np.mean((y[s17]-new[s17])**2))-np.sqrt(np.mean((y[s17]-base[s17])**2)):+.3f} s')
        # sanity: reconstruct base on P rows
    chk=np.clip(pm[Pidx]*os_[Pidx]+(1-pm[Pidx])*pb[Pidx],0,None)
    print('   reconstruction max|diff| on P rows:',np.abs(chk-base[Pidx]).max())

# ---- permutation null (arm b, moderate zone) + 2026 implication ----
row=1; base=oof[2].copy(); p=np.clip(spec[row],1e-6,1-1e-6); pm=np.zeros(len(y)); pm[Pidx]=p
zone=os_<=20000; sel_all=P&lirf&zone
d=[];xs0=np.array([arr_stamp['LIRF'][m] for m in range(1,13)])
for m in range(1,13):
    s=sel_all&(mon==m); n_t=(y[s]>7200).sum()
    if s.sum()<5 or n_t==0 or n_t==s.sum(): d.append(np.nan); continue
    d.append(brentq(lambda dd: sig(logit(pm[s])+dd).sum()-n_t,-10,10))
d=np.array(d); ok=np.isfinite(d); s17=np.isin(mon,[1,7])
def lomo_gain(xs):
    new=base.copy()
    for m in range(1,13):
        tr=ok&(np.arange(1,13)!=m); A=np.c_[np.ones(tr.sum()),xs[tr]]; c=np.linalg.lstsq(A,d[tr],rcond=None)[0]
        s=sel_all&(mon==m); pn=sig(logit(pm[s])+c[0]+c[1]*xs[m-1]); new[s]=np.clip(pn*os_[s]+(1-pn)*pb[s],0,None)
    return np.sqrt(np.mean((y-new)**2))-np.sqrt(np.mean((y-base)**2)), np.sqrt(np.mean((y[s17]-new[s17])**2))-np.sqrt(np.mean((y[s17]-base[s17])**2))
g0=lomo_gain(xs0); rng=np.random.default_rng(0)
null=np.array([lomo_gain(rng.permutation(xs0)) for _ in range(300)])
print(f'\nPERMUTATION (arm b, moderate): observed 12m {g0[0]:+.3f}, m1+7 {g0[1]:+.3f}; P(null<=obs) 12m {np.mean(null[:,0]<=g0[0]):.3f}, m1+7 {np.mean(null[:,1]<=g0[1]):.3f}; null median 12m {np.median(null[:,0]):+.3f}')
# 2026
rk=load('ranking',columns=['APT','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','MVT_ID_mvt','AOBT_3_flt'])
ra=rk[rk.PHASE_mvt.eq('ARR')]
st26=ra.assign(mon=ra.MVT_TIME_UTC_mvt.dt.month,s=(ra.BLOCK_TIME_UTC_mvt-ra.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=5).groupby(['APT','mon']).s.mean()
A=np.c_[np.ones(ok.sum()),xs0[ok]]; c=np.linalg.lstsq(A,d[ok],rcond=None)[0]
print('fit on all 2025 months: shift = %.2f + %.2f * arr_stamp'%(c[0],c[1]))
z=np.load('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/final/spec_x_ranking.npz',allow_pickle=True)
rd=rk[rk.PHASE_mvt.eq('DEP')].reset_index(drop=True)
assert (rd.MVT_ID_mvt.values==z['ids']).all()
os26=(rd.MVT_TIME_UTC_mvt-rd.SCHED_TIME_UTC_mvt).dt.total_seconds().values
for m in (1,7):
    x=st26['LIRF'][m]; dm=c[0]+c[1]*x
    s=z['in_P']&(rd.APT.astype(str).values=='LIRF')&(rd.MVT_TIME_UTC_mvt.dt.month.values==m)&(os26<=20000)
    p26=np.clip(z['p_spec'][s],1e-6,1-1e-6)
    print(f'2026 LIRF m{m}: arr_stamp {x:.4f} (2025 {xs0[m-1]:.4f}) -> shift {dm:+.2f}; moderate-zone P rows {s.sum()}, sum p {p26.sum():.1f} -> {sig(logit(p26)+dm).sum():.1f}')
