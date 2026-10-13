import numpy as np, pandas as pd
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
SEC=lambda s: (pd.to_datetime(s,utc=True)-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds()
cols=['PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','EOBT_1_flt','APT','TAXITIME_SEC_mvt']
t=pd.read_parquet(C+'training.parquet',columns=cols); t=t[t.PHASE_mvt=='DEP'].reset_index(drop=True)
z=np.load(C+'w2/v24s_oof.npz'); y,f,month=z['y'],z['f'],z['month']
mv=SEC(t.MVT_TIME_UTC_mvt).to_numpy(); lob=SEC(t.LOBT_flt).to_numpy(); ao=SEC(t.AOBT_3_flt).to_numpy()
lo=mv-lob-3606; hi=lo+7212; has=~np.isnan(lob)
g=f.copy(); g[has]=np.clip(f[has],np.maximum(lo[has],0),hi[has]); moved=has&(g!=f)
a3=np.where(np.isnan(ao),g,np.clip(mv-ao,np.maximum(lo,0),hi))
alt=np.where(moved,a3,g)
R=lambda p,mm: np.sqrt(((y-p)[mm]**2).mean())
sc=np.isin(month,[1,7]); al=np.ones(len(y),bool)
print('moved rows: boundary clip vs AOBT3-in-window:  12m %.3f vs %.3f | Jan+Jul %.3f vs %.3f'%(R(g,al),R(alt,al),R(g,sc),R(alt,sc)))
print('SSE on moved rows: v24 %.3g boundary %.3g aobt3 %.3g'%(((y-f)[moved]**2).sum(),((y-g)[moved]**2).sum(),((y-alt)[moved]**2).sum()))
big=has&((mv-lob)>43200)
print('2025 rows with MVT-LOBT > 12 h:',int(big.sum()))
d=pd.DataFrame({'apt':t.APT.astype(str),'month':month,'y':y,'f':f,'lo':lo,'hi':hi,'mvt_aobt3':mv-ao,'off_sched':mv-SEC(t.SCHED_TIME_UTC_mvt).to_numpy()})[big]
print(d.round(0).to_string())
print('--- delta variant (LOMO-estimated mean of y-lo on binding rows of the other 11 months)')
g=fc.copy(); dels={}
for m in range(1,13):
    tr=ch&(month!=m); dl=float(np.mean((y-lo)[tr])); dels[m]=round(dl)
    te=ch&(month==m); g[te]=np.minimum(lo[te]+dl, offl[te]+3606)
print('deltas', dels)
sse=(y-fc)**2-(y-g)**2
print('vs projection: held', r(g,held)-r(fc,held), 'scored', r(g,S)-r(fc,S), 'Jan', r(g,month==1)-r(fc,month==1), 'Jul', r(g,month==7)-r(fc,month==7), 'rows worse', (sse<-1e-6).sum(), 'of', ch.sum())
for nm,k in [('held',held),('scored',S)]:
    b=block_bootstrap(y[k],g[k],fc[k],block[k],n_boot=1000); print(nm, {kk:round(v,3) for kk,v in b.items()})
gain=(y-fc)**2-(y-g)**2
keep=np.ones(len(y),bool); keep[hi[np.argsort(gain[held])[::-1][:100]]]=False
print('R5 vs projection w/o top100 held', r(g,held&keep)-r(fc,held&keep))
print('months better', sum(r(g,month==m)<r(fc,month==m) for m in range(1,13)))
print('--- per month RMSE delta vs v24 (boundary / aobt3-in-window), rows moved')
for m in range(1,13):
    mm=month==m
    print(m, '%+.3f / %+.3f'%(R(g,mm)-R(f,mm),R(alt,mm)-R(f,mm)), int((moved&mm).sum()))
held=~np.isin(month,[3,6,9,12])
print('held: %+.3f / %+.3f'%(R(g,held)-R(f,held),R(alt,held)-R(f,held)))
de_alt=(y-g)**2-(y-alt)**2
print('aobt3 vs boundary on moved rows: better %d worse %d equal %d'%((de_alt>0).sum(),(de_alt<0).sum(),((de_alt==0)&moved).sum()))
print('moved rows by off_lobt sign: below-lo (move up) %d, above-hi (move down) %d'%(int((moved&(f<lo)).sum()),int((moved&(f>hi)).sum())))
