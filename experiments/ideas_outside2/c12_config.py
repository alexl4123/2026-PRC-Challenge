"""Runway-configuration regime: active departure/arrival runway sets per airport-hour (observable in ranking).
Ceiling via LOMO shrunk mean residual by (APT, config) and (APT, config, PIER, RUNWAY); plus booster arm."""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load, peak_rss_gb
C='/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
f=load('training',columns=['APT','PHASE_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','STAND_mvt'])
f['h']=f.MVT_TIME_UTC_mvt.dt.floor('h')
# runways with >=2 movements in the hour, per phase
g=f.groupby(['APT','h','PHASE_mvt'],observed=True).RUNWAY_mvt.agg(lambda s: '+'.join(sorted(s.astype(str).value_counts()[lambda v: v>=2].index)))
cfg=g.unstack('PHASE_mvt').fillna('')
cfg['config']=cfg.get('ARR','')+'/'+cfg.get('DEP','')
dep=f[f.PHASE_mvt.eq('DEP')].reset_index(drop=True)
k=pd.MultiIndex.from_arrays([dep.APT,dep.h])
dep['config']=cfg['config'].reindex(k).values
y=np.load(C+'altfam_y.npy'); mon=np.load(C+'altfam_month.npy'); v6=np.load(C+'exp62_oof.npy')[1]; res=y-v6
F=pd.read_parquet(OUT+'c2_features.parquet',columns=['off_sched','aobt_missing'])
P=(F.aobt_missing.values==1)&(F.off_sched.values>7200); clean=(y<=7200)&~P
pier=dep.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+|\d)',expand=False).fillna('NA')
print('distinct configs per airport:',dep.groupby('APT',observed=True).config.nunique().to_dict())
N=len(y); base=np.sqrt(np.mean(res[clean]**2))
def lomo(gk,k=100):
    g=pd.factorize(gk)[0]
    df=pd.DataFrame({'g':g[clean],'m':mon[clean],'r':res[clean]})
    tot=df.groupby('g').r.agg(['sum','count']); per=df.groupby(['g','m']).r.agg(['sum','count'])
    S=tot['sum'].reindex(g).fillna(0).to_numpy().copy(); n=tot['count'].reindex(g).fillna(0).to_numpy().astype(float)
    idx=pd.MultiIndex.from_arrays([g,mon]); S-=per['sum'].reindex(idx).fillna(0).values; n-=per['count'].reindex(idx).fillna(0).values
    c=S/(n+k); return np.sqrt(np.mean((res[clean]-c[clean])**2))-base
for name,gk in [('apt_config',dep.APT.astype(str)+'|'+dep.config),('apt_config_rwy',dep.APT.astype(str)+'|'+dep.config+'|'+dep.RUNWAY_mvt.astype(str)),
                ('apt_config_pier_rwy',dep.APT.astype(str)+'|'+dep.config+'|'+pier+'|'+dep.RUNWAY_mvt.astype(str)),
                ('apt_config_stand',dep.APT.astype(str)+'|'+dep.config+'|'+dep.STAND_mvt.astype(str))]:
    print(f'  LOMO {name:22s}  k=30 {lomo(gk,30):+.3f}  k=100 {lomo(gk,100):+.3f}  k=400 {lomo(gk,400):+.3f}')
