import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
E=pd.Timestamp('1970-01-01',tz='UTC')
f=load('training',columns=['APT','STAND_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','FLIGHT_mvt'])
f['tb']=(f.BLOCK_TIME_UTC_mvt-E).dt.total_seconds(); f['k']=f.APT.astype(str)+'|'+f.STAND_mvt.astype(str)
dep=f[f.PHASE_mvt.eq('DEP')].dropna(subset=['tb']).sort_values('tb'); arr=f[f.PHASE_mvt.eq('ARR')].dropna(subset=['tb']).sort_values('tb')
m=pd.merge_asof(dep[['k','tb','FLIGHT_mvt']],arr[['k','tb','FLIGHT_mvt']].rename(columns={'tb':'ta','FLIGHT_mvt':'fa'}),left_on='tb',right_on='ta',by='k',direction='nearest')
g=(m.ta-m.tb).abs()
print('departures whose BLOCK equals an arrival in-block at the same stand within 1s: %.4f ; within 60s: %.4f'%((g<=1).mean(),(g<=60).mean()))
print('  of those within 1s, same callsign: %.3f'%((m.fa==m.FLIGHT_mvt)[g<=1].mean()))
# null: shift arrivals by +1 day, same test
m2=pd.merge_asof(dep[['k','tb']],arr[['k','tb']].assign(tb=arr.tb+86400).rename(columns={'tb':'ta'}).sort_values('ta'),left_on='tb',right_on='ta',by='k',direction='nearest')
g2=(m2.ta-m2.tb).abs(); print('null (arrivals shifted 1 day): within 1s %.4f ; within 60s %.4f'%((g2<=1).mean(),(g2<=60).mean()))
# signed: next arrival in-block after our block, gap distribution
m3=pd.merge_asof(dep[['k','tb']],arr[['k','tb']].rename(columns={'tb':'ta'}),left_on='tb',right_on='ta',by='k',direction='forward')
gg=(m3.ta-m3.tb); print('next arrival at our stand after our (hidden) block: quantiles (s)', np.nanpercentile(gg,[5,10,25,50,75]).round(0))
