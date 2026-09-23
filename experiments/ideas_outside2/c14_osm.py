import sys, json, re, numpy as np, pandas as pd
sys.path.insert(0,'/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
from prc.io import load
d=load('training',phase='DEP',columns=['APT','STAND_mvt'])
def norm(s):
    s=str(s).upper().replace(' ','')
    return re.sub(r'(?<![0-9])0+(?=\d)','',s)
print('apt  osm_pos  osm_with_ref  our_stands  matched_exact  matched_norm  dep_rows_matched_norm')
for a in ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']:
    j=json.load(open(f'osm/{a}.json'))['elements']
    refs=[e.get('tags',{}).get('ref') for e in j]
    refs=[r for r in refs if r]
    R=set(refs); Rn=set(norm(r) for r in refs)
    g=d[d.APT.astype(str)==a].STAND_mvt.astype(str)
    st=pd.Series(g.unique())
    ex=st.isin(R).mean(); nm=st.map(norm).isin(Rn).mean(); rows=g.map(norm).isin(Rn).mean()
    print(f'{a} {len(j):7d} {len(refs):12d} {len(st):10d} {ex:13.2f} {nm:12.2f} {rows:12.3f}')
