import json, numpy as np
from pathlib import Path
C=Path('/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache')
S='/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/lever_search'
FAMS=['mix','nn','perapt','lintree','routed','xgboost','w3']
POOL=['ep30','ep40','ep50','ep60','ep80','ep40_d128','ep50_d128','ep40_l3','ep40_lr3']
z=np.load(C/'w2'/'v28s_oof.npz'); month=z['month']
def oof(folder,fam,seeds,suffix='.npy'):
    o=np.zeros(len(month))
    for s in seeds:
        for m in range(1,13):
            o[month==m]+=np.load(folder/f'{fam}__seed{s:02d}__m{m:02d}{suffix}')/len(seeds)
    return o
M={f:oof(C/'cv'/'adsb2',f,[0,1,2]) for f in FAMS}
M['seq']=np.mean([oof(C/'cv'/f'seqsw_{v}','seq',[0,1,2]) for v in POOL],axis=0)
np.savez(S+'/members.npz', X=np.column_stack([M[f] for f in FAMS+['seq']]).astype(np.float64), names=np.array(FAMS+['seq']))
