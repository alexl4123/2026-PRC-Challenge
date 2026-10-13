import numpy as np, pandas as pd
tr=pd.read_parquet('cache/major/dep_training.parquet',columns=['month','APT','MVT_TIME_UTC_mvt'])
print(tr.month.value_counts().sort_index().to_dict())
print(tr.MVT_TIME_UTC_mvt.dt.month.value_counts().sort_index().to_dict())
for p in ['cache/cv/adsb2/mix__seed00__m01.npy','cache/cv/adsb2/mix__seed00__m01.bulk.npy','cache/cv/w2/mix__seed00__m01.npy','cache/cv/seqsw_ep40/seq__seed00__m01.npy','cache/cv/adsb2/nn__seed00__m07.npy']:
    a=np.load(p,allow_pickle=True); print(p,a.dtype,a.shape,a[:3])
import os
print(sorted(set(f.split('__')[0] for f in os.listdir('cache/cv/adsb2'))))
print(sorted(os.listdir('cache/cv/w2'))[-10:])
print(sorted(os.listdir('cache/cv/c1'))[:10])
z=np.load('cache/w2/v28s_oof.npz'); print(np.unique(z['month']), z['f'][:5], z['y'][:5])
