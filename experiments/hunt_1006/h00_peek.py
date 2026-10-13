import sys; sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from prc.config import CACHE
z=np.load(CACHE/'w2'/'v28s_oof.npz',allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype, z[k][:3])
print(pq.read_schema(CACHE/'major'/'dep_training.parquet'))
print(pq.read_schema(CACHE/'training.parquet'))
d=pd.read_parquet(CACHE/'major'/'dep_training.parquet',columns=['MVT_ID_mvt','APT','y','off_sched'])
print(d.head()); print(len(d))
