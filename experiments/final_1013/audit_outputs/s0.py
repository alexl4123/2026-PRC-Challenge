import pyarrow.parquet as pq, numpy as np
for f in ['cache_final/major/dep_ranking.parquet','cache/major/dep_training.parquet','outputs_final/understanding-accordion_final.parquet','outputs/understanding-accordion_v29.parquet']:
    m=pq.ParquetFile(f); print(f, m.metadata.num_rows); print(m.schema_arrow.names)
z=np.load('cache/w2/v28s_oof.npz'); print({k:z[k].shape for k in z})
for d in ['adsb2_final','w2_final','nn_final','seq_ep40_final']:
    r=np.load(f'cache_final/{d}/ranking_ids.npy',allow_pickle=True); print(d,r.shape,r[:3])
print(np.load('cache_final/adsb2_final/mix__seed00.npy').shape)
print(open('cache_final/seq_ep40_final/_config.json').read()[:500])
