import numpy as np, pandas as pd
O='/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/'
F=pd.read_parquet(O+'ideas_outside2/c4_stand_ranking.parquet')
d=pd.read_parquet(O+'ideas_inside/ranking_DEP.parquet',columns=['MVT_TIME_UTC_mvt','APT'])
t=pd.to_datetime(d.MVT_TIME_UTC_mvt,unit='s'); mon=t.dt.month.values; day=t.dt.day.values
print('ranking rows',len(F), 'aligned?', len(F)==len(d))
big=30*86400
for c in F.columns:
    v=F[c].values
    if not np.issubdtype(v.dtype,np.floating): continue
    huge=np.abs(v)>big
    print(f'{c:16s} |v|>30d: {huge.sum():5d}  (Jan31 {int((huge&(mon==1)&(day==31)).sum())}, Jul1 {int((huge&(mon==7)&(day==1)).sum())}, other {int((huge&~(((mon==1)&(day==31))|((mon==7)&(day==1)))).sum())})  NaN {np.isnan(v).mean():.4f}')
T=pd.read_parquet(O+'ideas_outside2/c4_stand.parquet')
for c in ['sa_after_to','sn_minus_to','sd_next_to','sl_to','si_before_sched','sd_prev_to']:
    v=T[c].values; print(f'training {c:16s} |v|>30d share {np.mean(np.abs(v)>big):.5f}  p99 {np.nanpercentile(v,99):.0f}')
