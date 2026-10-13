import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 60)
MON = ['25-01','25-02','25-06','25-07','26-01','26-02','26-06','26-07']
# PIER check at EDDM / EDDF
tr = pd.read_parquet(R+'cache/training.parquet', columns=['PHASE_mvt','APT','STAND_mvt'], filters=[('PHASE_mvt','==','DEP')])
rk = pd.read_parquet(R+'cache_final/ranking.parquet', columns=['PHASE_mvt','APT','STAND_mvt','MVT_TIME_UTC_mvt'], filters=[('PHASE_mvt','==','DEP')])
pier = lambda s: s.astype(str).str.extract(r'^([A-Za-z]+|\d)', expand=False).fillna('NA')
seen = set(zip(tr.APT.astype(str), pier(tr.STAND_mvt)))
rk['pier'] = pier(rk.STAND_mvt); rk['ym'] = rk.MVT_TIME_UTC_mvt.dt.strftime('%y-%m')
rk['pier_unseen'] = [(a, p) not in seen for a, p in zip(rk.APT.astype(str), rk.pier)]
print('PIER unseen (apt-wise) share by apt/month'); print(rk.groupby([rk.APT.astype(str), 'ym']).pier_unseen.mean().unstack().round(4))
print(rk[rk.pier_unseen].groupby([rk.APT.astype(str), 'pier']).size())
e = rk[(rk.APT.astype(str)=='EDDM')]
print('EDDM stands 10x in 2025 training?', tr[(tr.APT.astype(str)=='EDDM') & tr.STAND_mvt.astype(str).str.match(r'^10\d$')].STAND_mvt.astype(str).value_counts().head())
print('EDDM pier distribution 2025 vs 2026-06/07'); 
print(pier(tr[tr.APT.astype(str)=='EDDM'].STAND_mvt).value_counts(normalize=True).head(8).round(3).to_dict())
print(e[e.ym.isin(['26-06','26-07'])].pier.value_counts(normalize=True).head(8).round(3).to_dict())
print('EDDM new stands first day', e[e.STAND_mvt.astype(str).str.match(r'^10[1-8]$')].MVT_TIME_UTC_mvt.min())
f = rk[(rk.APT.astype(str)=='EDDF') & rk.STAND_mvt.astype(str).isin(['J5','J10','V333','V335','V337'])]
print('EDDF new stands first day', f.MVT_TIME_UTC_mvt.min(), 'n', len(f))
del tr
# exp62 candidates (zone P)
a = pd.read_parquet(R+'cache/exp62_cand_training.parquet'); b = pd.read_parquet(R+'cache_final/exp62_cand_ranking.parquet')
print('exp62 rows train', len(a), 'rank', len(b)); print('rank months', b.month.value_counts().sort_index().to_dict(), 'train months', a.month.value_counts().sort_index().to_dict())
cols = [c for c in b.columns if c not in ('month','__index_level_0__','apt','k_op','k_stand','k_pier','k_ades')]
a['g'] = '25-' + a.month.astype(int).astype(str).str.zfill(2); b['g'] = '26-' + b.month.astype(int).astype(str).str.zfill(2)
x = pd.concat([a[a.month.isin([1,2,6,7])], b])
print('exp62 missing share by month'); print(x[cols].isna().groupby(x.g).mean().T.round(3).to_string())
print('exp62 median by month'); print(x[cols].apply(pd.to_numeric, errors='coerce').groupby(x.g).median().T.round(1).to_string())
print('exp62 rank rows by apt/month'); print(b.groupby(['apt','g']).size().unstack().fillna(0).astype(int))
print('exp62 train rows by apt/month(1,2,6,7)'); print(a[a.month.isin([1,2,6,7])].groupby(['apt','g']).size().unstack().fillna(0).astype(int))
# m11 state + dep base
for which, base in [('training', 'cache/'), ('ranking', 'cache_final/')]:
    d = pd.read_parquet(R+base+f'major/dep_{which}.parquet')
    s = pd.read_parquet(R+base+f'major/m11_state_{which}.parquet')
    yr = '25-' if which == 'training' else '26-'
    g = yr + d.month.astype(int).astype(str).str.zfill(2)
    keep = g.isin(MON)
    num = d.select_dtypes('number').drop(columns=['y'], errors='ignore')
    print(f'\n{which} dep base: missing by month'); print(num[keep].isna().groupby(g[keep]).mean().T.round(4).to_string())
    print(f'{which} dep base: median by month'); print(num[keep].groupby(g[keep]).median().T.round(1).to_string())
    print(f'{which} m11 state: missing by month'); print(s[keep].isna().groupby(g[keep]).mean().T.round(4).to_string())
    print(f'{which} m11 state: median/p90 by month'); print(s[keep].groupby(g[keep]).median().T.round(2).to_string()); print(s[keep].groupby(g[keep]).quantile(.9).T.round(2).to_string())
    if which == 'ranking':
        k = pd.read_parquet(R+'cache_final/ranking.parquet', columns=['MVT_ID_mvt','PHASE_mvt'], filters=[('PHASE_mvt','==','DEP')])
        print('dep_ranking row order == ranking DEP order:', np.array_equal(k.MVT_ID_mvt.to_numpy(), d.MVT_ID_mvt.to_numpy()))
