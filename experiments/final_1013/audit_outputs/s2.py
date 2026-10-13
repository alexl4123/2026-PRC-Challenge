import numpy as np, pandas as pd, os
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_outputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
cols = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt', 'off_sched', 'RUNWAY_mvt',
        'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'ADES_mvt']
rk = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=cols)
sub = pd.read_parquet(R + 'outputs_final/understanding-accordion_final.parquet')
print('sub rows', len(sub), 'unique ids', sub.MVT_ID_mvt.nunique(), 'rk unique', rk.MVT_ID_mvt.nunique(),
      'NaN preds', sub.TAXITIME_SEC_mvt.isna().sum())
print('ids sub-not-in-rk', (~sub.MVT_ID_mvt.isin(rk.MVT_ID_mvt)).sum(), 'rk-not-in-sub', (~rk.MVT_ID_mvt.isin(sub.MVT_ID_mvt)).sum())
tmpl = pd.read_parquet(R + '../final_submitting.parquet')
print('template rows', len(tmpl), 'cols', list(tmpl.columns), 'sub ids not in template', (~sub.MVT_ID_mvt.isin(tmpl.MVT_ID_mvt)).sum(),
      'template not in sub', (~tmpl.MVT_ID_mvt.isin(sub.MVT_ID_mvt)).sum())
rk = rk.merge(sub, on='MVT_ID_mvt', how='left')
p = rk.TAXITIME_SEC_mvt.to_numpy(float)
print('pred describe', pd.Series(p).describe(percentiles=[.001, .01, .5, .99, .999]).round(1).to_dict())
print('pred==0', (p == 0).sum(), '<60', (p < 60).sum(), '>3600', (p > 3600).sum(), '>7200', (p > 7200).sum(), '>20000', (p > 20000).sum())
rk['day'] = rk.MVT_TIME_UTC_mvt.dt.floor('D'); rk['month'] = rk.MVT_TIME_UTC_mvt.dt.month
rk['D'] = rk.AOBT_3_flt.isna(); rk['P'] = rk.D & (rk.off_sched > 7200); rk['nolobt'] = rk.LOBT_flt.isna()

# --- 2025 frame
tr = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=cols + ['y'])
z = np.load(R + 'cache/w2/v28s_oof.npz'); tr['f'] = z['f']; assert np.allclose(tr.y.to_numpy(), z['y'])
tr['day'] = tr.MVT_TIME_UTC_mvt.dt.floor('D'); tr['month'] = tr.MVT_TIME_UTC_mvt.dt.month
tr['D'] = tr.AOBT_3_flt.isna(); tr['P'] = tr.D & (tr.off_sched > 7200); tr['nolobt'] = tr.LOBT_flt.isna()

# extremes in 2025 OOF for reference
f = tr.f.to_numpy()
print('2025 OOF f describe', pd.Series(f).describe(percentiles=[.001, .01, .5, .99, .999]).round(1).to_dict())
print('2025 f==0', (f <= 0.5).sum(), '>7200', (f > 7200).sum(), 'y>7200', (tr.y > 7200).sum())
print('2026 pred >7200 by APT,month,D/P:'); print(rk[p > 7200].groupby(['APT', 'month', 'D', 'P']).size())
print('2026 pred==0 rows:'); print(rk[p == 0][['APT', 'MVT_TIME_UTC_mvt', 'D', 'P', 'off_sched']].head(20))
rk.to_parquet(O + 'rk.parquet'); tr[['APT', 'day', 'month', 'D', 'P', 'nolobt', 'y', 'f', 'MVT_TIME_UTC_mvt', 'off_sched']].to_parquet(O + 'tr_small.parquet')

# --- (2) structure per APT x month
def struct(d, val):
    g = d.groupby(['APT', 'month'])
    return pd.DataFrame({'n': g.size(), 'Dsh': g.D.mean(), 'Psh': g.P.mean(), 'noLOBT': g.nolobt.mean(), 'mean': g[val].mean()})
s26 = struct(rk, 'TAXITIME_SEC_mvt'); s25 = struct(tr[tr.month.isin([1, 2, 6, 7])], 'y'); s25f = struct(tr[tr.month.isin([1, 2, 6, 7])], 'f')
S = s26.join(s25, rsuffix='_25').join(s25f[['mean']].rename(columns={'mean': 'oof_25'}))
S['n_ratio'] = S.n / S.n_25
for c in ['Dsh', 'Psh', 'noLOBT', 'Dsh_25', 'Psh_25', 'noLOBT_25']:
    S[c] = (100 * S[c]).round(2)
print(S.round(2).to_string())
S.to_csv(O + 'struct_apt_month.csv')
