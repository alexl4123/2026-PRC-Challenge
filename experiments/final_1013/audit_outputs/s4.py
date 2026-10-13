import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_outputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
rk = pd.read_parquet(O + 'rk.parquet')
tr = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'off_aobt3', 'off_sched', 'y',
                     'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt'])
tr['f'] = np.load(R + 'cache/w2/v28s_oof.npz')['f']
r2 = pd.read_parquet(R + 'cache_final/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'off_aobt3'])
rk = rk.merge(r2, on='MVT_ID_mvt')
tr['day'] = tr.MVT_TIME_UTC_mvt.dt.floor('D')
d26 = pd.read_parquet(O + 'd26.parquet'); d25 = pd.read_parquet(O + 'd25.parquet')

# (a) label-free proxy: off_aobt3 = MVT - AOBT_3 on rows with AOBT_3 (clip to [0, 7200])
def prox(d, val):
    x = d[d.off_aobt3.notna()].copy(); x['px'] = x.off_aobt3.clip(0, 7200)
    g = x.groupby(['APT', 'day'])
    return pd.DataFrame({'px': g.px.mean(), 'v': g[val].mean(), 'nABC': g.size()}).reset_index()
p26 = prox(rk, 'TAXITIME_SEC_mvt'); p25y = prox(tr, 'y'); p25f = prox(tr, 'f')
p26['gap'] = p26.v - p26.px; p25y['gap'] = p25y.v - p25y.px; p25f['gap'] = p25f.v - p25f.px
print('daily gap = mean(pred or y) - mean(clip(MVT-AOBT3)) on A/B/C rows; per APT dist')
a = p25y.groupby('APT').gap.describe(percentiles=[.005, .5, .995])[['min', '0.5%', '50%', '99.5%', 'max']]
b = p25f.groupby('APT').gap.describe(percentiles=[.005, .5, .995])[['min', '0.5%', '50%', '99.5%', 'max']]
c = p26.groupby('APT').gap.describe(percentiles=[.005, .5, .995])[['min', '0.5%', '50%', '99.5%', 'max']]
print(pd.concat([a, b, c], axis=1, keys=['2025 y', '2025 oof', '2026 pred']).round(1).to_string())
lim = p25f.groupby('APT').gap.agg(['min', 'max'])
x = p26.merge(lim, left_on='APT', right_index=True)
print('\n2026 days with gap outside 2025 OOF [min,max]:'); print(x[(x.gap < x['min']) | (x.gap > x['max'])].round(1).to_string())
# (b) flagged days: component daily means vs neighbour medians
fl = [('EDDF', '2026-01-26'), ('EDDF', '2026-01-29'), ('EDDF', '2026-02-03'), ('EDDF', '2026-02-19'), ('LSZH', '2026-01-10'),
      ('LTFM', '2026-01-19'), ('LFPG', '2026-01-07'), ('LEMD', '2026-01-05'), ('LEMD', '2026-02-05'), ('EHAM', '2026-01-04'), ('EHAM', '2026-01-05'),
      ('LSZH', '2026-02-18'), ('LSZH', '2026-01-08'), ('EHAM', '2026-01-08'), ('EHAM', '2026-01-09')]
d26 = d26.merge(p26[['APT', 'day', 'px']], on=['APT', 'day'], how='left')
rows = []
for apt, dd in fl:
    g = d26[d26.APT == apt].sort_values('day').reset_index(drop=True); i = g.index[g.day.dt.strftime('%Y-%m-%d') == dd][0]
    nb = g.iloc[[j for j in range(max(0, i - 3), min(len(g), i + 4)) if j != i]]
    r = {'APT': apt, 'day': dd, 'n': g.n[i], 'n_nb': nb.n.median()}
    for c in ['pred', 'w2mix', 'a_mix', 'seq40', 'nnf', 'px', 'match', 'Dsh']:
        r[c] = round(g[c][i] - nb[c].median(), 1 if c not in ('match', 'Dsh') else 3)
    rows.append(r)
print('\nflagged days: daily mean minus neighbour median (px = clip(MVT-AOBT3) proxy)'); print(pd.DataFrame(rows).to_string())
# (c) low-match days: matched vs unmatched rows' d_aw
# (d) zeros: LSZH 2025 distribution of y near 0
for apt in ['LSZH']:
    t = tr[tr.APT == apt]
    print(f'\n{apt} 2025: y<60 {int((t.y < 60).sum())}, y<=0 {int((t.y <= 0).sum())} of {len(t)}; route D y<60 {int(((t.y < 60) & t.AOBT_3_flt.isna()).sum())} of {int(t.AOBT_3_flt.isna().sum())}; OOF f<1 {int((t.f < 1).sum())}')
z = rk[rk.TAXITIME_SEC_mvt == 0]; print('2026 pred==0 by APT', z.APT.value_counts().to_dict(), 'route D', int(z.D.sum()))
z = rk[rk.TAXITIME_SEC_mvt < 60]; print('2026 pred<60 by APT', z.APT.value_counts().to_dict(), 'route D', int(z.D.sum()))
z = tr[tr.f < 60]; print('2025 OOF<60 by APT', z.APT.value_counts().to_dict())
# big predictions
print('\n2026 pred > 20000:', rk[rk.TAXITIME_SEC_mvt > 20000].groupby(['APT', 'P']).size().to_dict(),
      ' 2025 OOF>20000', tr[tr.f > 20000].APT.value_counts().to_dict(), ' 2025 y>20000', tr[tr.y > 20000].APT.value_counts().to_dict())
big = rk[(rk.TAXITIME_SEC_mvt > 7200) & ~rk.P]
print('2026 pred>7200 outside zone P:', len(big)); print(big[['APT', 'MVT_TIME_UTC_mvt', 'D', 'off_sched', 'off_aobt3', 'TAXITIME_SEC_mvt']].sort_values('TAXITIME_SEC_mvt').to_string())
pz = rk[rk.P]; print('\nzone P 2026: n', len(pz), 'pred/off_sched ratio quantiles', (pz.TAXITIME_SEC_mvt / pz.off_sched).quantile([0, .1, .5, .9, 1]).round(2).to_dict())
