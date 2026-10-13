import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_outputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
rk = pd.read_parquet(O + 'rk.parquet'); tr = pd.read_parquet(O + 'tr_small.parquet')
ids = np.load(R + 'cache_final/adsb2_final/ranking_ids.npy', allow_pickle=True)
assert np.array_equal(ids, rk.MVT_ID_mvt.to_numpy())
for d in ['w2_final', 'nn_final', 'seq_ep40_final']:
    assert np.array_equal(np.load(R + f'cache_final/{d}/ranking_ids.npy', allow_pickle=True), ids)
FAM = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
def m3(fn): return np.mean([np.load(fn(s)).astype(float) for s in range(3)], axis=0)
C26 = {f'a_{f}': m3(lambda s: R + f'cache_final/adsb2_final/{f}__seed{s:02d}.npy') for f in FAM}
C26['w2mix'] = m3(lambda s: R + f'cache_final/w2_final/mix__seed{s:02d}.npy')
C26['seq40'] = m3(lambda s: R + f'cache_final/seq_ep40_final/seq__seed{s:02d}.npy')
nnf = m3(lambda s: R + f'cache_final/nn_final/nn_final_seed{s:02d}.npy')
for k, v in C26.items(): rk[k] = v
rk['nnf'] = nnf
# 2025 OOF components, months 1,2,6,7 (rows in dep_training order within month)
M = [1, 2, 6, 7]
trm = tr.month.to_numpy()
C25 = {k: np.full(len(tr), np.nan) for k in C26}
for m in range(1, 13):
    sel = np.flatnonzero(trm == m)
    for f in FAM:
        C25[f'a_{f}'][sel] = np.mean([np.load(R + f'cache/cv/adsb2/{f}__seed{s:02d}__m{m:02d}.npy').astype(float) for s in range(3)], axis=0)
    C25['w2mix'][sel] = np.mean([np.load(R + f'cache/cv/w2/mix__seed{s:02d}__m{m:02d}.npy').astype(float) for s in range(3)], axis=0)
    C25['seq40'][sel] = np.mean([np.load(R + f'cache/cv/seqsw_ep40/seq__seed{s:02d}__m{m:02d}.npy').astype(float) for s in range(3)], axis=0)
for k, v in C25.items(): tr[k] = v
# ADS-B features
fr = pd.read_parquet(R + 'cache_final/adsb_lol/features_ranking.parquet', columns=['MVT_ID_mvt', 'adsb_matched', 'adsb_has_ground'])
assert np.array_equal(fr.MVT_ID_mvt.to_numpy(), ids) or True
rk = rk.merge(fr, on='MVT_ID_mvt', how='left')
ft = pd.read_parquet(R + 'cache/adsb_lol/features_training.parquet', columns=['adsb_matched', 'adsb_has_ground'])
tr['adsb_matched'] = ft.adsb_matched.to_numpy(); tr['adsb_has_ground'] = ft.adsb_has_ground.to_numpy()
COMP = ['a_mix', 'a_nn', 'a_perapt', 'a_routed', 'a_xgboost', 'a_w3', 'w2mix', 'seq40']
for d in (rk, tr):
    d['spread'] = d[COMP].std(axis=1)
    d['d_aw'] = d.a_mix - d.w2mix          # ADS-B tabular vs non-ADS-B tabular
    d['d_sw'] = d.seq40 - d.w2mix          # sequence model vs non-ADS-B tabular
rk['pred'] = rk.TAXITIME_SEC_mvt; tr['pred'] = tr.f
def daily(d, extra):
    g = d.groupby(['APT', 'day'])
    out = pd.DataFrame({'n': g.size(), 'pred': g.pred.mean(), 'spread': g.spread.mean(), 'd_aw': g.d_aw.mean(), 'd_sw': g.d_sw.mean(),
                        'match': g.adsb_matched.mean(), 'ground': g.adsb_has_ground.mean(), 'Dsh': g.D.mean(), 'Psh': g.P.mean()})
    for e in extra: out[e] = g[e].mean()
    return out.reset_index()
d26 = daily(rk, ['w2mix', 'a_mix', 'seq40', 'nnf']); d25 = daily(tr, ['y', 'w2mix', 'a_mix', 'seq40'])
d26['dow'] = d26.day.dt.dayofweek; d25['dow'] = d25.day.dt.dayofweek
def neigh(df, col, k=3):
    # residual vs median of +-k neighbouring days (excluding self) within the same APT and contiguous block
    res = np.full(len(df), np.nan)
    for apt, g in df.groupby('APT'):
        g = g.sort_values('day'); v = g[col].to_numpy(); t = g.day.to_numpy(); idx = g.index.to_numpy()
        for i in range(len(g)):
            j = [x for x in range(max(0, i - k), min(len(g), i + k + 1)) if x != i and abs((t[x] - t[i]) / np.timedelta64(1, 'D')) <= k]
            if len(j) >= 2: res[np.searchsorted(df.index.to_numpy(), idx[i])] = v[i] - np.median(v[j])
    return res
d26 = d26.reset_index(drop=True); d25 = d25.reset_index(drop=True)
for col in ['pred', 'spread', 'd_aw', 'd_sw', 'match', 'n']:
    d26['r_' + col] = neigh(d26, col); d25['r_' + col] = neigh(d25, col)
d25['r_y'] = neigh(d25, 'y')
# n residual relative: use ratio to neighbour median
d26.to_parquet(O + 'd26.parquet'); d25.to_parquet(O + 'd25.parquet')
# reference distributions per APT from 2025 (all months) of residuals
ref = {}
for col in ['pred', 'spread', 'd_aw', 'd_sw', 'match']:
    q = d25.groupby('APT')['r_' + col].agg(lambda s: np.nanpercentile(np.abs(s), 99.5)).rename('q995_' + col)
    mx = d25.groupby('APT')['r_' + col].agg(lambda s: np.nanmax(np.abs(s))).rename('max_' + col)
    ref[col] = pd.concat([q, mx], axis=1)
qy = d25.groupby('APT')['r_y'].agg(lambda s: np.nanpercentile(np.abs(s), 99.5)).rename('q995_y_label')
print('2025 per-APT |daily residual| 99.5th pct and max:')
print(pd.concat([ref[c] for c in ref] + [qy], axis=1).round(1).to_string())
# level stats (not residual): d_aw and d_sw and spread absolute per APT in 2025 months 1,2,6,7 vs 2026
for col in ['spread', 'd_aw', 'd_sw', 'match', 'ground']:
    a = d25[d25.day.dt.month.isin(M)].groupby('APT')[col].describe(percentiles=[.01, .5, .99])[['min', '1%', '50%', '99%', 'max']]
    b = d26.groupby('APT')[col].describe(percentiles=[.01, .5, .99])[['min', '1%', '50%', '99%', 'max']]
    print(f'\n== daily {col}: 2025(1,2,6,7) vs 2026'); print(pd.concat([a, b], axis=1, keys=['2025', '2026']).round(3).to_string())
flags = []
for col in ['pred', 'spread', 'd_aw', 'd_sw', 'match']:
    x = d26.merge(ref[col], left_on='APT', right_index=True)
    fl = x[np.abs(x['r_' + col]) > x['max_' + col]]
    for _, r in fl.iterrows(): flags.append((col, r.APT, str(r.day.date()), int(r.n), round(r['r_' + col], 1), round(r['max_' + col], 1)))
F = pd.DataFrame(flags, columns=['stat', 'APT', 'day', 'n', 'resid', 'max2025'])
print('\nFLAGS: 2026 airport-days whose neighbour residual exceeds the 2025 max for that airport'); print(F.sort_values(['APT', 'day']).to_string())
F.to_csv(O + 'flags_daily.csv', index=False)
