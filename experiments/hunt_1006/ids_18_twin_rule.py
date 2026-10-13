"""ids_18 -- price a twin (return-to-stand) rule for the NM row of a narrow pair whose partner is route D.
Candidates (constants fit leave-one-month-out on 2025 pairs): r1 = off_sched - c ; r2 = off_lobt - c ; r3 = max(f, r1);
r4 = 0.5 (f + r1). 2025 OOF SSE deltas (12 m and Jan+Jul), rows worse; 2026 rows moved."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
tt = pd.read_parquet(H + 'cache/hunt/ids_08_twins2025.parquet')
lob = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt', 'off_lobt'])
tt = tt.merge(lob, on='MVT_ID_mvt', how='left')
def pairs(t):
    t = t[(t.gsize == 2) & (t.dmvt <= 600)]
    a = t[t.idrank == 0].set_index('gid'); b = t[t.idrank == 1].set_index('gid')
    keep = a.nm & ~b.nm      # first row NM, partner route D  (the observed structure; 99 of 105 in 2025)
    # label-free definition without ID order: exactly one NM row in the pair -> that row is 'nm', the other 'd'
    return a[keep], b[keep]
a, b = pairs(tt)
print('2025 pairs (NM first, D second):', len(a))
# label-free check: in pairs with exactly one NM row, is the NM row always the first by ID? (ID-free rule)
t = tt[(tt.gsize == 2) & (tt.dmvt <= 600)]
g = t.groupby('gid')
one = g.nm.transform('sum') == 1
print('pairs with exactly one NM row:', one.sum() // 2, ' NM row is first by ID:', t[one & t.nm].idrank.eq(0).mean())
x = pd.DataFrame({'y': a.y, 'f': a.f, 'os': a.off_sched, 'ol': a.off_lobt, 'm': a.month, 'apt': a.APT})
print('y - off_sched quantiles', np.quantile(x.y - x.os, [.1, .25, .5, .75, .9]).round(0))
print('y - off_lobt quantiles', np.quantile(x.y - x.ol, [.1, .25, .5, .75, .9]).round(0))
print('y - f quantiles', np.quantile(x.y - x.f, [.1, .25, .5, .75, .9]).round(0))
res = {}
for name in ['r1', 'r2', 'r3', 'r4', 'r5']:
    pred = np.zeros(len(x))
    for m in x.m.unique():
        tr_ = x[x.m != m]; te = (x.m == m).values
        c1 = np.median(tr_.os - tr_.y); c2 = np.median(tr_.ol - tr_.y); c5 = np.mean(tr_.y - tr_.f)
        r1 = x.os.values[te] - c1; r2 = x.ol.values[te] - c2
        pred[te] = {'r1': r1, 'r2': r2, 'r3': np.maximum(x.f.values[te], r1), 'r4': 0.5 * (x.f.values[te] + r1),
                    'r5': x.f.values[te] + c5}[name]
    # keep inside the LOBT window [ol - 3606, ol + 3606]
    pred = np.clip(pred, x.ol - 3606, x.ol + 3606)
    d_sse = (x.y - pred) ** 2 - (x.y - x.f) ** 2
    jj = x.m.isin([1, 7])
    res[name] = {'dSSE_12m': d_sse.sum(), 'dSSE_JanJul': d_sse[jj].sum(), 'worse': (d_sse > 0).sum(), 'n': len(x),
                 'months_better': sum(d_sse[x.m == m].sum() < 0 for m in x.m.unique()), 'months': x.m.nunique()}
r = pd.DataFrame(res).T
N25 = 2085047
r['dRMSE_12m_s'] = r.dSSE_12m / N25 / (2 * 234.37)
print(r.to_string())
# 2026
rt = pd.read_parquet(H + 'cache/hunt/ids_08_twins2026.parquet')
rk = pd.read_parquet(H + 'cache/major/dep_ranking.parquet', columns=['MVT_ID_mvt', 'off_lobt'])
rt = rt.merge(rk, on='MVT_ID_mvt', how='left')
a2, b2 = pairs(rt)
c1 = np.median(x.os - x.y)
r1 = np.clip(a2.off_sched - c1, a2.off_lobt - 3606, a2.off_lobt + 3606)
r3 = np.maximum(a2.f, r1)
print('\n2026 NM-first pairs:', len(a2), ' c1 =', c1)
out = pd.DataFrame({'apt': a2.APT, 'fl': a2.FLIGHT_mvt, 'f': a2.f, 'os': a2.off_sched, 'ol': a2.off_lobt, 'r1': r1.round(0), 'r3': r3.round(0)})
out['move_r3'] = out.r3 - out.f
print(out.to_string())
print('sum move^2 (r3):', (out.move_r3 ** 2).sum(), ' -> |dRMSE| scale', (out.move_r3 ** 2).sum() / 344841 / (2 * 235.4))
