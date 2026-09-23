"""Which clock does BLOCK match, by y band and route; SSE shares under v6."""
from common import *
d = load_dep(['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
              'AOBT_3_flt', 'EOBT_1_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy()
v6, v7, b = preds()
r = y - v6; sse = r ** 2; tot = sse.sum()
M, S, B, A3, E1 = [d[c].to_numpy() for c in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt']]
os_ = M - S; oa = M - A3
miss = np.isnan(A3)
route = np.where(miss, 'D', np.where(oa > 7200, 'B', np.where(oa <= 0, 'C', 'A')))
P = miss & (os_ > 7200)
clk = np.full(len(y), 'other', object)
near_s = np.abs(B - S) < 60; near_a = np.abs(B - A3) < 60; near_e = np.abs(B - E1) < 60
clk[near_e] = 'EOBT1'; clk[near_s] = 'SCHED'; clk[near_a] = 'AOBT3'
clk[near_a & near_s] = 'AOBT3=SCHED'
apt = d.APT.astype(str).to_numpy()
month = pd.to_datetime(M, unit='s').month
band = pd.cut(y, [-1e9, 1800, 7200, 1e9], labels=['<=1800', '1800-7200', '>7200']).astype(str)
df = pd.DataFrame(dict(route=route, clk=clk, band=band, sse=sse, apt=apt, P=P))
print('=== all 12 months: SSE share (% of total v6 SSE) and row counts, by band x route x clock')
g = df.groupby(['band', 'route', 'clk']).agg(n=('sse', 'size'), sse=('sse', 'sum'))
g['sse%'] = 100 * g.sse / tot; g['rmse'] = np.sqrt(g.sse / g.n)
print(g[g.n > 200].drop(columns='sse').round(2).to_string())
print('\n=== delayed band 1800<y<=7200 route A: clock shares by airport (rows %)')
m = (band == '1800-7200') & (route == 'A')
print(pd.crosstab(apt[m], clk[m], normalize='index').mul(100).round(1).to_string())
print('\n=== ALL route-A bulk rows y<=7200: clock shares by airport (rows %)')
m = (y <= 7200) & (route == 'A')
print(pd.crosstab(apt[m], clk[m], normalize='index').mul(100).round(1).to_string())
# NM delay: AOBT3 - SCHED. Among route A rows with delay > 600 s, how often is BLOCK at SCHED?
dl = A3 - S
m = (route == 'A') & (dl > 600) & (y <= 7200)
print(f'\n=== route A, NM delay > 600 s, y<=7200: {m.sum():,} rows, {100*sse[m].sum()/tot:.2f}% of SSE')
print(pd.crosstab(apt[m], clk[m], normalize='index').mul(100).round(1).to_string())
sub = pd.DataFrame(dict(apt=apt[m], clk=clk[m], sse=sse[m], r=r[m]))
gg = sub.groupby('clk').agg(n=('sse', 'size'), sse=('sse', 'sum'), bias=('r', 'mean'))
gg['sse%tot'] = 100 * gg.sse / tot; gg['rmse'] = np.sqrt(gg.sse / gg.n)
print(gg.drop(columns='sse').round(2).to_string())
# oracle: if the clock class were known inside this population, correct each class by its mean residual (per airport)
sub['cls'] = sub.apt + '|' + sub.clk
adj = sub.groupby('cls').r.transform('mean')
new = sse.copy(); new[m] = (r[m] - adj.to_numpy()) ** 2
print(f'oracle (class-mean bias removal per airport x clock, delayed-NM pop): RMSE {np.sqrt(tot/len(y)):.2f} -> {np.sqrt(new.sum()/len(y)):.2f}')
# stronger oracle: stamped rows predicted as off_sched, others unchanged
new2 = sse.copy(); st = m & (clk == 'SCHED'); new2[st] = (y[st] - os_[st]) ** 2
print(f'oracle2 (stamped rows := off_sched): RMSE -> {np.sqrt(new2.sum()/len(y)):.2f}')
# scored months
sc = np.isin(month, [1, 7])
print(f'scored months v6 RMSE {rmse(y[sc], v6[sc]):.2f}; delayed-NM pop share of scored SSE {100*sse[m&sc].sum()/sse[sc].sum():.2f}%')
print('\n=== route D not in P: band x clock (rows, SSE%)')
m = (route == 'D') & ~P
g = df[m].groupby(['band', 'clk']).agg(n=('sse', 'size'), sse=('sse', 'sum'))
g['sse%'] = 100 * g.sse / tot; g['rmse'] = np.sqrt(g.sse / g.n)
print(g.drop(columns='sse').round(2).to_string())
m2 = m & (y <= 7200)
osb = pd.cut(os_[m2], [-1e9, 0, 900, 1800, 3600, 7200])
print(pd.DataFrame(dict(osb=osb, clk=clk[m2], sse=sse[m2])).groupby(['osb', 'clk'], observed=True).agg(n=('sse', 'size'), ssep=('sse', lambda s: 100 * s.sum() / tot)).round(3).to_string())
print('peak GB', peak())
