"""Arrival in-block stamping at SCHED: does it exist, and does it track departure stamping?"""
from common import *
a = pd.read_parquet(f'{D}/training_ARR.parquet', columns=['APT', 'FLIGHT_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt'])
L, S, B = a.MVT_TIME_UTC_mvt.to_numpy(), a.SCHED_TIME_UTC_mvt.to_numpy(), a.BLOCK_TIME_UTC_mvt.to_numpy()
aapt = a.APT.astype(str).to_numpy()
lateL = np.abs(L - S) > 900           # landing far from scheduled in-block -> stamping detectable
ast = (np.abs(B - S) < 60)
print('ARR rows', len(a), ' landing-vs-SIBT |d|>900:', lateL.sum())
print('arrival in-block within 60s of SIBT when |landing - SIBT| > 900, by airport (%):')
print((pd.Series(ast[lateL]).groupby(aapt[lateL]).mean() * 100).round(2).to_string())
print('in-block BEFORE landing (impossible, i.e. stamped/garbled) share by airport (%):')
print((pd.Series((B < L)).groupby(aapt).mean() * 100).round(2).to_string())
# departure stamping indicator on route-A delayed rows
d = load_dep(['APT', 'FLIGHT_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
M, Sd, Bd, A3 = [d[c].to_numpy() for c in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt']]
y = d.TAXITIME_SEC_mvt.to_numpy()
v6, v7, b = preds(); r = y - v6
dapt = d.APT.astype(str).to_numpy()
pop = (~np.isnan(A3)) & (A3 - Sd > 600) & (M - A3 > 0) & (M - A3 <= 7200) & (y <= 7200)
dst = np.abs(Bd - Sd) < 60
print(f'\ndeparture pop (route A, NM delay>600, y<=7200): {pop.sum():,}; stamped share {dst[pop].mean()*100:.2f}%')
# day-level: airport-day arrival stamping rate vs airport-day departure stamping rate
aday = (L // 86400).astype(int); dday = (M // 86400).astype(int)
A_ = pd.DataFrame(dict(apt=aapt[lateL], day=aday[lateL], st=ast[lateL].astype(float)))
ag = A_.groupby(['apt', 'day']).st.agg(['mean', 'size']).rename(columns={'mean': 'arr_rate', 'size': 'arr_n'})
D_ = pd.DataFrame(dict(apt=dapt[pop], day=dday[pop], st=dst[pop].astype(float), r=r[pop]))
dg = D_.groupby(['apt', 'day']).agg(dep_rate=('st', 'mean'), dep_n=('st', 'size'), r=('r', 'mean'))
j = dg.join(ag, how='inner'); j = j[(j.arr_n >= 5) & (j.dep_n >= 5)]
print('airport-day correlation arr stamping rate vs dep stamping rate (Spearman), per airport:')
for ap, g in j.groupby(level=0):
    print(f'  {ap}: days {len(g):3d}  rho {g.arr_rate.corr(g.dep_rate, method="spearman"):+.3f}  mean arr {g.arr_rate.mean():.3f} dep {g.dep_rate.mean():.3f}  rho(arr_rate, mean resid) {g.arr_rate.corr(g.r, method="spearman"):+.3f}')
# flight-level: row-level join of departure to same-day arrival stamping rate at airport
jj = D_.merge(ag.reset_index(), on=['apt', 'day'], how='left')
print('row-level Spearman(dep stamped, airport-day arr stamping rate):', round(jj.st.corr(jj.arr_rate, method='spearman'), 4),
      ' with resid:', round(jj.r.corr(jj.arr_rate, method='spearman'), 4))
# same stand: previous arrival at our stand stamped?
key_a = pd.Series(aapt).str.cat(a.STAND_mvt.astype(str).to_numpy(), sep='|').to_numpy()
key_d = pd.Series(dapt).str.cat(d.STAND_mvt.astype(str).to_numpy(), sep='|').to_numpy()
ea = pd.DataFrame(dict(k=key_a, t=B, stp=(np.abs(B - S) < 60).astype(float), inb_minus_sched=B - S)).dropna(subset=['t']).sort_values('t')
q = pd.DataFrame(dict(k=key_d[pop], t=Sd[pop], i=np.where(pop)[0])).sort_values('t')
m = pd.merge_asof(q, ea, on='t', by='k', direction='backward', tolerance=12 * 3600)
m = m.sort_values('i')
st_pop = dst[pop]; r_pop = r[pop]
ok = m.stp.notna().to_numpy()
print(f'prev arrival at our stand within 12h found: {ok.mean()*100:.1f}%')
print('P(dep stamped | prev arr stamped) = %.3f  vs  P(dep stamped | prev arr not stamped) = %.3f' % (
    st_pop[ok][m.stp.to_numpy()[ok] == 1].mean(), st_pop[ok][m.stp.to_numpy()[ok] == 0].mean()))
print('by airport:')
for ap in np.unique(dapt[pop]):
    mm = ok & (dapt[pop] == ap)
    s1 = m.stp.to_numpy()[mm] == 1
    print(f'  {ap}: n {mm.sum():6d}  prev-arr-stamped share {s1.mean():.3f}  P(dep st|arr st) {st_pop[mm][s1].mean():.3f}  P(dep st|not) {st_pop[mm][~s1].mean():.3f}  mean resid|arr st {r_pop[mm][s1].mean():+.0f}  |not {r_pop[mm][~s1].mean():+.0f}')
print('peak GB', peak())
