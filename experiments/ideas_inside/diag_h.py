"""2026 ranking P rows: composition shift and what v7/v8 do there, vs 2025 OOF behaviour."""
from common import *
z = np.load(f'{R}/final/lgb_components.npz'); zs = np.load(f'{R}/final/spec_x_ranking.npz')
ids = z['ids']; pb = z['pb'].mean(0); et = z['et']; inP = z['in_P']; p7 = zs['p_spec']
assert np.array_equal(ids, zs['ids'])
net = np.mean([np.load(f'{R}/nn_final/nn_final_seed{s:02d}.npy') for s in range(3)], axis=0)
nid = np.load(f'{R}/nn_final/ranking_ids.npy')
net = pd.Series(net, index=nid).reindex(ids).to_numpy()
pb8 = 0.5 * pb + 0.5 * net
rk = load_dep(['APT', 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'], 'ranking').set_index('MVT_ID_mvt').reindex(ids)
apt = S_(rk.APT); mon = pd.to_datetime(rk.MVT_TIME_UTC_mvt.to_numpy(), unit='s').month
os_ = (rk.MVT_TIME_UTC_mvt - rk.SCHED_TIME_UTC_mvt).to_numpy()
g = et - pb8
add = p7 * g
df = pd.DataFrame(dict(apt=apt, mon=mon, p=p7, add=add, add2=add ** 2, os=os_))[inP]
print('2026 ranking P rows by airport x month: n, mean p (v7 spec), mean seconds added by the tail branch p*(et-pb8), SSE-if-all-REAL share')
N = len(ids)
t = df.groupby(['apt', 'mon']).agg(n=('p', 'size'), p_mean=('p', 'mean'), p_med=('p', 'median'), add_mean=('add', 'mean'), sse_fa=('add2', 'sum'))
t['rmse_pts_if_real'] = np.sqrt(t.sse_fa / N)
print(t.round(3).to_string())
tot_fa = df.add2.sum()
print(f'total P rows {inP.sum()}, sum (p*g)^2 = {tot_fa:.3e}; if ALL were REAL the tail-branch SSE alone = {np.sqrt(tot_fa/N):.2f} s of RMSE (v8 LB 297.83, SSE {297.83**2*N:.3e})')
zero = np.isin(df.apt, ['EDDF', 'EDDM', 'EHAM', 'LEBL', 'LEMD'])
print(f'zero-tail-history airports: {zero.sum()} P rows, mean p {df.p[zero].mean():.4f}, sum (p g)^2 {df.add2[zero].sum():.3e}')
# counterfactual: set p := 0 at the 5 zero-tail airports, assuming rows are REAL (y ~ pb8): dSSE = -sum (p g)^2 ; LB-delta approx
dsse = -df.add2[zero].sum()
print(f'  p:=0 there, if all REAL: LB delta ~ {np.sqrt((297.83**2*N + dsse)/N) - 297.83:+.3f} s')
# 2025 comparison from OOF: v7 spec p on P rows by airport, tail counts
d = load_dep(['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
P = d.AOBT_3_flt.isna().to_numpy() & (M - S > 7200)
spec = np.load(f'{R}/exp62_spec_P.npy'); pbo = np.load(f'{R}/exp62_pb.npy').astype(float)
Pi = np.where(P)[0]; ap = S_(d.APT)[Pi]; yP = y[Pi]; osP = (M - S)[Pi]
t2 = pd.DataFrame(dict(apt=ap, tail=yP > 7200, p=spec[1], real_bias=np.where(yP <= 7200, yP - pbo[Pi], np.nan), add=spec[1] * (osP - pbo[Pi])))
print('\n2025 P rows (OOF): n, tails, mean v7 p, mean added seconds, mean (y - pb) on REAL rows')
print(t2.groupby('apt').agg(n=('p', 'size'), tails=('tail', 'sum'), p_mean=('p', 'mean'), add_mean=('add', 'mean'), real_bias=('real_bias', 'mean')).round(3).to_string())
print('peak', peak())
