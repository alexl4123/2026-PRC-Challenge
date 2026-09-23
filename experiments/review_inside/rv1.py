"""Review of the inside report: per-month sign counts, concentration, callsign-turnover projection, route-A stamping check.
Reads the inside agent's cached arrays; fits nothing."""
import sys, os
os.environ['OMP_NUM_THREADS'] = '2'
I = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/ideas_inside'
sys.path.insert(0, I)
from common import *        # load_dep, preds, rmse, S_, K, R, D(=I)
d = load_dep(['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
B = d.BLOCK_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
mon = pd.to_datetime(M, unit='s').month.to_numpy(); apt = S_(d.APT); cs = K(apt, S_(d.FLIGHT_mvt))
N = len(y); P = np.isnan(A3) & (M - S > 7200); Pi = np.where(P)[0]; osP = (M - S)[Pi]
v6, v7, bl = preds(); v8 = np.where(P, v7, 0.5 * v7 + 0.5 * bl[1]); pb = np.load(f'{R}/exp62_pb.npy').astype(float)
sc = np.isin(mon, [1, 7])

def permonth(new, ref, label, extra_mask=None):
    dm = {m: rmse(y[mon == m], new[mon == m]) - rmse(y[mon == m], ref[mon == m]) for m in range(1, 13)}
    neg = sum(v < -0.005 for v in dm.values()); pos = sum(v > 0.005 for v in dm.values())
    dsse = (y - new) ** 2 - (y - ref) ** 2
    tot = dsse.sum(); o = np.argsort(dsse)
    top5 = dsse[o[:5]].sum(); top20 = dsse[o[:20]].sum(); worst5 = dsse[o[-5:]].sum()
    d12 = rmse(y, new) - rmse(y, ref); dsc = rmse(y[sc], new[sc]) - rmse(y[sc], ref[sc])
    # leave-one-row-out sensitivity: remove the single most helpful row
    wo = np.sqrt(((y - new) ** 2).sum() - (y[o[0]] - new[o[0]]) ** 2) / np.sqrt(N - 1) - np.sqrt(((y - ref) ** 2).sum() - (y[o[0]] - ref[o[0]]) ** 2) / np.sqrt(N - 1)
    print(f'{label}: 12m {d12:+.3f} scored {dsc:+.3f} | months better/worse {neg}/{pos} | per-month ' + ' '.join(f'{m}:{v:+.2f}' for m, v in dm.items()))
    print(f'    dSSE total {tot:.3e}; best 5 rows {top5:.3e} ({top5/tot:.0%}), best 20 {top20:.3e} ({top20/tot:.0%}), worst 5 {worst5:+.3e}; without the single best row 12m {wo:+.3f}')

print('================ IDEA 1 (weighted specialist), through the mixture, vs v7 (diag_e mix convention)')
E = np.load(f'{I}/diag_e_specP.npy'); ARMS = ['ctl_x', 'x_w_abs', 'x_w_sq', 'x_cont_sq', 'x_cs', 'x_cs_w_abs', 'base_w_abs']
def mix(p):
    new = v7.copy(); new[Pi] = np.clip(p * osP + (1 - p) * pb[Pi], 0, None); return new
ctl = mix(E[0])
for i, a in enumerate(ARMS[1:], 1):
    permonth(mix(E[i]), v7, f'  {a:11s}')
yP = y[Pi]; lirf = apt[Pi] == 'LIRF'
for i in (0, 2):
    p = E[i]
    for lo, hi in [(7200, 20000), (20000, 1e9)]:
        k = (osP > lo) & (osP <= hi)
        print(f'  {ARMS[i]:8s} off_sched ({lo},{hi}]: n {k.sum()}, mean p {p[k].mean():.3f}, tail rate {(yP[k] > 7200).mean():.3f}')

print('\n================ IDEA 4 (P2 specialist gbm_x) on v8~')
cand = pd.read_parquet(f'{I}/p2_cand_training.parquet'); P2 = cand.index.to_numpy(); os2 = cand.off_sched.to_numpy(float)
Q = np.load(f'{I}/diag_l_Q.npy')
for qi, nm in [(2, 'gbm_x'), (3, 'gbm_x_wabs')]:
    new = v8.copy(); new[P2] = Q[qi] * os2 + (1 - Q[qi]) * v8[P2]; permonth(new, v8, f'  {nm:10s}')
    dss = ((y - new) ** 2 - (y - v8) ** 2); print(f'    LIRF share of dSSE: {dss[apt == "LIRF"].sum() / dss.sum():.0%}')

print('\n================ IDEA 3 (stack) on v8~ : per month, vs base and vs null')
A = {a: v8 + np.load(f'{I}/diag_s_{a}_v8.npy') for a in ['null', 'all', 'all_nocs', 'stamp', 'clocks']}
for a in ['all', 'all_nocs', 'stamp', 'clocks']:
    permonth(A[a], v8, f'  {a:9s} vs base')
permonth(A['all'], A['null'], '  all       vs null')

print('\n================ COMBINATION (diag_o) per month vs v8~')
AP2 = A['all'].copy(); AP2[P2] = Q[2] * os2 + (1 - Q[2]) * A['all'][P2]
AP2W = AP2.copy(); AP2W[Pi] = np.clip(E[2] * osP + (1 - E[2]) * pb[Pi], 0, None)
permonth(AP2W, v8, '  #3+#4+#1  ')

print('\n================ CALLSIGN TURNOVER: idea 3 gain by LOMO-seen status, projected to 2026')
key_c, uk = pd.factorize(cs); cnt = np.zeros((len(uk), 13)); np.add.at(cnt, (key_c, mon), 1)
seen_lomo = (cnt[key_c].sum(1) - cnt[key_c, mon]) > 0
rk = load_dep(['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt'], 'ranking'); rcs = K(S_(rk.APT), S_(rk.FLIGHT_mvt)); rapt = S_(rk.APT)
seen26 = pd.Index(uk).isin(pd.Index(rcs)) if False else pd.Series(rcs).isin(set(uk)).to_numpy()
pop = ~P
for a in ['all', 'all_nocs']:
    dsse = (y - A[a]) ** 2 - (y - v8) ** 2
    tot_lomo = dsse.sum(); proj = 0.0; rows = []
    for ap in np.unique(apt):
        k = apt == ap
        ds, du = dsse[k & seen_lomo], dsse[k & ~seen_lomo]
        ms, mu = ds.mean(), (du.mean() if len(du) else 0.0)
        u26 = 1 - seen26[rapt == ap].mean(); ul = 1 - seen_lomo[k].mean(); n = k.sum()
        proj += n * ((1 - u26) * ms + u26 * mu); rows.append((ap, round(ul * 100, 1), round(u26 * 100, 1), round(ms, 1), round(mu, 1), round(dsse[k].sum() / 1e8, 2)))
    print(f'  {a}: (apt, LOMO unseen %, 2026 unseen %, dSSE/row seen, dSSE/row unseen, airport dSSE 1e8)')
    for r in rows: print('    ', r)
    base_rmse = rmse(y, v8)
    print(f'  {a}: LOMO 12m {np.sqrt(base_rmse**2 + tot_lomo / N) - base_rmse:+.3f}  -> with 2026 callsign turnover {np.sqrt(base_rmse**2 + proj / N) - base_rmse:+.3f}  (survival {proj / tot_lomo:.0%})')

print('\n================ ROUTE-A STAMPING CHECK (F1)')
dl = (A3 - S); rA = ~np.isnan(A3) & (M - A3 > 0)
k = rA & (dl > 600) & (y <= 7200)
db = B - S; st = k & (np.abs(db) < 60)
print(f'  route A, NM delay > 600 s, y <= 7200: {k.sum():,} rows; |BLOCK-SCHED| < 60 s: {st.sum():,} ({st.sum() / k.sum():.2%})')
print(f'  exact y == off_sched (|diff| < 1 s) among those: {(np.abs(y - (M - S)) < 1)[st].mean():.2%}; |BLOCK-SCHED| == 0 exactly: {(db[st] == 0).mean():.2%}')
for lo, hi in [(0, 1), (1, 60), (60, 120), (120, 180), (-60, -1), (-120, -60)]:
    kk = k & (db >= lo) & (db < hi) if lo >= 0 else k & (db >= lo) & (db < hi)
    print(f'    BLOCK-SCHED in [{lo},{hi}) s: {kk.sum():6,} rows  ({kk.sum() / max(hi - lo, 1):.1f} per second)')
prox = M - A3
print(f'  NM proxy (MVT-AOBT_3) median: stamped {np.median(prox[st]):.0f} s vs other delayed {np.median(prox[k & ~st]):.0f} s; y median stamped {np.median(y[st]):.0f} vs {np.median(y[k & ~st]):.0f}')
print(f'  v6 residual on stamped rows: mean {np.mean((y - v6)[st]):+.0f} s, RMSE {rmse(y[st], v6[st]):.0f}; SSE share {((y - v6)[st] ** 2).sum() / ((y - v6) ** 2).sum():.2%}')
print('  stamped share by airport:', {ap: round(float(st[k & (apt == ap)].mean() * 100), 1) for ap in np.unique(apt)})
print('peak GB', round(peak(), 2))
