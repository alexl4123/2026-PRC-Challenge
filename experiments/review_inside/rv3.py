import sys; I = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/ideas_inside'; sys.path.insert(0, I)
from common import *
d = load_dep(['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
mon = pd.to_datetime(M, unit='s').month.to_numpy(); apt = S_(d.APT); cs = K(apt, S_(d.FLIGHT_mvt)); sc = np.isin(mon, [1, 7]); N = len(y)
P = np.isnan(A3) & (M - S > 7200); Pi = np.where(P)[0]; osP = (M - S)[Pi]
base = np.load('base_v9s2.npy'); pb = np.load(f'{R}/exp62_pb.npy').astype(float)
C = {a: base + np.load(f'rv_s_{a}_v9s2.npy') for a in ['null', 'all', 'all_nocs']}
def rep(new, ref, lab):
    b1 = boot(y, new, ref, 300); b2 = boot(y[sc], new[sc], ref[sc], 300)
    dm = [rmse(y[mon == m], new[mon == m]) - rmse(y[mon == m], ref[mon == m]) for m in range(1, 13)]
    print(f'{lab:28s} 12m {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]  months better {sum(v < 0 for v in dm)}/12')
print(f'base v9~2: 12m {rmse(y, base):.3f}')
rep(C['null'], base, 'null vs base')
for a in ['all', 'all_nocs']:
    rep(C[a], base, f'{a} vs base'); rep(C[a], C['null'], f'{a} vs null')
# callsign-turnover projection
key_c, uk = pd.factorize(cs); cnt = np.zeros((len(uk), 13)); np.add.at(cnt, (key_c, mon), 1)
seen_lomo = (cnt[key_c].sum(1) - cnt[key_c, mon]) > 0
rk = load_dep(['APT', 'FLIGHT_mvt'], 'ranking'); rcs = K(S_(rk.APT), S_(rk.FLIGHT_mvt)); rapt = S_(rk.APT); seen26 = pd.Series(rcs).isin(set(uk)).to_numpy()
print('2026 scored rows with a callsign unseen in 2025: %.1f%% (LOMO within-2025: %.1f%%)' % (100 * (1 - seen26.mean()), 100 * (1 - seen_lomo.mean())))
for a in ['all', 'all_nocs']:
    dsse = (y - C[a]) ** 2 - (y - base) ** 2; proj = 0
    for ap in np.unique(apt):
        k = apt == ap; ms = dsse[k & seen_lomo].mean(); mu = dsse[k & ~seen_lomo].mean(); u = 1 - seen26[rapt == ap].mean()
        proj += k.sum() * ((1 - u) * ms + u * mu)
    r0 = rmse(y, base)
    print(f'  {a}: LOMO {np.sqrt(r0**2 + dsse.sum()/N) - r0:+.3f} -> projected with 2026 turnover {np.sqrt(r0**2 + proj/N) - r0:+.3f}')
    # conservative variant: unseen rows get ZERO gain (only seen rows keep theirs)
    cons = sum(dsse[(apt == ap) & seen_lomo].mean() * (apt == ap).sum() * seen26[rapt == ap].mean() for ap in np.unique(apt))
    print(f'      conservative (unseen rows gain nothing): {np.sqrt(r0**2 + cons/N) - r0:+.3f}')
# idea 4 and idea 1 on top of all_nocs, on the v9~2 base
cand = pd.read_parquet(f'{I}/p2_cand_training.parquet'); P2 = cand.index.to_numpy(); os2 = cand.off_sched.to_numpy(float)
Q = np.load(f'{I}/diag_l_Q.npy')[2]; E = np.load(f'{I}/diag_e_specP.npy')
A = C['all_nocs']; AP2 = A.copy(); AP2[P2] = Q * os2 + (1 - Q) * A[P2]
rep(AP2, A, '#4 on top of #3nocs')
W = AP2.copy(); W[Pi] = np.clip(E[2] * osP + (1 - E[2]) * pb[Pi], 0, None)
CT = AP2.copy(); CT[Pi] = np.clip(E[0] * osP + (1 - E[0]) * pb[Pi], 0, None)
rep(W, CT, '#1 on top (vs refit control)')
rep(CT, base, '#3nocs+#4 (+ctl P) vs base')
rep(W, base, '#3nocs+#4+#1 vs base')
# idea 1 without the single most helpful row
dss = (y - W) ** 2 - (y - CT) ** 2; o = np.argmin(dss); k = np.ones(N, bool); k[o] = False
print(f'#1 without its single best row (month {mon[o]}, {apt[o]}, y {y[o]:.0f}, off_sched {(M-S)[o]:.0f}): 12m {rmse(y[k], W[k]) - rmse(y[k], CT[k]):+.3f}; scored {rmse(y[k & sc], W[k & sc]) - rmse(y[k & sc], CT[k & sc]):+.3f}')
print('peak', round(peak(), 2))
