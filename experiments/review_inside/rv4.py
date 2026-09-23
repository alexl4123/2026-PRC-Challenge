import sys; I = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/ideas_inside'; sys.path.insert(0, I)
from common import *
d = load_dep(['APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
mon = pd.to_datetime(M, unit='s').month.to_numpy(); apt = S_(d.APT); cs = K(apt, S_(d.FLIGHT_mvt)); sc = np.isin(mon, [1, 7]); N = len(y)
P = np.isnan(A3) & (M - S > 7200); Pi = np.where(P)[0]; osP = (M - S)[Pi]
base = np.load('base_v9s2.npy'); pb = np.load(f'{R}/exp62_pb.npy').astype(float)
core = base + np.load('rv_s_core_v9s2.npy'); nocs = base + np.load('rv_s_all_nocs_v9s2.npy')
for nm, a in [('core (no callsign TE, no same-day arrival rates) vs base', core), ('all_nocs vs base', nocs)]:
    b1 = boot(y, a, base, 300); b2 = boot(y[sc], a[sc], base[sc], 300)
    dm = [rmse(y[mon == m], a[mon == m]) - rmse(y[mon == m], base[mon == m]) for m in range(1, 13)]
    print(f'{nm}: 12m {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}] scored {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}] months better {sum(v < 0 for v in dm)}/12')
    print('   by airport 12m:', {ap: round(float(np.sqrt(rmse(y, base)**2 + (((y - a)**2 - (y - base)**2) * (apt == ap)).sum() / N) - rmse(y, base)), 2) for ap in ['LIRF', 'LFPG', 'EGLL', 'LTFM', 'LEBL']})
key_c, uk = pd.factorize(cs); cnt = np.zeros((len(uk), 13)); np.add.at(cnt, (key_c, mon), 1); seen = (cnt[key_c].sum(1) - cnt[key_c, mon]) > 0
rk = load_dep(['APT', 'FLIGHT_mvt'], 'ranking'); rcs = K(S_(rk.APT), S_(rk.FLIGHT_mvt)); rapt = S_(rk.APT); s26 = pd.Series(rcs).isin(set(uk)).to_numpy()
dsse = (y - core) ** 2 - (y - base) ** 2; proj = sum((apt == ap).sum() * (s26[rapt == ap].mean() * dsse[(apt == ap) & seen].mean() + (1 - s26[rapt == ap].mean()) * dsse[(apt == ap) & ~seen].mean()) for ap in np.unique(apt))
r0 = rmse(y, base); print(f'core: projected with 2026 callsign turnover {np.sqrt(r0**2 + proj/N) - r0:+.3f}')
# idea 1 concentration in the scored months
E = np.load(f'{I}/diag_e_specP.npy'); v6, v7, bl = preds()
def mix(p): new = v7.copy(); new[Pi] = np.clip(p * osP + (1 - p) * pb[Pi], 0, None); return new
W, CT = mix(E[2]), mix(E[0]); ds = ((y - W) ** 2 - (y - CT) ** 2)
for m in (1, 7):
    k = mon == m; o = np.argsort(ds[k])[:3]; tot = ds[k].sum()
    print(f'idea 1, month {m}: total dSSE {tot:.3e}; best 3 rows {ds[k][o].sum():.3e} ({ds[k][o].sum()/tot:.0%}); rows: ' + '; '.join(f'{apt[k][i]} y {y[k][i]:.0f} os {(M-S)[k][i]:.0f} p {E[0][np.searchsorted(Pi, np.where(k)[0][i])]:.2f}->{E[2][np.searchsorted(Pi, np.where(k)[0][i])]:.2f}' for i in o))
print('peak', round(peak(), 2))
