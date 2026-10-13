"""SKEPTIC: does the B class really exclude stamping at LIRF? Stamp share of 2025 LIRF IFR route-D departures, B-class vs not,
by off_sched band; B rows where the OOF was nearly sure of the stamp (CFG4EF-like); NM-matched crossing rows at LIRF (mechanism)."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
V = np.load(R + 'cache/w2/v28s_oof.npz'); y = V['y']; f = V['f']
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'LOBT', 'FLIGHT_RULE_mvt', 'MVT', 'SCHED', 'MVT_ID_mvt'])
u = pd.read_parquet(H + 'h22_union25.parquet', columns=['i', 'onesided', 'stamp_excl'])
d['y'] = y; d['f'] = f; d['osd'] = d.MVT - d.SCHED
d['B'] = False; d.loc[u.i[~u.onesided & u.stamp_excl].to_numpy(), 'B'] = True
d['stamp'] = (d.y - d.osd).abs() <= 120
k = (d.APT == 'LIRF') & d.LOBT.isna() & (d.FLIGHT_RULE_mvt.astype(str) == 'I')
x = d[k]
print('2025 LIRF IFR route-D deps', len(x), 'B-class', int(x.B.sum()))
t = x.groupby([pd.cut(x.osd, [0, 3600, 5400, 7200, 9000, 12000, 1e6]), 'B'], observed=True).agg(n=('stamp', 'size'), stamped=('stamp', 'sum'))
t['share'] = (t.stamped / t.n).round(3); print(t)
print('\n2025 B rows (all airports) where OOF f within 600 s of off_sched and f - lo > 1000:')
uu = pd.read_parquet(H + 'h22_union25.parquet'); bb = uu[~uu.onesided & uu.stamp_excl]
print(bb[((bb.f - bb.off_sched).abs() < 600) & (bb.f - bb.lo > 1000)][['APT', 'FLIGHT_mvt', 'y', 'f', 'lo', 'hi', 'off_sched']])
print('max (f - lo) among B rows:', float((bb.f - bb.lo).max()), ' CFG4EF f - lo = 6072')
# non-B LIRF route-D stamp share where SCHED within [MVT-LOBT_arr +-3606] is impossible to test (no link) -> show v28-like analog:
z = x[(x.osd > 6000) & (x.osd < 7200)]
print('\nLIRF route-D rows with off_sched in (6000,7200): n', len(z), 'stamped', int(z.stamp.sum()), ' of which B', int(z.B.sum()), 'B stamped', int((z.B & z.stamp).sum()))
# NM-matched LIRF crossing rows: class
mm = (d.APT == 'LIRF') & d.LOBT.notna()
dt = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
du = ((dt.MVT_TIME_UTC_mvt - pd.to_timedelta(1800, 's')).dt.normalize() - dt.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()
q = d[mm.to_numpy() & (du >= 1) & (d.osd > 7200).to_numpy()]
print('\nLIRF NM-matched crossing rows (off_sched>7200):', len(q), ' stamped', int(q.stamp.sum()), ' y<=7200 (normal)', int((q.y <= 7200).sum()), ' y>43200', int((q.y > 43200).sum()))
print(' y - (MVT-LOBT) quantiles', (q.y - (q.MVT - q.LOBT)).quantile([.1, .5, .9]).round().tolist())
# T support: p14 of 2025 LIRF zone-P rows with off_sched > 28800 and no crossing
W = np.load(R + 'cache/w2/v24s_oof.npz', allow_pickle=True); Pidx, p14 = W['Pidx'], W['p14']
so = d.osd.to_numpy()[Pidx]; ap = d.APT.to_numpy()[Pidx]; dd = du[Pidx]
s = (ap == 'LIRF') & (so > 28800) & (dd < 1)
print('\nT support: LIRF zone P, off_sched>28800, no crossing: n', int(s.sum()), ' p14 min/median', round(float(p14[s].min()), 3), round(float(np.median(p14[s])), 3))
s2 = (ap == 'LIRF') & (so > 21600) & (dd < 1)
print('   off_sched>21600 no crossing p14 sorted (lowest 8):', np.round(np.sort(p14[s2])[:8], 3).tolist())
