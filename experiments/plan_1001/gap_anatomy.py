"""GAP-ANATOMY (read-only): where v24's error lives on 2025 OOF and in the expected 2026 mix; cheap label-free screens.
Reads cache only; writes nothing into the project (prints; optional npz into the scratchpad).
"""
import os, sys, time
ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/experiments/drivers_0926')
import numpy as np, pandas as pd
from prc.config import CACHE
from d02_decompose import mechanism, adsb_status
pd.set_option('display.width', 230); pd.set_option('display.max_rows', 200)
t0 = time.time()
SCREEN = [3, 6, 9, 12]

z = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, f, month, P, D = z['y'], z['f'], z['month'], z['P'], z['D']
v21 = np.load(CACHE / 'w2' / 'v21s_oof.npy').astype(float)
r = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet')
k = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_ranking.parquet')
assert np.allclose(r.y.to_numpy(float), y, equal_nan=True)
r['mech'] = mechanism(r); r['adsb'] = adsb_status(r); k['adsb'] = adsb_status(k)
day = r.mech.to_numpy() == 'day'
held = ~np.isin(month, SCREEN); sc = np.isin(month, [1, 7]); allr = np.ones(len(y), bool)
e24 = (y - f) ** 2; e21 = (y - v21) ** 2
R = lambda e, m: np.sqrt(e[m].mean())
print(f'v24 OOF: held {R(e24, held):.3f} scored {R(e24, sc):.3f} 12m {R(e24, allr):.3f} | w/o day: 12m {R(e24, ~day):.2f} '
      f'scored {R(e24, sc & ~day):.2f} | v21: 12m {R(e21, allr):.3f} scored {R(e21, sc):.3f} w/o day 12m {R(e21, ~day):.2f}')

# ---------------- A. mechanism shares v24 vs v21
print('\n== A. mechanism (12m), v24 vs v21')
for nm, scope in (('12m', allr), ('Jan+Jul', sc)):
    tot = e24[scope].sum(); n = scope.sum(); rm = np.sqrt(tot / n)
    g = pd.DataFrame({'m': r.mech.to_numpy()[scope], 'e': e24[scope], 'e21': e21[scope]}).groupby('m').agg(
        rows=('e', 'size'), sse=('e', 'sum'), rmse=('e', lambda v: np.sqrt(v.mean())), rmse21=('e21', lambda v: np.sqrt(v.mean())))
    g['sse_pct'] = 100 * g.sse / tot; g['zerofix'] = rm - np.sqrt((tot - g.sse) / n)
    print(nm, f'RMSE {rm:.2f}'); print(g.drop(columns='sse').sort_values('sse_pct', ascending=False).round(2).to_string())

# ---------------- A2. expected 2026, v24 cell MSE x 2026 counts (day-scale excluded), as t00
def cell(x):
    sob = pd.cut(x.off_sched, [-np.inf, 0, 900, 1800, 3600, 7200, np.inf], labels=False).fillna(-1).astype(int).astype(str)
    return x.apt.astype(str) + '|' + x.route.astype(str) + '|' + x.adsb.astype(str), sob
c25, b25 = cell(r); c26, b26 = cell(k)
fine25, fine26 = (c25 + '|' + b25).to_numpy(), (c26 + '|' + b26).to_numpy()
def expected(ee, mask=~day):
    mse = pd.Series(ee[mask]).groupby(fine25[mask]).mean(); msec = pd.Series(ee[mask]).groupby(c25.to_numpy()[mask]).mean()
    m26 = pd.Series(fine26).map(mse); miss = m26.isna().to_numpy()
    m26[miss] = pd.Series(c26.to_numpy()[miss]).map(msec).to_numpy()
    return m26.fillna(m26.mean()).to_numpy()
x24, x21 = expected(e24), expected(e21)
exp = pd.DataFrame({'apt': k.apt.astype(str).to_numpy(), 'route': k.route.astype(str).to_numpy(), 'adsb': k.adsb.to_numpy(),
                    'month': k.month.to_numpy(), 'e': x24, 'e21': x21})
tot = exp.e.sum()
print(f'\n== A2. EXPECTED 2026 day-free: v24 {np.sqrt(exp.e.mean()):.2f}  v21 {np.sqrt(exp.e21.mean()):.2f}')
ta = exp.groupby(['apt', 'route', 'adsb']).agg(rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot),
        rmse=('e', lambda v: np.sqrt(v.mean())), rmse21=('e21', lambda v: np.sqrt(v.mean()))).sort_values('sse_pct', ascending=False)
ta['cum'] = ta.sse_pct.cumsum(); print(ta.head(14).round(2).to_string())
tb = exp.groupby('apt').agg(rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot), rmse=('e', lambda v: np.sqrt(v.mean())),
        rmse21=('e21', lambda v: np.sqrt(v.mean())))
print(tb.sort_values('sse_pct', ascending=False).round(2).to_string())
tm = exp.groupby('month').agg(rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot), rmse=('e', lambda v: np.sqrt(v.mean())))
print(tm.round(2).to_string())
E26 = exp.e.mean()
for live in (244.2213, 219.64, 233.81, 244.19):
    print(f'live {live}: implied non-modelled MSE {live**2 - E26:,.0f} s^2/row = {100*(live**2-E26)/live**2:.1f}% of live SSE')

# ---------------- B. calibration: month-OOF binned residual correction (label-free at serve time)
print('\n== B. calibration by prediction bin (month-OOF bin means of y - f, fitted on rows y <= 12h)')
fitok = y <= 43200
def binned_corr(keys, nb_name):
    corr = np.zeros(len(y))
    for m in range(1, 13):
        tr = (month != m) & fitok; te = month == m
        mu = pd.Series((y - f)[tr]).groupby(keys[tr]).agg(['mean', 'size'])
        mu = mu['mean'].where(mu['size'] >= 200, 0.0)
        corr[te] = pd.Series(keys[te]).map(mu).fillna(0).to_numpy()
    g = np.clip(f + corr, 0, None); eg = (y - g) ** 2
    out = {s: R(eg, m) - R(e24, m) for s, m in (('held', held), ('scored', sc), ('Jan', month == 1), ('Jul', month == 7), ('12m', allr))}
    print(f'  {nb_name:38s} ' + ' '.join(f'{s} {v:+.3f}' for s, v in out.items()))
    return corr
route = np.where(P, 'P', np.where(D, 'D', 'A'))
qb = lambda v, n: np.asarray(pd.qcut(v, n, labels=False, duplicates='drop')).astype(int).astype(str)
fb40 = qb(f, 40)
apt = r.apt.astype(str).to_numpy().astype(str)
binned_corr(np.char.add(route, fb40), 'route x 40 f-bins')
binned_corr(np.char.add(np.char.add(route, apt), qb(f, 20)), 'route x apt x 20 f-bins')
print('  mean residual (y-f) by f decile, A/B/C outside P, y<=12h:')
A = ~D & ~P
dd = pd.DataFrame({'fb': pd.qcut(f[A], 10, labels=False), 'r': (y - f)[A], 'ok': fitok[A]})
print(dd[dd.ok].groupby('fb').r.agg(['mean', 'size']).round(1).T.to_string())

# ---------------- C. member disagreement
print('\n== C. member disagreement (A/B/C outside P)')
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
def oof(folder, fam, seeds, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o
M = {fm: oof(CACHE / 'cv' / 'adsb2', fm, [0, 1, 2]) for fm in FAMS}
M['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2]) for v in POOL], axis=0)
W5 = ['seq', 'routed', 'w3', 'xgboost', 'perapt']
stk = np.column_stack([M[m] for m in W5])
sd = stk.std(1); rng = stk.max(1) - stk.min(1)
dq = pd.qcut(sd[A], 10, labels=False)
t = pd.DataFrame({'q': dq, 'r': (y - f)[A], 'e': e24[A], 'sd': sd[A], 'ok': fitok[A] & ~day[A]})
tt = t[t.ok].groupby('q').agg(sd=('sd', 'median'), mean_r=('r', 'mean'), rmse=('e', lambda v: np.sqrt(v.mean())), n=('e', 'size'),
                             sse=('e', 'sum'))
tt['sse_pct'] = 100 * tt.sse / tt.sse.sum(); print(tt.drop(columns='sse').round(2).to_string())
sdq = np.full(len(y), 'x', dtype=object); sdq[A] = dq.astype(str)
binned_corr(np.char.add(np.char.add(route, apt), np.char.add(qb(f, 20), sdq.astype(str))), 'route x apt x 20 f-bins x sd-decile')
# member-level: who is right when they disagree? residual corr of each member with y - f
print('  member alone RMSE held-out outside P, and corr(member - f, y - f) on A outside P (y<=12h):')
ok = A & held & fitok
for m in FAMS + ['seq']:
    print(f'    {m:8s} rmse {np.sqrt(((y - M[m]) ** 2)[A & held].mean()):8.2f}  corr {np.corrcoef((M[m] - f)[ok], (y - f)[ok])[0, 1]:+.4f}')
# disagreement between seq and the tree consensus
tree = (M['routed'] + M['xgboost'] + M['perapt'] + M['mix'] + M['lintree']) / 5
gap = (M['seq'] - tree)
print('  corr(seq - trees, y - f) A held-out y<=12h by airport:')
for a_ in sorted(set(apt)):
    mm = ok & (apt == a_)
    print(f'    {a_} n={mm.sum():7d} corr {np.corrcoef(gap[mm], (y - f)[mm])[0, 1]:+.4f}  sd(gap) {gap[mm].std():6.1f}')

# ---------------- D. route D outside P and zone P
print('\n== D. SSE share by route x airport (12m and scored), v24')
for nm, scope in (('12m', allr), ('scored', sc), ('12m w/o day', ~day)):
    tot = e24[scope].sum()
    g = pd.DataFrame({'route': route[scope], 'apt': apt[scope], 'e': e24[scope]}).groupby(['route']).e.agg(['size', 'sum'])
    g['sse_pct'] = 100 * g['sum'] / tot; g['rmse'] = np.sqrt(g['sum'] / g['size'])
    print(nm); print(g.drop(columns='sum').round(2).to_string())
DnP = D & ~P
g = pd.DataFrame({'apt': apt[DnP], 'mech': r.mech.to_numpy()[DnP], 'e': e24[DnP]})
tot = e24[~day].sum()
gg = g[g.mech != 'day'].groupby(['apt', 'mech']).e.agg(['size', 'sum']); gg['pct_of_12m_dayfree'] = 100 * gg['sum'] / tot
print('route D outside P, day-free, by apt x mech (share of 12m day-free SSE):')
print(gg[gg.pct_of_12m_dayfree > 0.15].drop(columns='sum').sort_values('pct_of_12m_dayfree', ascending=False).round(3).to_string())

# ---------------- E. hot airport-days (label-free definition, 2025 thresholds applied to 2026)
print('\n== E. hot airport-days')
def daily(x):
    return x.groupby(['apt', 'date_loc'], observed=True).agg(n=('off_sched', 'size'), so_med=('off_sched', 'median'),
                                                            shD=('route', lambda v: np.isin(v, ['D', 'P']).mean()))
d25, d26 = daily(r), daily(k)
thr = d25.groupby(level=0).agg(so99=('so_med', lambda v: v.quantile(.99)), d99=('shD', lambda v: v.quantile(.99)))
def hotflag(dd):
    t_ = thr.reindex(dd.index.get_level_values(0))
    return ((dd.so_med.to_numpy() > t_.so99.to_numpy()) | (dd.shD.to_numpy() > t_.d99.to_numpy()))
d25['hot'] = hotflag(d25); d26['hot'] = hotflag(d26)
h25 = r.join(d25[['hot']], on=['apt', 'date_loc']).hot.fillna(False).to_numpy().astype(bool)
h26 = k.join(d26[['hot']], on=['apt', 'date_loc']).hot.fillna(False).to_numpy().astype(bool)
print(f'2025 hot days {d25.hot.sum()} rows {h25.mean()*100:.2f}% | 2026 hot days {d26.hot.sum()} rows {h26.mean()*100:.2f}%')
for nm, scope in (('12m', allr), ('12m w/o day', ~day)):
    tot = e24[scope].sum()
    print(f'  {nm}: hot-day rows carry {100*e24[scope & h25].sum()/tot:.2f}% of SSE; RMSE hot {R(e24, scope & h25):.1f} vs other {R(e24, scope & ~h25):.1f}'
          f'; v21 hot {R(e21, scope & h25):.1f}')
for a_ in ['EHAM', 'LFPG', 'LSZH', 'EDDF', 'EGLL', 'LTFM', 'LIRF', 'EDDM']:
    print(f'    {a_}: 2025 hot rows {(h25 & (apt == a_)).sum():6d} RMSE hot {R(e24, ~day & h25 & (apt == a_)):7.1f} other {R(e24, ~day & ~h25 & (apt == a_)):7.1f}'
          f' | 2026 hot rows {(h26 & (k.apt.astype(str).to_numpy() == a_)).sum():6d}')
# expected 2026 with hot-day refinement of the cells
fine25h = np.char.add(fine25.astype(str), np.where(h25, '|H', '|n')); fine26h = np.char.add(fine26.astype(str), np.where(h26, '|H', '|n'))
mse = pd.Series(e24[~day]).groupby(fine25h[~day]).agg(['mean', 'size']); mse = mse['mean'].where(mse['size'] >= 30)
m26 = pd.Series(fine26h).map(mse); miss = m26.isna().to_numpy(); m26[miss] = x24[miss]
print(f'  expected 2026 day-free with hot-day cells: {np.sqrt(m26.mean()):.2f} (without hot split {np.sqrt(x24.mean()):.2f}); '
      f'hot-day rows share of expected SSE {100*m26[h26].sum()/m26.sum():.1f}%')
print(f'\n{time.time()-t0:.0f} s')
