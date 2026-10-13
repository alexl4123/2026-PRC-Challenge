"""ids_06 -- does the position of a DEP row inside its MVT_ID batch carry BLOCK information beyond SCHED/MVT?
(a) tie-break among equal-SCHED neighbours: agreement of ID order with MVT order vs BLOCK order;
(b) neighbour-time offset e = median SCHED of +-5 ID-neighbours - own SCHED, vs residual y - f (v28 OOF);
(c) targeted: zone P rows, day-shift rows (|y - 86400| < 7200), Rome stamped rows."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
d = pd.read_parquet(H + 'cache/hunt/ids_04_sorted.parquet')
run = pd.read_parquet(H + 'cache/hunt/ids_05_runs.parquet')
d = d.merge(run, on='MVT_ID_mvt')
d['APT'] = d.APT.astype(str); d['PHASE_mvt'] = d.PHASE_mvt.astype(str)
d = d.sort_values('MVT_ID_mvt').reset_index(drop=True)
dep = d[d.PHASE_mvt.eq('DEP') & d.src.eq('t')].copy()
# attach OOF (training DEP row order = cache/major/dep_training.parquet)
mt = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt', 'y'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
mt['f'] = z['f']; mt['P'] = z['P']; mt['D'] = z['D']
assert np.allclose(mt.y.values, z['y'])
dep = dep.merge(mt[['MVT_ID_mvt', 'f', 'P', 'D']], on='MVT_ID_mvt', how='left')
dep = dep.sort_values('MVT_ID_mvt').reset_index(drop=True)
dep['y'] = dep.MVT_T - dep.BLOCK
dep['res'] = dep.y - dep.f
dep['off_sched'] = dep.MVT_T - dep.SCHED
print('dep rows', len(dep), 'with f', dep.f.notna().sum())

# (a) adjacent pairs in ID order within run, equal SCHED
same = (dep.run.values[1:] == dep.run.values[:-1]) & (dep.SCHED.values[1:] == dep.SCHED.values[:-1])
dm = np.sign(dep.MVT_T.values[1:] - dep.MVT_T.values[:-1])
db = np.sign(dep.BLOCK.values[1:] - dep.BLOCK.values[:-1])
apt = dep.APT.values[1:]
rows = []
for a in sorted(set(apt)):
    m = same & (apt == a) & (dm != 0) & (db != 0)
    rows.append({'apt': a, 'pairs': m.sum(), 'P(id up & MVT up)': np.mean(dm[m] > 0), 'P(id up & BLOCK up)': np.mean(db[m] > 0),
                 'P(BLOCK order != MVT order)': np.mean(db[m] != dm[m]),
                 'P(id agrees BLOCK | BLOCK!=MVT order)': np.mean(db[m & (db != dm)] > 0)})
print('(a) equal-SCHED adjacent pairs in ID order:\n', pd.DataFrame(rows).round(4).to_string())

# (b) neighbour offset
def nb_median(g, k=5):
    s = g.SCHED.values
    n = len(s)
    out = np.full(n, np.nan)
    pad = np.r_[np.full(k, np.nan), s, np.full(k, np.nan)]
    W = np.stack([pad[i:i + n] for i in range(2 * k + 1) if i != k], axis=1)
    return np.nanmedian(W, axis=1)
dep['nb'] = np.concatenate([nb_median(g) for _, g in dep.groupby('run', sort=False)])
dep['e'] = dep.nb - dep.SCHED
rows = []
for a, g in dep.groupby('APT'):
    g = g[g.f.notna()]
    rows.append({'apt': a, 'n': len(g), 'sd_e': g.e.std(), 'share|e|>3600': np.mean(np.abs(g.e) > 3600),
                 'rho(e,res)': spearmanr(g.e, g.res)[0], 'rho(e,y)': spearmanr(g.e, g.y)[0],
                 'rho(e,off_sched)': spearmanr(g.e, g.off_sched)[0],
                 'rho(|e|,|res|)': spearmanr(np.abs(g.e), np.abs(g.res))[0]})
print('(b) neighbour SCHED offset e vs residual:\n', pd.DataFrame(rows).round(4).to_string())
# (c) targeted classes
dep['dayshift'] = (dep.y - 86400).abs() < 7200
dep['stamp'] = (dep.BLOCK - dep.SCHED).abs() <= 60
for name, m in [('P', dep.P.fillna(False).astype(bool)), ('dayshift', dep.dayshift), ('LIRF P', dep.P.fillna(False).astype(bool) & dep.APT.eq('LIRF'))]:
    g = dep[m]
    print(f'\n(c) {name}: n={len(g)}; e quantiles', np.nanquantile(g.e, [.05, .25, .5, .75, .95]).round(0))
    if name != 'dayshift':
        t = g.assign(istail=g.y > 7200)
        print('   e by tail:', t.groupby('istail').e.describe().round(0).to_string())
        if t.istail.nunique() == 2:
            from sklearn.metrics import roc_auc_score
            print('   AUC(e -> tail)', roc_auc_score(t.istail, t.e), ' AUC(|e|)', roc_auc_score(t.istail, t.e.abs()),
                  ' AUC(stamp, e)', roc_auc_score(t.stamp, t.e) if t.stamp.nunique() == 2 else None)
    else:
        print(g[['APT', 'FLIGHT_mvt', 'SCHED', 'MVT_T', 'BLOCK', 'y', 'f', 'e']].assign(
            SCHED=pd.to_datetime(g.SCHED, unit='s'), MVT_T=pd.to_datetime(g.MVT_T, unit='s'),
            BLOCK=pd.to_datetime(g.BLOCK, unit='s')).to_string())
# LTFM DEP: why lower spearman? sample
g = dep[(dep.APT == 'LTFM') & (dep.mvt_month == '2025-07')]
print('\nLTFM DEP 2025-07: adjacent SCHED-up share', np.mean(np.diff(g.SCHED.values) >= 0))
print(g.head(30).assign(SCHED=pd.to_datetime(g.SCHED.head(30), unit='s'), MVT_T=pd.to_datetime(g.MVT_T.head(30), unit='s'))[['MVT_ID_mvt', 'FLIGHT_mvt', 'SCHED', 'MVT_T', 'y']].to_string())
dep[['MVT_ID_mvt', 'run', 'nb', 'e']].to_parquet(H + 'cache/hunt/ids_06_nb.parquet')
