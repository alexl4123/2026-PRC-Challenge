"""s04 -- tabular foundation model probe: TabPFN v2 (open weights, CPU) vs LightGBM.

One airport, the bulk regressor's domain (outside P, y <= 7200).  Test = N_TEST random rows
of the held-out month; train = random subsample of the other 11 months of that airport.

Arms on the SAME test rows:
  tabpfn(n)      TabPFN v2 regressor, context = n training rows
  lgb(n)         LightGBM on the same n rows
  lgb(full)      LightGBM on all of this airport's other-month rows (bulk domain)
  v9~            the production stand-in OOF (all airports, 11 months, LGB+NN blend)
  blends         0.5/0.5 and best-weight (fitted on half the test rows, scored on the other half)

TabPFN is installed OUTSIDE the shared venv (scratchpad --target dir, PYTHONPATH), v2 weights
(the only ungated version; v2.5+ are gated / non-commercial).

usage: PYTHONPATH=<scratch>/tfm python s04_tabpfn_probe.py APT MONTH N1,N2 [N_TEST]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb, torch
from prc.config import CACHE
from prc.io import peak_rss_gb

torch.set_num_threads(3)
APT_ = sys.argv[1]; MON = int(sys.argv[2]); NS = [int(x) for x in sys.argv[3].split(',')]
N_TEST = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
rng = np.random.default_rng(0)

X = pd.read_parquet(CACHE / 'altfam_X.parquet')
y = np.load(CACHE / 'altfam_y.npy').astype(float)
mon = np.load(CACHE / 'altfam_month.npy')
v9 = np.load(CACHE / 'major' / 'v9s_oof.npy')
oa = X.off_aobt3.to_numpy(); so = X.off_sched.to_numpy()
P = ~np.isfinite(oa) & (so > 7200)
dom = (X.APT.astype(str).to_numpy() == APT_) & ~P & (y <= 7200)
te_all = np.flatnonzero(dom & (mon == MON)); tr_all = np.flatnonzero(dom & (mon != MON))
te = np.sort(rng.choice(te_all, min(N_TEST, len(te_all)), replace=False))
cat_cols = [c for c in X.columns if str(X[c].dtype) == 'category']
X = X.drop(columns=['APT'])
cat_cols = [c for c in cat_cols if c != 'APT']
Xn = X.copy()
for c in cat_cols:                         # ordinal codes for TabPFN (NaN for missing)
    Xn[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
cat_idx = [list(X.columns).index(c) for c in cat_cols]
yt = y[te]
res = {}


def rm(p):
    return float(np.sqrt(np.mean((yt - p) ** 2)))


P_LGB = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, num_threads=3, verbose=-1, seed=0, max_cat_to_onehot=8, cat_smooth=10)
t0 = time.time()
full = lgb.train(P_LGB, lgb.Dataset(X.iloc[tr_all], y[tr_all], categorical_feature=cat_cols), 600)
res['lgb(full n=%d)' % len(tr_all)] = full.predict(X.iloc[te]); print('lgb full', round(time.time() - t0), 's', flush=True)
res['v9~ (production stand-in)'] = v9[te]

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
for n in NS:
    tr = np.sort(rng.choice(tr_all, min(n, len(tr_all)), replace=False))
    small_params = dict(P_LGB, num_leaves=15, min_data_in_leaf=10)
    m = lgb.train(small_params, lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols), 300)
    res[f'lgb(n={n})'] = m.predict(X.iloc[te])
    t0 = time.time()
    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', n_estimators=4,
                                                     ignore_pretraining_limits=True, categorical_features_indices=cat_idx)
    reg.fit(Xn.iloc[tr].to_numpy(), y[tr])
    pr = np.concatenate([reg.predict(Xn.iloc[te[i:i + 500]].to_numpy()) for i in range(0, len(te), 500)])
    res[f'tabpfn(n={n})'] = pr
    print(f'tabpfn n={n}: {time.time()-t0:.0f} s  rmse {rm(pr):.1f}  (lgb same n {rm(res[f"lgb(n={n})"]):.1f})', flush=True)

print(f'\n=== {APT_}, held-out month {MON}, {len(te)} test rows (bulk domain), sd(y) {yt.std():.1f}')
base = res['lgb(full n=%d)' % len(tr_all)]
for k, p in res.items():
    print(f'  {k:28s} rmse {rm(p):7.1f}   corr(resid, lgb-full resid) {np.corrcoef(yt - p, yt - base)[0,1]:.3f}')
h = np.arange(len(te)) % 2 == 0
for k in [k for k in res if k.startswith('tabpfn')]:
    for ref in [k2 for k2 in res if k2.startswith('lgb(full') or k2.startswith('v9')]:
        a, b = res[ref], res[k]
        ws = np.linspace(0, 1, 21)
        w = ws[np.argmin([np.mean((yt[h] - ((1 - w_) * a[h] + w_ * b[h])) ** 2) for w_ in ws])]
        print(f'  blend {ref} + {k}: 50/50 {rm(0.5*a+0.5*b):.1f} | w*={w:.2f} (fit on even rows) odd-rows rmse '
              f'{np.sqrt(np.mean((yt[~h] - ((1-w)*a[~h] + w*b[~h]))**2)):.1f} vs {ref} alone {np.sqrt(np.mean((yt[~h]-a[~h])**2)):.1f}')
np.savez(CACHE / 'strategy' / f's04_{APT_}_m{MON}.npz', te=te, **{k.replace(' ', '_'): v for k, v in res.items()})
print('peak', round(peak_rss_gb(), 2), 'GB')
