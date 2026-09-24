"""s05 -- season-proximity sample weights for the bulk regressor (new idea, never tested).

The month x airport interaction flips sign (northern airports winter-heavy / de-icing,
southern airports summer-heavy / congestion; 2026-09-23 log §14.4) and the model carries no
month feature.  For the final 2026 fit, Jan 2026 would up-weight Dec/Jan/Feb 2025 and Jul 2026
would up-weight Jun/Jul/Aug 2025.  In LOMO the held-out month's neighbours play that role.

weight(k | target m) = floor + (1 - floor) * exp(-d(k,m)^2 / (2 sigma^2)),  d = circular month distance.
Arms: none | sigma 1.5, floor 0.25 | sigma 3, floor 0.25.  Same LightGBM, same fixed training
subsample per fold, bulk domain (outside P, y <= 7200) for training, evaluation on the held-out
month's outside-P rows (all y and bulk), paired bootstrap vs 'none'.

usage: python s05_season_weight.py 1,7 [N_SUB]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE
from prc.validation import paired_bootstrap
from prc.io import peak_rss_gb

FOLDS = [int(x) for x in sys.argv[1].split(',')]
N_SUB = int(sys.argv[2]) if len(sys.argv) > 2 else 700_000
X = pd.read_parquet(CACHE / 'altfam_X.parquet')
y = np.load(CACHE / 'altfam_y.npy').astype(float)
mon = np.load(CACHE / 'altfam_month.npy').astype(int)
oa = X.off_aobt3.to_numpy(); so = X.off_sched.to_numpy()
P = ~np.isfinite(oa) & (so > 7200)
cat = [c for c in X.columns if str(X[c].dtype) == 'category']
PAR = dict(objective='regression', learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, cat_l2=10, num_threads=3, verbose=-1, seed=0)
ARMS = {'none': None, 's1.5': (1.5, 0.25), 's3': (3.0, 0.25)}
rng = np.random.default_rng(0)
out = []
for m in FOLDS:
    tr_all = np.flatnonzero((mon != m) & ~P & (y <= 7200))
    tr = np.sort(rng.choice(tr_all, min(N_SUB, len(tr_all)), replace=False))
    te = np.flatnonzero((mon == m) & ~P)
    dist = np.minimum(np.abs(mon[tr] - m), 12 - np.abs(mon[tr] - m))
    preds = {}
    for nm, a in ARMS.items():
        t0 = time.time()
        w = None if a is None else a[1] + (1 - a[1]) * np.exp(-dist ** 2 / (2 * a[0] ** 2))
        b = lgb.train(PAR, lgb.Dataset(X.iloc[tr], y[tr], weight=w, categorical_feature=cat), 300)
        preds[nm] = b.predict(X.iloc[te])
        print(f'fold {m} {nm}: {time.time()-t0:.0f} s', flush=True)
    yt = y[te]; bulk = yt <= 7200
    for nm in ARMS:
        if nm == 'none':
            continue
        pa = paired_bootstrap(yt, preds[nm], preds['none']); pbk = paired_bootstrap(yt[bulk], preds[nm][bulk], preds['none'][bulk])
        r = lambda p, k: np.sqrt(np.mean((yt[k] - p[k]) ** 2))
        allk = np.ones(len(yt), bool)
        print(f'  month {m} {nm} vs none: outside-P all {r(preds[nm], allk):.2f} vs {r(preds["none"], allk):.2f} '
              f'd {pa["diff"]:+.3f} [{pa["lo"]:+.3f},{pa["hi"]:+.3f}] | bulk d {pbk["diff"]:+.3f} [{pbk["lo"]:+.3f},{pbk["hi"]:+.3f}]', flush=True)
        out.append((m, nm, pa['diff'], pa['lo'], pa['hi'], pbk['diff']))
    np.savez(CACHE / 'strategy' / f's05_m{m}.npz', te=te, **preds)
print(pd.DataFrame(out, columns=['month', 'arm', 'd_all', 'lo', 'hi', 'd_bulk']).round(3).to_string(index=False))
print('peak', round(peak_rss_gb(), 2), 'GB')
