"""Rome, departures > 20 min late outside zone P: does an EXPLICIT stamp mixture beat v21's blended prediction?
(Exploratory MEASUREMENT.)  mixture = p_stamp * off_sched + (1 - p_stamp) * g,  p_stamp from t01 (regime arm, OOF),
g = LightGBM regressor trained on the NON-stamped candidates only (same columns + regime, leave-one-month-out).
Compared on the same rows with v21's OOF (cache/w2/v21s_oof.npy); also the 50/50 average and a 2-column NNLS
(fitted on the screening months, judged on the held-out months).

    python experiments/tail/t02_rome_stamp_mixture.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import nnls
from prc.config import CACHE, ID
from prc.io import peak_rss_gb
from t01_rome_stamp import regime

t0 = time.time()
from cluster.w2_features import assemble_w2, columns, W2_CATS
X, y, month, groups = assemble_w2(adsb=True, extra=True); cols, _ = columns(groups)
ids = pd.read_parquet(CACHE / 'v11' / 'feat_training.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
z = np.load(CACHE / 'tail' / 't01_rome_stamp_oof.npz'); cand, Pc = z['cand'], z['P'].astype(bool)
keep = ~Pc; rows = cand[keep]; p = z['regime'][keep]; st = z['y'][keep].astype(bool); m = month[rows]
Xc = X.loc[rows, cols].reset_index(drop=True); R = regime('training').set_index(ID).reindex(ids[rows])
for c in R.columns: Xc[c] = R[c].to_numpy()
del X
yc, so = y[rows], Xc['off_sched'].to_numpy()
g = np.zeros(len(yc)); cats = [c for c in W2_CATS if c in cols]
for mm in range(1, 13):
    tr, va = (m != mm) & ~st, m == mm
    mdl = lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
                         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=0, verbose=-1, num_threads=4),
                    lgb.Dataset(Xc.loc[tr], yc[tr], categorical_feature=cats), num_boost_round=600)
    g[va] = mdl.predict(Xc.loc[va])
f21 = np.load(CACHE / 'w2' / 'v21s_oof.npy').astype(float)[rows]
mix = p * so + (1 - p) * g
held = ~np.isin(m, [3, 6, 9, 12]); S = np.isin(m, [1, 7])
w, _ = nnls(np.column_stack([f21, mix])[~held], yc[~held]); nn = np.column_stack([f21, mix]) @ w
r = lambda f, k: np.sqrt(np.mean((yc[k] - f[k]) ** 2))
print(f'Rome candidates outside P: {len(rows):,} (stamped {st.mean():.3f}); NNLS weights v21 {w[0]:.3f} mixture {w[1]:.3f}')
for nm, f in (('v21 OOF', f21), ('stamp mixture', mix), ('50/50', 0.5 * f21 + 0.5 * mix), ('NNLS (screen-fitted)', nn)):
    print(f'  {nm:22s} held-out {r(f, held):8.2f}  Jan+Jul {r(f, S):8.2f}  | stamped rows {r(f, st & held):8.2f}  others {r(f, ~st & held):7.2f}')
tot = np.load(CACHE / 'v11' / 'base.npz')['y']; n = len(tot)
print(f'(Rome candidates outside P are {len(rows) / n:.2%} of all training rows) | {time.time() - t0:.0f}s, peak {peak_rss_gb():.2f} GB')
