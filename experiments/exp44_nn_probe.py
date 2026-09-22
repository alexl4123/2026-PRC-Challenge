"""EXP44 -- entity-embedding network probe: does it train, and how fast?

Same protocol as exp42 (CatBoost probe): 300k training rows, validate on month 7
-- the hardest month -- and refit LightGBM on the identical subsample so the
comparison is like for like. Per-epoch validation RMSE is printed so the shape
of the learning curve is visible; a network that is still improving at the last
epoch needs a longer budget, one that has turned up needs fewer.

The number that matters is not the solo RMSE. It is corr(residual_nn,
residual_lgb): a tree ensemble and an MLP are wrong in structurally different
ways, and if that correlation is well below the ~0.94 CatBoost shows, the
network earns its place in a blend even at a worse solo score.

Runtime ~20 min, peak ~4 GB.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np

from prc.config import TAIL_CAP
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.models.altfam import FULL, config_cols, design, numeric_design
from prc.models.embed_nn import fit_predict
from prc.validation import rmse

THREADS = 4
SEED = 0
N_SUB = 300_000
VAL_MONTH = 7

t0 = time.time()
X, y, month, groups = design()
cols = config_cols(groups, FULL)
cats = [c for c in CATS if c in cols]

va = month == VAL_MONTH
tr = (~va) & (y <= TAIL_CAP)
rng = np.random.default_rng(SEED)
sub = rng.choice(np.flatnonzero(tr), N_SUB, replace=False)
tr_mask = np.zeros(len(y), bool)
tr_mask[sub] = True
yva = y[va]
bulk = yva <= TAIL_CAP
print(f'train {int(tr_mask.sum()):,}  val {int(va.sum()):,} month {VAL_MONTH} '
      f'({int(bulk.sum()):,} bulk)   ({time.time() - t0:.0f}s)', flush=True)

# LightGBM on the identical subsample.
ts = time.time()
m = lgb.train(dict(objective='regression', metric='rmse', learning_rate=0.05,
                   num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   num_threads=THREADS, verbose=-1, seed=SEED, deterministic=True,
                   force_row_wise=True),
              lgb.Dataset(X.loc[tr_mask, cols], y[tr_mask], categorical_feature=cats),
              num_boost_round=350)
p_lgb = np.clip(m.predict(X.loc[va, cols]), 0, None)
print(f'LightGBM 350 rounds  {time.time() - ts:5.0f}s   bulk RMSE '
      f'{rmse(yva[bulk], p_lgb[bulk]):7.2f}', flush=True)
del m

ts = time.time()
codes, cards, Z, cat_cols, num_cols, _ = numeric_design(X[cols], cols, fit_mask=tr_mask)
print(f'numeric design: {len(cat_cols)} cat + {len(num_cols)} num, Z '
      f'{Z.nbytes / 1e6:.0f} MB  ({time.time() - ts:.0f}s)', flush=True)


def make_hook(tag):
    state = {'t': time.time()}

    def hook(ep, pred):
        r = rmse(yva[bulk], pred[bulk])
        c = np.corrcoef(yva[bulk] - pred[bulk], yva[bulk] - p_lgb[bulk])[0, 1]
        print(f'    {tag} epoch {ep:2d}  bulk RMSE {r:7.2f}  corr-resid vs lgb '
              f'{c:.4f}  ({time.time() - state["t"]:.0f}s)', flush=True)
        state['t'] = time.time()
    return hook


for tag, kw in [('small ', dict(hidden=(256, 128), epochs=10, lr=3e-3)),
                ('wide  ', dict(hidden=(512, 256, 128), epochs=10, lr=3e-3))]:
    ts = time.time()
    print(f'\nNN {tag.strip()} {kw}', flush=True)
    p_nn = fit_predict(codes, Z, y, tr_mask, va, seed=SEED, threads=THREADS,
                       eval_hook=make_hook(tag), **kw)
    print(f'  final bulk RMSE {rmse(yva[bulk], p_nn[bulk]):7.2f}   '
          f'({time.time() - ts:.0f}s)', flush=True)
    for w in (0.2, 0.3, 0.4, 0.5):
        b = (1 - w) * p_lgb + w * p_nn
        print(f'    blend {1 - w:.1f}*lgb + {w:.1f}*nn -> bulk RMSE '
              f'{rmse(yva[bulk], b[bulk]):7.2f}', flush=True)

print(f'\npeak RSS {peak_rss_gb():.2f} GB   total {time.time() - t0:.0f}s')
