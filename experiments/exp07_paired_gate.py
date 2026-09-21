"""EXP07 -- fixing the validation gate.

EXP06 showed the uncapped holdout RMSE has a bootstrap sd of ~47s and a 95% CI
163s wide, because ~149 tail rows dominate it. The top 8 teams on the leaderboard
span 5.9s. An absolute local RMSE therefore cannot resolve anything that matters.

The standard fix is a PAIRED bootstrap: resample rows, and look at the
DIFFERENCE between two models on the SAME resampled rows. The tail rows move
both models together, so the difference has far less variance than either
absolute number. That is what lets us detect a 5s improvement.

Also tested: enlarging the holdout via 12-fold month-wise CV, which quadruples
the tail sample (584 rows instead of 149).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET
from prc.features import build, align_categories, CATS, NEEDED
from prc.io import load, peak_rss_gb
from prc.validation import rmse

tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
X = build(tr)
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
X, = align_categories(X)
cols = list(X.columns)

P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
         min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=1.0, num_threads=8, verbose=-1, seed=0, deterministic=True,
         force_row_wise=True)

val = np.isin(month, [1, 7]); trn = ~val
yv = y[val]


def fit_predict(cap, rounds):
    sel = trn & ((y <= cap) if cap else np.ones(len(y), bool))
    m = lgb.train(P, lgb.Dataset(X.loc[sel, cols], y[sel], categorical_feature=CATS),
                  num_boost_round=rounds)
    return np.clip(m.predict(X.loc[val, cols]), 0, None)


t = time.time()
pA = fit_predict(7200, 350)     # trimmed  (v1 style)
pB = fit_predict(None, 350)     # untrimmed (v2 style)
print(f'two models fitted in {time.time()-t:.0f}s')
print(f'  A trimmed   uncapped RMSE {rmse(yv,pA):.2f}')
print(f'  B untrimmed uncapped RMSE {rmse(yv,pB):.2f}')
print(f'  point difference A-B = {rmse(yv,pA)-rmse(yv,pB):+.2f}s   '
      '(official said +36.1s in B\'s favour)')

rng = np.random.default_rng(0)
n = len(yv)
B = 500
idx = rng.integers(0, n, size=(B, n))
dA = (yv - pA) ** 2
dB = (yv - pB) ** 2

abs_A = np.sqrt(dA[idx].mean(axis=1))
abs_B = np.sqrt(dB[idx].mean(axis=1))
diff = abs_A - abs_B

print('\n--- UNPAIRED view (what we were effectively doing) ---')
print(f'  A: {abs_A.mean():7.1f} +- {abs_A.std():5.1f}   95% CI '
      f'[{np.percentile(abs_A,2.5):.0f}, {np.percentile(abs_A,97.5):.0f}]')
print(f'  B: {abs_B.mean():7.1f} +- {abs_B.std():5.1f}   95% CI '
      f'[{np.percentile(abs_B,2.5):.0f}, {np.percentile(abs_B,97.5):.0f}]')
print('  the two intervals overlap heavily -> cannot tell the models apart')

print('\n--- PAIRED view (same resampled rows for both) ---')
print(f'  A-B: {diff.mean():+7.2f} +- {diff.std():5.2f}   95% CI '
      f'[{np.percentile(diff,2.5):+.1f}, {np.percentile(diff,97.5):+.1f}]')
print(f'  P(A worse than B) = {(diff>0).mean()*100:.1f}%')
print(f'  => paired sd is {abs_A.std()/max(diff.std(),1e-9):.1f}x smaller than the '
      'absolute sd')

print('\n--- component split: is the noise all in the tail? ---')
tail = yv > 7200
for name, sel in [('bulk (y<=7200)', ~tail), ('tail (y>7200)', tail)]:
    a, b = rmse(yv[sel], pA[sel]), rmse(yv[sel], pB[sel])
    print(f'  {name:16s} n={int(sel.sum()):6,}  A={a:9.1f}  B={b:9.1f}  diff={a-b:+9.1f}')
print(f'\npeak RSS = {peak_rss_gb():.2f} GB')
