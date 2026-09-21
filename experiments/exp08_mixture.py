"""EXP08 -- two-stage mixture model.

EXP07's component split is the whole story:

    training target      bulk RMSE   tail RMSE
    trimmed  @7200          258.2      24158
    untrimmed               296.4      17410

Trimming buys 38s on the 344,270 bulk rows and loses 6,748s on the 149 tail rows.
Neither single model can have both. The RMSE-optimal prediction is the
conditional mean, which factorises:

    E[y|x] = P(T|x) * E[y|T,x] + (1 - P(T|x)) * E[y|~T,x]        T = {y > 7200}

  E[y|~T,x] : L2 regressor trained on bulk rows only -- the good bulk model
  E[y| T,x] : tail rows have y ~= off_sched (median |y-off_sched| = 4s among
              the AOBT_3-missing ones, which are 82% of the tail)
  P(T|x)    : binary classifier. Its CALIBRATION is critical: we multiply it by
              values around 20,000s, so a probability that is wrong by 0.05
              costs 1,000s on that row.

Evaluated uncapped on the Jan+Jul 2025 holdout, compared with a paired bootstrap.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build, align_categories, CATS, NEEDED
from prc.io import load, peak_rss_gb
from prc.validation import rmse, decompose, paired_bootstrap
pd.set_option('display.width', 200)

t0 = time.time()
tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
X = build(tr)
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
off_sched = X.off_sched.to_numpy()
X, = align_categories(X)
COLS = list(X.columns)

val = np.isin(month, [1, 7]); trn = ~val
yv, osv = y[val], off_sched[val]
print(f'train {trn.sum():,}  holdout {val.sum():,} ({int((yv>TAIL_CAP).sum())} tail)  '
      f'{time.time()-t0:.0f}s')

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)

# ---- reference models ------------------------------------------------------
print('\n--- reference models ---')
bulk_sel = trn & (y <= TAIL_CAP)
m_bulk = lgb.train(REG, lgb.Dataset(X.loc[bulk_sel, COLS], y[bulk_sel],
                                    categorical_feature=CATS), num_boost_round=350)
p_bulk = np.clip(m_bulk.predict(X.loc[val, COLS]), 0, None)

m_all = lgb.train(REG, lgb.Dataset(X.loc[trn, COLS], y[trn], categorical_feature=CATS),
                  num_boost_round=350)
p_all = np.clip(m_all.predict(X.loc[val, COLS]), 0, None)
for nm, p in [('A trimmed', p_bulk), ('B untrimmed', p_all)]:
    d = decompose(yv, p)
    print(f'  {nm:14s} RMSE {d["rmse"]:7.2f}  | bulk {d["rmse_bulk"]:6.1f} '
          f'(contrib {d["contrib_bulk"]:6.1f}) | tail {d["rmse_tail"]:8.0f} '
          f'(contrib {d["contrib_tail"]:6.1f})')

# ---- stage 1: P(tail | x) --------------------------------------------------
print('\n--- stage 1: P(tail|x) classifier ---')
is_tail = (y > TAIL_CAP).astype(int)
m_clf = lgb.train(CLF, lgb.Dataset(X.loc[trn, COLS], is_tail[trn],
                                   categorical_feature=CATS), num_boost_round=300)
pt = m_clf.predict(X.loc[val, COLS])
print(f'  positives in train: {is_tail[trn].sum()} of {trn.sum():,} '
      f'({is_tail[trn].mean()*100:.4f}%)')
tail_v = yv > TAIL_CAP
print(f'  holdout: mean p={pt.mean():.6f}  actual rate={tail_v.mean():.6f}  '
      f'(calibration ratio {pt.mean()/max(tail_v.mean(),1e-12):.2f})')
for q in [0.9, 0.99, 0.999, 0.9999]:
    thr = np.quantile(pt, q)
    sel = pt >= thr
    print(f'  top {(1-q)*100:6.2f}% by p (n={int(sel.sum()):6,}): captures '
          f'{int(tail_v[sel].sum()):3d}/{int(tail_v.sum())} tail rows, '
          f'precision {tail_v[sel].mean()*100:6.2f}%')

# ---- stage 2: E[y | tail, x] ----------------------------------------------
print('\n--- stage 2: E[y|tail,x] ---')
tail_trn = trn & (y > TAIL_CAP)
print(f'  tail training rows: {tail_trn.sum()}')
resid = y[tail_trn] - off_sched[tail_trn]
print(f'  y - off_sched on tail rows: median {np.nanmedian(resid):.0f}s  '
      f'mean {np.nanmean(resid):.0f}s')
# E[y|tail] anchored on off_sched, falling back to the tail mean where off_sched
# is unusable (missing or non-positive).
tail_mean = y[tail_trn].mean()
e_tail = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, tail_mean)
print(f'  anchored estimate on holdout tail rows: RMSE '
      f'{rmse(yv[tail_v], e_tail[tail_v]):.0f}s  (vs {rmse(yv[tail_v], p_all[tail_v]):.0f} '
      'for the untrimmed regressor)')

# ---- combine ---------------------------------------------------------------
print('\n--- mixture: p*E[y|tail] + (1-p)*E[y|bulk] ---')
results = {}
for shrink in [0.0, 0.25, 0.5, 0.75, 1.0]:
    p_eff = pt * shrink
    mix = np.clip(p_eff * e_tail + (1 - p_eff) * p_bulk, 0, None)
    d = decompose(yv, mix)
    results[shrink] = mix
    print(f'  shrink={shrink:4.2f}  RMSE {d["rmse"]:7.2f}  | bulk {d["rmse_bulk"]:6.1f} '
          f'| tail {d["rmse_tail"]:8.0f}')

best = min(results, key=lambda s: rmse(yv, results[s]))
print(f'\nbest shrink = {best}')
for nm, p in [('vs A trimmed', p_bulk), ('vs B untrimmed', p_all)]:
    r = paired_bootstrap(yv, results[best], p)
    print(f'  mixture {nm:16s}: diff {r["diff"]:+8.2f} +- {r["sd"]:5.2f}  '
          f'95% CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]  P(mixture worse)={r["p_a_worse"]*100:.1f}%')
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
