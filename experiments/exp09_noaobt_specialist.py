"""EXP09 -- a dedicated model for the AOBT_3-missing subpopulation.

EXP08's global classifier is well calibrated but blunt: the tail is 0.025% of
training rows, so even a good classifier leaves P(T|x) small on most tail rows
and the mixture barely moves toward the tail estimate.

But the tail is not spread evenly. The whole _flt block goes missing together
(FLIGHT_ID unmatched), and in that subpopulation the tail rate is ~2%, eighty
times the global rate:

    all DEP rows            2,085,047   584 tail   0.028%
    AOBT_3 missing             22,470   481 tail   2.14%     <- 82% of the tail
    AOBT_3 present          2,062,577   103 tail   0.005%

So split the problem in two and give each half a model suited to it:

  AOBT_3 PRESENT : anchor on off_aobt3, correction sd 385s. Ordinary regression.
  AOBT_3 MISSING : no _flt columns at all. The question is binary -- did the
                   airport stamp BLOCK_TIME at SCHED_TIME (so y == off_sched)
                   or not? Fit that classifier on 22k rows where positives are
                   2% rather than 0.025%.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build, align_categories, CATS, NEEDED
from prc.io import load, peak_rss_gb
from prc.validation import rmse, decompose, paired_bootstrap
pd.set_option('display.width', 210)

t0 = time.time()
tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
X = build(tr)
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
osd = X.off_sched.to_numpy()
miss = X.aobt_missing.to_numpy().astype(bool)
X, = align_categories(X)
COLS = list(X.columns)
# features actually informative when the _flt block is absent
COLS_M = ['APT', 'RUNWAY_mvt', 'PIER', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
          'off_sched', 'log_off_sched', 'hour', 'dow']
CATS_M = [c for c in CATS if c in COLS_M]

val = np.isin(month, [1, 7]); trn = ~val
yv = y[val]
print(f'holdout {val.sum():,} rows, {int((yv>TAIL_CAP).sum())} tail, '
      f'{int(miss[val].sum()):,} with AOBT_3 missing')

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)

# ---------- baseline: one bulk regressor everywhere -------------------------
bulk_sel = trn & (y <= TAIL_CAP)
m_bulk = lgb.train(REG, lgb.Dataset(X.loc[bulk_sel, COLS], y[bulk_sel],
                                    categorical_feature=CATS), num_boost_round=350)
p_bulk = np.clip(m_bulk.predict(X.loc[val, COLS]), 0, None)
m_all = lgb.train(REG, lgb.Dataset(X.loc[trn, COLS], y[trn], categorical_feature=CATS),
                  num_boost_round=350)
p_all = np.clip(m_all.predict(X.loc[val, COLS]), 0, None)

# ---------- the split ------------------------------------------------------
print('\n--- tail concentration by subpopulation (training rows) ---')
for nm, sel in [('AOBT_3 present', trn & ~miss), ('AOBT_3 missing', trn & miss)]:
    t = (y[sel] > TAIL_CAP)
    print(f'  {nm:16s} n={sel.sum():9,}  tail {int(t.sum()):4d}  ({t.mean()*100:.3f}%)')

# Target for the specialist: did BLOCK land on SCHED (so y == off_sched)?
on_sched = np.abs(y - osd) <= 300
print(f'\n--- specialist target: |y - off_sched| <= 300s ---')
for nm, sel in [('AOBT_3 present', trn & ~miss), ('AOBT_3 missing', trn & miss)]:
    print(f'  {nm:16s} rate = {on_sched[sel].mean()*100:6.2f}%')

CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
           min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
sel_m = trn & miss
m_clf = lgb.train(CLF, lgb.Dataset(X.loc[sel_m, COLS_M], on_sched[sel_m].astype(int),
                                   categorical_feature=CATS_M), num_boost_round=300)
pm = m_clf.predict(X.loc[val & miss, COLS_M])
act = on_sched[val & miss]
print(f'\n  specialist trained on {sel_m.sum():,} rows '
      f'({on_sched[sel_m].mean()*100:.1f}% positive)')
print(f'  holdout n={len(pm):,}: mean p={pm.mean():.4f}  actual={act.mean():.4f}  '
      f'(calibration {pm.mean()/max(act.mean(),1e-9):.2f})')
for q in [0.5, 0.8, 0.9, 0.95]:
    thr = np.quantile(pm, q); s = pm >= thr
    print(f'    top {(1-q)*100:5.1f}%: n={int(s.sum()):5,}  precision {act[s].mean()*100:5.1f}%  '
          f'recall {act[s].sum()/max(act.sum(),1)*100:5.1f}%')

# ---------- assemble --------------------------------------------------------
print('\n--- assembled prediction ---')
osv = osd[val]
e_on = np.where(np.isfinite(osv), osv, np.nan)     # if on-schedule stamp, y == off_sched
best = None
for w in [0.0, 0.5, 0.75, 1.0]:
    pred = p_bulk.copy()
    mv = miss[val]
    p_use = pm * w
    cand = np.where(np.isfinite(e_on[mv]), p_use * e_on[mv] + (1 - p_use) * p_bulk[mv],
                    p_bulk[mv])
    pred[mv] = cand
    pred = np.clip(pred, 0, None)
    d = decompose(yv, pred)
    print(f'  w={w:4.2f}  RMSE {d["rmse"]:7.2f}  | bulk {d["rmse_bulk"]:6.1f} '
          f'| tail {d["rmse_tail"]:8.0f} (contrib {d["contrib_tail"]:6.1f})')
    if best is None or d['rmse'] < best[1]:
        best = (w, d['rmse'], pred)

w, _, pred = best
print(f'\nbest w={w}')
for nm, p in [('A trimmed', p_bulk), ('B untrimmed', p_all)]:
    r = paired_bootstrap(yv, pred, p)
    print(f'  specialist vs {nm:14s}: {r["diff"]:+8.2f} +- {r["sd"]:5.2f}  '
          f'95% CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
np.save('cache/exp09_pred.npy', pred)
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
