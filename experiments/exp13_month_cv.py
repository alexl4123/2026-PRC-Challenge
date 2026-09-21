"""EXP13 -- leave-one-month-out CV as the local check.

Replaces the Jan+Jul-2025 holdout (344k rows, 149 tail) with out-of-fold
predictions over all twelve months (2.08M rows, 584 tail). Three models are run
through the identical folds so they can be compared with a paired bootstrap:

    A  trimmed    -- train rows restricted to y <= 7200   (what v1 was)
    B  untrimmed  -- all training rows                    (what v2 was)
    C  mixture    -- P(T|x)*E[y|T,x] + (1-P(T|x))*E[y|~T,x]  (what v3 was)

Also answers the seasonality question directly: per-month RMSE, so we can see
whether summer really is harder and by how much.

Runtime ~20 min. Run detached:
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \
    systemd-inhibit --what=idle:sleep:handle-lid-switch \
    /home/thinklex/prc2026-venv/bin/python experiments/exp13_month_cv.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build, align_categories, CATS, NEEDED
from prc.io import load, peak_rss_gb
from prc.validation import rmse, decompose, paired_bootstrap, month_cv_oof, per_month
pd.set_option('display.width', 210)

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
X, = align_categories(X)
COLS = list(X.columns)
X = X[COLS].reset_index(drop=True)
print(f'{len(y):,} departures, {int((y>TAIL_CAP).sum())} tail rows  ({time.time()-t0:.0f}s)')

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)


def fp_trimmed(Xt, yt, Xv):
    k = yt <= TAIL_CAP
    m = lgb.train(REG, lgb.Dataset(Xt[k], yt[k], categorical_feature=CATS),
                  num_boost_round=350)
    return np.clip(m.predict(Xv), 0, None)


def fp_untrimmed(Xt, yt, Xv):
    m = lgb.train(REG, lgb.Dataset(Xt, yt, categorical_feature=CATS), num_boost_round=350)
    return np.clip(m.predict(Xv), 0, None)


def fp_mixture(Xt, yt, Xv):
    k = yt <= TAIL_CAP
    mb = lgb.train(REG, lgb.Dataset(Xt[k], yt[k], categorical_feature=CATS),
                   num_boost_round=350)
    mc = lgb.train(CLF, lgb.Dataset(Xt, (yt > TAIL_CAP).astype(int),
                                    categorical_feature=CATS), num_boost_round=300)
    pb = np.clip(mb.predict(Xv), 0, None)
    pt = mc.predict(Xv)
    osv = Xv['off_sched'].to_numpy()
    tm = yt[yt > TAIL_CAP].mean()
    et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, tm)
    return np.clip(pt * et + (1 - pt) * pb, 0, None)


oofs = {}
for name, fn in [('A trimmed', fp_trimmed), ('B untrimmed', fp_untrimmed),
                 ('C mixture', fp_mixture)]:
    print(f'\n=== {name} ===', flush=True)
    oofs[name] = month_cv_oof(X, y, month, fn)
    d = decompose(y, oofs[name])
    print(f'  OOF RMSE {d["rmse"]:7.2f} | bulk {d["rmse_bulk"]:6.1f} '
          f'(contrib {d["contrib_bulk"]:6.1f}) | tail {d["rmse_tail"]:8.0f} '
          f'(contrib {d["contrib_tail"]:6.1f}) | n_tail {d["n_tail"]}', flush=True)

print('\n' + '=' * 100)
print('PER-MONTH OOF RMSE (the seasonality question)')
print('=' * 100)
pm = per_month(y, oofs['C mixture'], month)
print(pm.round(1).to_string(index=False))
print(f"\n  bulk-only RMSE by month: min {pm.rmse_bulk.min():.1f} (month "
      f"{int(pm.loc[pm.rmse_bulk.idxmin(),'month'])})  max {pm.rmse_bulk.max():.1f} "
      f"(month {int(pm.loc[pm.rmse_bulk.idxmax(),'month'])})")
print(f"  scored months -- Jan {pm.loc[pm.month==1,'rmse_bulk'].iloc[0]:.1f}, "
      f"Jul {pm.loc[pm.month==7,'rmse_bulk'].iloc[0]:.1f} (bulk); "
      f"tail rows Jan {int(pm.loc[pm.month==1,'n_tail'].iloc[0])}, "
      f"Jul {int(pm.loc[pm.month==7,'n_tail'].iloc[0])}")

print('\n' + '=' * 100)
print('PAIRED COMPARISONS -- OOF (2.08M rows) vs the old Jan+Jul holdout (344k)')
print('=' * 100)
jj = np.isin(month, [1, 7])
for a, b in [('C mixture', 'B untrimmed'), ('C mixture', 'A trimmed'),
             ('B untrimmed', 'A trimmed')]:
    r_all = paired_bootstrap(y, oofs[a], oofs[b])
    r_jj = paired_bootstrap(y[jj], oofs[a][jj], oofs[b][jj])
    print(f'  {a:12s} vs {b:12s}')
    print(f'      OOF all months : {r_all["diff"]:+8.2f} +- {r_all["sd"]:5.2f}  '
          f'CI [{r_all["lo"]:+8.1f}, {r_all["hi"]:+8.1f}]')
    print(f'      Jan+Jul only   : {r_jj["diff"]:+8.2f} +- {r_jj["sd"]:5.2f}  '
          f'CI [{r_jj["lo"]:+8.1f}, {r_jj["hi"]:+8.1f}]   '
          f'<- sd {r_jj["sd"]/max(r_all["sd"],1e-9):.1f}x wider')
np.save('cache/exp13_oof.npy', np.vstack([oofs[k] for k in oofs]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0/60:.0f}s')
