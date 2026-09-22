"""EXP18 -- does leave-one-month-out CV undervalue regime-shift features?

v4 beat v3 by 30.0s officially where OOF predicted 8.4s, and the local->official
offset shrank from +56.05 to +34.44. The difference between v3 and v4 is the
weather + ATFM block (plus 5-seed averaging, worth ~1.5s by OOF).

HYPOTHESIS: those features encode how disrupted a given day or month was, and
their value shows up only when train and test differ in disruption level. LOMO
predicts a held-out 2025 month from other 2025 months -- broadly the same
regime -- so it cannot see that value. The real task is 2025 -> 2026, where
July 2026 carried 70% less weather delay than July 2025.

TEST: compare the same two feature sets under two protocols.
  LOMO      : each month held out, trained on the other eleven  (no time gap)
  FORWARD   : train Jan-Sep, predict Oct-Dec                    (extrapolation)
If the hypothesis holds, the full feature set should gain MORE under FORWARD
than under LOMO on the same months.

This matters beyond curiosity: if LOMO understates regime-shift features, we
should stop using it as the sole gate for them.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.validation import rmse, decompose, paired_bootstrap
pd.set_option('display.width', 200)

t0 = time.time()
full = load('training')
cong = build_cong(full)
tr = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
del full
X = build_base(tr)
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
BASE = list(X.columns)
wx = attach_wx(tr, prepare(download()))
atfm = attach_atfm(tr)
X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index), atfm.set_axis(X.index)],
              axis=1)
CONG_ = [c for c in X.columns if c not in BASE and not c.startswith(('wx_', 'atfm_'))]
REGIME = [c for c in X.columns if c.startswith(('wx_', 'atfm_'))]
del cong, wx, atfm
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
X, = align_categories(X)
LEAN, RICH = BASE + CONG_, BASE + CONG_ + REGIME
print(f'{len(y):,} rows | lean {len(LEAN)} | rich {len(RICH)} '
      f'({len(REGIME)} regime-shift feats) | {time.time()-t0:.0f}s', flush=True)

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=5, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(REG, objective='binary', num_leaves=63, min_data_in_leaf=50)
CLF['metric'] = 'auc'


def fit_predict(trn_mask, val_mask, reg_cols):
    yt = y[trn_mask]
    k = yt <= TAIL_CAP
    rc = [c for c in CATS if c in reg_cols]
    cc = [c for c in CATS if c in BASE]
    mb = lgb.train(REG, lgb.Dataset(X.loc[trn_mask, reg_cols][k], yt[k],
                                    categorical_feature=rc), num_boost_round=350)
    mc = lgb.train(CLF, lgb.Dataset(X.loc[trn_mask, BASE], (yt > TAIL_CAP).astype(int),
                                    categorical_feature=cc), num_boost_round=300)
    pb = np.clip(mb.predict(X.loc[val_mask, reg_cols]), 0, None)
    pt = mc.predict(X.loc[val_mask, BASE])
    osv = X.loc[val_mask, 'off_sched'].to_numpy()
    et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
    return np.clip(pt * et + (1 - pt) * pb, 0, None)


# Evaluate BOTH protocols on the SAME months (Oct-Dec) so the comparison is fair.
EVAL = np.isin(month, [10, 11, 12])
print('\nevaluating both protocols on Oct-Dec (the only months a forward split '
      'can hold out)\n')
res = {}
for proto, trn in [('FORWARD  (train Jan-Sep)', month <= 9),
                   ('LOMO-like (train all other months)', ~EVAL)]:
    for name, cols in [('lean', LEAN), ('rich', RICH)]:
        t = time.time()
        p = fit_predict(trn, EVAL, cols)
        d = decompose(y[EVAL], p)
        res[(proto, name)] = p
        print(f'  {proto:36s} {name:5s}  RMSE {d["rmse"]:7.2f}  bulk {d["rmse_bulk"]:6.1f} '
              f'tail {d["rmse_tail"]:8.0f}   ({time.time()-t:.0f}s)', flush=True)

print('\n' + '=' * 88)
for proto in ['FORWARD  (train Jan-Sep)', 'LOMO-like (train all other months)']:
    r = paired_bootstrap(y[EVAL], res[(proto, 'rich')], res[(proto, 'lean')])
    print(f'  {proto:36s} rich - lean = {r["diff"]:+7.2f} +- {r["sd"]:5.2f}  '
          f'CI [{r["lo"]:+6.1f}, {r["hi"]:+6.1f}]')
print('\n  If the regime-shift hypothesis holds, FORWARD should show a LARGER')
print('  gain from the rich set than LOMO does.')
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
