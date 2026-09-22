"""EXP51 -- are the two bad bulk subpopulations underfit, or intrinsically hard?

Measured on out-of-fold predictions from the shipped model (bulk rows only):
    AOBT_3 present   2,062,474 rows   RMSE  216.6
    AOBT_3 missing      21,989 rows   RMSE  905.2     1.05% of rows, 15.7% of SE
    LIRF               160,224 rows   RMSE  464.0     7.7% of rows, 30.1% of SE

Two explanations, with opposite implications:
  UNDERFIT  the global model spends its capacity on the 99% and never learns
            these subpopulations' structure -> a dedicated model should beat it,
            and specialist models are worth building.
  HARD      the rows are genuinely unpredictable (when AOBT_3 is missing the
            whole _flt block goes with it, so a third of the features are NaN)
            -> a dedicated model will not beat it, and we should stop here.

TEST: for each subpopulation, compare the GLOBAL model's predictions on it
against a model trained ONLY on that subpopulation, same folds, bulk rows only.
Cheap: the specialist trains on 20k-160k rows, not 1.9M.

Three folds (months 2, 4, 9) -- deliberately not 1, 7 or 10, which are reported
elsewhere.
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
from prc.validation import rmse, paired_bootstrap
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
wx = attach_wx(tr, prepare(download()))
atfm = attach_atfm(tr)
X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index), atfm.set_axis(X.index)],
              axis=1)
del cong, wx, atfm
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
apt = tr.APT.astype(str).to_numpy()
miss = X.aobt_missing.to_numpy().astype(bool)
X, = align_categories(X)
COLS = list(X.columns)
CATF = [c for c in CATS if c in COLS]
bulk = y <= TAIL_CAP
print(f'{len(y):,} rows | {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)

P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
         min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, seed=0,
         deterministic=True, force_row_wise=True)

SUBPOPS = [('AOBT_3 missing', miss), ('LIRF', apt == 'LIRF')]
FOLDS = [2, 4, 9]
out = {}
for name, sub in SUBPOPS:
    gl, sp, yy = [], [], []
    for m in FOLDS:
        va, trn = month == m, month != m
        ev = va & bulk & sub
        # global model: trained on everything, as shipped
        g = lgb.train(P, lgb.Dataset(X.loc[trn & bulk, COLS], y[trn & bulk],
                                     categorical_feature=CATF), num_boost_round=350)
        # specialist: same features, trained ONLY on this subpopulation.
        # Fewer leaves because it sees 1-8% of the rows; 255 would overfit.
        sub_trn = trn & bulk & sub
        ps = dict(P, num_leaves=63, min_data_in_leaf=40)
        s = lgb.train(ps, lgb.Dataset(X.loc[sub_trn, COLS], y[sub_trn],
                                      categorical_feature=CATF), num_boost_round=350)
        gl.append(np.clip(g.predict(X.loc[ev, COLS]), 0, None))
        sp.append(np.clip(s.predict(X.loc[ev, COLS]), 0, None))
        yy.append(y[ev])
        print(f'  {name:16s} month {m}: n_train_sub={int(sub_trn.sum()):7,}  '
              f'n_eval={int(ev.sum()):6,}  global {rmse(yy[-1],gl[-1]):7.1f}  '
              f'specialist {rmse(yy[-1],sp[-1]):7.1f}', flush=True)
    gl, sp, yy = np.concatenate(gl), np.concatenate(sp), np.concatenate(yy)
    r = paired_bootstrap(yy, sp, gl)
    out[name] = (rmse(yy, gl), rmse(yy, sp), r)
    print(f'  >>> {name}: global {rmse(yy,gl):.1f} -> specialist {rmse(yy,sp):.1f}   '
          f'{r["diff"]:+.1f} +- {r["sd"]:.1f}  CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]\n', flush=True)

print('=' * 78)
for k, (g, s, r) in out.items():
    verdict = 'UNDERFIT -> build a specialist' if r['hi'] < 0 else 'HARD -> stop here'
    print(f'  {k:16s} {g:7.1f} -> {s:7.1f}   {verdict}')
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
