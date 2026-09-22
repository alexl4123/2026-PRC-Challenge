"""EXP52 -- v5 = v4 with the one hyperparameter change exp50 confirmed.

The 480-task grid settled the configuration:

    arm              10-seed OOF   bulk    single-seed sd
    r-full__c-cong      287.45    237.20        2.88
    r-full__c-base      287.81    236.60        0.97     <- chosen
    r-cong__c-cong      289.15    239.29        2.81
    r-cong__c-base      289.60    238.69        0.91

r-full__c-cong is 0.36s better and that difference is deep inside the noise,
while its seed-to-seed spread is three times larger. Take the stable one.

So: the REGRESSOR gets the full 72-feature set (base + congestion + METAR +
ATFM), the CLASSIFIER keeps the lean 18-feature base set. Feeding the
classifier more columns does not improve it -- it only triples its variance,
because it learns from 435 positives in 1.74M rows.

Seeds: 5 rather than 10. The grid shows r-full__c-base gains 289.34 -> 287.90
at five seeds and 287.81 at ten; the last five seeds are worth 0.09s and cost
half the compute, which matters with two other jobs on this laptop.

EXP50 tuned the bulk regressor for the first time -- 255 leaves / 350 rounds /
lr 0.05 had been a day-one guess frozen through every experiment since. The
winner is 511 leaves / 1400 rounds / lr 0.025, and it replicates:

    month 4 (sizing)   -2.06 +- 0.20  CI [-2.43, -1.69]
    month 2 (confirm)  -1.04 +- 0.28  CI [-1.59, -0.49]
    month 9 (confirm)  -1.69 +- 0.23  CI [-2.13, -1.22]

The runner-up (511/700/lr 0.05) FAILED confirmation on month 2 (+0.48, CI
crossing zero), which is why the sizing-month argmax was not shipped directly.

Note 1023 leaves scored WORSE than 511 (200.86 vs 200.66), so the model is
information-limited rather than capacity-limited. Expect ~2s, not more.

This is the ONLY change from v4, so if it is submitted the official delta
measures the hyperparameters and nothing else.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP, ID
from prc.features import build as build_base, align_categories, CATS, NEEDED
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.submit import build as build_submission

N_SEEDS = 5
t0 = time.time()


def assemble(which, extra_cols):
    full = load(which)
    cong = build_cong(full)
    d = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    del full
    X = build_base(d)
    X['aobt_missing'] = d.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = d.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    base_cols = list(X.columns)
    wx = attach_wx(d, prepare(download()))
    atfm = attach_atfm(d)
    X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
                   atfm.set_axis(X.index)], axis=1)
    return d[extra_cols], X, base_cols


tr_meta, Xtr, BASE = assemble('training', [TARGET])
rk_meta, Xrk, _ = assemble('ranking', [ID])
Xtr, Xrk = align_categories(Xtr, Xrk)
FULL = list(Xtr.columns)
y = tr_meta[TARGET].astype(float).to_numpy()
os_rk = Xrk.off_sched.to_numpy()
print(f'train {Xtr.shape}  ranking {Xrk.shape}  | reg {len(FULL)} feats, '
      f'clf {len(BASE)} feats | {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB',
      flush=True)

REG = dict(objective='regression', metric='rmse', learning_rate=0.025, num_leaves=511,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1,
           deterministic=True, force_row_wise=True)

# Round counts scaled from the 350/300 used per CV fold: folds train on ~11/12
# of the data, the final models train on all of it.
SCALE = 12 / 11
bulk = y <= TAIL_CAP
is_tail = (y > TAIL_CAP).astype(int)
tail_mean = y[y > TAIL_CAP].mean()
rcat = [c for c in CATS if c in FULL]
ccat = [c for c in CATS if c in BASE]

preds = []
for seed in range(N_SEEDS):
    t = time.time()
    r = dict(REG, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    c = dict(CLF, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    mb = lgb.train(r, lgb.Dataset(Xtr.loc[bulk, FULL], y[bulk], categorical_feature=rcat),
                   num_boost_round=int(1400 * SCALE))
    mc = lgb.train(c, lgb.Dataset(Xtr[BASE], is_tail, categorical_feature=ccat),
                   num_boost_round=int(300 * SCALE))
    pb = np.clip(mb.predict(Xrk[FULL]), 0, None)
    pt = mc.predict(Xrk[BASE])
    et = np.where(np.isfinite(os_rk) & (os_rk > TAIL_CAP), os_rk, tail_mean)
    preds.append(np.clip(pt * et + (1 - pt) * pb, 0, None))
    print(f'  seed {seed}: mean {preds[-1].mean():.1f}s  p(tail)>0.5 on '
          f'{int((pt>0.5).sum())} rows  ({time.time()-t:.0f}s)', flush=True)

pred = np.mean(preds, axis=0)
spread = np.std(preds, axis=0)
print(f'\n{N_SEEDS}-seed average: mean {pred.mean():.1f}s  median {np.median(pred):.1f}s  '
      f'max {pred.max():.0f}s')
print(f'  per-row seed spread: median {np.median(spread):.1f}s  '
      f'p99 {np.percentile(spread,99):.0f}s')
build_submission(rk_meta[ID].to_numpy(), pred, version=5, clip=(0.0, 200000.0))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
