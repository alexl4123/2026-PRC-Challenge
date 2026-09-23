"""EXP54 -- v6 = v5 with the tail specialist the cluster confirmed.

WHAT CHANGED FROM v5
--------------------
Exactly one thing: P(y > 7200 | x) is replaced, inside the 4,944-row population

    P = {AOBT_3_flt missing} AND {off_sched > 7200}

by prc.models.tail_specialist. The bulk regressor, the global classifier, the
tail anchor `et`, the feature sets, the hyper-parameters and the seed count are
all identical to exp52. So the official delta measures the specialist and
nothing else.

THE EVIDENCE
------------
Cluster campaign `tailspec`, 12-fold leave-one-month-out, paired bootstrap of
the specialist arm (4 seeds) against the matched baseline arm:

    spec-r-full__c-base   275.62   (bulk 236.94, tail 8416.4)
    r-full__c-base        288.32   (bulk 236.48, tail 9858.8)
    paired  -13.06 +- 5.98   CI [-25.7, -4.0]   P(worse) = 0.0%

The bulk is untouched (+0.46, noise); the entire gain is tail, 9859 -> 8416.

WHY THE LEADERBOARD GAIN SHOULD EXCEED -13.06
---------------------------------------------
-13.06 is the average over twelve 2025 months. It is NOT uniform. Per month the
specialist arm scores 353.3 in January and 334.5 in July, against a 10-seed
r-full__c-base grid reading of 352.3 and 417.4. The gain is concentrated almost
entirely in July -- and the scored set is January + July 2026. January is the
month the specialist provably cannot help, July is the month it transforms.

TWO CAVEATS, STATED PLAINLY
---------------------------
1. The campaign's own baseline arm has ONE complete seed against the
   specialist's four, so part of the 13.06 is seed averaging. Seed averaging is
   worth about -0.90 on this arm, so roughly -12 survives as the specialist's
   own effect. The paired bootstrap CI [-25.7, -4.0] clears zero either way.
2. The campaign ran on the PRE-exp50 regressor (255 leaves / 350 rounds /
   lr 0.05), not v5's (511 / 1400 / 0.025). The specialist only edits the tail
   branch, which is close to orthogonal to the bulk regressor, so the effect is
   expected to carry -- but it is an assumption, not a measurement, and the
   official score is what tests it.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP, ID
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.submit import build as build_submission
from prc.models.tail_specialist import TailSpecialist, population, apply_to_mixture

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

SCALE = 12 / 11
bulk = y <= TAIL_CAP
is_tail = (y > TAIL_CAP).astype(int)
tail_mean = y[y > TAIL_CAP].mean()
rcat = [c for c in CATS if c in FULL]
ccat = [c for c in CATS if c in BASE]

# ---- the specialist -------------------------------------------------------
# Fit once on all training rows. It averages three internal seeds itself and
# does not depend on the outer seed, so it sits outside the loop.
apt_tr = Xtr['APT'].astype(str).to_numpy()
apt_rk = Xrk['APT'].astype(str).to_numpy()
os_tr = Xtr['off_sched'].to_numpy()
miss_tr = Xtr['aobt_missing'].to_numpy().astype(bool)
miss_rk = Xrk['aobt_missing'].to_numpy().astype(bool)

P_tr = population(os_tr, miss_tr)
print(f'\ntraining population P: {int(P_tr.sum()):,} rows, '
      f'{int(is_tail[P_tr].sum()):,} of {int(is_tail.sum()):,} tail rows '
      f'({100*is_tail[P_tr].mean():.1f}% rate inside P)', flush=True)

ts = time.time()
spec = TailSpecialist(threads=4).fit(Xtr, y, apt_tr, os_tr, miss_tr)
p_spec, in_P_rk = spec.predict(Xrk, apt_rk, os_rk, miss_rk)
print(f'specialist fitted in {time.time()-ts:.0f}s | ranking rows inside P: '
      f'{int(in_P_rk.sum()):,} of {len(Xrk):,}', flush=True)
if in_P_rk.sum() == 0:
    sys.exit('FATAL: no ranking rows inside P -- v6 would be identical to v5. '
             'Check off_sched / aobt_missing on the ranking set.')
print(f'  specialist p(tail) inside P: median {np.median(p_spec[in_P_rk]):.4f}  '
      f'mean {p_spec[in_P_rk].mean():.4f}  max {p_spec[in_P_rk].max():.4f}', flush=True)

# ---- the mixture, unchanged except for pt inside P ------------------------
preds, preds_v5 = [], []
for seed in range(N_SEEDS):
    t = time.time()
    r = dict(REG, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    c = dict(CLF, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    mb = lgb.train(r, lgb.Dataset(Xtr.loc[bulk, FULL], y[bulk], categorical_feature=rcat),
                   num_boost_round=int(1400 * SCALE))
    mc = lgb.train(c, lgb.Dataset(Xtr[BASE], is_tail, categorical_feature=ccat),
                   num_boost_round=int(300 * SCALE))
    pb = np.clip(mb.predict(Xrk[FULL]), 0, None)
    pt_global = mc.predict(Xrk[BASE])
    pt = apply_to_mixture(pt_global, p_spec, in_P_rk)
    et = np.where(np.isfinite(os_rk) & (os_rk > TAIL_CAP), os_rk, tail_mean)
    preds.append(np.clip(pt * et + (1 - pt) * pb, 0, None))
    preds_v5.append(np.clip(pt_global * et + (1 - pt_global) * pb, 0, None))
    d_med = float(np.median(pt[in_P_rk] - pt_global[in_P_rk]))
    print(f'  seed {seed}: mean {preds[-1].mean():.1f}s  '
          f'median dp inside P {d_med:+.4f}  ({time.time()-t:.0f}s)', flush=True)

pred = np.mean(preds, axis=0)
pred_v5 = np.mean(preds_v5, axis=0)
spread = np.std(preds, axis=0)
moved = np.abs(pred - pred_v5) > 1.0
print(f'\n{N_SEEDS}-seed average: mean {pred.mean():.1f}s  median {np.median(pred):.1f}s  '
      f'max {pred.max():.0f}s')
print(f'  per-row seed spread: median {np.median(spread):.1f}s  '
      f'p99 {np.percentile(spread,99):.0f}s')
print(f'  rows differing from the v5 recipe by >1s: {int(moved.sum()):,} '
      f'({100*moved.mean():.3f}%)  mean shift {np.mean(pred[moved]-pred_v5[moved]):+.0f}s')
build_submission(rk_meta[ID].to_numpy(), pred, version=6, clip=(0.0, 200000.0))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
