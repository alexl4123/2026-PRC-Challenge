"""EXP62 step 2 -- 12-fold leave-one-month-out test of the extended specialist.

Step 1 (exp62_screen, claude_log 2026-09-23 §19) found that per-flight features
observable at scoring time separate ARTEFACT from REAL inside P far better than
the shipped specialist: inside-P Brier 0.03346 -> 0.02768, CI clear of zero,
LIRF AUC 0.662 -> 0.854. This asks whether that survives the real mixture.

    arm a  = v6: shipped TailSpecialist inside P, fitted per fold
    arm b  = v6 with TailSpecialistX(extra = FAMILIES['all']) inside P
    (v5    = global classifier everywhere; reported for reference only)
Everything outside P is identical across a and b by construction.

PHASE A (expensive, ~1 h, resumable): per fold, the v5/v6 bulk regressor and
global classifier exactly as exp54 configures them for seed 0 (REG 511 leaves /
lr 0.025 / 1400 rounds on y <= 7200 with all 72 features; CLF 63 leaves / lr
0.05 / 300 rounds on the 18 base features), 11-month training so no 12/11
round scaling. Each fold is written atomically to cache/exp62_folds/mMM.npz;
a rerun skips finished folds. Assembled to cache/exp62_pb.npy and
cache/exp62_ptg.npy (float32, all 2,085,047 DEP rows, load('training') DEP order).

PHASE B (seconds): specialists per fold on the P rows of the training months,
et = off_sched inside the anchor zone else the training months' tail mean
(exp53 convention), mixture, then the report.

DECISION: adopt b only if the 12-month paired bootstrap (b - a) CI clears zero.

Run:
    systemd-inhibit --what=idle:sleep:handle-lid-switch --why=exp62 \
      systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
      /home/thinklex/prc2026-venv/bin/python experiments/exp62_cv.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, CACHE
from prc.features import NEEDED, CATS, build as build_base, align_categories
from prc.io import load, peak_rss_gb
from prc.validation import decompose, paired_bootstrap, per_month, rmse
from prc.models.tail_specialist import (TailSpecialist, TailSpecialistX, SPEC_FEATURES,
                                        population, apply_to_mixture)
from prc.p_features import te_features, FAMILIES

pd.set_option('display.width', 220)
t0 = time.time()
THREADS = 6
N_DEP = 2_085_047
FOLD_DIR = CACHE / 'exp62_folds'
FOLD_DIR.mkdir(exist_ok=True)
EXTRA = FAMILIES['all']

REG = dict(objective='regression', metric='rmse', learning_rate=0.025, num_leaves=511,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1,
           deterministic=True, force_row_wise=True, seed=0, bagging_seed=1,
           feature_fraction_seed=2)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1,
           deterministic=True, force_row_wise=True, seed=0, bagging_seed=1,
           feature_fraction_seed=2)


def log(msg):
    print(f'[{time.time()-t0:6.0f}s | RSS {peak_rss_gb():.2f} GB] {msg}', flush=True)


def fold_file(m):
    return FOLD_DIR / f'm{m:02d}.npz'


# ============================================================================
# PHASE A -- bulk regressor + global classifier per fold, cached
# ============================================================================
todo = [m for m in range(1, 13) if not fold_file(m).exists()]
if todo:
    from prc.congestion import build as build_cong
    from prc.weather import download, prepare, attach as attach_wx
    from prc.disruption import attach as attach_atfm
    log(f'phase A: folds to fit {todo}')
    full = load('training')
    cong = build_cong(full)
    d = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    del full
    X = build_base(d)
    X['aobt_missing'] = d.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = d.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    BASE = list(X.columns)
    wx = attach_wx(d, prepare(download()))
    atfm = attach_atfm(d)
    X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
                   atfm.set_axis(X.index)], axis=1)
    del cong, wx, atfm
    FULL = list(X.columns)
    if (len(FULL), len(BASE)) != (72, 18):
        sys.exit(f'FATAL: feature counts {len(FULL)}/{len(BASE)} differ from v6 (72/18)')
    yA = d[TARGET].astype(float).to_numpy()
    monthA = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    del d
    (X,) = align_categories(X)
    if len(yA) != N_DEP:
        sys.exit(f'FATAL: {len(yA)} DEP rows, expected {N_DEP}')
    rcat = [c for c in CATS if c in FULL]
    ccat = [c for c in CATS if c in BASE]
    log(f'features assembled: {X.shape}, reg {len(FULL)} / clf {len(BASE)}')
    for m in todo:
        tf = time.time()
        va = monthA == m
        tr = ~va
        k = tr & (yA <= TAIL_CAP)
        mb = lgb.train(REG, lgb.Dataset(X.loc[k, FULL], yA[k], categorical_feature=rcat),
                       num_boost_round=1400)
        mc = lgb.train(CLF, lgb.Dataset(X.loc[tr, BASE], (yA[tr] > TAIL_CAP).astype(int),
                                        categorical_feature=ccat), num_boost_round=300)
        pb = np.clip(mb.predict(X.loc[va, FULL]), 0, None).astype(np.float32)
        ptg = mc.predict(X.loc[va, BASE]).astype(np.float32)
        tmp = FOLD_DIR / f'm{m:02d}.tmp.npz'
        np.savez(tmp, pb=pb, ptg=ptg, idx=np.where(va)[0])
        os.replace(tmp, fold_file(m))
        del mb, mc
        log(f'fold {m:02d}: train {int(tr.sum()):,} -> val {int(va.sum()):,}  '
            f'bulk-RMSE of pb on val y<=7200 {rmse(yA[va & (yA <= TAIL_CAP)], pb[yA[va] <= TAIL_CAP]):.2f}  '
            f'({time.time()-tf:.0f}s)')
    del X

pb = np.full(N_DEP, np.nan, np.float32)
ptg = np.full(N_DEP, np.nan, np.float32)
for m in range(1, 13):
    z = np.load(fold_file(m))
    pb[z['idx']], ptg[z['idx']] = z['pb'], z['ptg']
if not (np.isfinite(pb).all() and np.isfinite(ptg).all()):
    sys.exit('FATAL: pb/ptg incomplete after assembling the 12 fold files')
np.save(CACHE / 'exp62_pb.npy', pb)
np.save(CACHE / 'exp62_ptg.npy', ptg)
log('phase A complete: cache/exp62_pb.npy, cache/exp62_ptg.npy written')

# ============================================================================
# PHASE B -- specialists per fold
# ============================================================================
d = load('training', phase='DEP', columns=sorted(set(NEEDED + [TARGET])))
y = d[TARGET].astype(float).to_numpy()
month = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
X = build_base(d)
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
(X,) = align_categories(X)
os_all = X.off_sched.to_numpy(float)
miss_all = d.AOBT_3_flt.isna().to_numpy()
del d
inP = population(os_all, miss_all)

cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet')
pool = pd.read_parquet(CACHE / 'exp62_pool_training.parquet')
Pidx = cand.index.to_numpy()
if not np.array_equal(np.where(inP)[0], Pidx):
    sys.exit('FATAL: cached P rows do not match population() on this load -- rebuild '
             'cache/exp62_cand_training.parquet with experiments/exp62_screen.py')
if not np.array_equal(month[Pidx], cand.month.to_numpy()):
    sys.exit('FATAL: month mismatch between cached candidates and this load')
cand = cand.reset_index(drop=True)
yP = y[Pidx]
tP = (yP > TAIL_CAP).astype(int)
pool_art = (np.abs(y[pool['pos'].to_numpy()] - pool.off_sched.to_numpy()) < 60).astype(float)
XP = pd.concat([X.loc[Pidx, SPEC_FEATURES].reset_index(drop=True),
                cand[[c for c in EXTRA if not c.startswith('te_')]]], axis=1)
aptP = cand.apt.to_numpy(object)
osP = cand.off_sched.to_numpy(float)
missP = np.ones(len(cand), bool)
monP = cand.month.to_numpy()
del X
log(f'phase B: P rows {len(cand):,} (tail {int(tP.sum())}), extra features {len(EXTRA)}')

spec_a = np.full(len(cand), np.nan)
spec_b = np.full(len(cand), np.nan)
et = np.full(N_DEP, np.nan)
for m in range(1, 13):
    va, tr = monP == m, monP != m
    Xf = XP.copy()
    pk = (pool.month != m).to_numpy()
    te_tr, te_va = te_features(cand[tr], tP[tr], pool[pk], pool_art[pk], cand[va])
    for c in te_tr.columns:
        col = np.full(len(cand), np.nan)
        col[tr], col[va] = te_tr[c].to_numpy(), te_va[c].to_numpy()
        Xf[c] = col
    sa = TailSpecialist(threads=THREADS).fit(Xf[tr], yP[tr], aptP[tr], osP[tr], missP[tr])
    spec_a[va] = sa.predict(Xf[va], aptP[va], osP[va], missP[va])[0]
    sb = TailSpecialistX(extra=EXTRA, threads=THREADS).fit(Xf[tr], yP[tr], aptP[tr], osP[tr],
                                                          missP[tr])
    spec_b[va] = sb.predict(Xf[va], aptP[va], osP[va], missP[va])[0]
    # anchor: exp53 convention, tail mean of the TRAINING months
    vm = month == m
    tail_mean = y[(month != m) & (y > TAIL_CAP)].mean()
    et[vm] = np.where(np.isfinite(os_all[vm]) & (os_all[vm] > TAIL_CAP), os_all[vm], tail_mean)
log('specialists fitted for 12 folds')

pa_full = np.full(N_DEP, np.nan); pa_full[Pidx] = spec_a
pbx_full = np.full(N_DEP, np.nan); pbx_full[Pidx] = spec_b
ptg64, pb64 = ptg.astype(float), pb.astype(float)
pt_a = apply_to_mixture(ptg64, pa_full, inP)
pt_b = apply_to_mixture(ptg64, pbx_full, inP)
pred = {'v5': np.clip(ptg64 * et + (1 - ptg64) * pb64, 0, None),
        'a_v6': np.clip(pt_a * et + (1 - pt_a) * pb64, 0, None),
        'b_v6x': np.clip(pt_b * et + (1 - pt_b) * pb64, 0, None)}
np.save(CACHE / 'exp62_oof.npy', np.vstack([pred[k] for k in pred]))
np.save(CACHE / 'exp62_spec_P.npy', np.vstack([spec_a, spec_b]))
log('saved cache/exp62_oof.npy (rows v5, a_v6, b_v6x) and cache/exp62_spec_P.npy (a, b on P rows)')
assert np.array_equal(pred['a_v6'][~inP], pred['b_v6x'][~inP]), 'arms differ outside P'

# ============================================================================
# REPORT
# ============================================================================
print('\n' + '=' * 100)
print('12-MONTH OOF (leave-one-month-out, seed 0)')
print('=' * 100)
for k, p in pred.items():
    dd = decompose(y, p)
    print(f'  {k:6s} RMSE {dd["rmse"]:8.3f} | bulk {dd["rmse_bulk"]:7.2f} (n {dd["n_bulk"]:,}) '
          f'| tail {dd["rmse_tail"]:8.1f} (n {dd["n_tail"]})')
e53 = np.load(CACHE / 'exp53_oof.npy')[1]
print(f'  sanity: exp53 v5 OOF (bagging/ff seeds and threads differ) RMSE {rmse(y, e53):.3f}; '
      f'this run v5 {rmse(y, pred["v5"]):.3f}')


def pb_line(label, yy, pa, pbb, n_boot=1000):
    r = paired_bootstrap(yy, pa, pbb, n_boot=n_boot)
    print(f'  {label:44s} {r["diff"]:+8.3f} +- {r["sd"]:6.3f}  CI [{r["lo"]:+8.3f}, {r["hi"]:+8.3f}]  '
          f'P(first worse) {r["p_a_worse"]*100:5.1f}%')
    return r


print('\nPAIRED BOOTSTRAP (first minus second; negative = first better; 1000 resamples)')
r12 = pb_line('12-month   b_v6x - a_v6', y, pred['b_v6x'], pred['a_v6'])
pb_line('12-month   a_v6  - v5   (v6 specialist effect)', y, pred['a_v6'], pred['v5'])
sc = np.isin(month, [1, 7])
rsc = pb_line('scored 1+7 b_v6x - a_v6', y[sc], pred['b_v6x'][sc], pred['a_v6'][sc])
for m in (1, 7):
    s = month == m
    pb_line(f'month {m:2d}   b_v6x - a_v6', y[s], pred['b_v6x'][s], pred['a_v6'][s])
for k in ('a_v6', 'b_v6x'):
    print(f'  scored 1+7 RMSE {k}: {rmse(y[sc], pred[k][sc]):.3f}')

print('\nPER MONTH')
pm = pd.DataFrame({k: per_month(y, p, month).set_index('month').rmse for k, p in pred.items()})
pm['b-a'] = pm['b_v6x'] - pm['a_v6']
pm['n_P'] = pd.Series(monP).value_counts().sort_index()
print(pm.round(2).to_string())

print('\nINSIDE P: false-alarm SSE (non-tail P rows) vs missed-tail SSE (tail P rows)')
print('  contribution = sqrt(SSE / N_all) on the 12-month scale; scored = months 1+7 only')
rows = []
for k, p in pred.items():
    for scope, sel in (('12m', np.ones(N_DEP, bool)), ('scored', sc)):
        e = (y - p) ** 2
        fa = sel & inP & (y <= TAIL_CAP)
        mt = sel & inP & (y > TAIL_CAP)
        n = int(sel.sum())
        rows.append({'arm': k, 'scope': scope, 'n_FA': int(fa.sum()), 'n_MT': int(mt.sum()),
                     'FA_SSE_1e9': e[fa].sum() / 1e9, 'MT_SSE_1e9': e[mt].sum() / 1e9,
                     'FA_contrib': np.sqrt(e[fa].sum() / n), 'MT_contrib': np.sqrt(e[mt].sum() / n),
                     'outsideP_SSE_1e9': e[sel & ~inP].sum() / 1e9, 'RMSE': np.sqrt(e[sel].mean())})
fam = pd.DataFrame(rows)
print(fam.round(3).to_string(index=False))

lj = inP & (month == 7) & (y <= 1800)
X_lirf = np.zeros(N_DEP, bool); X_lirf[Pidx[aptP == 'LIRF']] = True
lj &= X_lirf
print(f'\nLIRF July, P rows with y <= 1800 (the "REAL" flights): n={int(lj.sum())}, mean truth '
      f'{y[lj].mean():.0f}s, mean pred a {pred["a_v6"][lj].mean():.0f}s -> b {pred["b_v6x"][lj].mean():.0f}s; '
      f'mean pt a {pt_a[lj].mean():.3f} -> b {pt_b[lj].mean():.3f}')
lt = X_lirf & (y > TAIL_CAP)
print(f'LIRF tail rows in P: n={int(lt.sum())}, mean pt a {pt_a[lt].mean():.3f} -> b {pt_b[lt].mean():.3f}')

adopt = r12['hi'] < 0
print('\n' + '=' * 100)
print(f'DECISION: 12-month b - a = {r12["diff"]:+.3f} +- {r12["sd"]:.3f}, CI [{r12["lo"]:+.3f}, '
      f'{r12["hi"]:+.3f}] -> {"ADOPT b (CI clears zero)" if adopt else "DO NOT ADOPT (CI crosses zero)"}')
print(f'  scored months (not decisive): {rsc["diff"]:+.3f} +- {rsc["sd"]:.3f}')
print(f'\npeak RSS {peak_rss_gb():.2f} GB   total {(time.time()-t0)/60:.1f} min')
