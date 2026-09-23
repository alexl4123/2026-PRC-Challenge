"""EXP64 -- v7 = v6 + exp62 arm b, built as a LEADERBOARD TEST.

WHAT CHANGED FROM v6 (experiments/exp54_submit_v6.py)
-----------------------------------------------------
Exactly one thing: inside P = {AOBT_3_flt missing} AND {off_sched > 7200},
TailSpecialistX with the pre-specified `all` family (prc.p_features.FAMILIES['all'],
42 per-flight features) replaces TailSpecialist. assemble(), REG/CLF, the 5
seeds, SCALE, et and the clip are exp54's, at num_threads=4 as in exp54.
Target encodings / stamping rates for the final fit come from ALL 12 training
months with the same Beta(8) shrinkage as in-fold (prc.p_features.te_features:
ranking rows from all training rows, training rows by inner leave-one-month-out).

WHY A LEADERBOARD TEST, NOT AN OOF-JUSTIFIED SHIP
------------------------------------------------
exp62 step 2 (claude_log 2026-09-23 §20): 12-month b - a = -1.43 +- 0.88,
CI [-3.00, +0.44] -- does NOT clear zero; scored months (1+7) -6.25 +- 1.94,
CI [-10.61, -3.07]. The ranking is on Jan+Jul 2026 with each team's best score
kept, so one submission measures the final quantity directly. User-approved via
the main session.

CHECKS (the script refuses to write v7 if check 1 fails)
  1. Outside P, v7 == shipped v6 exactly (int32, aligned by MVT_ID_mvt).
     The in-run v6 recipe is also compared with shipped v6 on EVERY row.
  2. Inside P: rows changed, raised/lowered, shift, >7200/>20000/>60000 counts,
     probability-change distribution, LIRF share.
  3. prc.submit.build(..., version=7, clip=(0, 200000)) + validate(Path(...)).
DOES NOT UPLOAD.

Run (background; log cache/exp64_v7.log):
    systemd-inhibit --what=idle:sleep:handle-lid-switch --why=v7 \
      systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
      /home/thinklex/prc2026-venv/bin/python experiments/exp64_submit_v7.py
Per-seed (pb, ptg) are saved to cache/exp64_seeds/ so a crash resumes.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP, ID, CACHE, OUTPUTS
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.submit import build as build_submission, validate
from prc.models.tail_specialist import (TailSpecialist, TailSpecialistX, SPEC_FEATURES,
                                        population, apply_to_mixture)
from prc.p_features import load_set, candidates, te_features, FAMILIES, TE_COLS

pd.set_option('display.width', 220)
N_SEEDS = 5
EXTRA = FAMILIES['all']
V6_PATH = OUTPUTS / 'understanding-accordion_v6.parquet'
V7_PATH = OUTPUTS / 'understanding-accordion_v7.parquet'
SEED_DIR = CACHE / 'exp64_seeds'
SEED_DIR.mkdir(exist_ok=True)
t0 = time.time()


def log(msg):
    print(f'[{time.time()-t0:6.0f}s | RSS {peak_rss_gb():.2f} GB] {msg}', flush=True)


if len(EXTRA) != 42:
    sys.exit(f'FATAL: FAMILIES["all"] has {len(EXTRA)} features, exp62 arm b had 42')
if V7_PATH.exists():
    sys.exit(f'FATAL: {V7_PATH} already exists -- move it aside deliberately before rebuilding')


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


# ---- P-row features first (small; freed before the big assemble) -----------
drk, depm_rk, _ = load_set('ranking')
cand_rk, _, P_rk_c = candidates(drk, depm_rk)
del drk
f_tr, f_pool = CACHE / 'exp62_cand_training.parquet', CACHE / 'exp62_pool_training.parquet'
if f_tr.exists() and f_pool.exists():
    cand_tr, pool = pd.read_parquet(f_tr), pd.read_parquet(f_pool)
    src = 'cache (exp62)'
else:
    dtr, depm_tr, _ = load_set('training')
    cand_tr, pool, _ = candidates(dtr, depm_tr)
    del dtr
    src = 'rebuilt'
log(f'candidates: ranking P {len(cand_rk):,} (built now), training P {len(cand_tr):,} ({src})')

# ---- exp54 feature build ---------------------------------------------------
tr_meta, Xtr, BASE = assemble('training', [TARGET])
rk_meta, Xrk, _ = assemble('ranking', [ID])
Xtr, Xrk = align_categories(Xtr, Xrk)
FULL = list(Xtr.columns)
y = tr_meta[TARGET].astype(float).to_numpy()
os_rk = Xrk.off_sched.to_numpy()
print(f'train {Xtr.shape}  ranking {Xrk.shape}  | reg {len(FULL)} feats, '
      f'clf {len(BASE)} feats | {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)
if (len(FULL), len(BASE)) != (72, 18):
    sys.exit(f'FATAL: feature counts {len(FULL)}/{len(BASE)} differ from v6 (72/18)')

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

apt_tr = Xtr['APT'].astype(str).to_numpy()
apt_rk = Xrk['APT'].astype(str).to_numpy()
os_tr = Xtr['off_sched'].to_numpy()
miss_tr = Xtr['aobt_missing'].to_numpy().astype(bool)
miss_rk = Xrk['aobt_missing'].to_numpy().astype(bool)
P_tr = population(os_tr, miss_tr)
P_rk = population(os_rk, miss_rk)
if not np.array_equal(np.where(P_tr)[0], cand_tr.index.to_numpy()):
    sys.exit('FATAL: training candidates do not match P on this load')
if not (np.array_equal(np.where(P_rk)[0], cand_rk.index.to_numpy()) and np.array_equal(P_rk, P_rk_c)):
    sys.exit('FATAL: ranking candidates do not match P on this load')
print(f'\ntraining population P: {int(P_tr.sum()):,} rows, {int(is_tail[P_tr].sum()):,} of '
      f'{int(is_tail.sum()):,} tail rows | ranking P: {int(P_rk.sum()):,} of {len(Xrk):,}', flush=True)

# ---- v6 specialist, exactly exp54's call -----------------------------------
spec6 = TailSpecialist(threads=4).fit(Xtr, y, apt_tr, os_tr, miss_tr)
p6, in_P6 = spec6.predict(Xrk, apt_rk, os_rk, miss_rk)
assert np.array_equal(in_P6, P_rk)
print(f'  v6 specialist p(tail) inside P: median {np.median(p6[P_rk]):.4f}  '
      f'mean {p6[P_rk].mean():.4f}  max {p6[P_rk].max():.4f}   (exp54 printed 0.0057 / 0.0766 / 0.9805)',
      flush=True)

# ---- v7 specialist: TailSpecialistX on the P rows, TE from all 12 months ---
itr, irk = cand_tr.index.to_numpy(), cand_rk.index.to_numpy()
ctr, crk = cand_tr.reset_index(drop=True), cand_rk.reset_index(drop=True)
tP = (y[itr] > TAIL_CAP).astype(int)
pool_art = (np.abs(y[pool['pos'].to_numpy()] - pool.off_sched.to_numpy()) < 60).astype(float)
te_tr, te_rk = te_features(ctr, tP, pool, pool_art, crk)
raw = [c for c in EXTRA if not c.startswith('te_')]
XPtr = pd.concat([Xtr.loc[itr, SPEC_FEATURES].reset_index(drop=True), ctr[raw],
                  te_tr.reset_index(drop=True)], axis=1)
XPrk = pd.concat([Xrk.loc[irk, SPEC_FEATURES].reset_index(drop=True), crk[raw],
                  te_rk.reset_index(drop=True)], axis=1)

print('\nFEATURE COVERAGE on the ranking P rows (TE: share whose key was seen in training)')
cov = []
for c in EXTRA:
    if c in TE_COLS or c == 'te_op_rd':
        k = TE_COLS.get(c, 'k_op')
        seen = set(pool.k_op) if c == 'te_op_rd' else set(ctr[k])
        crk_c, ctr_c = float(crk[k].isin(seen).mean()), 1.0
        lr_c = float(crk.loc[crk.apt == 'LIRF', k].isin(seen).mean())
    else:
        crk_c, ctr_c = float(XPrk[c].notna().mean()), float(XPtr[c].notna().mean())
        lr_c = float(XPrk.loc[(crk.apt == 'LIRF').to_numpy(), c].notna().mean())
    cov.append({'feature': c, 'cov_ranking_P': crk_c, 'cov_ranking_LIRF': lr_c, 'cov_training_P': ctr_c})
cov = pd.DataFrame(cov)
cov.to_csv(CACHE / 'exp64_coverage.csv', index=False)
print(cov.round(3).to_string(index=False))
bad = cov[(cov.cov_training_P > 0.5) & (cov.cov_ranking_P < 0.5 * cov.cov_training_P)]
if len(bad):
    sys.exit(f'FATAL: coverage collapses on ranking for {list(bad.feature)}')
print(f'  {len(cov)} features; ranking-P coverage min {cov.cov_ranking_P.min():.3f}, '
      f'median {cov.cov_ranking_P.median():.3f}; {int((cov.cov_ranking_P >= 0.9).sum())} at >= 0.90',
      flush=True)

ones_tr, ones_rk = np.ones(len(itr), bool), np.ones(len(irk), bool)
spec7 = TailSpecialistX(extra=EXTRA, threads=4).fit(XPtr, y[itr], apt_tr[itr], os_tr[itr], ones_tr)
p7 = np.full(len(Xrk), np.nan)
p7[irk] = spec7.predict(XPrk, apt_rk[irk], os_rk[irk], ones_rk)[0]
print(f'  v7 specialist p(tail) inside P: median {np.nanmedian(p7):.4f}  mean {np.nanmean(p7):.4f}  '
      f'max {np.nanmax(p7):.4f}', flush=True)
np.save(CACHE / 'exp64_p6_p7.npy', np.vstack([p6, p7]))

# ---- the mixture: exp54 verbatim, pt differs only inside P ------------------
et = np.where(np.isfinite(os_rk) & (os_rk > TAIL_CAP), os_rk, tail_mean)
preds7, preds6 = [], []
for seed in range(N_SEEDS):
    t = time.time()
    f = SEED_DIR / f'seed{seed}.npz'
    if f.exists():
        z = np.load(f)
        pb, pt_global = z['pb'], z['ptg']
        how = 'resumed'
    else:
        r = dict(REG, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
        c = dict(CLF, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
        mb = lgb.train(r, lgb.Dataset(Xtr.loc[bulk, FULL], y[bulk], categorical_feature=rcat),
                       num_boost_round=int(1400 * SCALE))
        mc = lgb.train(c, lgb.Dataset(Xtr[BASE], is_tail, categorical_feature=ccat),
                       num_boost_round=int(300 * SCALE))
        pb = np.clip(mb.predict(Xrk[FULL]), 0, None)
        pt_global = mc.predict(Xrk[BASE])
        del mb, mc
        tmp = SEED_DIR / f'seed{seed}.tmp.npz'
        np.savez(tmp, pb=pb, ptg=pt_global)
        os.replace(tmp, f)
        how = 'fitted'
    pt7 = apply_to_mixture(pt_global, p7, P_rk)
    pt6 = apply_to_mixture(pt_global, p6, P_rk)
    preds7.append(np.clip(pt7 * et + (1 - pt7) * pb, 0, None))
    preds6.append(np.clip(pt6 * et + (1 - pt6) * pb, 0, None))
    log(f'seed {seed} ({how}): v7 mean {preds7[-1].mean():.1f}s | v6 recipe mean '
        f'{preds6[-1].mean():.1f}s ({time.time()-t:.0f}s)')

pred7 = np.mean(preds7, axis=0)
pred6 = np.mean(preds6, axis=0)
np.save(CACHE / 'exp64_pred_v6recipe_v7.npy', np.vstack([pred6, pred7]))
log('saved cache/exp64_pred_v6recipe_v7.npy (row 0 v6 recipe, row 1 v7)')

# ---- CHECK 1 (before anything is written): outside P, v7 == shipped v6 -----
ids = rk_meta[ID].to_numpy()
v6s = pd.read_parquet(V6_PATH).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
if v6s.index.duplicated().any() or not pd.Index(ids).isin(v6s.index).all():
    sys.exit('FATAL: shipped v6 ids do not cover the ranking departures one-to-one')
a6 = v6s.reindex(ids).to_numpy().astype(np.int32)
r7 = np.rint(np.clip(pred7, 0.0, 200000.0)).astype(np.int32)     # what build() will write
r6 = np.rint(np.clip(pred6, 0.0, 200000.0)).astype(np.int32)
n_recipe = int((r6 != a6).sum())
n_out = int((r7[~P_rk] != a6[~P_rk]).sum())
print('\n' + '=' * 90)
print('CHECK 1 -- identity with the shipped v6 parquet (int32, aligned by MVT_ID_mvt)')
print('=' * 90)
print(f'  in-run v6 recipe vs shipped v6, ALL {len(ids):,} rows: {n_recipe:,} differ'
      + (f' (max |diff| {np.abs(r6.astype(np.int64) - a6).max()} s)' if n_recipe else ''))
print(f'  v7 vs shipped v6, the {int((~P_rk).sum()):,} rows OUTSIDE P: {n_out:,} differ')
if n_out:
    bad = np.where(~P_rk & (r7 != a6))[0]
    print(pd.DataFrame({'id': ids[bad[:20]], 'apt': apt_rk[bad[:20]], 'v6': a6[bad[:20]],
                        'v7': r7[bad[:20]]}).to_string(index=False))
    sys.exit('STOP: v7 differs from shipped v6 outside P -- v7 NOT written. Investigate before '
             'building (thread count / data / code drift since 09:48).')

# ---- CHECK 3: build + validate ---------------------------------------------
path = build_submission(ids, pred7, version=7, clip=(0.0, 200000.0))
print('\nexplicit re-validation:', flush=True)
assert Path(path).resolve() == V7_PATH.resolve(), f'build wrote {path}, expected {V7_PATH}'
validate(Path(V7_PATH))
v7s = pd.read_parquet(path).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
b7 = v7s.reindex(ids).to_numpy().astype(np.int32)
assert np.array_equal(b7, r7), 'on-disk v7 differs from the pre-build rounding'
n_out_disk = int((b7[~P_rk] != a6[~P_rk]).sum())
print(f'  re-read from disk: v7 vs v6 outside P differ on {n_out_disk} rows')
if n_out_disk:
    sys.exit('STOP: on-disk v7 differs from v6 outside P')

# ---- CHECK 2: inside P ------------------------------------------------------
print('\n' + '=' * 90)
print(f'CHECK 2 -- inside P ({int(P_rk.sum()):,} ranking rows), v7 vs shipped v6 (on-disk int values)')
print('=' * 90)
iP = np.where(P_rk)[0]
d = b7[iP].astype(np.int64) - a6[iP].astype(np.int64)
ch = np.abs(d) > 1
assert np.array_equal(irk, iP)
mon_rk = crk.month.to_numpy()
aptP = apt_rk[iP]
dp = p7[iP] - p6[iP]
print(f'  rows changed by > 1 s     : {int(ch.sum()):,} of {len(iP):,}  '
      f'(raised {int((d > 1).sum()):,}, lowered {int((d < -1).sum()):,})')
print(f'  shift, all P rows         : mean {d.mean():+.1f} s, median {np.median(d):+.0f} s, '
      f'sum {d.sum()/1e3:+.0f}k s')
print(f'  shift, changed rows       : mean {d[ch].mean():+.1f} s, median {np.median(d[ch]):+.0f} s, '
      f'max raise {d.max():+,} s, max lower {d.min():+,} s')
print(f'  {"threshold":>10s} {"v6":>6s} {"v7":>6s}   (P rows; outside P identical by check 1)')
for thr in (7200, 20000, 60000):
    print(f'  {">" + str(thr):>10s} {int((a6[iP] > thr).sum()):6d} {int((b7[iP] > thr).sum()):6d}')
q = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
print('  p7 - p6 quantiles        : ' + '  '.join(f'q{k}={v:+.3f}' for k, v in zip(q, np.percentile(dp, q))))
for thr in (0.01, 0.05, 0.1, 0.2, 0.3):
    print(f'    |dp| > {thr:<4}: {int((np.abs(dp) > thr).sum()):5d}  '
          f'(up {int((dp > thr).sum())}, down {int((dp < -thr).sum())})')
lir = aptP == 'LIRF'
print(f'  LIRF share                : {int(lir.sum())} of {len(iP)} P rows ({100*lir.mean():.1f}%); '
      f'{int((ch & lir).sum())} of {int(ch.sum())} changed rows ({100*(ch & lir).sum()/max(ch.sum(),1):.1f}%); '
      f'{100*np.abs(d[lir]).sum()/max(np.abs(d).sum(),1):.1f}% of total |shift|')
tab = pd.DataFrame({'apt': aptP, 'mon': mon_rk, 'd': d, 'ch': ch, 'dp': dp,
                    'p6': p6[iP], 'p7': p7[iP]})
print('\n  by airport:')
print(tab.groupby('apt').agg(n=('d', 'size'), changed=('ch', 'sum'), mean_shift=('d', 'mean'),
                             mean_p6=('p6', 'mean'), mean_p7=('p7', 'mean'),
                             max_absdp=('dp', lambda s: s.abs().max())).round(3).to_string())
print('\n  by month:')
print(tab.groupby('mon').agg(n=('d', 'size'), changed=('ch', 'sum'), mean_shift=('d', 'mean'),
                             mean_p6=('p6', 'mean'), mean_p7=('p7', 'mean')).round(3).to_string())
print('\n  LIRF by month:')
print(tab[lir].groupby('mon').agg(n=('d', 'size'), changed=('ch', 'sum'), mean_shift=('d', 'mean'),
                                  mean_p6=('p6', 'mean'), mean_p7=('p7', 'mean'),
                                  sd_p6=('p6', 'std'), sd_p7=('p7', 'std')).round(3).to_string())
top = np.argsort(-np.abs(d))[:12]
print('\n  largest changes:')
print(pd.DataFrame({'apt': aptP[top], 'mon': mon_rk[top], 'off_sched': os_rk[iP][top],
                    'v6': a6[iP][top], 'v7': b7[iP][top], 'p6': p6[iP][top],
                    'p7': p7[iP][top]}).round(3).to_string(index=False))
print(f'\n  whole file: v6 min/med/max {a6.min()}/{int(np.median(a6))}/{a6.max()}  '
      f'v7 min/med/max {b7.min()}/{int(np.median(b7))}/{b7.max()}')
print(f'\nNOT UPLOADED. peak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
