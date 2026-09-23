"""EXP62 step 1 -- can anything observable separate ARTEFACT from REAL inside P?

P = {AOBT_3_flt missing} AND {off_sched > 7200}: 4,944 training rows, 450 tail.
Inside P a row is either
    ARTEFACT  the airport stamped BLOCK at SCHED, so y ~= off_sched, or
    REAL      the airport recorded a genuine off-block, y ~ 1,000 s.
The shipped TailSpecialist sees only airport x off_sched structure (plus stand,
pier, runway, type, hour, dow). This screen asks whether per-flight
information that is observable at scoring time adds to it.

MEASURED PREMISE CORRECTION (read before trusting any "operator" result from
earlier experiments): inside P the ENTIRE _flt block is NaN -- including
AIRCRAFT_OPERATOR_flt, MARKET_SEGMENT_flt and WK_TBL_CAT_flt -- in training
AND ranking. So three of the specialist's twelve inputs are constant-NaN there,
and exp38's "(airport, operator)" propensity was a no-op inside P. The operator
has to be read from the callsign prefix in FLIGHT_mvt instead.

LEAKAGE GUARD: departure BLOCK_TIME is the target. It is set to NaT on every
DEP row immediately after loading, before any feature code runs, so no
candidate can read it -- for our flight or any other. Arrival BLOCK_TIME
(in-block) is kept: it is present in ranking.parquet.

CANDIDATE FAMILIES (pre-specified; each tested as an addition to the
specialist's LightGBM component, table + isotonic components unchanged):
  rot    rotation timing at our stand (arrival in-blocks, other departures)
  pier   the same at pier level
  inb    inbound existence: arrivals at our stand in the last 3/6/12 h
  op     in-fold target-encoded stamping rate by callsign prefix (inside P,
         and over all route-D departures with off_sched > 1800)
  te     in-fold TE by operator, stand, pier, destination (shrunk to airport)
  sched  scheduled hour / minute / round-hour
  cs     callsign form: missing, length, garbled (letter at position 4),
         alphanumeric suffix; flight rule
  other  same-callsign records, operator's route-D share nearby, MVT_ID rank
         residual, the arrival record of our own flight at its destination
  all    everything

Evaluation: 12-fold leave-one-month-out INSIDE P only. Brier, log-loss, AUC,
paired bootstrap over rows (B - A; negative = candidate better), and the
implied mixture RMSE inside P with a PROXY bulk prediction pb = 1000 s.
Go/no-go: some family improves inside-P Brier with a 95% CI clear of zero.

Run:
    systemd-inhibit --what=idle:sleep:handle-lid-switch --why=exp62 \
      systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
      /home/thinklex/prc2026-venv/bin/python experiments/exp62_screen.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb  # noqa: F401  (kept: the specialist trains LightGBM)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prc.config import TAIL_CAP, CACHE
from prc.features import build as build_base, align_categories
from prc.io import peak_rss_gb
from prc.validation import paired_bootstrap
from prc.models.tail_specialist import TailSpecialist, TailSpecialistX as SpecX, SPEC_FEATURES
from prc.p_features import load_set, candidates, te_features, FAMILIES, TE_COLS

pd.set_option('display.width', 250)
t0 = time.time()
PB_PROXY = 1000.0
THREADS = 6


def log(msg):
    print(f'[{time.time()-t0:6.0f}s | RSS {peak_rss_gb():.2f} GB] {msg}', flush=True)


def brier(p, t):
    return float(np.mean((p - t) ** 2))


def logloss(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def boot_mean_diff(d, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {'diff': float(d.mean()), 'sd': float(bs.std()), 'lo': float(np.percentile(bs, 2.5)),
            'hi': float(np.percentile(bs, 97.5)), 'lo_bonf': float(np.percentile(bs, 0.25)),
            'hi_bonf': float(np.percentile(bs, 99.75))}


def safe_auc(t, x):
    ok = np.isfinite(x)
    if ok.sum() < 20 or len(np.unique(t[ok])) < 2 or len(np.unique(x[ok])) < 2:
        return np.nan
    return float(roc_auc_score(t[ok], x[ok]))


if __name__ == '__main__':
    # ---------------- candidates: training ----------------
    ftr = CACHE / 'exp62_cand_training.parquet'
    fpool = CACHE / 'exp62_pool_training.parquet'
    frk = CACHE / 'exp62_cand_ranking.parquet'
    d, depm, y_dep = load_set('training')
    log(f'training loaded: {len(d):,} rows, {int(depm.sum()):,} DEP')
    if ftr.exists() and fpool.exists():
        cand = pd.read_parquet(ftr); pool = pd.read_parquet(fpool)
        log('training candidates read from cache')
    else:
        cand, pool, _ = candidates(d, depm)
        cand.to_parquet(ftr); pool.to_parquet(fpool)
        log(f'training candidates built: {cand.shape}')
    # specialist inputs, exactly as exp54 builds them (training vocabulary)
    dep = d[depm].reset_index(drop=True)
    del d
    X = build_base(dep)
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    (X,) = align_categories(X)
    XP = X.loc[cand.index, SPEC_FEATURES].reset_index(drop=True)
    del X, dep
    y = y_dep[cand.index.to_numpy()]
    pool_y = y_dep[pool['pos'].to_numpy()]
    del y_dep
    cand = cand.reset_index(drop=True)
    log(f'P rows {len(cand):,}  tail {int((y > TAIL_CAP).sum())}  '
        f'artefact {int((np.abs(y - cand.off_sched) < 60).sum())}  | '
        f'route-D pool (off_sched>1800) {len(pool):,}')

    # ---------------- candidates: ranking ----------------
    if frk.exists():
        cand_rk = pd.read_parquet(frk)
    else:
        drk, depm_rk, _ = load_set('ranking')
        cand_rk, _, _ = candidates(drk, depm_rk)
        cand_rk.to_parquet(frk)
        del drk
    log(f'ranking P rows {len(cand_rk):,}')

    t_tail = (y > TAIL_CAP).astype(int)
    t_art = (np.abs(y - cand.off_sched.to_numpy()) < 60).astype(int)
    pool_art = (np.abs(pool_y - pool.off_sched.to_numpy()) < 60).astype(float)
    apt = cand.apt.to_numpy(object)
    os_p = cand.off_sched.to_numpy(float)
    miss = np.ones(len(cand), bool)
    month = cand.month.to_numpy()
    lirf = apt == 'LIRF'

    # ---------------- LOMO inside P ----------------
    arms = ['base'] + list(FAMILIES)
    oof = {a: np.full(len(cand), np.nan) for a in arms}
    te_oof = {c: np.full(len(cand), np.nan) for c in list(TE_COLS) + ['te_op_rd']}
    for m in range(1, 13):
        va = month == m
        tr = ~va
        tf = time.time()
        Xf = pd.concat([XP, cand[[c for c in FAMILIES['all'] if not c.startswith('te_')]]], axis=1)
        pk = (pool.month != m).to_numpy()
        te_tr, te_va = te_features(cand[tr], t_tail[tr], pool[pk], pool_art[pk], cand[va])
        for c in te_tr.columns:
            col = np.full(len(cand), np.nan)
            col[tr], col[va] = te_tr[c].to_numpy(), te_va[c].to_numpy()
            Xf[c] = col
        for c in te_oof:
            te_oof[c][va] = Xf[c].to_numpy()[va]
        for a in arms:
            if a == 'base':
                sp = TailSpecialist(threads=THREADS)
            else:
                sp = SpecX(extra=FAMILIES[a], threads=THREADS)
            sp.fit(Xf[tr], y[tr], apt[tr], os_p[tr], miss[tr])
            oof[a][va] = sp.predict(Xf[va], apt[va], os_p[va], miss[va])[0]
        log(f'fold {m:02d}: val {int(va.sum()):4d} rows, {int(t_tail[va].sum()):3d} tail '
            f'({time.time()-tf:.0f}s)')
    np.save(CACHE / 'exp62_screen_oof.npy', np.vstack([oof[a] for a in arms]))
    np.save(CACHE / 'exp62_screen_te_oof.npy', np.vstack([te_oof[c] for c in te_oof]))
    np.save(CACHE / 'exp62_screen_y.npy', y)
    log(f'saved OOF for arms {arms}')

    # ---------------- report 1: coverage and univariate AUC ----------------
    print('\n' + '=' * 110)
    print('UNIVARIATE SCREEN inside P  (AUC on non-null rows; TE columns are out-of-fold)')
    print('=' * 110)
    feats = [c for c in FAMILIES['all']]
    rk_cov = {}
    for c in feats:
        if c in TE_COLS or c == 'te_op_rd':
            k = TE_COLS.get(c, 'k_op')
            if c == 'te_op_rd':
                seen = set(pool.k_op)
            else:
                seen = set(cand[k])
            rk_cov[c] = float(cand_rk[k].isin(seen).mean())
        else:
            rk_cov[c] = float(cand_rk[c].notna().mean())
    rows = []
    for c in feats:
        x = te_oof[c] if c in te_oof else cand[c].to_numpy(float)
        a1, a2, a3 = safe_auc(t_tail, x), safe_auc(t_tail[lirf], x[lirf]), safe_auc(t_art, x)
        rows.append({'family': next(f for f in FAMILIES if f != 'all' and c in FAMILIES[f]),
                     'feature': c, 'cov_tr': float(np.isfinite(x).mean()), 'cov_rk': rk_cov[c],
                     'cov_tr_lirf': float(np.isfinite(x[lirf]).mean()),
                     'auc_tail': a1, 'auc_tail_lirf': a2, 'auc_artefact': a3})
    uni = pd.DataFrame(rows)
    # reference: the specialist's own OOF probability and off_sched
    ref = pd.DataFrame([
        {'family': 'REF', 'feature': 'specialist_oof', 'cov_tr': 1.0, 'cov_rk': 1.0, 'cov_tr_lirf': 1.0,
         'auc_tail': safe_auc(t_tail, oof['base']), 'auc_tail_lirf': safe_auc(t_tail[lirf], oof['base'][lirf]),
         'auc_artefact': safe_auc(t_art, oof['base'])},
        {'family': 'REF', 'feature': 'off_sched', 'cov_tr': 1.0, 'cov_rk': 1.0, 'cov_tr_lirf': 1.0,
         'auc_tail': safe_auc(t_tail, os_p), 'auc_tail_lirf': safe_auc(t_tail[lirf], os_p[lirf]),
         'auc_artefact': safe_auc(t_art, os_p)}])
    uni = pd.concat([ref, uni], ignore_index=True)
    uni.to_csv(CACHE / 'exp62_screen_univariate.csv', index=False)
    print(uni.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # ---------------- report 2: incremental value over the specialist ----------------
    print('\n' + '=' * 110)
    print('INCREMENTAL over the shipped specialist, 12-fold LOMO inside P '
          f'(n={len(cand)}, tail={int(t_tail.sum())}; LIRF n={int(lirf.sum())}, '
          f'tail={int(t_tail[lirf].sum())})')
    print('diff = candidate - base; NEGATIVE = candidate better. 95% CI; [Bonf] = 99.5% CI '
          f'(9 families tested)')
    print(f'mixture RMSE inside P uses a PROXY bulk prediction pb = {PB_PROXY:.0f} s, '
          'not the real regressor')
    print('=' * 110)
    base = oof['base']
    mix = {a: oof[a] * os_p + (1 - oof[a]) * PB_PROXY for a in arms}
    res = []
    for a in arms:
        p = oof[a]
        db = (p - t_tail) ** 2 - (base - t_tail) ** 2
        bb = boot_mean_diff(db)
        pc = np.clip(p, 1e-6, 1 - 1e-6); bc = np.clip(base, 1e-6, 1 - 1e-6)
        ll = -(t_tail * np.log(pc) + (1 - t_tail) * np.log(1 - pc))
        llb = -(t_tail * np.log(bc) + (1 - t_tail) * np.log(1 - bc))
        bl = boot_mean_diff(ll - llb)
        dbl = db[lirf]
        bbl = boot_mean_diff(dbl)
        pr = paired_bootstrap(y, mix[a], mix['base'], n_boot=1000) if a != 'base' else None
        e = (y - mix[a]) ** 2
        res.append({
            'arm': a, 'n_feat': len(FAMILIES.get(a, [])),
            'brier': brier(p, t_tail), 'd_brier': bb['diff'], 'd_brier_lo': bb['lo'],
            'd_brier_hi': bb['hi'], 'bonf_lo': bb['lo_bonf'], 'bonf_hi': bb['hi_bonf'],
            'logloss': logloss(p, t_tail), 'd_ll': bl['diff'], 'd_ll_lo': bl['lo'], 'd_ll_hi': bl['hi'],
            'auc': safe_auc(t_tail, p), 'auc_lirf': safe_auc(t_tail[lirf], p[lirf]),
            'brier_lirf': brier(p[lirf], t_tail[lirf]), 'd_brier_lirf': bbl['diff'],
            'd_brier_lirf_lo': bbl['lo'], 'd_brier_lirf_hi': bbl['hi'],
            'mixRMSE_P': float(np.sqrt(e.mean())),
            'd_mix': pr['diff'] if pr else 0.0, 'd_mix_lo': pr['lo'] if pr else 0.0,
            'd_mix_hi': pr['hi'] if pr else 0.0,
            'FA_SSE_1e9': float(e[t_tail == 0].sum() / 1e9),
            'MT_SSE_1e9': float(e[t_tail == 1].sum() / 1e9)})
    inc = pd.DataFrame(res)
    inc.to_csv(CACHE / 'exp62_screen_incremental.csv', index=False)
    with pd.option_context('display.float_format', lambda v: f'{v:.5f}' if abs(v) < 1 else f'{v:.2f}'):
        print(inc[['arm', 'n_feat', 'brier', 'd_brier', 'd_brier_lo', 'd_brier_hi', 'bonf_lo',
                   'bonf_hi', 'logloss', 'd_ll', 'd_ll_lo', 'd_ll_hi', 'auc', 'auc_lirf']]
              .to_string(index=False))
        print()
        print(inc[['arm', 'brier_lirf', 'd_brier_lirf', 'd_brier_lirf_lo', 'd_brier_lirf_hi',
                   'mixRMSE_P', 'd_mix', 'd_mix_lo', 'd_mix_hi', 'FA_SSE_1e9', 'MT_SSE_1e9']]
              .to_string(index=False))
    passing = inc[(inc.arm != 'base') & (inc.d_brier_hi < 0)]
    print('\nGO/NO-GO: families whose inside-P Brier 95% CI is clear of zero (improving): '
          f'{list(passing.arm) if len(passing) else "NONE"}')
    print(f'\npeak RSS {peak_rss_gb():.2f} GB   total {(time.time()-t0)/60:.1f} min')
