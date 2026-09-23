"""EXP63 step 2 unit -- one (option, fold) of the THREE-WAY target treatment.

Also the single home of every exp63 definition (feature build, boundaries,
regions, classifier, component regressors), so that the screen
(experiments/exp63_screen.py), this task and the collector cannot drift apart.

HYPOTHESIS
----------
v6 treats the target two ways: bulk (y <= 7200) by one L2 regressor, artefact
(y > 7200) by the tail branch. exp63 splits the bulk into NORMAL and DELAYED
(long but genuine, ~30 min - 2 h) and estimates the same conditional mean as a
probability-weighted mixture:

    E[y|x] = pt*et + (1-pt) * [ (1-q)*m_n(x) + q*m_d(x) ]

    pt, et : the v6 tail branch, UNCHANGED (global classifier, TailSpecialist
             inside P, anchor et). Taken from exp62's caches by the collector.
    q      : P(y > b_apt | y <= 7200, x)   -- the delayed-band classifier
    m_n    : regressor trained on y <= b_apt
    m_d    : regressor trained on b_apt < y <= 7200

NOTE ON NESTING: the three probabilities are p_t = pt, p_d = (1-pt)q,
p_n = (1-pt)(1-q). That is the same three-way partition as the originally
proposed P(y > b) then P(y > 7200 | y > b), nested in the other order. The
order is forced by the requirement that the v6 tail machinery is kept exactly:
with the other order p_t would come from a NEW classifier.

MEASURED FACTS IT RESTS ON (claude_log/2026-09-23_dev.log sections 13-18)
  * 1800 < y <= 7200 is 3.53% of rows and 24.4% of squared error (v5 OOF).
  * 1800 s is the 99.24th pct at LSZH but the 89.41st at EGLL -> the boundary
    must be airport-relative and computed IN-FOLD from training months.
  * The share of >1800 s flights flips seasonal sign by airport.
  * The LOO airport x day x hour residual ceiling is -4.9 s: the delayed state
    is per-flight, not shared. exp63 only pays if the band is DISCRIMINABLE
    per flight, beyond what the bulk regressor already ranks.

Three formulations, selected by --options:
  A:<bnd>:<region>   three-component soft mixture (above)
  B:<bnd>:<region>   one v6 bulk regressor with the NESTED out-of-fold q as an
                     extra feature (inner leave-one-month-out on the 11
                     training months; boundaries recomputed inside each inner fold)
  N                  NULL CONTROL: the v6 bulk regressor refitted with seed+1000.
                     Its paired difference against exp62's v6 fit is the size of
                     pure refit noise that the row bootstrap does not see.
  <bnd>    flat1800 | p95 | p97   (per-airport percentile of y <= 7200, in-fold)
  <region> pooled | ns | season   (which airports share one q classifier)

GATE / DECISION RULE
  Step 1 (exp63_screen.py) gates this: step 2 is only run for an option whose
  realistic mixture beats a matched two-way baseline on screening months
  3, 6, 9, 12 with a paired-bootstrap CI clear of zero.
  Step 2 (exp63_collect.py) adopts an option only if its 12-month paired
  bootstrap against v6 has a CI clear of zero (and it clearly exceeds N).

OUTPUT
  cache/exp63/units/<tag>__sSS__mMM.npz  (idx, pbulk, q, [mn, md], meta json)
  written atomically; an existing file is skipped, so re-running is safe.

USAGE
  # one cluster task
  python cluster/exp63_task.py --options A:p95:pooled --months 7 --threads 4
  # locally, several units in one process (features assembled once)
  python cluster/exp63_task.py --options A:p95:pooled,N --months 1-12 --threads 6
  # plumbing test (50k rows, months 3 and 7, tiny round counts)
  python cluster/exp63_task.py --smoke
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, CACHE, AIRPORTS
from prc.features import build as build_base, align_categories, CATS
from prc.io import load, peak_rss_gb

N_DEP = 2_085_047
UNIT_DIR = CACHE / 'exp63' / 'units'
SMOKE_DIR = CACHE / 'exp63_smoke'
SMOKE_ROWS = 50_000
SMOKE_MONTHS = (3, 7)

BOUNDARIES = ('flat1800', 'p95', 'p97')
# Which airports share one delayed-band classifier. 'ns' is the geographic split
# as proposed. 'season' moves EGLL to the summer-heavy group, because the
# measured seasonal sign puts it there (share > 1800 s: Jan 10.20% vs Jul 13.51%,
# dev.log 2026-09-23 s14.4). Edit here to change a split.
REGIONS = {
    'pooled': None,
    'ns': {'north': ('EDDF', 'EDDM', 'EHAM', 'LSZH', 'LFPG', 'EGLL'),
           'south': ('LIRF', 'LEBL', 'LEMD', 'LTFM')},
    'season': {'winter': ('EDDF', 'EDDM', 'EHAM', 'LSZH', 'LFPG'),
               'summer': ('LIRF', 'LEBL', 'LEMD', 'LTFM', 'EGLL')},
}
for _r, _g in REGIONS.items():
    if _g is not None:
        _all = [a for v in _g.values() for a in v]
        assert sorted(_all) == sorted(AIRPORTS), f'region map {_r} must partition AIRPORTS'

# Round budgets. 'full' is the real protocol; 'smoke' only proves the plumbing.
BUDGET = {
    'full':  dict(clf_rounds=300, md_max=3000, md_patience=100, reg_rounds=None),
    'smoke': dict(clf_rounds=15, md_max=30, md_patience=5, reg_rounds=15),
}


def log(t0, msg):
    print(f'[{time.time()-t0:6.0f}s | peak RSS {peak_rss_gb():.2f} GB] {msg}', flush=True)


# ----------------------------------------------------------------------------
# Features: the v6 build, copied from experiments/exp54_submit_v6.py::assemble
# (identical to exp62_cv.py phase A) so exp63 stays self-contained.
# ----------------------------------------------------------------------------
def assemble():
    from prc.congestion import build as build_cong
    from prc.weather import download, prepare, attach as attach_wx
    from prc.disruption import attach as attach_atfm
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
    y = d[TARGET].astype(float).to_numpy()
    month = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    del d
    (X,) = align_categories(X)
    if len(y) != N_DEP:
        sys.exit(f'FATAL: {len(y):,} DEP rows, expected {N_DEP:,} -- stale cache/training.parquet?')
    apt = np.asarray(X['APT'].astype(str), dtype=object)
    bad = sorted(set(apt) - set(AIRPORTS))
    if bad:
        sys.exit(f'FATAL: unexpected airports {bad}')
    return SimpleNamespace(X=X, y=y, month=month, apt=apt, FULL=FULL, BASE=BASE,
                           cats=[c for c in CATS if c in FULL],
                           gidx=np.arange(len(y)))


def subsample(ctx, n_rows=SMOKE_ROWS, seed=0):
    """Smoke only: a random subset of DEP rows. gidx keeps the global positions,
    so every saved array stays aligned to the 2,085,047-row DEP order."""
    rows = np.sort(np.random.default_rng(seed).choice(len(ctx.y), n_rows, replace=False))
    ctx.X = ctx.X.iloc[rows].reset_index(drop=True)
    for k in ('y', 'month', 'apt', 'gidx'):
        setattr(ctx, k, getattr(ctx, k)[rows])
    return ctx


# ----------------------------------------------------------------------------
# Boundaries: per-airport, from TRAINING rows only.
# ----------------------------------------------------------------------------
def boundaries(y, apt, train, kind):
    """{airport: b_apt}. Percentiles are over y <= TAIL_CAP (EUROCONTROL's own
    'not a taxi time' filter), training rows only."""
    if kind not in BOUNDARIES:
        raise ValueError(f'unknown boundary {kind!r}; choose from {BOUNDARIES}')
    out = {}
    for a in AIRPORTS:
        if kind == 'flat1800':
            out[a] = 1800.0
            continue
        s = train & (apt == a) & (y <= TAIL_CAP)
        if s.sum() < 200:
            raise RuntimeError(f'only {int(s.sum())} training rows at {a} for {kind}')
        out[a] = float(np.percentile(y[s], {'p95': 95.0, 'p97': 97.0}[kind]))
    return out


def row_boundary(apt, bmap):
    b = pd.Series(apt).map(bmap).to_numpy(float)
    if not np.isfinite(b).all():
        raise RuntimeError('rows with an airport missing from the boundary map')
    return b


# ----------------------------------------------------------------------------
# Model configurations
# ----------------------------------------------------------------------------
def _common(seed, threads):
    return dict(feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=threads, verbose=-1, deterministic=True, force_row_wise=True,
                seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)


def reg_params(cap, seed, threads, budget):
    """cap='v6'     : the shipped bulk regressor (511 leaves / lr 0.025 / 1400).
       cap='screen' : the pre-exp50 regressor (255 / 0.05 / 350), ~4x cheaper;
                      used in step 1 for BOTH arms so the comparison is matched."""
    p = dict(objective='regression', metric='rmse', min_data_in_leaf=100, **_common(seed, threads))
    if cap == 'v6':
        p.update(learning_rate=0.025, num_leaves=511); rounds = 1400
    elif cap == 'screen':
        p.update(learning_rate=0.05, num_leaves=255); rounds = 350
    else:
        raise ValueError(cap)
    return p, (budget['reg_rounds'] or rounds)


def clf_params(seed, threads):
    """The v6 classifier config (63 leaves / lr 0.05 / 300 rounds), on all 72
    features: the delayed band has ~70k positives, not the tail's 435, and its
    candidate signals (de-icing weather, congestion) live in the wx/cong/atfm
    groups the tail classifier deliberately leaves out."""
    return dict(objective='binary', metric='binary_logloss', learning_rate=0.05,
                num_leaves=63, min_data_in_leaf=50, **_common(seed, threads))


def md_params(seed, threads):
    """Delayed-band regressor: ~70k rows, so 63 leaves (about the bulk
    regressor's rows-per-leaf); rounds by in-fold early stopping."""
    return dict(objective='regression', metric='l2', learning_rate=0.025,
                num_leaves=63, min_data_in_leaf=100, **_common(seed, threads))


# ----------------------------------------------------------------------------
# Fits. All masks are boolean arrays over the rows of ctx.
# ----------------------------------------------------------------------------
def _train(params, rounds, ctx, cols, fit, target, valid=None):
    ds = lgb.Dataset(ctx.X.loc[fit, cols], target, categorical_feature=ctx.cats)
    if valid is None:
        return lgb.train(params, ds, num_boost_round=rounds)
    vmask, vtarget, patience = valid
    dv = lgb.Dataset(ctx.X.loc[vmask, cols], vtarget, reference=ds)
    return lgb.train(params, ds, num_boost_round=rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(patience, verbose=False)])


def fit_reg(ctx, cols, fit, pred, params, rounds):
    m = _train(params, rounds, ctx, cols, fit, ctx.y[fit])
    return np.clip(m.predict(ctx.X.loc[pred, cols]), 0, None)


def fit_q(ctx, cols, b_row, train, pred, region, seed, threads, budget):
    """P(y > b_apt | y <= 7200, x): fitted on training rows with y <= 7200,
    one classifier per region group. Returns q for the rows in `pred`."""
    groups = [('all', None)] if REGIONS[region] is None else list(REGIONS[region].items())
    q = np.full(len(ctx.y), np.nan)
    info = {}
    for g, apts in groups:
        ing = np.ones(len(ctx.y), bool) if apts is None else np.isin(ctx.apt, apts)
        k = train & ing & (ctx.y <= TAIL_CAP)
        lab = (ctx.y[k] > b_row[k]).astype(int)
        if lab.sum() < 20 or lab.sum() == len(lab):
            raise RuntimeError(f'q[{region}/{g}]: {int(lab.sum())} positives of {len(lab)}')
        m = _train(clf_params(seed, threads), budget['clf_rounds'], ctx, cols, k, lab)
        pv = pred & ing
        if pv.any():
            q[pv] = m.predict(ctx.X.loc[pv, cols])
        info[g] = dict(n=int(k.sum()), pos=int(lab.sum()))
    out = q[pred]
    if not np.isfinite(out).all():
        raise RuntimeError('q has NaN on prediction rows -- region map does not cover them')
    return out, info


def fit_md(ctx, cols, b_row, train, pred, val_month, seed, threads, budget):
    """m_d: rounds chosen by early stopping on ONE training month (the one after
    the validation month, cyclically), then refitted on all training-month band
    rows with rounds * 11/10. Never sees the validation month."""
    band = train & (ctx.y > b_row) & (ctx.y <= TAIL_CAP)
    inner = val_month % 12 + 1
    ki, ke = band & (ctx.month != inner), band & (ctx.month == inner)
    if ke.sum() < 20 or ki.sum() < 200:
        raise RuntimeError(f'm_d: {int(ki.sum())} fit / {int(ke.sum())} stop rows')
    p = md_params(seed, threads)
    m = _train(p, budget['md_max'], ctx, cols, ki, ctx.y[ki],
               valid=(ke, ctx.y[ke], budget['md_patience']))
    best = int(m.best_iteration or budget['md_max'])
    used = max(1, int(round(best * 11 / 10)))
    del m
    md = fit_reg(ctx, cols, band, pred, p, used)
    return md, dict(best=best, used=used, n=int(band.sum()), inner_month=inner)


# ----------------------------------------------------------------------------
# Options and one unit of work
# ----------------------------------------------------------------------------
def parse_option(s):
    s = s.strip()
    if s == 'N':
        return SimpleNamespace(form='N', boundary=None, region=None, tag='N', text=s)
    try:
        form, bnd, reg = s.split(':')
    except ValueError:
        raise SystemExit(f'bad option {s!r}: expected A:<bnd>:<region>, B:<bnd>:<region> or N')
    if form not in ('A', 'B') or bnd not in BOUNDARIES or reg not in REGIONS:
        raise SystemExit(f'bad option {s!r}: form A|B, boundary {BOUNDARIES}, '
                         f'region {tuple(REGIONS)}')
    return SimpleNamespace(form=form, boundary=bnd, region=reg, tag=f'{form}-{bnd}-{reg}', text=s)


def parse_months(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out or any(m < 1 or m > 12 for m in out):
        raise SystemExit(f'bad --months {s!r}')
    return sorted(set(out))


def unit_path(unit_dir, opt, seed, m):
    return unit_dir / f'{opt.tag}__s{seed:02d}__m{m:02d}.npz'


def save_npz(path, idx, arrays, meta):
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, idx=np.asarray(idx, np.int64),
             meta=np.array(json.dumps(meta, default=float)),
             **{k: np.asarray(v, np.float32) for k, v in arrays.items()})
    os.replace(tmp, path)


def run_unit(ctx, opt, m, seed, threads, budget, t0):
    """Fit one (option, fold). Returns (arrays for the validation rows, meta)."""
    y, FULL = ctx.y, ctx.FULL
    val = ctx.month == m
    train = ~val
    timing, meta = {}, dict(option=opt.text, month=m, seed=seed, n_val=int(val.sum()))

    if opt.form == 'N':
        ts = time.time()
        p, r = reg_params('v6', seed + 1000, threads, budget)
        pbulk = fit_reg(ctx, FULL, train & (y <= TAIL_CAP), val, p, r)
        timing['reg'] = time.time() - ts
        meta['timing_s'] = timing
        return dict(pbulk=pbulk), meta

    bmap = boundaries(y, ctx.apt, train, opt.boundary)
    b_row = row_boundary(ctx.apt, bmap)
    meta['boundary'] = bmap

    ts = time.time()
    q, qinfo = fit_q(ctx, FULL, b_row, train, val, opt.region, seed, threads, budget)
    timing['q'] = time.time() - ts
    meta['q_info'] = qinfo

    if opt.form == 'A':
        ts = time.time()
        p, r = reg_params('v6', seed, threads, budget)
        mn = fit_reg(ctx, FULL, train & (y <= b_row), val, p, r)
        timing['m_n'] = time.time() - ts
        ts = time.time()
        md, mdinfo = fit_md(ctx, FULL, b_row, train, val, m, seed, threads, budget)
        timing['m_d'] = time.time() - ts
        meta['m_d'] = mdinfo
        pbulk = (1 - q) * mn + q * md
        meta['timing_s'] = timing
        return dict(pbulk=pbulk, q=q, mn=mn, md=md), meta

    # B: nested out-of-fold q on the training months, then one v6 regressor.
    ts = time.time()
    qcol = np.full(len(y), np.nan)
    for j in sorted(set(ctx.month[train].tolist())):
        tr_j = train & (ctx.month != j)
        pr_j = train & (ctx.month == j)
        b_j = row_boundary(ctx.apt, boundaries(y, ctx.apt, tr_j, opt.boundary))
        qcol[pr_j], _ = fit_q(ctx, FULL, b_j, tr_j, pr_j, opt.region, seed, threads, budget)
    qcol[val] = q
    timing['q_inner'] = time.time() - ts
    if not np.isfinite(qcol).all():
        raise RuntimeError('B: nested q incomplete')
    ts = time.time()
    ctx.X['q_delay'] = qcol
    try:
        p, r = reg_params('v6', seed, threads, budget)
        pbulk = fit_reg(ctx, FULL + ['q_delay'], train & (y <= TAIL_CAP), val, p, r)
    finally:
        del ctx.X['q_delay']
    timing['reg'] = time.time() - ts
    meta['timing_s'] = timing
    return dict(pbulk=pbulk, q=q), meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--options', default='A:p95:pooled',
                    help='comma-separated, e.g. A:p95:pooled,B:p95:pooled,N')
    ap.add_argument('--months', default='1-12', help='e.g. 7 or 1-12 or 3,6,9,12')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)))
    ap.add_argument('--smoke', action='store_true',
                    help='plumbing test: 50k rows, months 3 and 7, tiny round counts, '
                         'writes under cache/exp63_smoke/')
    a = ap.parse_args(argv)
    t0 = time.time()

    opts = [parse_option(s) for s in a.options.split(',') if s.strip()]
    if a.smoke:
        months = list(SMOKE_MONTHS)
        if a.options == 'A:p95:pooled':
            opts = [parse_option(s) for s in ('A:p95:pooled', 'B:p95:ns', 'N')]
        unit_dir, budget = SMOKE_DIR / 'units', BUDGET['smoke']
        import shutil                     # a smoke must refit, never skip stale units
        shutil.rmtree(unit_dir, ignore_errors=True)
    else:
        months = parse_months(a.months)
        unit_dir, budget = UNIT_DIR, BUDGET['full']
    unit_dir.mkdir(parents=True, exist_ok=True)

    todo = [(o, m) for o in opts for m in months
            if not unit_path(unit_dir, o, a.seed, m).exists()]
    for o in opts:
        for m in months:
            if (o, m) not in todo:
                print(f'{unit_path(unit_dir, o, a.seed, m).name} exists, skipping', flush=True)
    if not todo:
        return
    log(t0, f'{len(todo)} unit(s) to fit: {[(o.tag, m) for o, m in todo]} | threads {a.threads}')

    ctx = assemble()
    if a.smoke:
        ctx = subsample(ctx)
    log(t0, f'features assembled: {ctx.X.shape}')

    for o, m in todo:
        tu = time.time()
        arrays, meta = run_unit(ctx, o, m, a.seed, a.threads, budget, t0)
        val = ctx.month == m
        for k, v in arrays.items():
            if v.shape != (int(val.sum()),) or not np.isfinite(v).all():
                raise RuntimeError(f'{o.tag} m{m}: {k} has shape {v.shape} or non-finite values')
        meta.update(seconds=time.time() - tu, peak_rss_gb=peak_rss_gb(), smoke=a.smoke)
        out = unit_path(unit_dir, o, a.seed, m)
        save_npz(out, ctx.gidx[val], arrays, meta)
        yv = ctx.y[val]
        bulk = yv <= TAIL_CAP
        rm = float(np.sqrt(np.mean((yv[bulk] - arrays['pbulk'][bulk]) ** 2)))
        extra = ''
        if 'q' in arrays:
            b = row_boundary(ctx.apt[val], meta['boundary'])
            lab = (yv > b)[bulk]
            extra = (f' | q mean {arrays["q"][bulk].mean():.4f} vs band rate {lab.mean():.4f}')
        log(t0, f'{out.name}: n={int(val.sum()):,} bulk-RMSE(pbulk) {rm:.2f}{extra} '
                f'| {meta["timing_s"]} | {meta["seconds"]:.0f}s')
    log(t0, 'done')


if __name__ == '__main__':
    main()
