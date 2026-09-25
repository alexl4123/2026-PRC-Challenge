"""Shared harness for the Discord-insights evaluation (2026-09-25).

RESIDUAL HARNESS (ideas 2, 4, 5, 6 of the evaluation):
  base   = cache/w2/v15s_oof.npy, the v15-structure 2025 leave-one-month-out prediction
           (row order of load('training', phase='DEP'); asserted against cache/v11/base.npz)
  target = y - base, fitted on rows outside P with y <= 7200 (bulk; the tail zone P keeps its
           specialist and is never corrected)
  model  = one LightGBM per arm, 12 folds leave-one-month-out, a fixed 500k-row subsample of
           the fold's training rows (the SAME rows for every arm, so arms are paired)
  served = base + correction on rows outside P; P rows keep the base. Not shrunk.
  CONTROL arm = the context columns only (APT, off_aobt3, off_sched, base). Every candidate is
  reported against the base AND against the control; candidate - control is the number that
  attributes a gain to the candidate columns.

Scoring (evaluate): paired bootstrap on ALL flights, 12 months / scored months (1, 7) /
January / July, months better of 12, and a concentration check (the gain after removing the
100 rows it helped most; the share of the net SSE gain carried by the top 100 rows).

Nothing here writes into cache/; per-arm predictions go to the session scratchpad (SCRATCH).
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from prc.config import CACHE
from prc.validation import paired_bootstrap
from prc.io import peak_rss_gb

SCRATCH = Path(os.environ.get('DISCORD_SCRATCH',
               '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/'
               'ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/discord_eval'))
SCRATCH.mkdir(parents=True, exist_ok=True)
N_SUB = 500_000
PARAMS = dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              lambda_l2=5.0, feature_fraction=0.9, num_threads=6, verbose=-1, seed=0,
              deterministic=True, force_row_wise=True)
ROUNDS = 300


def load_base():
    """y, month, P, D, base (v15 OOF) and the context frame, all in load('training', 'DEP') order."""
    b = np.load(CACHE / 'v11' / 'base.npz')
    y, month, P, D = b['y'], b['month'], b['P'], b['D']
    base = np.load(CACHE / 'w2' / 'v15s_oof.npy').astype(float)
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet',
                         columns=['MVT_ID_mvt', 'APT', 'off_sched', 'off_aobt3', 'y'])
    assert len(dd) == len(y) == len(base)
    assert np.allclose(dd.y.to_numpy(float), y, equal_nan=True), 'dep_training not aligned with base.npz'
    ctx = pd.DataFrame({'APT': pd.Categorical(dd.APT.astype(str)),
                        'off_aobt3': dd.off_aobt3.to_numpy(float),
                        'off_sched': dd.off_sched.to_numpy(float),
                        'base': base})
    return dict(y=y, month=month, P=P, D=D, base=base, ctx=ctx, ids=dd.MVT_ID_mvt.to_numpy())


def residual_arms(B, arms, tag, rounds=ROUNDS, n_sub=N_SUB, log=print):
    """arms: {name: DataFrame of extra columns (row-aligned) or None for the control}.
    Returns {name: served prediction}. Caches each arm's prediction under SCRATCH/<tag>__<name>.npy."""
    y, month, P, base, ctx = B['y'], B['month'], B['P'], B['base'], B['ctx']
    r = y - base
    fit_ok = ~P & (y <= 7200) & np.isfinite(r)
    out = {}
    for name, extra in arms.items():
        f = SCRATCH / f'{tag}__{name}.npy'
        if f.exists():
            out[name] = np.load(f); log(f'  {name}: cached'); continue
        fold_fn = extra if callable(extra) else None           # fold-dependent columns (nested TE)
        if fold_fn is None:
            X = ctx if extra is None else pd.concat([ctx, extra.reset_index(drop=True)], axis=1)
        corr = np.zeros(len(y))
        t0 = time.time()
        for m in range(1, 13):
            if fold_fn is not None:
                X = pd.concat([ctx, fold_fn(m).reset_index(drop=True)], axis=1)
            trn = np.flatnonzero(fit_ok & (month != m))
            trn = np.sort(np.random.default_rng(m).choice(trn, size=min(n_sub, len(trn)), replace=False))
            va = np.flatnonzero((month == m) & ~P)
            mdl = lgb.train(PARAMS, lgb.Dataset(X.iloc[trn], r[trn], categorical_feature=['APT']),
                            num_boost_round=rounds)
            corr[va] = mdl.predict(X.iloc[va])
        pred = np.clip(base + corr, 0, None)
        np.save(f, pred.astype(np.float32))
        out[name] = pred
        log(f'  {name}: {X.shape[1]} cols, {time.time() - t0:.0f} s, peak RSS {peak_rss_gb():.2f} GB')
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def concentration(y, a, b, k=100):
    """(RMSE diff a-b after dropping the k rows a helped most, share of the net SSE gain in those k
    rows, RMSE diff a-b after dropping the k rows with the largest |change in squared error| in
    EITHER direction -- the symmetric version, which does not bias against every change)."""
    d = (y - a) ** 2 - (y - b) ** 2
    order = np.argsort(d)                  # most negative (most helped) first
    keep = np.ones(len(y), bool); keep[order[:k]] = False
    net = d.sum()
    share = float(d[order[:k]].sum() / net) if net < 0 else float('nan')
    keep2 = np.ones(len(y), bool); keep2[np.argsort(-np.abs(d))[:k]] = False
    return (rmse(y[keep], a[keep]) - rmse(y[keep], b[keep]), share,
            rmse(y[keep2], a[keep2]) - rmse(y[keep2], b[keep2]))


def compare(B, a, b, n_boot=500):
    """Dict of paired-bootstrap results a - b (negative = a better) on all flights."""
    y, month = B['y'], B['month']
    res = {}
    for scope, k in (('12m', np.ones(len(y), bool)), ('scored', np.isin(month, [1, 7])),
                     ('jan', month == 1), ('jul', month == 7)):
        bb = paired_bootstrap(y[k], a[k], b[k], n_boot=n_boot)
        res[scope] = (bb['diff'], bb['lo'], bb['hi'])
    per = [rmse(y[month == m], a[month == m]) - rmse(y[month == m], b[month == m]) for m in range(1, 13)]
    res['months_better'] = int(sum(v < 0 for v in per))
    res['worst_month'] = float(max(per))
    res['drop100_12m'], res['top100_share_12m'], res['dropabs100_12m'] = concentration(y, a, b)
    S = np.isin(month, [1, 7])
    res['drop100_scored'], res['top100_share_scored'], res['dropabs100_scored'] = concentration(y[S], a[S], b[S])
    return res


def fmt(res):
    f = lambda t: f'{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]'
    return (f"12m {f(res['12m'])} | scored {f(res['scored'])} | Jan {f(res['jan'])} | Jul {f(res['jul'])} | "
            f"{res['months_better']}/12 better (worst {res['worst_month']:+.2f}) | w/o top-100 rows: "
            f"12m {res['drop100_12m']:+.2f}, scored {res['drop100_scored']:+.2f} | w/o top-100 |d|: "
            f"12m {res['dropabs100_12m']:+.2f}, scored {res['dropabs100_scored']:+.2f}")


def report_arms(B, preds, control='control', log=print, save=None):
    """Each arm vs the base and vs the control."""
    y, month, base = B['y'], B['month'], B['base']
    S = np.isin(month, [1, 7])
    log(f'  base v15s: 12m {rmse(y, base):.3f}  scored {rmse(y[S], base[S]):.3f}')
    rows = {}
    for name, p in preds.items():
        rows[name] = {'vs_base': compare(B, p, base)}
        log(f'  {name:28s} vs base   : {fmt(rows[name]["vs_base"])}')
        if name != control and control in preds:
            rows[name]['vs_control'] = compare(B, p, preds[control])
            log(f'  {name:28s} vs CONTROL: {fmt(rows[name]["vs_control"])}')
    if save:
        json.dump(rows, open(SCRATCH / save, 'w'), indent=1)
    return rows
