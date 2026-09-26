"""Metrics and the validation protocol.

THE ONE RULE: evaluate on a target capped at TAIL_CAP. The uncapped metric on
2025 data is ~95% noise from ~150 rows, and the 2026 scored labels demonstrably
do not contain that tail (the leaderboard leader scores 239.13 while Jan+Jul 2025
has an oracle floor of 391.0). Validating uncapped makes you reject good changes.
See claude_log EXP01.
"""
import numpy as np
import pandas as pd

from .config import TAIL_CAP


def rmse(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def oracle_floor(y, thr: float = TAIL_CAP) -> float:
    """Lowest RMSE reachable if every row <= thr is predicted perfectly and rows
    above thr can only be predicted at their group mean. Depends on labels alone."""
    y = np.asarray(y, float)
    tail = y[y > thr]
    if len(tail) == 0:
        return 0.0
    return float(np.sqrt(((tail - tail.mean()) ** 2).sum() / len(y)))


def score(y, p, month=None, cap: float = TAIL_CAP) -> dict:
    """Headline capped RMSE plus the per-month split.

    January and July behave very differently (a 2026 competitor reports 224 vs
    297 on comparable months), so a blended number hides regressions.
    """
    y = np.asarray(y, float); p = np.asarray(p, float)
    keep = y <= cap
    out = {'rmse_capped': rmse(y[keep], p[keep]),
           'rmse_uncapped': rmse(y, p),
           'n': int(keep.sum()), 'n_dropped': int((~keep).sum())}
    if month is not None:
        month = np.asarray(month)
        for m in np.unique(month[keep]):
            sel = keep & (month == m)
            out[f'rmse_m{int(m):02d}'] = rmse(y[sel], p[sel])
    return out


def seasonal_split(df, time_col='MVT_TIME_UTC_mvt'):
    """Primary gate: hold out Jan+Jul 2025, matching the scored months' seasonality."""
    m = pd.to_datetime(df[time_col]).dt.month
    return ~m.isin([1, 7]), m.isin([1, 7])


def forward_split(df, time_col='MVT_TIME_UTC_mvt'):
    """Secondary gate: train Jan-Sep, test Oct-Dec. Tests temporal extrapolation."""
    m = pd.to_datetime(df[time_col]).dt.month
    return m <= 9, m >= 10


# ---------------------------------------------------------------------------
# Paired bootstrap. EXP06/EXP07: the uncapped holdout RMSE has a bootstrap sd of
# ~47s because ~149 tail rows dominate it, while the top 8 leaderboard teams span
# 5.9s. An ABSOLUTE local RMSE therefore cannot resolve anything that matters.
# Resampling the same rows for both models cuts the sd by ~2.7x, because the tail
# rows move both models together. Always compare models this way.
# ---------------------------------------------------------------------------

def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 500, seed: int = 0) -> dict:
    """Bootstrap the RMSE difference (a - b) on shared resamples.

    Returns the mean difference, its sd, a 95% interval, and P(a worse than b).
    A negative mean means `a` is better.
    """
    y = np.asarray(y, float)
    da = (y - np.asarray(pred_a, float)) ** 2
    db = (y - np.asarray(pred_b, float)) ** 2
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(da[idx].mean()) - np.sqrt(db[idx].mean())
    return {'diff': float(diffs.mean()), 'sd': float(diffs.std()),
            'lo': float(np.percentile(diffs, 2.5)),
            'hi': float(np.percentile(diffs, 97.5)),
            'p_a_worse': float((diffs > 0).mean())}


def block_bootstrap(y, pred_a, pred_b, block, n_boot: int = 1000, seed: int = 0) -> dict:
    """paired_bootstrap with whole BLOCKS resampled (airport x local day). Squared error clusters by
    airport-day (the worst 5% of airport-days carry 23% of ordinary SSE), so the row bootstrap is too
    narrow for bulk changes; this is the honest interval (RMSE review 2026-09-26). Same keys, plus
    'point' (the plain RMSE difference) and 'n_blocks'."""
    y = np.asarray(y, float)
    da = (y - np.asarray(pred_a, float)) ** 2
    db = (y - np.asarray(pred_b, float)) ** 2
    _, inv = np.unique(block, return_inverse=True)
    SA, SB, N = np.bincount(inv, da), np.bincount(inv, db), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    nb = len(N)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        n = w @ N
        diffs[i] = np.sqrt(w @ SA / n) - np.sqrt(w @ SB / n)
    return {'diff': float(diffs.mean()), 'sd': float(diffs.std()),
            'lo': float(np.percentile(diffs, 2.5)), 'hi': float(np.percentile(diffs, 97.5)),
            'p_a_worse': float((diffs > 0).mean()),
            'point': float(np.sqrt(SA.sum() / N.sum()) - np.sqrt(SB.sum() / N.sum())), 'n_blocks': int(nb)}


def decompose(y, pred, thr: float = TAIL_CAP) -> dict:
    """Split RMSE into its bulk and tail contributions.

    total^2 = (n_bulk * rmse_bulk^2 + n_tail * rmse_tail^2) / n
    The tail term is what separates us from the leaderboard, so it gets reported
    explicitly rather than being buried in a single number.
    """
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    tail = y > thr
    n = len(y)
    out = {'rmse': rmse(y, pred),
           'n_bulk': int((~tail).sum()), 'rmse_bulk': rmse(y[~tail], pred[~tail]),
           'n_tail': int(tail.sum()),
           'rmse_tail': rmse(y[tail], pred[tail]) if tail.any() else 0.0}
    out['contrib_bulk'] = float(np.sqrt(out['n_bulk'] * out['rmse_bulk'] ** 2 / n))
    out['contrib_tail'] = float(np.sqrt(out['n_tail'] * out['rmse_tail'] ** 2 / n))
    return out


# ---------------------------------------------------------------------------
# Month-wise cross-validation.
#
# The Jan+Jul-2025 holdout has only ~149 tail rows, and those rows drive the
# metric, so its bootstrap sd is ~47s. Rotating the holdout over all twelve
# months makes every row an out-of-fold prediction: 2.08M rows and 584 tail
# rows instead of 344k and 149. Noise should fall by roughly sqrt(4) = 2x.
#
# CAVEAT, and it is a real one: leave-one-month-out trains on months that come
# AFTER the held-out month, whereas the true task extrapolates from 2025 into
# 2026. So OOF numbers are optimistic in an absolute sense. They are still the
# right tool for RANKING two candidate models, because the same leakage applies
# to both. Use `forward_split` as the secondary check on temporal drift.
# ---------------------------------------------------------------------------

def month_cv_oof(X, y, month, fit_predict, months=None, verbose=True):
    """Out-of-fold predictions, holding out one calendar month at a time.

    fit_predict(X_tr, y_tr, X_va) -> predictions for X_va.
    Returns an array of OOF predictions aligned to y (NaN for any month skipped).
    """
    import time as _t
    y = np.asarray(y, float)
    month = np.asarray(month)
    oof = np.full(len(y), np.nan)
    todo = sorted(set(month.tolist())) if months is None else list(months)
    for m in todo:
        va = month == m
        tr = ~va
        t0 = _t.time()
        oof[va] = fit_predict(X[tr], y[tr], X[va])
        if verbose:
            print(f'    month {int(m):02d}: train {tr.sum():,} -> val {va.sum():,}  '
                  f'RMSE {rmse(y[va], oof[va]):8.2f}  ({_t.time()-t0:.0f}s)', flush=True)
    return oof


def per_month(y, pred, month, thr: float = TAIL_CAP) -> pd.DataFrame:
    """RMSE by calendar month, with the bulk/tail split alongside.

    Reported separately because the two scored months behave very differently --
    July has consistently run ~50s harder than January for us.
    """
    y = np.asarray(y, float); pred = np.asarray(pred, float); month = np.asarray(month)
    rows = []
    for m in sorted(set(month.tolist())):
        s = (month == m) & np.isfinite(pred)
        if not s.any():
            continue
        d = decompose(y[s], pred[s], thr)
        rows.append({'month': int(m), 'n': int(s.sum()), 'rmse': d['rmse'],
                     'n_tail': d['n_tail'], 'rmse_bulk': d['rmse_bulk'],
                     'rmse_tail': d['rmse_tail'], 'contrib_tail': d['contrib_tail']})
    return pd.DataFrame(rows)
