"""Shared plumbing for the alternative-model-family experiments (exp40+).

The question these experiments ask is narrow: swap the BULK REGRESSOR of the
mixture for a different model family and see whether the bulk term moves. The
tail classifier and the tail estimate are held fixed at the current best
configuration (r-full__c-base), so any difference measured is attributable to
the regressor alone.

    E[y|x] = P(T|x)*E[y|T,x] + (1 - P(T|x))*E[y|~T,x]
             |___ held fixed ___|            |__ the thing we vary __|

Two things live here:

1. `design()` -- the 72-column feature matrix, cached to disk. It reproduces
   cluster/cv_task.assemble() exactly (asserted column-for-column by exp40)
   but loads a slim column subset for the congestion pass, which is what
   keeps the peak under the laptop's budget.
2. `numeric_design()` -- the same matrix re-expressed for models that cannot
   take pandas categoricals: integer codes for the categoricals, median-filled
   and robust-scaled floats for the rest.

MEMORY. The full frame is ~2.08M x 72. Held as categoricals + float32 that is
about 700 MB; as float64 with object strings it is several GB. Everything here
stays in the first regime on purpose. Peak RSS for a fresh build is ~4.5 GB,
for a cached load ~1.6 GB.
"""
import json

import numpy as np
import pandas as pd

from ..config import CACHE, TAIL_CAP, TARGET
from ..features import CATS, NEEDED, align_categories, build as build_base

# Screening folds. Month 1 and month 7 are the two scored months and the two
# hardest (Jan: 29 enormous tail rows; Jul: worst bulk at 299.8 AND 120 tail
# rows). Month 10 is the most typical month in the data (OOF RMSE 236.2 against
# a 12-month mean of ~280). Any expensive model is screened on these three and
# the SAME three are used for every comparison.
SCREEN_MONTHS = (1, 7, 10)
ALL_MONTHS = tuple(range(1, 13))

# Value a NaN numeric becomes under missing='sentinel'. Outside the +-8 clip
# numeric_design applies, so 'absent' is its own region of input space rather
# than a plausible-looking ordinary value.
MISSING_SENTINEL = -10.0

_XP = CACHE / 'altfam_X.parquet'
_YP = CACHE / 'altfam_y.npy'
_MP = CACHE / 'altfam_month.npy'
_GP = CACHE / 'altfam_groups.json'

# Columns congestion.build() actually touches. Loading only these instead of the
# whole 4.17M-row movement table is the difference between a 3 GB peak and an
# 8 GB one, because build() starts with a full .copy().
_CONG_COLS = ['PHASE_mvt', 'APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt',
              'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt']


def _assemble():
    """Build the design matrix from scratch. Mirrors cluster/cv_task.assemble."""
    from ..congestion import build as build_cong
    from ..disruption import attach as attach_atfm
    from ..io import load
    from ..weather import attach as attach_wx, download, prepare

    full = load('training', columns=_CONG_COLS)
    cong = build_cong(full).reset_index(drop=True)
    dep = full.PHASE_mvt.eq('DEP')
    n_dep = int(dep.sum())
    # Keep the DEP takeoff times from THIS load to check the next one lines up.
    # The congestion block and the base block come from two separate load()
    # calls, and if their row ORDER ever diverged, 26 congestion columns would
    # be silently misaligned against 18 base ones: no exception, no warning,
    # just a quietly worse model. A row-COUNT check does not catch that.
    dep_t = full.loc[dep, 'MVT_TIME_UTC_mvt'].to_numpy()
    del full, dep

    tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
    if len(tr) != n_dep or len(cong) != n_dep:
        raise RuntimeError(f'row-count disagreement: dep={n_dep} tr={len(tr)} cong={len(cong)}')
    if not (tr.MVT_TIME_UTC_mvt.to_numpy() == dep_t).all():
        raise RuntimeError('the two load() paths returned DEP rows in different '
                           'orders; congestion features would be misaligned')
    del dep_t

    X = build_base(tr)
    X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    base = list(X.columns)

    wx = attach_wx(tr, prepare(download())).reset_index(drop=True)
    atfm = attach_atfm(tr).reset_index(drop=True)
    groups = {'base': base, 'cong': list(cong.columns),
              'wx': list(wx.columns), 'atfm': list(atfm.columns)}

    y = tr[TARGET].astype(float).to_numpy()
    month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy().astype('int8')
    del tr

    X = pd.concat([X.reset_index(drop=True), cong, wx, atfm], axis=1)
    X, = align_categories(X)
    # float64 everywhere is pointless here: every continuous feature is a count
    # or a duration in seconds, well inside float32.
    for c in X.columns:
        if str(X[c].dtype) in ('float64', 'float32'):
            X[c] = X[c].astype('float32')
    return X, y, month, groups


def design(rebuild: bool = False):
    """(X, y, month, groups), cached on disk. X keeps pandas `category` dtype."""
    if rebuild or not _XP.exists():
        X, y, month, groups = _assemble()
        X.to_parquet(_XP, compression='zstd', index=False)
        np.save(_YP, y)
        np.save(_MP, month)
        _GP.write_text(json.dumps(groups))
        return X, y, month, groups
    X = pd.read_parquet(_XP)
    return X, np.load(_YP), np.load(_MP), json.loads(_GP.read_text())


def config_cols(groups, names):
    """Column list for a feature configuration, e.g. ('base','cong','wx','atfm')."""
    return [c for g in names for c in groups[g]]


FULL = ('base', 'cong', 'wx', 'atfm')   # the regressor's feature set (r-full)
BASE = ('base',)                        # the classifier's feature set (c-base)


# ---------------------------------------------------------------------------
# The fixed parts of the mixture.
# ---------------------------------------------------------------------------

def tail_estimate(off_sched, tail_mean):
    """E[y | tail, x]. off_sched where it is itself tail-sized, else the mean.

    Rationale (see claude_log section 11): where AOBT_3 is missing and the flight
    went out very late, the airport stamped BLOCK_TIME at SCHED_TIME, so
    y == off_sched almost exactly -- median |y - off_sched| = 4 s over the 481
    such rows.
    """
    off_sched = np.asarray(off_sched, float)
    return np.where(np.isfinite(off_sched) & (off_sched > TAIL_CAP), off_sched, tail_mean)


def mixture(p_tail, pred_bulk, off_sched, tail_mean):
    """Combine the three fixed pieces into the conditional mean."""
    et = tail_estimate(off_sched, tail_mean)
    return np.clip(np.asarray(p_tail, float) * et
                   + (1 - np.asarray(p_tail, float)) * np.asarray(pred_bulk, float), 0, None)


# ---------------------------------------------------------------------------
# Numeric re-expression, for torch and for anything else that cannot eat a
# pandas categorical.
# ---------------------------------------------------------------------------

def numeric_design(X, cols, fit_mask=None, missing='median', cats=None):
    """Split `cols` into (codes, cards, Z, num_cols).

    `fit_mask` selects the rows the imputation median and the scale are computed
    from. Pass the training rows of the fold: a median taken over the held-out
    month too is a (small) leak, and small leaks in a screening protocol are how
    a family gets credited with a gain it does not have.

    codes : int32 (n, n_cat) -- category code + 1, so 0 is the unseen/NaN slot
    cards : list of embedding table sizes (n_levels + 1)
    Z     : float32 (n, n_num) -- median-imputed, robust-scaled, clipped to +-8

    Robust (median / IQR) rather than mean / sd because off_sched has a range of
    +-130,000 s driven by a handful of rows; a z-score on that puts 99.9% of the
    mass in a pinhead around zero and a plain standardisation makes the network
    untrainable. Clipping at +-8 IQRs keeps the tail rows distinguishable
    without letting them dominate the first layer.

    `missing` decides what a NaN becomes.

      'median'   -- the obvious choice, and MEASURED TO BE THE WRONG ONE here.
                    exp47: the network is 2.8 s BETTER than LightGBM on the
                    522,756 bulk rows where AOBT_3 exists and 40.9 s WORSE on
                    the 7,145 where it does not. Median-filling off_aobt3 puts
                    those rows in the middle of the distribution, a confident
                    wrong signal that fights the aobt_missing indicator.
                    LightGBM sends a NaN down its own branch and never has the
                    problem.
      'sentinel' -- fill with MISSING_SENTINEL, which sits outside the +-8 clip,
                    so 'absent' is its own region of input space that the first
                    layer can separate linearly.
    """
    cats = CATS if cats is None else cats      # W2 passes CATS + ['ADES_mvt']; default unchanged
    cat_cols = [c for c in cols if c in cats]
    num_cols = [c for c in cols if c not in cats]

    codes = np.empty((len(X), len(cat_cols)), dtype=np.int32)
    cards = []
    for j, c in enumerate(cat_cols):
        cc = X[c].cat.codes.to_numpy()
        codes[:, j] = cc + 1
        cards.append(int(X[c].cat.categories.size) + 1)

    fit = np.ones(len(X), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    Z = np.empty((len(X), len(num_cols)), dtype=np.float32)
    stats = {}
    for j, c in enumerate(num_cols):
        v = X[c].to_numpy(dtype=np.float32, na_value=np.nan)
        f = v[fit & np.isfinite(v)]
        if f.size:
            med = float(np.median(f))
            q1, q3 = np.percentile(f, [25, 75])
            scale = float(q3 - q1) or 1.0
        else:
            med, scale = 0.0, 1.0
        ok = np.isfinite(v)
        Z[:, j] = np.where(ok, np.clip((v - med) / scale, -8, 8),
                           0.0 if missing == 'median' else MISSING_SENTINEL)
        stats[c] = (med, scale)
    return codes, cards, Z, cat_cols, num_cols, stats


# ---------------------------------------------------------------------------
# Fold-prediction store. Each experiment writes one .npy per (name, month) so a
# later script can assemble any combination without refitting, and so a partial
# run is never lost. Same idea as cluster/cv_task.py, local scale.
# ---------------------------------------------------------------------------

FOLDS = CACHE / 'altfam'


def fold_path(name: str, m: int):
    FOLDS.mkdir(parents=True, exist_ok=True)
    return FOLDS / f'{name}__m{int(m):02d}.npy'


def save_fold(name: str, m: int, arr):
    np.save(fold_path(name, m), np.asarray(arr, dtype=np.float32))


def have_fold(name: str, m: int) -> bool:
    return fold_path(name, m).exists()


def load_folds(name: str, month, months=SCREEN_MONTHS):
    """Stitch per-month predictions into one array over the rows of `months`.

    Returns None if any month is missing, rather than a silently short array.
    """
    month = np.asarray(month)
    sel = np.isin(month, months)
    out = np.full(int(sel.sum()), np.nan, dtype=np.float64)
    pos = month[sel]
    for m in months:
        p = fold_path(name, m)
        if not p.exists():
            return None
        out[pos == m] = np.load(p)
    return out


def screen(month, months=SCREEN_MONTHS):
    return np.isin(np.asarray(month), months)


def report(y, pred, label: str, width: int = 34):
    """One line of decompose(), the standard summary for every model here."""
    from ..validation import decompose
    d = decompose(y, pred)
    print(f'  {label:<{width}s} RMSE {d["rmse"]:8.2f} | bulk {d["rmse_bulk"]:7.2f} '
          f'| tail {d["rmse_tail"]:8.0f} | contrib b/t {d["contrib_bulk"]:6.1f}/'
          f'{d["contrib_tail"]:6.1f}', flush=True)
    return d
