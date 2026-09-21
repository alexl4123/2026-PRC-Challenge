"""The EXP13 base feature matrix, in one place.

EXP13/EXP14 established 296.22 OOF for the two-stage mixture on exactly these
eighteen columns, and that number is the thing any new architecture has to beat.
Reproducing the block by hand in every new experiment is how a comparison
quietly stops being a comparison, so it lives here and exp20+ and the cluster
task all call it.

Deliberately NOT including congestion / weather / ATFM: those are being
evaluated separately (EXP14/EXP15) and folding them in here would confound an
architecture change with a feature change.

Row order is the row order of `prc.io.load('training', phase='DEP')`, which is
also the order of `cache/exp13_oof.npy`, so cached baseline predictions can be
compared row-for-row without re-fitting them.
"""
import numpy as np

from ..config import TARGET
from ..features import build, align_categories, NEEDED
from ..io import load


def assemble_base(with_target=True):
    """Return (X, y, month) for all 2025 departures.

    y is NaN-free float seconds; month is the calendar month of the takeoff,
    which is the fold key for leave-one-month-out CV.
    """
    tr = load('training', phase='DEP', columns=NEEDED + ([TARGET] if with_target else []))
    X = build(tr)

    # Flags and offsets a tree cannot construct for itself. `aobt_missing` is the
    # single most informative bit in the file for the tail: 481 of 584 tail rows
    # have no NM pushback record.
    X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()

    # Signed log of the two offsets. A monotone transform is a no-op for a tree
    # on its own, but these are kept because they were part of the feature set
    # that produced the 296.22 baseline and removing them would change it.
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())

    y = tr[TARGET].astype(float).to_numpy() if with_target else None
    month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    X, = align_categories(X)
    return X.reset_index(drop=True), y, month
