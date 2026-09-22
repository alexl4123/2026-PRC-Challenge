"""Multi-anchor mixture: model WHICH clock BLOCK_TIME was stamped from.

EXP33 measured, for every training row, |BLOCK_TIME - c| for each candidate
clock c the scorer also gives us. Over all 2.08M departures the nearest
candidate is

    AOBT_3 51.45% | SCHED 22.45% | none-within-300s 12.75% | EOBT_1 11.73%
    LOBT 1.57% | IOBT 0.04%

and on the 584 tail rows it is SCHED 450, none 123, EOBT_1 10, SCHED-1d 1.

So the taxi time is not one quantity with a heavy tail: it is a MIXTURE OVER
WHICH CLOCK the airport recorded, and every clock except the real pushback is
known at scoring time. That makes the RMSE-optimal prediction

    E[y|x] = sum_c P(c|x) * (MVT - c)  +  P(real|x) * E[y | real, x]

which is the existing two-component mixture generalised from one anchor
(SCHED) to all of them. The extra anchors are what category X needs: 97 tail
rows where AOBT_3 exists and says the taxi was ~1,055s, but BLOCK was stamped
at LOBT/SCHED hours earlier. MVT - LOBT predicts those rows to 1,525s RMSE
against the 7,973s of MVT - AOBT_3.

Also builds the constructed delay features (AOBT_3 - LOBT and friends). A tree
cannot form a difference of two raw columns, and the difference is exactly the
signal: "the aircraft actually pushed back long after the time it had filed" is
what precedes the airport stamping the filed time instead.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP
from prc.features import build as build_base, align_categories, CATS, NEEDED
from prc.io import load

# Candidate clocks, in the priority order used to break near-ties when several
# land within TOL of BLOCK_TIME. AOBT_3 first because it is the real pushback
# whenever it is present, so a row it explains needs no anchor at all.
ANCHORS = ['AOBT_3', 'LOBT', 'EOBT_1', 'SCHED']
ANCHOR_COL = {'AOBT_3': 'AOBT_3_flt', 'LOBT': 'LOBT_flt',
              'EOBT_1': 'EOBT_1_flt', 'SCHED': 'SCHED_TIME_UTC_mvt'}
# Anchors whose offset is used DIRECTLY as the prediction. AOBT_3 is excluded:
# BLOCK - AOBT_3 still has a 385s spread, so the regressor beats the raw offset
# there, and NONE has no offset at all. Both route to the regressor.
DIRECT = ['LOBT', 'EOBT_1', 'SCHED']
CLASSES = ANCHORS + ['NONE']
TOL = 120.0

EXTRA_COLS = ['IOBT_flt', 'BLOCK_TIME_UTC_mvt']


def assemble(with_extra_features=True):
    """Feature matrix, anchor offsets, and the metadata the mixture needs."""
    cols = sorted(set(NEEDED + [TARGET] + EXTRA_COLS))
    tr = load('training', phase='DEP', columns=cols)
    X = build_base(tr)
    X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    base_cols = list(X.columns)

    if with_extra_features:
        X['off_iobt'] = (tr.MVT_TIME_UTC_mvt - tr.IOBT_flt).dt.total_seconds()
        # Delays between the clocks. These are the differences a tree cannot
        # build for itself, and "pushed back long after the filed time" is the
        # precondition for the airport stamping the filed time instead.
        pairs = [('aobt3', 'AOBT_3_flt', 'sched', 'SCHED_TIME_UTC_mvt'),
                 ('aobt3', 'AOBT_3_flt', 'lobt', 'LOBT_flt'),
                 ('aobt3', 'AOBT_3_flt', 'eobt', 'EOBT_1_flt'),
                 ('aobt3', 'AOBT_3_flt', 'iobt', 'IOBT_flt'),
                 ('lobt', 'LOBT_flt', 'sched', 'SCHED_TIME_UTC_mvt'),
                 ('eobt', 'EOBT_1_flt', 'sched', 'SCHED_TIME_UTC_mvt'),
                 ('lobt', 'LOBT_flt', 'iobt', 'IOBT_flt')]
        for an, ac, bn, bc in pairs:
            X[f'd_{an}_{bn}'] = (tr[ac] - tr[bc]).dt.total_seconds()
        # How far apart the candidate clocks are: when they agree there is
        # nothing at stake, when they disagree the class choice is everything.
        offs_tmp = np.vstack([(tr.MVT_TIME_UTC_mvt - tr[ANCHOR_COL[a]]).dt.total_seconds()
                              for a in ANCHORS])
        X['anchor_spread'] = np.nanmax(offs_tmp, axis=0) - np.nanmin(offs_tmp, axis=0)
        X['n_anchor_avail'] = np.isfinite(offs_tmp).sum(axis=0).astype('int8')
    X, = align_categories(X)
    X = X.reset_index(drop=True)

    offs = pd.DataFrame({a: (tr.MVT_TIME_UTC_mvt - tr[ANCHOR_COL[a]]).dt.total_seconds()
                         for a in ANCHORS})
    dist = pd.DataFrame({a: (tr.BLOCK_TIME_UTC_mvt - tr[ANCHOR_COL[a]])
                         .dt.total_seconds().abs() for a in ANCHORS})
    meta = pd.DataFrame({
        'y': tr[TARGET].astype(float).to_numpy(),
        'month': tr.MVT_TIME_UTC_mvt.dt.month.to_numpy(),
        'off_sched': X.off_sched.to_numpy(),
        'off_aobt3': X.off_aobt3.to_numpy(),
        'delta': (tr.BLOCK_TIME_UTC_mvt - tr.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy(),
        'apt': tr.APT.astype(str).to_numpy(),
        'aobt_missing': X.aobt_missing.to_numpy().astype(bool),
    })
    return X, meta, offs.reset_index(drop=True), dist.reset_index(drop=True), base_cols


def anchor_label(dist, tol=TOL):
    """Which clock BLOCK_TIME was stamped from: the nearest candidate within
    `tol`, ties broken by the ANCHORS priority order; NONE if none is close."""
    D = dist[ANCHORS].to_numpy(float)
    D = np.where(np.isfinite(D), D, np.inf)
    lab = D.argmin(axis=1)
    return np.where(D.min(axis=1) <= tol, lab, len(ANCHORS)).astype(np.int8)


def anchor_mixture(W, offs, r, floor=0.0):
    """sum_c w_c * (MVT - c) over the DIRECT anchors, remainder on the regressor.

    Weight on an anchor whose clock is missing is moved to the regressor rather
    than dropped, so the weights always sum to one and the prediction stays a
    proper conditional mean.
    """
    O = offs[DIRECT].to_numpy(float)
    idx = [CLASSES.index(a) for a in DIRECT]
    w = W[:, idx].copy()
    usable = np.isfinite(O) & (O > floor)
    w = np.where(usable, w, 0.0)
    O = np.where(usable, O, 0.0)
    return np.clip((w * O).sum(axis=1) + (1.0 - w.sum(axis=1)) * r, 0, None)


MCLF = dict(objective='multiclass', num_class=len(CLASSES), metric='multi_logloss',
            learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, verbose=-1, deterministic=True, force_row_wise=True)


def fit_anchor_clf(X, lab, tr_mask, cols, seed=0, threads=4, rounds=300):
    p = dict(MCLF, seed=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2, num_threads=threads)
    cats = [c for c in CATS if c in cols]
    return lgb.train(p, lgb.Dataset(X.loc[tr_mask, cols], lab[tr_mask],
                                    categorical_feature=cats),
                     num_boost_round=rounds)
