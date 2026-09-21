"""Target reparameterisation: regress the correction, not the level.

WHAT THIS TESTS
---------------
`off_aobt3 = MVT_TIME - AOBT_3` is the top-gain feature in the model at 19.9%,
and for good reason: AOBT_3 is the same physical event as the hidden off-block
time, measured by Network Manager instead of the airport, with 98.9% coverage
and a disagreement of only sd 384.5 s (claude_log EXP02). Yet the trees are
asked to predict y, which means a large share of their capacity goes on
reconstructing the identity map y ~= off_aobt3 before any of it can be spent on
the residual that actually varies.

So regress

    r = y - off_aobt3        where AOBT_3 exists  (i.e. r = AOBT_3 - BLOCK_TIME)
    r = y                    where it does not

and add the anchor back at prediction time. The loss is UNCHANGED -- shifting a
row's target and its prediction by the same known constant leaves the squared
error identical -- so this is not a reweighting in disguise. What changes is the
function the trees have to represent, which is now a bounded correction on 98.9%
of rows instead of a level spanning three orders of magnitude.

This is deliberately not a monotone transform of a feature, which would be a
no-op for a threshold-based learner. It is a per-row change of the target.

WHAT IT DOES NOT FIX
--------------------
The tail. 481 of 584 tail rows have no AOBT_3, so their anchor is zero and they
are untouched: the reparameterisation cannot rescue rows whose anchor is
missing, which is precisely the subpopulation the tail lives in. This is a bulk
lever, and it is used as the bulk component of a mixture rather than on its own.
"""
import numpy as np


def anchor(X) -> np.ndarray:
    """Known-at-scoring-time offset to subtract from the target.

    Zero where AOBT_3 is absent, so those rows fall back to predicting y
    directly. Using off_sched as a fallback anchor was considered and rejected:
    where AOBT_3 is missing, off_sched is exactly the column that runs to 88,000
    s, so anchoring on it would import that spread into the target rather than
    removing spread from it.
    """
    a = X['off_aobt3'].to_numpy(dtype=float)
    return np.where(np.isfinite(a), a, 0.0)


def fit_predict(params, Xt, yt, Xv, num_boost_round=350, categorical_feature=None,
                trim=None):
    """Fit on the anchored residual, predict, add the anchor back.

    `trim` caps the TRAINING rows by y (not by r), so that this stays directly
    comparable with the trimmed bulk regressor the v3 mixture uses.
    """
    import lightgbm as lgb
    at, av = anchor(Xt), anchor(Xv)
    rt = np.asarray(yt, float) - at
    sel = slice(None) if trim is None else np.asarray(yt, float) <= trim
    ds = lgb.Dataset(Xt[sel] if trim is not None else Xt,
                     rt[sel] if trim is not None else rt,
                     categorical_feature=categorical_feature or 'auto')
    m = lgb.train(params, ds, num_boost_round=num_boost_round)
    return np.clip(m.predict(Xv) + av, 0.0, None)
