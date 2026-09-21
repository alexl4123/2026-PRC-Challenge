"""Distributional regression: discretise the target, classify, integrate.

WHAT THIS TESTS
---------------
Every model tried so far estimates E[y|x] with a squared-error loss, and the
whole project has been fighting one consequence of that choice:

    training target     bulk RMSE   tail RMSE      (claude_log EXP07)
    trimmed @7200          258.2      24158
    untrimmed              296.4      17410

The mechanism is the L2 gradient. For squared error the gradient of a row is its
residual, so a row with y = 88,000 s pulls on the ensemble roughly a hundred
times as hard as an ordinary 900 s row. 584 such rows in 2.08M are enough to
dominate the early boosting iterations -- which is exactly why the untrimmed
model early-stops at 60 rounds where the trimmed one survives 342. The
two-stage mixture works around this by splitting the population; it does not
remove the cause.

Cross-entropy over a discretised target removes the cause. Each row contributes
a gradient bounded by 1 in probability space no matter how extreme its label is,
so a single model can learn the bulk and the tail at once with no trade-off to
manage. And because the fitted object is a whole conditional distribution rather
than a point, the RMSE-optimal prediction falls out by integration:

    E[y|x] = sum_k P(bin k | x) * E[y | bin k, x]

which is the same quantity the hand-built mixture approximates with two
components, except the weights and the component means are learnt rather than
assumed, and there are K of them rather than two.

FALSIFIABLE PREDICTION, stated before running anything: if the gradient argument
above is right, a binned model trained on ALL rows should reach a bulk RMSE near
the TRIMMED L2 regressor's 233.7 (OOF) rather than the untrimmed one's 258.4,
while keeping a tail estimate at least as good as the mixture's 10,107.

CHOOSING THE GRID
-----------------
A binned model cannot resolve detail finer than its bins, so the grid sets a
hard floor: predict every row at the centroid of its own true bin and you get

    quantile bins,     K=100 : bulk floor  87.2 s
    quantile bins,     K=207 : bulk floor  66.4 s
    linear+geometric,  K= 56 : bulk floor  28.9 s   <- DEFAULT_GRID
    linear+geometric,  K= 62 : bulk floor  28.8 s

Quantile bins are the obvious choice and the wrong one. They put ~6 s of width
at the mode, where the extra resolution buys nothing, and ~2,000 s of width at
the 99.9th percentile, where it costs everything. A fixed linear grid over the
core with a geometric continuation above beats them at half the class count.
28.9 s in quadrature against a ~243 s model is +1.7 s, i.e. negligible; the
number of classes is the real budget, since LightGBM grows one tree per class
per iteration.

The grid is a FIXED CONSTRUCTION, not a statistic of the data: no quantiles, no
data-dependent edges, hence nothing to leak across a CV fold. Only the bin
representatives are fitted, and those come from the training fold alone.
"""
import numpy as np

from ..config import TAIL_CAP

# Upper edge of the last bin. Fixed above the largest 2025 label (131,167 s) so
# that the grid does not depend on the data; a 2026 row beyond it would simply
# be clipped into the top bin, which is the correct behaviour anyway.
Y_MAX = 150_000.0

# core_width : bin width, in seconds, over the dense part of the distribution
# core_hi    : where the linear core stops (99.7% of rows are below 4,000 s)
# n_mid      : geometric bins between core_hi and TAIL_CAP
# n_tail     : geometric bins between TAIL_CAP and Y_MAX
DEFAULT_GRID = dict(core_width=100.0, core_hi=4000.0, n_mid=6, n_tail=10)


def make_edges(core_width=100.0, core_hi=4000.0, n_mid=6, n_tail=10,
               tail_cap=TAIL_CAP, y_max=Y_MAX) -> np.ndarray:
    """Bin edges: linear over the core, geometric above it.

    The first edge is -1 rather than 0 so that y == 0 (there are such rows)
    lands inside bin 0 rather than on its boundary.
    """
    core = np.arange(core_width, core_hi + 0.5 * core_width, core_width)
    mid = np.geomspace(core_hi, tail_cap, n_mid + 1)[1:]
    tail = np.geomspace(tail_cap, y_max, n_tail + 1)[1:]
    return np.unique(np.concatenate([[-1.0], core, mid, tail]))


def bin_codes(y, edges) -> np.ndarray:
    """Class label for each row: the index of the bin containing y."""
    return np.clip(np.searchsorted(edges, np.asarray(y, float), side='right') - 1,
                   0, len(edges) - 2).astype(np.int32)


def centroids(y_train, edges) -> np.ndarray:
    """E[y | bin k] from the TRAINING fold only -- these are fitted quantities.

    Empty bins fall back to the bin midpoint. They attract essentially no
    predicted mass, so the fallback is a formality rather than a modelling
    choice, but leaving them at zero would silently bias the expectation.
    """
    y_train = np.asarray(y_train, float)
    k = bin_codes(y_train, edges)
    nb = len(edges) - 1
    cnt = np.bincount(k, minlength=nb).astype(float)
    tot = np.bincount(k, weights=y_train, minlength=nb)
    mid = 0.5 * (np.maximum(edges[:-1], 0.0) + edges[1:])
    return np.where(cnt > 0, tot / np.maximum(cnt, 1.0), mid)


class DistributionalRegressor:
    """LightGBM softmax over target bins; prediction is the integrated mean.

    `tail_rep` chooses E[y | bin k, x] for bins above TAIL_CAP, and it is the
    one place where domain knowledge enters. The three settings exist so that
    "does the distribution add anything beyond the existing hand-built mixture"
    is an ablation rather than an opinion:

      'centroid'     E[y|k] from training. Pure -- no domain input at all.
      'sched_global' off_sched wherever off_sched > TAIL_CAP, else the centroid.
                     This is exactly the tail anchor the v3 mixture uses, so it
                     isolates the effect of replacing a binary tail weight with
                     a full distribution.
      'sched_inbin'  off_sched, but only where off_sched falls inside bin k.
                     Uses the model's resolution: the predicted bin says which
                     magnitude, off_sched refines within it, and a schedule
                     stamp that disagrees with the predicted magnitude is
                     ignored instead of trusted.

    Bulk bins always use the training centroid; the 28.9 s floor that implies is
    below the noise of anything we can measure.
    """

    def __init__(self, edges=None, params=None, num_boost_round=150,
                 tail_cap=TAIL_CAP):
        self.edges = make_edges(**DEFAULT_GRID) if edges is None else np.asarray(edges, float)
        self.n_bins = len(self.edges) - 1
        self.tail_cap = tail_cap
        self.num_boost_round = num_boost_round
        self.params = dict(default_params(self.n_bins), **(params or {}))
        self.params['num_class'] = self.n_bins
        self.booster_ = None
        self.centroids_ = None

    def fit(self, X, y, categorical_feature=None):
        import lightgbm as lgb
        self.centroids_ = centroids(y, self.edges)
        ds = lgb.Dataset(X, bin_codes(y, self.edges),
                         categorical_feature=categorical_feature or 'auto')
        self.booster_ = lgb.train(self.params, ds, num_boost_round=self.num_boost_round)
        return self

    def predict_proba(self, X) -> np.ndarray:
        # float32 keeps the (n_rows x n_bins) matrix small; 190k x 56 in float64
        # is 85 MB per fold and there is no precision to be gained here.
        return self.booster_.predict(X).astype(np.float32)

    def predict(self, X, tail_rep='sched_inbin', off_sched=None) -> np.ndarray:
        return self.expectation(self.predict_proba(X), tail_rep=tail_rep,
                                off_sched=off_sched)

    def expectation(self, proba, tail_rep='sched_inbin', off_sched=None) -> np.ndarray:
        """Integrate a predicted distribution into a conditional mean."""
        c = self.centroids_
        out = proba.astype(np.float64) @ c
        if tail_rep != 'centroid':
            if off_sched is None:
                raise ValueError(f'tail_rep={tail_rep!r} needs off_sched')
            os_ = np.asarray(off_sched, float)
            # Correct only the tail bins in place. Building a full (n x K)
            # matrix of row-dependent representatives would cost ~1 GB per fold
            # for a correction that touches ten columns.
            for k in np.where(self.edges[:-1] >= self.tail_cap)[0]:
                if tail_rep == 'sched_global':
                    m = np.isfinite(os_) & (os_ > self.tail_cap)
                elif tail_rep == 'sched_inbin':
                    m = np.isfinite(os_) & (os_ >= self.edges[k]) & (os_ < self.edges[k + 1])
                else:
                    raise ValueError(f'unknown tail_rep {tail_rep!r}')
                out[m] += proba[m, k] * (os_[m] - c[k])
        return np.clip(out, 0.0, None)


def default_params(n_bins, seed=0, threads=8):
    """Softmax over bins.

    num_leaves is much smaller than the 255 used for the L2 regressor on
    purpose: LightGBM grows one tree per class per iteration, so the cost is
    n_bins x num_leaves x num_boost_round. A single class boundary is also a far
    simpler function than the conditional mean, so the capacity is better spent
    on rounds than on leaves.
    """
    return dict(objective='multiclass', num_class=n_bins, metric='multi_logloss',
                learning_rate=0.1, num_leaves=63, min_data_in_leaf=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, num_threads=threads, verbose=-1, seed=seed,
                bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
                deterministic=True, force_row_wise=True)
