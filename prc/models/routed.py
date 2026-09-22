"""Routed residual model: four disjoint routes, each with a consistent target.

WHY THIS SHAPE
--------------
Our shipped model is one global soft mixture. A competitor scoring 278.01 has a
July bulk of 299.75 against our 299.8 -- identical -- yet beats us by 48s on that
month purely through routing. Two structural problems with the single mixture:

1. The bulk regressor is trained on y <= 7200 and predicts raw seconds, so its
   leaf values are bounded by the training target. It CANNOT emit a long taxi
   time however strong the evidence. Every long row therefore depends entirely
   on the classifier firing.
2. The classifier runs globally where the tail rate is 0.028%. A calibrated
   probability buys a factor (1-q), i.e. 0.03%. Only discrimination pays, and
   discriminating a 1-in-3500 event across 2M rows is the hardest version of
   the problem.

Routing fixes both. `proxy = MVT_TIME - AOBT_3_flt` is Network Manager's view of
the same physical quantity we predict, so it partitions the data by MECHANISM:

    A  0 < proxy <= 7200   98.9% of rows,  0.005% tail   -> regress the RESIDUAL
                           y - proxy (sd 374s vs the target's 546s), then add
                           the proxy back. Inherits the anchor's level, so the
                           model only has to learn the correction.
    B  proxy > 7200        19 training rows. Same residual model applied by
                           extrapolation -- too few to fit anything of its own.
    C  proxy <= 0          1,058 rows. The proxy is unusable (NM off-block after
                           takeoff), so regress y directly.
    D  proxy missing       22,470 rows holding 481 of the 584 tail rows, a 2.14%
                           tail rate -- SEVENTY TIMES the global rate. This is
                           where the mixture belongs, and only here.

An earlier attempt (prc/models/anchored.py, exp21) used ONE model with the
residual target defined differently on the AOBT-present and AOBT-missing rows.
It lost on every month, because the model had to learn a discontinuity. Routing
is the same idea with each population given its own consistently-defined target.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from ..config import TAIL_CAP

ROUTE_A, ROUTE_B, ROUTE_C, ROUTE_D = 'A', 'B', 'C', 'D'


def route_of(proxy: np.ndarray) -> np.ndarray:
    """Partition rows by the NM proxy. Disjoint and exhaustive by construction."""
    r = np.full(len(proxy), ROUTE_D, dtype='<U1')
    fin = np.isfinite(proxy)
    r[fin & (proxy <= 0)] = ROUTE_C
    r[fin & (proxy > 0) & (proxy <= TAIL_CAP)] = ROUTE_A
    r[fin & (proxy > TAIL_CAP)] = ROUTE_B
    return r


class RoutedModel:
    """Fit/predict with four routes. `reg_cols` may differ from `clf_cols`."""

    def __init__(self, reg_cols, clf_cols, cats, seed=0, threads=8,
                 leaves=511, rounds=1400, lr=0.025):
        self.reg_cols, self.clf_cols, self.cats = reg_cols, clf_cols, cats
        self.seed, self.threads = seed, threads
        self.leaves, self.rounds, self.lr = leaves, rounds, lr

    def _reg(self, **kw):
        p = dict(objective='regression', metric='rmse', learning_rate=self.lr,
                 num_leaves=self.leaves, min_data_in_leaf=100, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 num_threads=self.threads, verbose=-1, seed=self.seed,
                 bagging_seed=self.seed + 1, feature_fraction_seed=self.seed + 2,
                 deterministic=True, force_row_wise=True)
        p.update(kw)
        return p

    def fit(self, X, y, proxy, sched_off):
        rt = route_of(proxy)
        rc = [c for c in self.cats if c in self.reg_cols]
        cc = [c for c in self.cats if c in self.clf_cols]

        # --- A: the residual model. 98.9% of rows, and the only large fit. -----
        a = rt == ROUTE_A
        self.m_resid = lgb.train(
            self._reg(), lgb.Dataset(X.loc[a, self.reg_cols], (y - proxy)[a],
                                     categorical_feature=rc),
            num_boost_round=self.rounds)

        # --- C: proxy unusable, regress y directly. Small, so a small model. ---
        c = rt == ROUTE_C
        self.m_direct = None
        if c.sum() >= 200:
            self.m_direct = lgb.train(
                self._reg(num_leaves=31, min_data_in_leaf=20),
                lgb.Dataset(X.loc[c, self.reg_cols], y[c], categorical_feature=rc),
                num_boost_round=300)
        self.c_fallback = float(np.median(y[c])) if c.any() else float(np.median(y))

        # --- D: the specialist. This is where the tail lives (2.14% rate). -----
        d = rt == ROUTE_D
        d_bulk = d & (y <= TAIL_CAP)
        self.m_d_bulk = lgb.train(
            self._reg(num_leaves=63, min_data_in_leaf=40),
            lgb.Dataset(X.loc[d_bulk, self.reg_cols], y[d_bulk], categorical_feature=rc),
            num_boost_round=400)
        self.m_d_clf = lgb.train(
            dict(self._reg(objective='binary', num_leaves=31, min_data_in_leaf=20,
                           learning_rate=0.05), metric='auc'),
            lgb.Dataset(X.loc[d, self.clf_cols], (y[d] > TAIL_CAP).astype(int),
                        categorical_feature=cc),
            num_boost_round=300)
        dt = d & (y > TAIL_CAP)
        self.d_tail_mean = float(y[dt].mean()) if dt.any() else 20000.0
        return self

    def predict(self, X, proxy, sched_off):
        rt = route_of(proxy)
        out = np.full(len(X), np.nan)

        # A and B share the residual model; B is pure extrapolation of the same
        # correction, which is the point -- it inherits the proxy's magnitude.
        ab = (rt == ROUTE_A) | (rt == ROUTE_B)
        if ab.any():
            out[ab] = proxy[ab] + self.m_resid.predict(X.loc[ab, self.reg_cols])

        c = rt == ROUTE_C
        if c.any():
            out[c] = (self.m_direct.predict(X.loc[c, self.reg_cols])
                      if self.m_direct is not None else self.c_fallback)

        d = rt == ROUTE_D
        if d.any():
            pb = self.m_d_bulk.predict(X.loc[d, self.reg_cols])
            pt = self.m_d_clf.predict(X.loc[d, self.clf_cols])
            so = sched_off[d]
            et = np.where(np.isfinite(so) & (so > TAIL_CAP), so, self.d_tail_mean)
            out[d] = pt * et + (1 - pt) * pb

        return np.clip(out, 0, None)
