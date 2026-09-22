"""Specialist P(tail|x) for the one population that decides the tail.

WHY THIS EXISTS
---------------
`TAXITIME_SEC = MVT_TIME - BLOCK_TIME` exactly, and `BLOCK_TIME` is hidden.
When the airport has no off-block record it stamps BLOCK_TIME at SCHED_TIME,
and the taxi time then equals `off_sched = MVT_TIME - SCHED_TIME`, which can be
a whole day. EXP30-35 localised every one of those rows:

    P = {AOBT_3_flt missing} AND {off_sched > 7200}
        4,944 rows of 2,085,047 (0.24%), containing 450 of the 584 tail rows.
        Tail rate inside P: 9.1%.   Outside P: 0.0064%.

Outside P the mixture's tail branch is inert -- the probabilities are 1e-4 and
the anchor is never used. So the whole tail problem is a 4,944-row binary
classification, and fitting it on 1.9M rows with 435 positives is the reason the
shipped classifier is weak: it has to spend its capacity keeping 1.9M irrelevant
rows at zero, and it ends up UNDER-confident at large off_sched, which is
exactly where a wrong probability costs tens of thousands of seconds.

This module leaves the shipped mixture untouched everywhere and replaces P(T|x)
only inside P. That is what distinguishes it from EXP09's AOBT-missing
specialist MODEL, which replaced the global model and therefore could not touch
the tail rows outside its own population; it scored worse and was abandoned.

WHAT IS IN THE ESTIMATOR
------------------------
Three weak estimators averaged, because 4,944 rows support very little:
  * a small LightGBM (15 leaves) with a monotone constraint on off_sched,
    averaged over three seeds -- 450 positives is few enough that seed noise
    is visible;
  * an (airport x off_sched bucket) empirical rate with Beta shrinkage to the
    airport rate and then to the global rate;
  * isotonic regression of the rate on off_sched, fitted separately for LIRF
    and everything else.
The only structure exp35 found inside P that is strong AND stable is "later
relative to schedule -> more likely a stamping artifact" plus a large airport
effect (LIRF 65.9% tail inside P; EDDF/EDDM/EHAM/LEBL/LEMD 0.0% across 2,319
rows). All three estimators encode that and nothing more.

EVERY input is available at scoring time: ranking.parquet carries AOBT_3_flt,
MVT_TIME_UTC_mvt and SCHED_TIME_UTC_mvt for PHASE='DEP'.
"""
import lightgbm as lgb
import numpy as np

from prc.config import TAIL_CAP

# Features that actually vary inside P. Everything derived from AOBT_3, EOBT_1
# or LOBT is NaN there -- when the NM record is absent the whole _flt block goes
# with it -- so they are left out rather than fed in as constant NaN columns.
SPEC_FEATURES = ['APT', 'PIER', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
                 'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt',
                 'off_sched', 'log_off_sched', 'hour', 'dow']
SPEC_CAT = ['APT', 'PIER', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
            'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt']
OS_BUCKETS = np.array([7200, 8000, 9000, 10000, 12000, 15000, 20000, 30000, np.inf])
SEEDS = (0, 1, 2)

SPEC_PARAMS = dict(objective='binary', metric='auc', learning_rate=0.05,
                   num_leaves=15, min_data_in_leaf=30, feature_fraction=0.9,
                   bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, deterministic=True, force_row_wise=True)


def population(off_sched, aobt_missing, cap=TAIL_CAP):
    """The decision-relevant population P. Both inputs are known at scoring time."""
    off_sched = np.asarray(off_sched, float)
    return np.asarray(aobt_missing, bool) & np.isfinite(off_sched) & (off_sched > cap)


class TailSpecialist:
    """Fit on training rows, predict P(y > TAIL_CAP | x) for rows inside P."""

    def __init__(self, threads=4, seeds=SEEDS, rounds=200, prior_w=8.0):
        self.threads, self.seeds, self.rounds, self.prior_w = threads, seeds, rounds, prior_w

    def fit(self, X, y, apt, off_sched, aobt_missing):
        P = population(off_sched, aobt_missing)
        if P.sum() < 50:
            raise ValueError(f'only {int(P.sum())} training rows inside P -- refusing to '
                             'fit a specialist on that; check off_sched/aobt_missing')
        t = (np.asarray(y, float) > TAIL_CAP).astype(int)
        self.global_rate_ = float(t[P].mean())
        Xp, tp = X.loc[P, SPEC_FEATURES], t[P]
        os_p, apt_p = np.asarray(off_sched, float)[P], np.asarray(apt)[P]

        mono = [1 if c in ('off_sched', 'log_off_sched') else 0 for c in SPEC_FEATURES]
        self.gbms_ = [
            lgb.train(dict(SPEC_PARAMS, seed=s, bagging_seed=s + 1,
                           feature_fraction_seed=s + 2, num_threads=self.threads,
                           monotone_constraints=mono),
                      lgb.Dataset(Xp, tp, categorical_feature=SPEC_CAT),
                      num_boost_round=self.rounds)
            for s in self.seeds]

        # (airport, off_sched bucket) table, shrunk twice.
        b = np.digitize(os_p, OS_BUCKETS) - 1
        self.table_, self.apt_rate_ = {}, {}
        g, w = self.global_rate_, self.prior_w
        for a in np.unique(apt_p):
            sa = apt_p == a
            ga = (tp[sa].sum() + w * g) / (sa.sum() + w)
            self.apt_rate_[a] = float(ga)
            for bb in np.unique(b[sa]):
                s = sa & (b == bb)
                self.table_[(a, int(bb))] = float((tp[s].sum() + w * ga) / (s.sum() + w))

        # Isotonic on off_sched, LIRF vs the rest.
        from sklearn.isotonic import IsotonicRegression
        self.iso_ = {}
        for grp in ('LIRF', 'OTHER'):
            k = (apt_p == 'LIRF') if grp == 'LIRF' else (apt_p != 'LIRF')
            if k.sum() >= 30:
                ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                ir.fit(os_p[k], tp[k].astype(float))
                self.iso_[grp] = ir
        return self

    def _table_predict(self, apt_v, os_v):
        b = np.digitize(os_v, OS_BUCKETS) - 1
        out = np.full(len(os_v), self.global_rate_)
        for i, (a, bb) in enumerate(zip(apt_v, b)):
            out[i] = self.table_.get((a, int(bb)),
                                     self.apt_rate_.get(a, self.global_rate_))
        return out

    def _iso_predict(self, apt_v, os_v):
        out = np.full(len(os_v), self.global_rate_)
        for grp in ('LIRF', 'OTHER'):
            m = (apt_v == 'LIRF') if grp == 'LIRF' else (apt_v != 'LIRF')
            if m.any() and grp in self.iso_:
                out[m] = self.iso_[grp].predict(os_v[m])
        return out

    def predict(self, X, apt, off_sched, aobt_missing):
        """P(tail|x) for rows inside P; NaN elsewhere (caller keeps its own value)."""
        P = population(off_sched, aobt_missing)
        out = np.full(len(X), np.nan)
        if not P.any():
            return out, P
        Xp = X.loc[P, SPEC_FEATURES]
        os_v, apt_v = np.asarray(off_sched, float)[P], np.asarray(apt)[P]
        gbm = np.mean([m.predict(Xp) for m in self.gbms_], axis=0)
        out[P] = np.clip((gbm + self._table_predict(apt_v, os_v)
                          + self._iso_predict(apt_v, os_v)) / 3.0, 0.0, 1.0)
        return out, P


def apply_to_mixture(p_global, p_spec, in_P):
    """Shipped probabilities everywhere, specialist probabilities inside P."""
    p = np.asarray(p_global, float).copy()
    k = in_P & np.isfinite(p_spec)
    p[k] = p_spec[k]
    return p
