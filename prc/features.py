"""Feature construction shared by training and scoring.

CRITICAL INVARIANT: every feature here must be computable from columns that are
present in ranking.parquet for PHASE='DEP'. That excludes BLOCK_TIME_UTC_mvt and
TAXITIME_SEC_mvt, which are blanked. It INCLUDES MVT_TIME_UTC_mvt (the takeoff
time) and the whole _flt block, notably AOBT_3_flt.

Train and score must go through this same function, or the two feature matrices
drift apart and the model silently degrades.
"""
import numpy as np
import pandas as pd

# Columns that must be loaded for build() to work.
NEEDED = ['APT', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt',
          'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'FLIGHT_TYPE_flt',
          'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt']

CATS = ['APT', 'RUNWAY_mvt', 'PIER', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
        'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt']
NUMS = ['off_aobt3', 'off_sched', 'off_eobt', 'hour', 'dow']
FEATURES = CATS + NUMS


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to df's rows (index preserved)."""
    out = pd.DataFrame(index=df.index)

    # 'Pier' = leading letter-or-digit block of the stand id. At most of these
    # airports this is the terminal/pier; between-pier mean taxi-out differs by
    # up to 999 s at EHAM. Generalises to stands seen rarely or not at all.
    out['PIER'] = (df.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]+|\d)', expand=False)
                   .fillna('NA'))
    for c in CATS:
        if c != 'PIER':
            out[c] = df[c].astype(str)

    # Offsets from the known takeoff time back to the various estimates of
    # off-block. off_aobt3 is the strongest single feature in the dataset.
    t = df.MVT_TIME_UTC_mvt
    for src, name in [('AOBT_3_flt', 'off_aobt3'), ('SCHED_TIME_UTC_mvt', 'off_sched'),
                      ('EOBT_1_flt', 'off_eobt')]:
        out[name] = (t - df[src]).dt.total_seconds()
    out['hour'] = t.dt.hour.astype('int16')
    out['dow'] = t.dt.dayofweek.astype('int16')
    return out[FEATURES]


def align_categories(train_X: pd.DataFrame, *others: pd.DataFrame):
    """Cast CATS to a single shared category vocabulary built from training.

    Categories unseen in training become NaN, which LightGBM handles. Without
    this, pandas assigns different integer codes to the same string in the two
    frames and the model reads garbage at scoring time.
    """
    frames = [train_X] + list(others)
    for c in CATS:
        vocab = pd.Index(sorted(train_X[c].dropna().unique()))
        for f in frames:
            f[c] = pd.Categorical(f[c], categories=vocab)
    return frames
