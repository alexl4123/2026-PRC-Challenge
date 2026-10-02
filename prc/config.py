"""Paths and constants. Data deliberately lives OUTSIDE this repo.

The challenge Terms & Conditions forbid using or publishing the 2026 dataset
outside the competition until the organizers publish it. Keeping the parquets in
the parent directory makes it structurally impossible to commit them by accident.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]        # .../prc2026
DATA = REPO.parent                                # .../00_src   <- parquets live here
CACHE = REPO / 'cache'                            # gitignored, rebuildable
OUTPUTS = REPO / 'outputs'                        # gitignored: models, submissions
for _p in (CACHE, OUTPUTS):
    _p.mkdir(exist_ok=True)

PYTHON = __import__('os').environ.get('PRC_PYTHON', __import__('sys').executable)  # laptop: /home/thinklex/prc2026-venv (system anaconda pyarrow 19 cannot read the files)

AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']

TARGET = 'TAXITIME_SEC_mvt'
ID = 'MVT_ID_mvt'

# Rows above this are the pathological tail; see claude_log EXP01 for the argument.
TAIL_CAP = 7200

# Columns that are cheap to keep as categoricals rather than python strings.
CAT_COLS = ['FLIGHT_RULE_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'AIRCRAFT_TYPE_mvt',
            'RUNWAY_mvt', 'STAND_mvt', 'ADEP_flt', 'ADES_flt', 'ADES_FILED_flt',
            'MARKET_SEGMENT_flt', 'FLIGHT_RULE_flt', 'FLIGHT_TYPE_flt',
            'AIRCRAFT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'APT']
