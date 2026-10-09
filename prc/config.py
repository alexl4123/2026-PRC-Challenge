"""Paths and constants. Data deliberately lives OUTSIDE this repo.

The challenge Terms & Conditions forbid using or publishing the 2026 dataset
outside the competition until the organizers publish it. Keeping the parquets in
the parent directory makes it structurally impossible to commit them by accident.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]        # .../prc2026
DATA = REPO.parent                                # .../00_src   <- parquets live here
# FINAL PHASE (2026-10-09): PRC_RANKING=final points the whole pipeline at the organisers' final_ranking.parquet
# (Jan, Feb, Jun, Jul 2026) with its own derived cache and outputs, so the initial-phase artefacts stay untouched.
# Training-side inputs are mirrored into cache_final/ by tools/make_final_cache.py.
FINAL = __import__('os').environ.get('PRC_RANKING', 'initial') == 'final'
CACHE_BASE = REPO / 'cache'                       # the initial-phase cache (training-side inputs live here)
CACHE = REPO / ('cache_final' if FINAL else 'cache')       # gitignored, rebuildable
OUTPUTS = REPO / ('outputs_final' if FINAL else 'outputs')  # gitignored: models, submissions
RANKING_FILE = 'final_ranking.parquet' if FINAL else 'ranking.parquet'
SUBMIT_TEMPLATE = 'final_submitting.parquet' if FINAL else 'submitting.parquet'
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
