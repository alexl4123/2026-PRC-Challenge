"""A3 arm 2: wake-sequence and schedule-backlog columns, plus the gap_prev_to month-boundary fix.

Measured 2026-09-25 (reports/2026-09-25_discord_insights_eval.md, residual harness on v15s_oof, minus control):
wake + backlog-at-schedule -0.71 [-0.84,-0.59] 12 months, -0.81 scored, January and July each better. Chosen after
the full backlog set failed January, so it is a hypothesis for the A3 CV, judged only against arm 1.

All columns are label-free and use only fields present for departures in BOTH the training and the ranking file
(RUNWAY_mvt, MVT/SCHED times, AOBT_3, EOBT_1, WK_TBL_CAT, aircraft type). Definitions follow the evaluation's
experiments/discord_eval/features.py (our own code), with two changes for production:
  - "previous departure on the same runway" is ignored when it lies more than 6 h back (MAX_GAP), which removes
    the January -> July jump inside the ranking file and any overnight or closure gap;
  - gap_prev_to (prc/congestion.py) gets the same 6 h cap -- the bug the evaluation found (the first July
    departure per runway saw a ~5-month gap).
The type -> wake-category fallback (rows without an NM record) is taken from the TRAINING departures and cached,
so the ranking file uses the same map.
"""
import json
import numpy as np
import pandas as pd

from prc.config import CACHE

WTC = {'L': 0, 'M': 1, 'H': 2, 'J': 3}
SEP = {(3, 2): 120, (3, 1): 180, (3, 0): 180, (2, 1): 120, (2, 0): 120, (1, 0): 120}   # ICAO Doc 4444 departures
MAX_GAP = 6 * 3600
CAP = 4 * 3600            # an overdue flight counts for at most its last 4 h (keeps day-late artefacts out)
MAP_FILE = CACHE / 'w2' / 'wtc_modal_by_type.json'
EXTRA_COLS = ['wk_own', 'wk_prev', 'wk_sep_req', 'wk_gap_prev', 'wk_hv_prev10', 'wk_sep_excess', 'wk_pair',
              'bl_ovd_at_sched', 'bl_ovd15_at_sched', 'bl_gate_at_sched', 'bl_eobt_ovd_at_sched', 'bl_rwy_ovd_at_sched']
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')


def _secs(s):
    return (s - EPOCH).dt.total_seconds().to_numpy()


def _stab(starts, ends, q):
    """Number of intervals [s, e) containing q (valid for s < e)."""
    ok = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    s, e = np.sort(starts[ok]), np.sort(ends[ok])
    res = np.searchsorted(s, q, side='left') - np.searchsorted(e, q, side='right')
    return np.where(np.isfinite(q), res, np.nan).astype(float)


def wtc_map(d, which):
    if which == 'training':
        t = d.dropna(subset=['WK_TBL_CAT_flt'])
        m = t.groupby(t.AIRCRAFT_TYPE_mvt.astype(str)).WK_TBL_CAT_flt.agg(lambda s: s.astype(str).value_counts().index[0])
        MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(m.to_dict(), open(MAP_FILE, 'w'))
        return m.to_dict()
    if not MAP_FILE.exists():
        raise SystemExit(f'FATAL: {MAP_FILE} missing -- build the training frame first (assemble_w2_pair does)')
    return json.load(open(MAP_FILE))


def build(d, which):
    """d = the departures frame of `which` (training or ranking), as in w2_features._one. Returns a frame of
    EXTRA_COLS in d's row order."""
    modal = wtc_map(d, which)
    cat = d.WK_TBL_CAT_flt.astype(object)
    cat = cat.where(cat.notna(), d.AIRCRAFT_TYPE_mvt.astype(str).map(modal))
    w = cat.map(WTC).astype(float).to_numpy()
    t = _secs(d.MVT_TIME_UTC_mvt)
    key = (d.APT.astype(str) + '|' + d.RUNWAY_mvt.astype(str)).to_numpy()
    o = np.lexsort((t, key))
    ks, ts, ws = key[o], t[o], w[o]
    gap = np.where(np.r_[False, ks[1:] == ks[:-1]], np.r_[np.nan, np.diff(ts)], np.nan)
    gap = np.where(gap > MAX_GAP, np.nan, gap)
    prev_w = np.where(np.isfinite(gap), np.r_[np.nan, ws[:-1]], np.nan)
    hv = (ws >= 2).astype(float)
    n_hv10 = np.full(len(ts), np.nan)
    for k in np.unique(ks):
        idx = np.flatnonzero(ks == k)
        tk, ck = ts[idx], np.cumsum(hv[idx])
        lo = np.searchsorted(tk, tk - 600, side='left')
        n_hv10[idx] = ck - hv[idx] - np.r_[0.0, ck][lo]
    sep = np.array([SEP.get((int(a), int(b)), 60) if np.isfinite(a) and np.isfinite(b) else np.nan
                    for a, b in zip(prev_w, ws)])
    out = pd.DataFrame(index=np.arange(len(t)))
    for name, arr in (('wk_own', ws), ('wk_prev', prev_w), ('wk_sep_req', sep), ('wk_gap_prev', gap), ('wk_hv_prev10', n_hv10)):
        col = np.empty(len(t)); col[o] = arr; out[name] = col
    out['wk_sep_excess'] = out.wk_gap_prev - out.wk_sep_req
    out['wk_pair'] = out.wk_prev * 4 + out.wk_own

    mvt, sch, a3, eo = (_secs(d[c]) for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt'))
    for c in EXTRA_COLS[7:]:
        out[c] = np.nan
    for grp, keyv in (('apt', d.APT.astype(str).to_numpy()), ('rwy', key)):
        for k in np.unique(keyv):
            i = np.flatnonzero(keyv == k)
            s_ovd = np.maximum(sch[i], mvt[i] - CAP)
            if grp == 'apt':
                out.loc[i, 'bl_ovd_at_sched'] = _stab(s_ovd, mvt[i], sch[i])
                out.loc[i, 'bl_ovd15_at_sched'] = _stab(np.maximum(sch[i] + 900, mvt[i] - CAP), mvt[i], sch[i])
                out.loc[i, 'bl_gate_at_sched'] = _stab(np.maximum(sch[i], a3[i] - CAP), a3[i], sch[i])
                out.loc[i, 'bl_eobt_ovd_at_sched'] = _stab(np.maximum(eo[i], mvt[i] - CAP), mvt[i], sch[i])
            else:
                out.loc[i, 'bl_rwy_ovd_at_sched'] = _stab(s_ovd, mvt[i], sch[i])
    return out[EXTRA_COLS].astype('float32')
