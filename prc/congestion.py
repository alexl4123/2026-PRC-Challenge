"""Congestion and queue features built from the full movement stream.

EVERY feature here is computable on ranking.parquet. That file keeps arrivals
completely intact -- landing time, in-block time, stand, runway -- and for
departures it keeps the takeoff time, the runway and AOBT_3_flt (pushback). So
the operational state of each airport during the scored months is fully
reconstructible; only the departure block time is hidden.

Motivation (EXP13): July's BULK RMSE is 306.9 against March's 208.7, with tail
rows excluded. That 98s is the largest structural gap we have measured and
summer congestion is the obvious candidate. Nothing in the model currently
captures it.

Feature families
  *_to_*    takeoffs (departure MVT_TIME) in a trailing window
  *_ldg_*   landings (arrival MVT_TIME) in a trailing window
  *_inblk_* arrivals reaching their stand in a trailing window
  *_push_*  pushbacks (AOBT_3_flt) in a trailing window
  N_queue   Idris-style count of takeoffs on the same runway between this
            aircraft's own pushback and its own takeoff
  n_taxiing number of aircraft simultaneously taxiing out at our pushback
  gap_prev_to  seconds since the previous takeoff on the same runway
"""
import numpy as np
import pandas as pd

EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
WINDOWS = (900, 1800, 3600)


def _secs(s: pd.Series) -> pd.Series:
    """Epoch seconds. These columns are timestamp[us]; never use astype('int64')
    and divide by 1e9 -- that silently makes every duration 1000x too small."""
    return (s - EPOCH).dt.total_seconds()


def _window_counts(events: pd.DataFrame, at: pd.DataFrame, key: str, tcol: str,
                   prefix: str, windows=WINDOWS) -> pd.DataFrame:
    """For each row of `at`, count rows of `events` sharing `key` whose `tcol`
    falls in (t - w, t]. Implemented as two merge_asof lookups into a per-key
    running count, so it is O(n log n) rather than a join per window."""
    ev = events[[key, tcol]].dropna(subset=[tcol]).sort_values(tcol).copy()
    ev['_c'] = ev.groupby(key, observed=True).cumcount() + 1
    ev = ev.rename(columns={tcol: '_t'})

    q = at[[key, 'at_t']].copy()
    q['_i'] = np.arange(len(q))
    q = q.sort_values('at_t')
    hi = pd.merge_asof(q, ev, left_on='at_t', right_on='_t', by=key, direction='backward')
    hi_c = pd.Series(hi['_c'].to_numpy(), index=hi['_i'].to_numpy())

    out = {}
    for w in windows:
        qw = q.assign(_lo=q.at_t - w).sort_values('_lo')
        lo = pd.merge_asof(qw, ev, left_on='_lo', right_on='_t', by=key, direction='backward')
        lo_c = pd.Series(lo['_c'].to_numpy(), index=lo['_i'].to_numpy())
        idx = np.arange(len(at))
        out[f'{prefix}_{w // 60}m'] = (hi_c.reindex(idx).fillna(0).to_numpy()
                                       - lo_c.reindex(idx).fillna(0).to_numpy())
    return pd.DataFrame(out, index=at.index)


def build(full: pd.DataFrame) -> pd.DataFrame:
    """`full` = both phases for one period. Returns features for the DEP rows,
    indexed to match `full[full.PHASE_mvt == 'DEP']` after reset_index(drop=True)."""
    f = full.copy()
    f['t_mvt'] = _secs(f.MVT_TIME_UTC_mvt)
    f['t_blk'] = _secs(f.BLOCK_TIME_UTC_mvt)
    f['t_push'] = _secs(f.AOBT_3_flt)
    f['k_apt'] = f.APT.astype(str)
    f['k_rwy'] = f.APT.astype(str) + '|' + f.RUNWAY_mvt.astype(str)

    dep = f[f.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    arr = f[f.PHASE_mvt.eq('ARR')].reset_index(drop=True)
    out = pd.DataFrame(index=dep.index)

    # --- trailing-window activity, evaluated at our own takeoff time ---------
    dep_at = dep.assign(at_t=dep.t_mvt)
    for ev, key, tcol, pref in [
            (dep, 'k_rwy', 't_mvt', 'dep_rwy_to'), (dep, 'k_apt', 't_mvt', 'dep_apt_to'),
            (arr, 'k_apt', 't_mvt', 'arr_apt_ldg'), (arr, 'k_rwy', 't_mvt', 'arr_rwy_ldg'),
            (arr, 'k_apt', 't_blk', 'arr_apt_inblk'),
            (dep, 'k_apt', 't_push', 'dep_apt_push'), (dep, 'k_rwy', 't_push', 'dep_rwy_push')]:
        out = out.join(_window_counts(ev, dep_at, key, tcol, pref))

    # --- Idris N: takeoffs on our runway between our pushback and our takeoff --
    ev = dep[['k_rwy', 't_mvt']].dropna().sort_values('t_mvt').copy()
    ev['_c'] = ev.groupby('k_rwy', observed=True).cumcount() + 1
    ev = ev.rename(columns={'t_mvt': '_t'})
    q = dep[['k_rwy', 't_mvt', 't_push']].copy()
    q['_i'] = np.arange(len(q))
    hi = pd.merge_asof(q.sort_values('t_mvt'), ev, left_on='t_mvt', right_on='_t',
                       by='k_rwy', direction='backward')
    lo = pd.merge_asof(q.dropna(subset=['t_push']).sort_values('t_push'), ev,
                       left_on='t_push', right_on='_t', by='k_rwy', direction='backward')
    hi_c = pd.Series(hi['_c'].to_numpy(), index=hi['_i'].to_numpy())
    lo_c = pd.Series(lo['_c'].to_numpy(), index=lo['_i'].to_numpy())
    idx = np.arange(len(dep))
    out['N_queue_rwy'] = (hi_c.reindex(idx) - lo_c.reindex(idx)).to_numpy()

    # --- how many aircraft are taxiing out when we push back ------------------
    a = dep[['k_apt', 't_push', 't_mvt']].dropna()
    ee = pd.concat([a[['k_apt', 't_push']].rename(columns={'t_push': 't'}).assign(d=1),
                    a[['k_apt', 't_mvt']].rename(columns={'t_mvt': 't'}).assign(d=-1)])
    ee = ee.sort_values('t')
    ee['occ'] = ee.groupby('k_apt', observed=True)['d'].cumsum()
    ee = ee.drop_duplicates(subset=['k_apt', 't'], keep='last')
    qd = dep[['k_apt', 't_push']].copy()
    qd['_i'] = np.arange(len(qd))
    mm = pd.merge_asof(qd.dropna(subset=['t_push']).sort_values('t_push'),
                       ee[['k_apt', 't', 'occ']], left_on='t_push', right_on='t',
                       by='k_apt', direction='backward')
    out['n_taxiing_out'] = pd.Series(mm['occ'].to_numpy(),
                                     index=mm['_i'].to_numpy()).reindex(idx).to_numpy()

    # --- seconds since the previous takeoff on our runway ---------------------
    d2 = dep[['k_rwy', 't_mvt']].copy()
    d2['_i'] = np.arange(len(d2))
    d2 = d2.sort_values('t_mvt')
    prev = d2.groupby('k_rwy', observed=True)['t_mvt'].shift(1)
    out['gap_prev_to'] = pd.Series((d2.t_mvt - prev).to_numpy(),
                                   index=d2['_i'].to_numpy()).reindex(idx).to_numpy()

    # --- normalised load: activity relative to this airport's own typical rate -
    for c in ['dep_apt_to_60m', 'arr_apt_ldg_60m']:
        med = out.groupby(dep.APT.to_numpy(), observed=True)[c].transform('median')
        out[c + '_rel'] = out[c] / med.replace(0, np.nan)
    return out
