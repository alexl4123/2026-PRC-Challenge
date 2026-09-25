"""Candidate feature columns for the Discord-insights evaluation, built from the PROVIDED 2025
training file only (plus the cached METAR archive and, for one arm, the ADS-B stand table).
No label is used by any function here EXCEPT stand_te(), whose target encoding is computed
nested out-of-month inside each fold (the held-out month's labels never enter).

  wake_features()      idea 5: wake category of the previous departure on the same runway
  backlog_features()   idea 6: overdue / gate-hold backlog at our SCHED, our AOBT_3, our takeoff
  weather_features()   idea 4: temperature-dewpoint spread and frost indicators (at takeoff)
  stand_te_fold(m)     idea 2: (airport, stand, runway) target encoding, nested for fold m
  adsb_distance()      idea 2b: straight-line stand -> runway lift-off point (ADS-B positions)

All are row-aligned with load('training', phase='DEP') (asserted against dep_training.parquet).
Every column here is also computable on ranking.parquet (SCHED, MVT, AOBT_3, EOBT_1, RUNWAY, STAND,
AIRCRAFT_TYPE and WK_TBL_CAT are all present for departures there).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

from prc.config import CACHE
from prc.io import load
from common import SCRATCH

EPOCH = pd.Timestamp('1970-01-01', tz='UTC')


def _secs(s):
    return (s - EPOCH).dt.total_seconds().to_numpy()


def _dep(cols, which='training'):
    d = load(which, phase='DEP', columns=['MVT_ID_mvt'] + cols)
    ref = pd.read_parquet(CACHE / 'major' / f'dep_{which}.parquet', columns=['MVT_ID_mvt'])
    assert np.array_equal(d.MVT_ID_mvt.to_numpy(), ref.MVT_ID_mvt.to_numpy()), 'row order differs'
    return d


# ------------------------------------------------------------------ idea 5: wake sequence
WTC = {'L': 0, 'M': 1, 'H': 2, 'J': 3}
# ICAO Doc 4444 departure wake separation (s) for (leader, follower); 60 s = no wake minimum
SEP = {(3, 2): 120, (3, 1): 180, (3, 0): 180, (2, 1): 120, (2, 0): 120, (1, 0): 120}


def wake_features(which='training'):
    f = SCRATCH / ('feat_wake.parquet' if which == 'training' else f'feat_wake_{which}.parquet')
    if f.exists():
        return pd.read_parquet(f)
    full = load('training', columns=['AIRCRAFT_TYPE_mvt', 'WK_TBL_CAT_flt'])     # type map: training file
    # type -> modal category over ALL movements (fills rows without an NM record); label-free
    tt = full.dropna(subset=['WK_TBL_CAT_flt'])
    modal = tt.groupby(['AIRCRAFT_TYPE_mvt'], observed=True).WK_TBL_CAT_flt.agg(lambda s: s.value_counts().index[0])
    del full, tt
    d = _dep(['APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt', 'WK_TBL_CAT_flt'], which)
    cat = d.WK_TBL_CAT_flt.astype(object)
    cat = cat.where(cat.notna(), d.AIRCRAFT_TYPE_mvt.astype(object).map(modal))
    w = cat.map(WTC).astype(float).to_numpy()
    t = _secs(d.MVT_TIME_UTC_mvt)
    key = (d.APT.astype(str) + '|' + d.RUNWAY_mvt.astype(str)).to_numpy()
    o = np.lexsort((t, key))
    ks, ts, ws = key[o], t[o], w[o]
    same = np.r_[False, ks[1:] == ks[:-1]]
    prev_w = np.where(same, np.r_[np.nan, ws[:-1]], np.nan)
    gap = np.where(same, np.r_[np.nan, np.diff(ts)], np.nan)
    # heavy-or-super departures on the same runway in the preceding 10 min (running count)
    hv = (ws >= 2).astype(float)
    n_hv10 = np.full(len(ts), np.nan)
    for k in np.unique(ks):
        idx = np.flatnonzero(ks == k)
        tk, ck = ts[idx], np.cumsum(hv[idx])
        lo = np.searchsorted(tk, tk - 600, side='left')
        before = np.r_[0.0, ck][lo]
        n_hv10[idx] = ck - hv[idx] - before          # exclude self
    sep = np.array([SEP.get((int(a), int(b)), 60) if np.isfinite(a) and np.isfinite(b) else np.nan
                    for a, b in zip(prev_w, ws)])
    out = pd.DataFrame(index=np.arange(len(t)))
    for name, arr in (('wk_own', ws), ('wk_prev', prev_w), ('wk_sep_req', sep),
                      ('wk_gap_prev', gap), ('wk_hv_prev10', n_hv10)):
        col = np.empty(len(t)); col[o] = arr; out[name] = col
    out['wk_sep_excess'] = out.wk_gap_prev - out.wk_sep_req
    out['wk_pair'] = (out.wk_prev * 4 + out.wk_own)          # 16 codes, numeric (trees split it)
    out.to_parquet(f, index=False)
    return out


# ------------------------------------------------------------------ idea 6: backlog
CAP = 4 * 3600          # an overdue flight counts for at most its last 4 h (keeps day-late artefacts out)


def _stab(starts, ends, q):
    """#intervals [s, e) with s < q <= ... : count of s < q minus count of e <= q (valid for s < e)."""
    ok = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    s, e = np.sort(starts[ok]), np.sort(ends[ok])
    res = np.searchsorted(s, q, side='left') - np.searchsorted(e, q, side='right')
    return np.where(np.isfinite(q), res, np.nan).astype(float)


def backlog_features(which='training'):
    f = SCRATCH / ('feat_backlog.parquet' if which == 'training' else f'feat_backlog_{which}.parquet')
    if f.exists():
        return pd.read_parquet(f)
    d = _dep(['APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt'], which)
    mvt, sch, a3, eo = (_secs(d[c]) for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt'))
    n = len(mvt)
    out = {k: np.full(n, np.nan) for k in (
        'bl_ovd_at_sched', 'bl_ovd_at_a3', 'bl_ovd_at_mvt', 'bl_ovd15_at_sched',
        'bl_gate_at_sched', 'bl_gate_at_a3', 'bl_eobt_ovd_at_sched', 'bl_rwy_ovd_at_sched', 'bl_rwy_ovd_at_a3')}
    for grp_name, key in (('apt', d.APT.astype(str).to_numpy()),
                          ('rwy', (d.APT.astype(str) + '|' + d.RUNWAY_mvt.astype(str)).to_numpy())):
        for k in np.unique(key):
            i = np.flatnonzero(key == k)
            s_ovd = np.maximum(sch[i], mvt[i] - CAP)                 # past schedule, not yet airborne
            self_in = lambda q: ((s_ovd < q) & (q < mvt[i])).astype(float)
            if grp_name == 'apt':
                out['bl_ovd_at_sched'][i] = _stab(s_ovd, mvt[i], sch[i])
                out['bl_ovd_at_a3'][i] = _stab(s_ovd, mvt[i], a3[i]) - self_in(a3[i])
                out['bl_ovd_at_mvt'][i] = _stab(s_ovd, mvt[i], mvt[i])
                s15 = np.maximum(sch[i] + 900, mvt[i] - CAP)          # > 15 min past schedule
                out['bl_ovd15_at_sched'][i] = _stab(s15, mvt[i], sch[i])
                s_g = np.maximum(sch[i], a3[i] - CAP)                 # past schedule, not yet pushed (gate hold)
                out['bl_gate_at_sched'][i] = _stab(s_g, a3[i], sch[i])
                out['bl_gate_at_a3'][i] = _stab(s_g, a3[i], a3[i])
                s_e = np.maximum(eo[i], mvt[i] - CAP)                 # past EOBT, not yet airborne
                out['bl_eobt_ovd_at_sched'][i] = _stab(s_e, mvt[i], sch[i])
            else:
                out['bl_rwy_ovd_at_sched'][i] = _stab(s_ovd, mvt[i], sch[i])
                out['bl_rwy_ovd_at_a3'][i] = _stab(s_ovd, mvt[i], a3[i]) - self_in(a3[i])
    o = pd.DataFrame(out)
    o.to_parquet(f, index=False)
    return o


# ------------------------------------------------------------------ idea 4: weather spread
def weather_features():
    f = SCRATCH / 'feat_wx.parquet'
    if f.exists():
        return pd.read_parquet(f)
    from prc.weather import download, prepare, attach
    obs = prepare(download())
    d = _dep(['APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt'])
    wx = attach(d, obs)
    out = pd.DataFrame({'wx_tmpc': wx.wx_tmpc.to_numpy(), 'wx_dwpc': wx.wx_dwpc.to_numpy()})
    out['wx_spread'] = out.wx_tmpc - out.wx_dwpc
    out['wx_frost_risk'] = ((out.wx_tmpc <= 2) & (out.wx_spread <= 3)).astype(float)
    # coldest temperature and smallest spread over the preceding 12 h at the station (overnight frost)
    o2 = obs[['station', 't', 'tmpc', 'dwpc']].dropna().copy()
    o2['spr'] = o2.tmpc - o2.dwpc
    o2 = o2.sort_values(['station', 't']).set_index('t')
    roll = o2.groupby('station')[['tmpc', 'spr']].rolling('12h').min().reset_index()
    roll = roll.rename(columns={'tmpc': 'tmin12', 'spr': 'sprmin12'}).sort_values('t')
    q = pd.DataFrame({'station': d.APT.astype(str).to_numpy(), 't': d.MVT_TIME_UTC_mvt, '_i': np.arange(len(d))})
    roll['t'] = roll['t'].astype(q.t.dtype)
    m = pd.merge_asof(q.sort_values('t'), roll, on='t', by='station', direction='backward',
                      tolerance=pd.Timedelta('2h')).set_index('_i').sort_index()
    out['wx_tmin12'] = m.tmin12.to_numpy()
    out['wx_sprmin12'] = m.sprmin12.to_numpy()
    out['wx_frost_night'] = ((out.wx_tmin12 <= 0) & (out.wx_sprmin12 <= 2)).astype(float)
    out.to_parquet(f, index=False)
    return out


# ------------------------------------------------------------------ idea 2: stand x runway TE
class StandTE:
    """Nested out-of-month target encoding of (APT, STAND, RUNWAY), shrunk to (APT, RUNWAY), shrunk
    to APT. Statistic: mean of min(y, 3600) over rows with y <= 7200 (normal taxis; GREKI's
    'median taxi time from this stand to this runway' as a mean so it can be computed from sums).
    For fold m: rows of month m use months != m; rows of month k != m use months not in {k, m}."""

    def __init__(self, y, month, K=30.0):
        d = _dep(['APT', 'STAND_mvt', 'RUNWAY_mvt'])
        apt = d.APT.astype(str); st = d.STAND_mvt.astype(str); rw = d.RUNWAY_mvt.astype(str)
        ok = np.isfinite(y) & (y <= 7200)
        v = np.where(ok, np.minimum(y, 3600), 0.0); c = ok.astype(float)
        self.month, self.K = month, K
        self.levels = {}
        for name, key in (('a', apt), ('ar', apt + '|' + rw), ('asr', apt + '|' + st + '|' + rw),
                          ('as', apt + '|' + st)):
            codes, uniq = pd.factorize(key)
            S = np.zeros((len(uniq), 13)); N = np.zeros((len(uniq), 13))
            np.add.at(S, (codes, month), v); np.add.at(N, (codes, month), c)
            self.levels[name] = (codes, S, N)

    def _sums(self, name, m):
        codes, S, N = self.levels[name]
        own = self.month
        s = S[codes].sum(1) - S[codes, own] - np.where(own != m, S[codes, m], 0.0)
        n = N[codes].sum(1) - N[codes, own] - np.where(own != m, N[codes, m], 0.0)
        return s, n

    def fold(self, m):
        sa, na = self._sums('a', m); te_a = sa / np.maximum(na, 1)
        sr, nr = self._sums('ar', m); te_r = (sr + self.K * te_a) / (nr + self.K)
        ss, ns = self._sums('as', m); te_s = (ss + self.K * te_a) / (ns + self.K)
        st, nt = self._sums('asr', m); te_t = (st + self.K * te_r) / (nt + self.K)
        return pd.DataFrame({'te_srw': te_t, 'te_srw_minus_rw': te_t - te_r, 'te_rw': te_r,
                             'te_stand': te_s, 'te_srw_logn': np.log1p(nt)})


# ------------------------------------------------------------------ idea 2b: ADS-B distance
def adsb_distance():
    """Straight line from the stand's median ADS-B pushback position (cache/adsb_lol/stands.parquet,
    9-day test build) to the median ADS-B lift-off point of that (airport, runway) on the six
    labelled 2025 match days. Positions only (no label, no BLOCK). NaN where either is unknown."""
    st = pd.read_parquet(CACHE / 'adsb_lol' / 'stands.parquet').reset_index()
    mt = pd.read_parquet(CACHE / 'adsb_lol' / 'match' / 'training_6days.parquet',
                         columns=['APT', 'RUNWAY_mvt', 'ev_x_air', 'ev_y_air'])
    rw = mt.dropna().groupby([mt.APT.astype(str), mt.RUNWAY_mvt.astype(str)])[['ev_x_air', 'ev_y_air']].median()
    rw = rw[mt.dropna().groupby([mt.APT.astype(str), mt.RUNWAY_mvt.astype(str)]).size() >= 20]
    d = _dep(['APT', 'STAND_mvt', 'RUNWAY_mvt'])
    ks = pd.MultiIndex.from_arrays([d.APT.astype(str), d.STAND_mvt.astype(str)])
    kr = pd.MultiIndex.from_arrays([d.APT.astype(str), d.RUNWAY_mvt.astype(str)])
    sti = st.set_index(['_apt', '_stand'])
    sx = sti.sx.reindex(ks).to_numpy(); sy = sti.sy.reindex(ks).to_numpy()
    rx = rw.ev_x_air.reindex(kr).to_numpy(); ry = rw.ev_y_air.reindex(kr).to_numpy()
    return pd.DataFrame({'dist_stand_liftoff': np.hypot(sx - rx, sy - ry)})
