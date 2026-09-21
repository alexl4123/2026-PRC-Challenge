"""METAR weather from the Iowa Environmental Mesonet ASOS archive.

Public domain, no API key. Verified: 48 observations per station per day, 30
minutes apart, all ten airports, both scored months.

Joined at each flight's TAKEOFF time (the most recent observation at or before
it), not as a daily average -- taxi-out is a 10-30 minute process and the
weather that matters is the weather at the time.

De-icing is the mechanism we most want to capture. EUROCONTROL explicitly
excludes flights de-iced after off-block from its own taxi-out indicator, which
is institutional evidence that the effect is large and distinct. January 2026
was cold, so it should matter for the January third of the score.
"""
import io
import time
import urllib.request

import numpy as np
import pandas as pd

from .config import CACHE, AIRPORTS

METAR_DIR = CACHE / 'external' / 'metar'
BASE = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
# NOTE: 'p01i' (hourly precipitation) is requested but NOT used -- it comes back
# as literally 0.0 for all 292,353 observations at these ten European stations,
# which do not report that group. Keep it out of the feature set.
FIELDS = ['tmpc', 'dwpc', 'drct', 'sknt', 'gust', 'vsby', 'wxcodes', 'skyl1']

# Weather codes that imply de-icing or a frozen surface, from the METAR w'w' group.
SNOW_CODES = ('SN', 'SG', 'PL', 'GS', 'GR', 'IC')
FREEZE_CODES = ('FZ',)


def _fetch(station: str, y0, m0, y1, m1, attempts: int = 5) -> pd.DataFrame:
    """One station. Retries with exponential backoff: this is a free public
    service and it returns 503 if hit too quickly."""
    q = (f'{BASE}?station={station}' + ''.join(f'&data={f}' for f in FIELDS) +
         f'&year1={y0}&month1={m0}&day1=1&year2={y1}&month2={m1}&day2=1'
         '&tz=UTC&format=onlycomma&missing=M&trace=T&report_type=3&report_type=4')
    last = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(q, timeout=300) as r:
                return pd.read_csv(io.StringIO(r.read().decode()), low_memory=False)
        except Exception as e:                       # noqa: BLE001 - want any transport error
            last = e
            wait = 10 * 2 ** i
            print(f'    {station}: {type(e).__name__} {e}; retrying in {wait}s', flush=True)
            time.sleep(wait)
    raise RuntimeError(f'{station}: failed after {attempts} attempts') from last


def download(force: bool = False) -> pd.DataFrame:
    """All ten stations, 2025-01 through 2026-08. Cached on disk.

    Do NOT delete this cache to fix a downstream processing bug -- refetching
    all ten stations at once gets the requests throttled (HTTP 503), and the raw
    data was never the problem.
    """
    METAR_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for st in AIRPORTS:
        p = METAR_DIR / f'{st}.parquet'
        if p.exists() and not force:
            out.append(pd.read_parquet(p))
            continue
        df = _fetch(st, 2025, 1, 2026, 9)
        df.to_parquet(p, index=False)
        print(f'  {st}: {len(df):,} observations -> {p.name}', flush=True)
        out.append(df)
        time.sleep(5)                      # be polite to a free public service
    return pd.concat(out, ignore_index=True)


def _num(s):
    return pd.to_numeric(s.replace({'M': np.nan, 'T': '0.001'}), errors='coerce')


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean observations, sorted by station and time, ready for merge_asof."""
    d = pd.DataFrame({'station': raw.station.astype(str)})
    d['t'] = pd.to_datetime(raw.valid, utc=True)
    for c in ['tmpc', 'dwpc', 'drct', 'sknt', 'gust', 'vsby', 'skyl1']:
        d[c] = _num(raw[c]) if c in raw else np.nan
    wx = raw.get('wxcodes', pd.Series('', index=raw.index)).fillna('').astype(str)
    d['wx_snow'] = wx.str.contains('|'.join(SNOW_CODES), regex=True).astype('int8')
    d['wx_freeze'] = wx.str.contains('|'.join(FREEZE_CODES), regex=True).astype('int8')
    d['wx_rain'] = wx.str.contains('RA|DZ', regex=True).astype('int8')
    d['wx_fog'] = wx.str.contains('FG|BR', regex=True).astype('int8')
    d['wx_ts'] = wx.str.contains('TS', regex=True).astype('int8')
    d['gust'] = d.gust.fillna(0.0)
    # The operational trigger: cold enough to freeze AND frozen precipitation
    # actually being reported. (An earlier version also OR-ed in hourly
    # precipitation, which is all-zero here and contributed nothing.)
    d['deice_likely'] = ((d.tmpc <= 3) &
                         ((d.wx_snow == 1) | (d.wx_freeze == 1))).astype('int8')
    d['cold'] = (d.tmpc <= 3).astype('int8')
    d['frozen_precip'] = ((d.wx_snow == 1) | (d.wx_freeze == 1)).astype('int8')
    return d.dropna(subset=['t']).sort_values(['t']).reset_index(drop=True)


def _runway_heading(rwy: pd.Series) -> pd.Series:
    """Runway designator -> heading in degrees. '24' -> 240, '18C' -> 180."""
    n = rwy.astype(str).str.extract(r'^(\d{1,2})', expand=False)
    return pd.to_numeric(n, errors='coerce') * 10.0


def attach(dep: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Most recent observation at or before each departure's takeoff time.

    Returns a frame indexed like `dep` (which must have a clean RangeIndex).
    """
    obs = obs.copy()
    # merge_asof(on='t') keeps the LEFT frame's key, so the observation's own
    # timestamp has to travel as a separate column or it is silently lost.
    obs['obs_t'] = obs['t']

    q = pd.DataFrame({'station': dep.APT.astype(str),
                      't': dep.MVT_TIME_UTC_mvt})
    q['_i'] = np.arange(len(q))
    q = q.sort_values('t')
    m = pd.merge_asof(q, obs, on='t', by='station', direction='backward',
                      tolerance=pd.Timedelta('2h'))
    m = m.set_index('_i').sort_index()

    out = m.drop(columns=['station']).reset_index(drop=True)

    # How stale is the matched observation? Useful in its own right (a long gap
    # means the airport stopped reporting, which correlates with disruption) and
    # it replaces the raw timestamp, which LightGBM cannot consume.
    out['age_min'] = (out['t'] - out['obs_t']).dt.total_seconds() / 60.0
    out = out.drop(columns=['t', 'obs_t'])

    # Crosswind and headwind relative to the departure runway. The runway
    # designator encodes its heading: '24' -> 240 degrees.
    hdg = _runway_heading(dep.RUNWAY_mvt).to_numpy()
    ang = np.deg2rad(out.drct.to_numpy() - hdg)
    out['crosswind'] = np.abs(out.sknt.to_numpy() * np.sin(ang))
    out['headwind'] = out.sknt.to_numpy() * np.cos(ang)

    out.columns = ['wx_' + c if not c.startswith('wx_') else c for c in out.columns]
    bad = [c for c in out.columns if out[c].dtype.kind not in 'ifb']
    if bad:
        raise TypeError(f'non-numeric columns would reach the model: {bad}')
    dead = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
    if dead:
        print(f'  weather WARNING: constant columns carry no signal: {dead}')
    return out
