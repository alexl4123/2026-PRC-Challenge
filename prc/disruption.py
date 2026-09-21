"""Daily ATFM disruption from EUROCONTROL, joined on (airport, date).

Source: Airport_Arrival_ATFM_Delay.xlsx, extracted once to
cache/external/atfm_daily.csv by experiments/ (see IDEAS.md 5a). Covers
2025-01-01 to 2026-08-31, so both scored months are present.

Cause codes: A accident, C ATC capacity, D de-icing, E equipment (ATC),
G aerodrome capacity, I industrial action (ATC), M military, N industrial
action (non-ATC), O other, P special event, R ATC routeings, S ATC staffing,
T equipment (non-ATC), V environmental, W weather.

Why it matters: July 2026 had 70% less weather delay than July 2025
(56,404 vs 186,066 min) and 46% less aerodrome-capacity delay. We train on
July 2025 -- the hardest month in the data -- and predict a much calmer July
2026, so there is a systematic over-prediction risk on the ~69% of our score
that July carries.

CAVEAT: this is ARRIVAL ATFM delay, a proxy for airport-wide disruption, not a
departure-side measure.
"""
import numpy as np
import pandas as pd

from .config import CACHE

CSV = CACHE / 'external' / 'atfm_daily.csv'
CAUSES = ['C', 'D', 'E', 'G', 'M', 'O', 'P', 'R', 'S', 'T', 'V', 'W', 'I', 'N']


def load_daily() -> pd.DataFrame:
    d = pd.read_csv(CSV, parse_dates=['date'])
    d['date'] = d.date.dt.date
    keep = ['date', 'apt', 'flt_arr'] + [f'dly_{c}' for c in CAUSES if f'dly_{c}' in d]
    d = d[keep].copy()
    cols = [c for c in d.columns if c.startswith('dly_')]
    d[cols] = d[cols].fillna(0.0)
    d['dly_total'] = d[cols].sum(axis=1)
    # Per-arrival intensity, so a big airport and a small one are comparable.
    d['dly_per_flt'] = d.dly_total / d.flt_arr.replace(0, np.nan)
    d['dly_W_per_flt'] = d.dly_W / d.flt_arr.replace(0, np.nan)
    d['dly_G_per_flt'] = d.dly_G / d.flt_arr.replace(0, np.nan)
    d['dly_strike'] = d.get('dly_I', 0) + d.get('dly_N', 0) + d.get('dly_S', 0)
    return d


def attach(dep: pd.DataFrame) -> pd.DataFrame:
    """Features indexed like `dep` (clean RangeIndex expected)."""
    daily = load_daily()
    key = pd.DataFrame({'apt': dep.APT.astype(str),
                        'date': dep.MVT_TIME_UTC_mvt.dt.date})
    m = key.merge(daily, on=['apt', 'date'], how='left')
    cols = ['dly_total', 'dly_per_flt', 'dly_W', 'dly_W_per_flt', 'dly_G',
            'dly_G_per_flt', 'dly_C', 'dly_S', 'dly_strike', 'flt_arr']
    out = m[[c for c in cols if c in m]].copy()
    out.columns = ['atfm_' + c for c in out.columns]
    out.index = dep.index
    return out
