"""Day list for the full adsb.lol pipeline, one date per line (the Slurm array index = line number - 1).

    python experiments/adsb/adsb_days.py                 # all days (429)
    python experiments/adsb/adsb_days.py --missing       # days whose points/<date>.parquet is absent
    python experiments/adsb/adsb_days.py --count

'full' = 2024-12-31 .. 2025-12-31 (training year + the eve of 1 Jan for the pre-midnight window),
         2026-01-01 .. 2026-01-31 and 2026-06-30 .. 2026-07-31 (scored months + eve of 1 Jul).
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'cache' / 'adsb_lol'


def span(a: date, b: date):
    while a <= b:
        yield a
        a += timedelta(days=1)


def days(which='full'):
    out = list(span(date(2024, 12, 31), date(2025, 12, 31)))
    out += list(span(date(2026, 1, 1), date(2026, 1, 31)))
    out += list(span(date(2026, 6, 30), date(2026, 7, 31)))
    if which == 'scored':
        out = [d for d in out if d.year == 2026 or d == date(2025, 12, 31)]
    return [d.isoformat() for d in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', default='full', choices=['full', 'scored'])
    ap.add_argument('--missing', action='store_true')
    ap.add_argument('--count', action='store_true')
    a = ap.parse_args()
    ds = days(a.which)
    if a.missing:
        ds = [d for d in ds if not (BASE / 'points' / f'{d}.parquet').exists()]
    print(len(ds) if a.count else '\n'.join(ds))


if __name__ == '__main__':
    main()
