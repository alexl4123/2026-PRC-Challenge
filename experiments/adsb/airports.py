"""Reference points of the ten challenge airports.

ARP = aerodrome reference point (published AIP values, rounded to ~10 m);
elevation in feet AMSL. Used only to cut a circle around each airport and to turn
barometric altitude into height above the field. Radius 12 km covers the full
movement area of every airport including EHAM's Polderbaan (~6.5 km from the ARP)
and LTFM's outer runways.
"""
import numpy as np

AIRPORTS = {
    #        lat        lon       elev_ft
    'EDDF': (50.0333,   8.5706,   364),
    'EDDM': (48.3538,  11.7861,  1487),
    'EGLL': (51.4775,  -0.4614,    83),
    'EHAM': (52.3086,   4.7639,   -11),
    'LEBL': (41.2971,   2.0785,    12),
    'LEMD': (40.4719,  -3.5626,  1998),
    'LFPG': (49.0097,   2.5479,   392),
    'LIRF': (41.8003,  12.2389,    13),
    'LSZH': (47.4647,   8.5492,  1416),
    'LTFM': (41.2622,  28.7425,   325),
}
CODES = list(AIRPORTS)
LAT = np.array([AIRPORTS[c][0] for c in CODES])
LON = np.array([AIRPORTS[c][1] for c in CODES])
ELEV = np.array([AIRPORTS[c][2] for c in CODES], dtype=float)

R_KEEP_M = 12_000.0      # keep points within this radius of the ARP
H_KEEP_FT = 6_000.0      # ... and below this height above the field (or on ground)
EARTH_R = 6_371_000.0


def local_xy(lat, lon, apt: str):
    """Equirectangular metres east/north of the ARP (error < 1 m at 12 km)."""
    lat0, lon0, _ = AIRPORTS[apt]
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y
