"""Summarise a readsb-style live JSON (adsb.lol / airplanes.live / adsb.fi point query)
or a readsb-hist snapshot: count aircraft on ground / low near each airport ARP."""
import json, sys, math, gzip
ARP = {'EDDF': (50.0333, 8.5706, 364), 'EDDM': (48.3538, 11.7861, 1487), 'EGLL': (51.4775, -0.4614, 83),
       'EHAM': (52.3086, 4.7639, -11), 'LEBL': (41.2971, 2.0785, 12), 'LEMD': (40.4719, -3.5626, 1998),
       'LFPG': (49.0097, 2.5479, 392), 'LIRF': (41.8003, 12.2389, 13), 'LSZH': (47.4647, 8.5492, 1416),
       'LTFM': (41.2622, 28.7425, 325)}
def dist_km(la1, lo1, la2, lo2):
    x = math.radians(lo2 - lo1) * math.cos(math.radians((la1 + la2) / 2)); y = math.radians(la2 - la1)
    return 6371 * math.hypot(x, y)
def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try:
        with op(p, 'rt') as f: return json.load(f)
    except (OSError, UnicodeDecodeError):
        with open(p, 'rt') as f: return json.load(f)
def summarise(d, apts=ARP, rkm=5.0):
    ac = d.get('ac') or d.get('aircraft') or []
    out = {}
    for a, (la, lo, el) in apts.items():
        n5 = ng = nlow = 0; src = {}
        for x in ac:
            if 'lat' not in x or 'lon' not in x: continue
            if dist_km(la, lo, x['lat'], x['lon']) > rkm: continue
            n5 += 1
            alt = x.get('alt_baro')
            if alt == 'ground': ng += 1; src[x.get('type', '?')] = src.get(x.get('type', '?'), 0) + 1
            elif isinstance(alt, (int, float)) and alt - el < 1500: nlow += 1
        out[a] = dict(within_km=rkm, n=n5, on_ground=ng, low_lt1500ft=nlow, ground_src=src)
    return out
if __name__ == '__main__':
    for p in sys.argv[1:]:
        d = load(p)
        print(p, 'now=', d.get('now'), 'total_ac=', len(d.get('ac') or d.get('aircraft') or []))
        for a, r in summarise(d).items():
            if r['n']: print('  ', a, r)
