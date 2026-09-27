"""Per snapshot (readsb aircraft.json / readsb-hist), per airport: aircraft within R km of the ARP that are
on the ground (alt_baro == 'ground', or alt missing with gs < 50 kt), split by source type, moving (gs >= 1)
vs stationary, and excluding service vehicles / fixed transmitters (no ICAO type or type in SERV/TWR/GND)."""
import json, gzip, math, sys, collections, os
ARP = {'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861), 'EGLL': (51.4775, -0.4614), 'EHAM': (52.3086, 4.7639),
       'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626), 'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389),
       'LSZH': (47.4647, 8.5492), 'LTFM': (41.2622, 28.7425)}
R = 4.5
VEH = {'SERV', 'TWR', 'GND', 'VEH', None, ''}
def load(p):
    with open(p, 'rb') as f: b = f.read()
    if b[:2] == b'\x1f\x8b': b = gzip.decompress(b)
    return json.loads(b)
def run(paths):
    rows = []
    for p in paths:
        d = load(p); ac = d.get('aircraft') or d.get('ac') or []
        for a, (la, lo) in ARP.items():
            c = collections.Counter()
            for x in ac:
                if 'lat' not in x: continue
                dk = 6371 * math.hypot(math.radians(x['lon'] - lo) * math.cos(math.radians(la)), math.radians(x['lat'] - la))
                if dk > R: continue
                alt = x.get('alt_baro'); gs = x.get('gs')
                ground = alt == 'ground' or (alt is None and gs is not None and gs < 50)
                if not ground: continue
                veh = x.get('t') in VEH
                key = ('veh' if veh else 'acft', x.get('type', '?'), 'mov' if (gs or 0) >= 1 else 'stat')
                c[key] += 1
            acft = sum(v for k, v in c.items() if k[0] == 'acft')
            rows.append((os.path.basename(p), a, acft, dict(c)))
    return rows
if __name__ == '__main__':
    rows = run(sys.argv[1:])
    apts = sys.argv[0] and ['LTFM', 'LFPG', 'LIRF', 'EHAM']
    for f, a, n, c in rows:
        if a in apts:
            print(f'{f:28s} {a} aircraft_on_ground={n:3d}  ' + ' '.join(f'{"/".join(k)}={v}' for k, v in sorted(c.items())))
