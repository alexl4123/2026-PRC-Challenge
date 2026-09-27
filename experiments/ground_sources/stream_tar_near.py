"""Stream a readsb globe_history tar (stdin), parse every trace_full_*.json(.gz) and keep points within
R km of the ten challenge ARPs and < 6000 ft above field or on ground. Prints per-airport counts of
aircraft with ground points and with low points, by source. Writes the kept points to a small parquet/csv.
Usage: curl -sL URL | python stream_tar_near.py OUT.csv"""
import sys, tarfile, gzip, json, math, collections, csv
ARP = {'EDDF': (50.0333, 8.5706, 364), 'EDDM': (48.3538, 11.7861, 1487), 'EGLL': (51.4775, -0.4614, 83),
       'EHAM': (52.3086, 4.7639, -11), 'LEBL': (41.2971, 2.0785, 12), 'LEMD': (40.4719, -3.5626, 1998),
       'LFPG': (49.0097, 2.5479, 392), 'LIRF': (41.8003, 12.2389, 13), 'LSZH': (47.4647, 8.5492, 1416),
       'LTFM': (41.2622, 28.7425, 325)}
R_KM = 12.0
def near(lat, lon):
    for a, (la, lo, el) in ARP.items():
        if abs(lat - la) < 0.12 and abs(lon - lo) < 0.17:
            dk = 6371 * math.hypot(math.radians(lon - lo) * math.cos(math.radians(la)), math.radians(lat - la))
            if dk <= R_KM: return a, el, dk
    return None
def main(out):
    tf = tarfile.open(fileobj=sys.stdin.buffer, mode='r|*')
    n_files = 0; ground = collections.defaultdict(set); low = collections.defaultdict(set)
    src_g = collections.defaultdict(collections.Counter); npts = collections.Counter()
    w = csv.writer(open(out, 'w', newline='')); w.writerow(['apt', 'hex', 't', 'lat', 'lon', 'alt', 'gs', 'track', 'src', 'dist_km'])
    for m in tf:
        if not m.isfile() or 'trace_full_' not in m.name: continue
        b = tf.extractfile(m).read(); n_files += 1
        if b[:2] == b'\x1f\x8b': b = gzip.decompress(b)
        try: d = json.loads(b)
        except Exception: continue
        hx = d.get('icao'); t0 = d.get('timestamp', 0)
        for p in d.get('trace', []):
            r = near(p[1], p[2])
            if not r: continue
            a, el, dk = r; alt = p[3]
            isg = alt == 'ground'
            if not isg and not (isinstance(alt, (int, float)) and alt - el < 6000): continue
            src = p[9] if len(p) > 9 else None
            npts[a] += 1
            (ground if isg else low)[a].add(hx)
            if isg: src_g[a][src] += 1
            w.writerow([a, hx, round(t0 + p[0], 2), p[1], p[2], alt, p[4], p[5], src, round(dk, 2)])
    print('trace files:', n_files)
    for a in ARP:
        print(f'{a}: points={npts[a]} aircraft_with_ground={len(ground[a])} aircraft_low_only={len(low[a] - ground[a])} ground_src={dict(src_g[a])}')
if __name__ == '__main__':
    main(sys.argv[1])
