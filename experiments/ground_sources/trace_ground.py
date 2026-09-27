"""For a readsb trace_full json: list ground/low segments near an airport, per take-off:
first ground point, first movement, speed profile, source mix, spacing."""
import json, sys, math, datetime as dt, collections
APT = {'LTFM': (41.2622, 28.7425, 325), 'LFPG': (49.0097, 2.5479, 392), 'LIRF': (41.8003, 12.2389, 13)}
def main(path, apt='LTFM', verbose=False):
    d = json.load(open(path)); la, lo, el = APT[apt]; t0 = d['timestamp']
    pts = []
    for p in d['trace']:
        t, lat, lon, alt, gs = p[0], p[1], p[2], p[3], p[4]
        dk = 6371 * math.hypot(math.radians(lon - lo) * math.cos(math.radians(la)), math.radians(lat - la))
        if dk > 8: continue
        src = p[9] if len(p) > 9 else None
        cs = (p[8] or {}).get('flight') if isinstance(p[8], dict) else None
        gnd = alt == 'ground' or (alt is None and gs is not None and gs < 60) or (isinstance(alt, (int, float)) and alt - el < 150 and (gs or 0) < 60)
        pts.append((t0 + t, lat, lon, alt, gs, p[5], src, gnd, dk, cs))
    print(path, d.get('r'), d.get('t'), 'points<8km:', len(pts))
    # split into ground runs separated by >600 s or airborne
    runs, cur = [], []
    for q in pts:
        if q[7]:
            if cur and q[0] - cur[-1][0] > 600: runs.append(cur); cur = []
            cur.append(q)
        else:
            if cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    for r in runs:
        ts = [q[0] for q in r]; srcs = collections.Counter(q[6] for q in r)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        mv = [q for q in r if (q[4] or 0) >= 3]
        f = lambda x: dt.datetime.utcfromtimestamp(x).strftime('%H:%M:%S')
        print(f'  run {f(ts[0])}-{f(ts[-1])} dur={ts[-1]-ts[0]:.0f}s n={len(r)} src={dict(srcs)} '
              f'gap_med={sorted(gaps)[len(gaps)//2] if gaps else None} gap_max={max(gaps) if gaps else None} '
              f'first_gs={r[0][4]} first_move={f(mv[0][0]) if mv else None} max_gs={max((q[4] or 0) for q in r)} cs={next((q[9] for q in r if q[9]), None)}')
        if verbose:
            for q in r[:12]: print('     ', f(q[0]), round(q[1], 5), round(q[2], 5), q[3], q[4], q[5], q[6])
if __name__ == '__main__':
    apt = sys.argv[1]
    for p in sys.argv[2:]: main(p, apt, verbose=True)
