"""Leaderboard snapshot: our submissions, rank, neighbours, the front, and overnight movement.

    python tools/leaderboard.py                 # snapshot
    python tools/leaderboard.py --wait v10      # poll until that file is scored (max 25 min)
    python tools/leaderboard.py --since 16      # also list who moved in the last 16 hours
Uses only the public competition API. Retries through transient DNS/network errors.
"""
import json, sys, time, urllib.request, statistics as st
from datetime import datetime, timezone, timedelta
BASE = ("https://datacomp.opensky-network.org/api/competitions/"
        "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard")
ME = 'understanding-accordion'

def _fetch():
    rows, cur = [], None
    for _ in range(80):
        d = json.load(urllib.request.urlopen(BASE + "?limit=200" + (f"&cursor={cur}" if cur else ""), timeout=30))
        items = d.get('items') or d.get('data') or []
        rows += items; cur = d.get('nextCursor')
        if not cur or not items: break
    return [r for r in rows if r.get('score') is not None]

def fetch():
    for attempt in range(20):
        try: return _fetch()
        except Exception as e:
            print(f'  fetch failed ({type(e).__name__}: {e}); retry {attempt+1}/20 in 30 s', flush=True)
            time.sleep(30)
    raise SystemExit('leaderboard unreachable for 10 minutes')

def best_by_team(rows, until=None):
    best = {}
    for r in rows:
        if until and r['processedAt'] > until: continue
        t, s = r['teamName'], r['score']
        if t not in best or s < best[t]: best[t] = s
    return best

def main():
    wait = sys.argv[sys.argv.index('--wait') + 1] if '--wait' in sys.argv else None
    since = float(sys.argv[sys.argv.index('--since') + 1]) if '--since' in sys.argv else None
    t0 = time.time()
    while True:
        rows = fetch()
        mine = sorted([r for r in rows if r['teamName'] == ME], key=lambda r: r['processedAt'])
        if not wait or any(f'_{wait}.parquet' in (r.get('filename') or '') for r in mine): break
        if time.time() - t0 > 1500: print(f'{wait} not scored after 25 min'); break
        time.sleep(30)
    best = best_by_team(rows)
    rank = sorted(best.items(), key=lambda kv: kv[1])
    pos = [i for i, (t, _) in enumerate(rank, 1) if t == ME][0]
    print(f"snapshot {max(r['processedAt'] for r in rows)[:19]}Z | {len(rows):,} submissions, {len(best)} teams")
    for r in mine[-4:]: print(f"  {r['processedAt'][:16]}  {r.get('filename',''):38s} {r['score']}")
    print(f"RANK {pos} of {len(best)} | best {best[ME]:.4f} | leader {rank[0][0]} {rank[0][1]:.4f} "
          f"(gap {best[ME]-rank[0][1]:.2f}) | median {st.median(best.values()):.2f}")
    for i, (t, s) in enumerate(rank, 1):
        if abs(i - pos) <= 2: print(f"  {i:4d}  {t:26s} {s:9.4f}{'  <-- us' if t == ME else ''}")
    print("top 8: " + ", ".join(f"{t} {s:.2f}" for t, s in rank[:8]))
    if since:
        cut = (datetime.now(timezone.utc) - timedelta(hours=since)).strftime('%Y-%m-%dT%H:%M:%S')
        before = best_by_team(rows, until=cut)
        moved = sorted(((before.get(t, float('inf')) - s, t, before.get(t), s) for t, s in best.items()
                        if s < before.get(t, float('inf'))), reverse=True)
        print(f"\nimproved in the last {since:g} h: {len(moved)} teams; biggest steps:")
        for d, t, b, s in moved[:8]:
            print(f"  {t:26s} {('new' if b is None else f'{b:.2f}'):>8} -> {s:8.2f}  ({'new' if b is None else f'-{d:.2f}'})")
        pr = sorted(before.items(), key=lambda kv: kv[1])
        was = [i for i, (t, _) in enumerate(pr, 1) if t == ME]
        print(f"  our rank {was[0] if was else '?'} -> {pos}")

if __name__ == '__main__':
    main()
