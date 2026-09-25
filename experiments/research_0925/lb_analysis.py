"""Leaderboard analysis for the 2026-09-25 research report (public API only, read-only).

    python experiments/research_0925/lb_analysis.py [--json rows.json] [--since 2026-09-24T08:00:00]

Without --json it fetches the full submission history via tools/leaderboard.fetch().
Prints: the current front (top 25), cut-offs (rank = floor(p * n_teams)) for p = 10/20/25%,
their history at daily checkpoints, movers since --since, and linear projections to the deadline.
"""
import sys, json, math, os, re
from datetime import datetime, timedelta, timezone
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
import leaderboard as lb

ME = 'understanding-accordion'
DEADLINE = datetime(2026, 10, 11, 22, 59, tzinfo=timezone.utc)   # 23:59 CET = 22:59 UTC
PS = (0.10, 0.20, 0.25)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def board(rows, until=None):
    best = lb.best_by_team(rows, until=until)
    return sorted(best.items(), key=lambda kv: kv[1])


def cutoffs(rank):
    n = len(rank)
    out = {}
    for p in PS:
        k = max(1, math.floor(p * n))
        out[p] = (k, rank[k - 1][1])
    return n, out


def main():
    path = arg('--json')
    rows = json.load(open(path)) if path else lb.fetch()
    since = arg('--since', '2026-09-24T08:00:00')
    now_s = max(r['processedAt'] for r in rows)
    now = ts(now_s)
    print(f'snapshot {now_s[:19]}Z  submissions {len(rows):,}  teams {len(board(rows))}')
    print(f'usedPairs values: {sorted(set(r.get("usedPairs") for r in rows))}')

    rank = board(rows)
    pos = {t: i for i, (t, _) in enumerate(rank, 1)}
    n, cut = cutoffs(rank)
    print(f'\nCUT-OFFS now (n={n}):  ' + '  '.join(f'top {int(p*100)}% rank<={k}: {s:.4f}' for p, (k, s) in cut.items()))
    me = dict(rank)[ME]
    print(f'us rank {pos[ME]} score {me:.4f}; gap to top10% {me - cut[0.10][1]:.2f}, top20% {me - cut[0.20][1]:.2f}, '
          f'top25% {me - cut[0.25][1]:.2f}; leader {rank[0][0]} {rank[0][1]:.4f} (gap {me - rank[0][1]:.2f})')

    # front: top 25 with submission counts, last submission time, best-time and score 24h/72h ago
    before = dict(board(rows, until=since))
    by_team = {}
    for r in rows:
        by_team.setdefault(r['teamName'], []).append(r)
    print(f'\nFRONT (best now | best at {since}Z | delta | n subs | subs since | last sub | first sub)')
    for i, (t, s) in enumerate(rank[:30], 1):
        rr = sorted(by_team[t], key=lambda r: r['processedAt'])
        nsince = sum(1 for r in rr if r['processedAt'] > since)
        b = before.get(t)
        d = '' if b is None else f'{s - b:+7.2f}'
        print(f'  {i:3d} {t:26s} {s:9.4f} {("new" if b is None else f"{b:9.4f}"):>9} {d:>8} {len(rr):4d} {nsince:4d}  '
              f'{rr[-1]["processedAt"][:16]}  {rr[0]["processedAt"][:10]}')
    print('  ...')
    for i, (t, s) in enumerate(rank, 1):
        if 30 < i and abs(i - pos[ME]) <= 3:
            b = before.get(t)
            print(f'  {i:3d} {t:26s} {s:9.4f} {("new" if b is None else f"{b:9.4f}"):>9}')

    # movers since
    moved = sorted(((before.get(t, float('inf')) - s, t, before.get(t), s) for t, s in rank
                    if s < before.get(t, float('inf'))), reverse=True)
    print(f'\nMOVERS since {since}Z: {len(moved)} teams improved; new teams: '
          f'{sum(1 for m in moved if m[2] is None)}')
    rb = board(rows, until=since)
    posb = {t: i for i, (t, _) in enumerate(rb, 1)}
    for d, t, b, s in moved[:40]:
        print(f'  {t:26s} {("new" if b is None else f"{b:9.2f}"):>9} -> {s:9.2f}  '
              f'{("" if b is None else f"-{d:6.2f}"):>8}  rank {posb.get(t, "-")} -> {pos[t]}')

    # cut-off history at checkpoints
    print('\nCUT-OFF HISTORY (end of UTC day unless stated)')
    print('  checkpoint            n   leader   r10  top10%   r20  top20%   r25  top25%  median   us(rank)')
    hist = []
    day = datetime(2026, 9, 8, 23, 59, 59, tzinfo=timezone.utc)
    cps = []
    while day < now:
        cps.append(day)
        day += timedelta(days=1)
    cps += [ts('2026-09-24T08:00:00Z'), now + timedelta(seconds=1)]
    cps = sorted(set(cps))
    for c in cps:
        cs = c.strftime('%Y-%m-%dT%H:%M:%S')
        rk = board(rows, until=cs)
        n_, ct = cutoffs(rk)
        med = float(np.median([s for _, s in rk]))
        us = dict(rk).get(ME)
        up = [i for i, (t, _) in enumerate(rk, 1) if t == ME]
        hist.append((c, n_, rk[0][1], ct[0.10][1], ct[0.20][1], ct[0.25][1], med))
        print(f'  {cs[:16]}  {n_:4d} {rk[0][1]:8.2f}  {ct[0.10][0]:3d} {ct[0.10][1]:7.2f}  {ct[0.20][0]:3d} '
              f'{ct[0.20][1]:7.2f}  {ct[0.25][0]:3d} {ct[0.25][1]:7.2f}  {med:6.2f}   '
              f'{"" if us is None else f"{us:.2f} ({up[0]})"}')

    # projections: linear fits on the last 7 and 14 days of end-of-day cut-offs (+ now)
    days_left = (DEADLINE - now).total_seconds() / 86400
    print(f'\nPROJECTION to the deadline ({DEADLINE:%Y-%m-%d %H:%M}Z, {days_left:.1f} days from the snapshot)')
    H = np.array([[(h[0] - now).total_seconds() / 86400] + list(h[1:]) for h in hist])
    for win in (4, 7, 14):
        m = H[:, 0] >= -win
        x = H[m, 0]
        out = []
        for j, name in ((1, 'n'), (2, 'leader'), (3, 'top10'), (4, 'top20'), (5, 'top25')):
            b, a = np.polyfit(x, H[m, j], 1)
            out.append(f'{name} {a:.2f} {b:+.3f}/d -> {a + b * days_left:.2f}')
        print(f'  last {win:2d} d: ' + ' | '.join(out))

    # team-count projection -> cut-off rank at the deadline
    b, a = np.polyfit(H[H[:, 0] >= -7, 0], H[H[:, 0] >= -7, 1], 1)
    n_dl = a + b * days_left
    print(f'  teams at deadline (7-d linear) ~{n_dl:.0f} -> top-10% rank {math.floor(0.1 * n_dl)}, '
          f'top-20% rank {math.floor(0.2 * n_dl)}, top-25% rank {math.floor(0.25 * n_dl)}')
    # score today at those ranks (what the cut-off would be with more teams but no movement)
    for p in PS:
        k = math.floor(p * n_dl)
        print(f'    today, score at rank {k}: {rank[k - 1][1]:.2f}')

    # the band around the top-10% cut: who sits at ranks 12-30 and how fast they have moved (last 7 days)
    c7 = (now - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
    c3 = (now - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%S')
    b7 = dict(board(rows, until=c7))
    b3 = dict(board(rows, until=c3))
    print('\nRANKS 10-40: best now, change over the last 3 and 7 days')
    for i, (t, s) in enumerate(rank[9:40], 10):
        d3 = '' if t not in b3 else f'{s - b3[t]:+7.2f}'
        d7 = '' if t not in b7 else f'{s - b7[t]:+7.2f}'
        print(f'  {i:3d} {t:26s} {s:9.4f}  3d {d3:>8}  7d {d7:>8}')

    # outside view: of teams in [270, 290] 7 days ago, how many improved by >= x
    pool = [t for t, s in b7.items() if 265 <= s <= 290]
    gains = np.array([b7[t] - dict(rank)[t] for t in pool])
    print(f'\nOUTSIDE VIEW: {len(pool)} teams at 265-290 seven days ago; gain over the 7 days: '
          f'median {np.median(gains):.2f}, >=5 s {np.mean(gains >= 5):.0%}, >=10 s {np.mean(gains >= 10):.0%}, '
          f'>=15 s {np.mean(gains >= 15):.0%}, max {gains.max():.2f}')
    pool = [t for t, s in b7.items() if s <= 265]
    gains = np.array([b7[t] - dict(rank)[t] for t in pool])
    print(f'  teams <=265 seven days ago ({len(pool)}): median gain {np.median(gains):.2f}, '
          f'>=3 s {np.mean(gains >= 3):.0%}, max {gains.max():.2f}')


def extras(rows):
    now = ts(max(r['processedAt'] for r in rows)) + timedelta(seconds=1)
    rank = board(rows)
    sc = np.array([s for _, s in rank])
    print('\nDENSITY: teams with best score in each band now')
    for lo, hi in ((225, 240), (240, 250), (250, 255), (255, 260), (260, 265), (265, 270), (270, 275), (275, 280)):
        print(f'  [{lo},{hi}): {int(((sc >= lo) & (sc < hi)).sum())}')
    print('\nFIXED-RANK history (score at rank 19 / 26 / 38 / 47 at end of each UTC day)')
    day = datetime(2026, 9, 14, 23, 59, 59, tzinfo=timezone.utc)
    while day < now + timedelta(days=1):
        c = min(day, now).strftime('%Y-%m-%dT%H:%M:%S')
        rk = board(rows, until=c)
        print(f'  {c[:16]}  n={len(rk):3d}  r19 {rk[18][1]:.2f}  r26 {rk[25][1]:.2f}  r38 {rk[37][1]:.2f}  r47 {rk[46][1]:.2f}')
        day += timedelta(days=1)
    print('\nTRAJECTORIES of big recent movers (submissions since 09-21, score, filename version)')
    for t in ('jolly-lobster', 'quick-boat', 'humble-rocket', 'nice-umbrella', 'dynamic-lobster', 'strong-crane',
              'bubbly-telephone', 'gentle-ladder', 'vigorous-telephone', 'affectionate-dollhouse', 'loyal-anchor',
              'upstanding-firefly', 'zesty-puzzle', 'humble-giraffe', 'kind-mango'):
        rr = sorted([r for r in rows if r['teamName'] == t and r['processedAt'] >= '2026-09-21'], key=lambda r: r['processedAt'])
        best = float('inf'); out = []
        for r in rr:
            v = re.findall(r'(v\d+[a-z0-9_]*)', r.get('filename') or '')
            tag = v[0] if v else (r.get('filename') or '')[-20:]
            mark = '*' if r['score'] < best else ''
            best = min(best, r['score'])
            out.append(f"{r['processedAt'][5:13]} {tag} {r['score']:.2f}{mark}")
        print(f'  {t}: ' + ' | '.join(out))


if __name__ == '__main__':
    if '--extras' in sys.argv:
        extras(json.load(open(arg('--json'))) if arg('--json') else lb.fetch())
        sys.exit()
    main()
