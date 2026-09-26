"""Leaderboard snapshot for the 2026-09-26 review (public API only, read-only).

    python experiments/drivers_0926/lb_snapshot.py [--json rows.json] [--dump rows.json]

Reports: team count, our rank/score, leader and top 10, movement in the last 24 h and since
2026-09-24T00:00Z, cut-offs for top 5/10/20/25 % (rank = ceil(p * n_teams)) and our gap,
team counts in score bands, and the history of the top-10 % cut-off at 12-hour checkpoints.
"""
import sys, os, json, math
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
import leaderboard as lb

ME = 'understanding-accordion'
PS = (0.05, 0.10, 0.20, 0.25)
BANDS = ((225, 240), (240, 251), (251, 260))


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def iso(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S')


def board(rows, until=None, drop=None):
    return sorted(((t, v) for t, v in lb.best_by_team(rows, until=until).items() if t != drop), key=lambda kv: kv[1])


def cut(rank, p):
    k = max(1, math.ceil(p * len(rank)))
    return k, rank[k - 1][1]


def movers(rows, now_rank, until, label, show_below=300.0):
    before = dict(board(rows, until=until))
    pos_before = {t: i for i, (t, _) in enumerate(board(rows, until=until), 1)}
    pos_now = {t: i for i, (t, _) in enumerate(now_rank, 1)}
    imp = [(t, before.get(t), s) for t, s in now_rank if t not in before or s < before[t]]
    new = [x for x in imp if x[1] is None]
    print(f'\nMOVEMENT {label} (since {until}Z): {len(imp)} teams improved ({len(new)} new); '
          f'teams then {len(before)} -> now {len(now_rank)}')
    for t, b, s in imp:
        if s >= show_below:
            continue
        step = 'new' if b is None else f'-{b - s:.2f}'
        was = pos_before.get(t, '-')
        print(f'  {t:26s} {("new" if b is None else f"{b:8.2f}"):>8} -> {s:8.2f} ({step:>8})  rank {was} -> {pos_now[t]}')
    rest = [x for x in imp if x[2] >= show_below]
    if rest:
        print(f'  (+{len(rest)} improvers now at >= {show_below:.0f} not listed)')


def main():
    path = arg('--json')
    rows = json.load(open(path)) if path else lb.fetch()
    if arg('--dump'):
        json.dump(rows, open(arg('--dump'), 'w'))
    now_s = max(r['processedAt'] for r in rows)
    now = datetime.fromisoformat(now_s[:19]).replace(tzinfo=timezone.utc)
    rank = board(rows)
    n = len(rank)
    pos = {t: i for i, (t, _) in enumerate(rank, 1)}
    me = dict(rank)[ME]
    print(f'snapshot (latest processedAt) {now_s[:19]}Z | {len(rows):,} scored submissions | {n} teams')
    print(f'US: rank {pos[ME]} of {n}, score {me:.4f} | leader {rank[0][0]} {rank[0][1]:.4f} (gap {me - rank[0][1]:.2f})')
    print('TOP 10: ' + '; '.join(f'{i}. {t} {s:.4f}' for i, (t, s) in enumerate(rank[:10], 1)))

    print(f'\nCUT-OFFS (rank = ceil(p*n), n={n}; score of the team at that rank; gap = ours - cut-off):')
    for p in PS:
        k, s = cut(rank, p)
        kf = max(1, math.floor(p * n))
        others = board(rows, drop=ME)
        entry = others[k - 1][1]          # we are in the top k iff our score < k-th best of the OTHER teams
        print(f'  top {p*100:4.0f}%: rank <= {k:3d}  score at that rank {s:.4f}  | entry threshold (k-th best other team) '
              f'{entry:.4f}, ours - threshold {me - entry:+.2f}   [floor rule: rank {kf}, {rank[kf-1][1]:.4f}]')

    print('\nBANDS (teams, excluding us):')
    for lo, hi in BANDS:
        c = sum(1 for t, s in rank if lo <= s < hi and t != ME)
        print(f'  [{lo}, {hi}): {c}')

    movers(rows, rank, iso(now - timedelta(hours=24)), 'last 24 h')
    movers(rows, rank, '2026-09-24T00:00:00', 'since 2026-09-24T00:00Z')

    print('\nTOP-10% CUT-OFF HISTORY (ceil rule with n at that time; also fixed rank 20 = ceil(0.1*192)):')
    pts = []
    for h in range(96, -1, -12):
        t = iso(now - timedelta(hours=h))
        r = board(rows, until=t)
        k, s = cut(r, 0.10)
        p_me = {tt: i for i, (tt, _) in enumerate(r, 1)}.get(ME)
        o = board(rows, until=t, drop=ME)
        ent = o[math.ceil(0.10 * len(r)) - 1][1]
        pts.append((h, s, r[19][1], ent))
        print(f'  {t}Z  n={len(r):3d}  rank {k:2d}: {s:.4f}  rank20: {r[19][1]:.4f}  entry(others, excl. us): {ent:.4f}'
              f'  leader {r[0][1]:.2f}  us rank {p_me}')
    for span in (72, 48, 24):
        a = [x for x in pts if x[0] == span][0]
        b = pts[-1]
        print(f'  rate over last {span} h: ceil-rule {(b[1] - a[1]) / (span / 24):+.2f}/day, '
              f'rank20 {(b[2] - a[2]) / (span / 24):+.2f}/day, entry threshold excl. us {(b[3] - a[3]) / (span / 24):+.2f}/day')


if __name__ == '__main__':
    main()
