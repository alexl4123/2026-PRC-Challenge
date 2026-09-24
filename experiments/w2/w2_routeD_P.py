"""W2 extensions: route D outside P, and the real-taxi branch inside P -- on the all-flights scale.

v14 used W2 only on routes A/B/C outside P (pre-registered). Two rows groups still serve the old
members:
  route D outside P   v11~ there = v9~ + the route-D (Amsterdam) corrector + stack arm c
  P (the tail zone)   p * off_sched + (1 - p) * pb, pb = the old v9 bulk (0.5 LGB + 0.5 network)

BASELINE = the v14 structure on 2025 OOF: v11~ everywhere, W2 blend on A/B/C outside P, and inside
P the v13/v14 probabilities (TabICL per-airport at Rome, pooled elsewhere; W4 OOF) with the old pb.
CANDIDATES (each changes one row group):
  D1  route D outside P: W2 blend (NNLS route-D weights from w2_eval) + the corrector's delta
      (v10~ - v9~), i.e. the corrector is kept, the old blend and the stack are replaced
  D0  as D1 without the corrector (information only: 2025 CV cannot see the corrector's 2026 value)
  P1  inside P: pb := 0.5 * W2 LightGBM bulk + 0.5 * W2 network bulk (the .bulk.npy fold files)
PRE-REGISTERED RULE (per candidate D1, P1): adopt if better on BOTH the 12 months and the scored
months with the 12-month paired-bootstrap interval below 0. D0 is reported, not selectable.

    python experiments/w2/w2_routeD_P.py --seeds 0
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from prc.config import CACHE
from prc.validation import paired_bootstrap

NEW = CACHE / 'cv' / 'w2'


def bulk_oof(fam, seeds, month):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(NEW / f'{fam}__seed{s:02d}__m{m:02d}.bulk.npy') / len(seeds)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    a = ap.parse_args()
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    z = np.load(CACHE / 'w2' / 'w2_eval.npz')
    if list(z['seeds']) != a.seeds:
        sys.exit(f"FATAL: w2_eval.npz was built with seeds {list(z['seeds'])}; rerun w2_eval.py --seeds {' '.join(map(str, a.seeds))}")
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    A = ~D & ~P
    DnP = D & ~P

    # inside P: v13/v14 probabilities from the W4 OOF (Rome own, elsewhere pool), old pb
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; assert np.array_equal(np.flatnonzero(P), Pidx), 'P definitions differ'
    rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    import pandas as pd
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)

    base = v11.copy()
    base[A] = np.clip(z['new'][A], 0, None)
    base[Pidx] = v11[Pidx] + (mixP(p14, pb_old) - mixP(p_cur, pb_old))       # v13/v14 p inside P

    cands = {}
    c = base.copy(); c[DnP] = np.clip(z['new'][DnP] + (v10[DnP] - v9[DnP]), 0, None); cands['D1 W2 on route D + corrector'] = c
    c = base.copy(); c[DnP] = np.clip(z['new'][DnP], 0, None); cands['D0 W2 on route D, no corrector (info)'] = c
    pb_new = 0.5 * bulk_oof('mix', a.seeds, month)[Pidx] + 0.5 * bulk_oof('nn', a.seeds, month)[Pidx]
    c = base.copy(); c[Pidx] = base[Pidx] + (mixP(p14, pb_new) - mixP(p14, pb_old)); cands['P1 W2 bulk inside P'] = c

    S = np.isin(month, [1, 7])
    r = lambda f, k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
    print(f'seeds {a.seeds} | baseline (v14 structure, 2025 OOF): 12m {r(base, slice(None)):.2f}  scored {r(base, S):.2f}')
    print(f'rows: route D outside P {DnP.sum():,}, P {P.sum():,}; mean |pb_new - pb_old| in P {np.mean(np.abs(pb_new - pb_old)):.0f} s')
    for nm, f in cands.items():
        out = []
        for scope, k in (('12m', np.ones(len(y), bool)), ('scored', S)):
            bb = paired_bootstrap(y[k], f[k], base[k])
            out.append(f'{scope} {bb["diff"]:+.2f} [{bb["lo"]:+.2f},{bb["hi"]:+.2f}]')
        per = [r(f, month == m) - r(base, month == m) for m in range(1, 13)]
        print(f'  {nm:40s} ' + '   '.join(out) + f'   months better {sum(v < 0 for v in per)}/12, worst {max(per):+.2f}')


if __name__ == '__main__':
    main()
