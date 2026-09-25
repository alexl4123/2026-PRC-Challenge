"""Idea 7 (Piyush, Discord l.504): apply the validation discipline to today's shipped decisions.

  require January AND July to improve separately; reject gains carried by < 100 rows;
  permutation control for classifier-based changes.

Decisions re-scored (the arrays are rebuilt exactly as experiments/w2/w2_eval.py, w2_routeD_P.py and
w3_eval.py build them, 3 seeds; nothing is refitted, the saved weights are used):
  W2   v14: W2 blend replaces blend + stack on routes A/B/C outside P  (b = v11~, a = v11~ with A := W2)
  D1   v15: route D outside P := W2 blend + corrector delta          (on the v14 structure)
  P1   v15: inside P, pb := W2 bulk                                   (on the v14 structure)
  W3   v16: blend7 (w3 added, saved weights) vs blend6, v15 structure
  PERM control for the one classifier change in the structure (v13/v14 TabICL p inside P): the p
       vector permuted within airport among P rows (20 draws) vs the real p, both against p_cur.

    python experiments/discord_eval/discipline.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'w2'))
import numpy as np, pandas as pd
from common import compare, fmt, rmse, SCRATCH
from prc.config import CACHE
from prc.io import peak_rss_gb
from w2_eval import oof, NEW, FAMS
from w2_routeD_P import bulk_oof


def main():
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    B = dict(y=y, month=month)
    z = np.load(CACHE / 'w2' / 'w2_eval.npz'); assert list(z['seeds']) == [0, 1, 2]
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    A, DnP = ~D & ~P, D & ~P
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; assert np.array_equal(np.flatnonzero(P), Pidx)
    rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    pb_new = 0.5 * bulk_oof('mix', [0, 1, 2], month)[Pidx] + 0.5 * bulk_oof('nn', [0, 1, 2], month)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)

    # W2 (v14 decision)
    shipped_A = v11.copy(); shipped_A[A] = np.clip(z['shipped'][A], 0, None)
    v14 = v11.copy(); v14[A] = np.clip(z['new'][A], 0, None)
    print(f'check: v11~ vs clip(shipped) on A rows, max |diff| {np.max(np.abs(v11[A] - shipped_A[A])):.2f}')
    # v14 structure (w2_routeD_P baseline) and the two v15 candidates
    base14 = v14.copy(); base14[Pidx] = v11[Pidx] + (mixP(p14, pb_old) - mixP(p_cur, pb_old))
    d1 = base14.copy(); d1[DnP] = np.clip(z['new'][DnP] + (v10[DnP] - v9[DnP]), 0, None)
    p1 = base14.copy(); p1[Pidx] = base14[Pidx] + (mixP(p14, pb_new) - mixP(p14, pb_old))
    # v15 structure (blend6) and v16 (blend7), as w3_eval.py
    W6 = json.load(open(CACHE / 'w2' / 'w2_weights.json'))['weights']
    W7 = json.load(open(CACHE / 'w2' / 'w3_weights.json'))['weights']
    mem = {f: oof(NEW, f, [0, 1, 2], month) for f in FAMS}
    mem['w3'] = oof(NEW, 'w3', [0, 1, 2], month)

    def structure(weights):
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            blend = sum(weights[key].get(m, 0.0) * mem[m] for m in mem)
            f[g] = np.clip(blend[g] + ((v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        f[Pidx] = v11[Pidx] + (mixP(p14, pb_new) - mixP(p_cur, pb_old))
        return f
    f6, f7 = structure(W6), structure(W7)
    v15s = np.load(CACHE / 'w2' / 'v15s_oof.npy').astype(float)
    print(f'check: rebuilt v15 structure vs cache/w2/v15s_oof.npy max |diff| {np.max(np.abs(f6 - v15s)):.3f}')

    # classifier change inside P (v13/v14 TabICL p vs p_cur), real vs within-airport permutation
    tfm_real = base14.copy()                                # p14 with pb_old
    tfm_cur = base14.copy(); tfm_cur[Pidx] = v11[Pidx]      # p_cur with pb_old (= v11~ inside P)

    out = {}
    for name, a_, b_ in (('W2 (v14) A/B/C', v14, shipped_A), ('D1 (v15) route D', d1, base14),
                         ('P1 (v15) W2 bulk in P', p1, base14), ('D1+P1 (v15 vs v14)', f6, base14),
                         ('W3 (v16) blend7-blend6', f7, f6), ('TabICL p in P (v13/v14)', tfm_real, tfm_cur),
                         ('ALL today v16 vs v11~', f7, v11)):
        r = compare(B, a_, b_); out[name] = r
        nchg = int(np.sum(np.abs(a_ - b_) > 1e-6))
        print(f'{name:26s} rows changed {nchg:>9,} | {fmt(r)} | top-100 share of net gain 12m {r["top100_share_12m"]:.2f} scored {r["top100_share_scored"]:.2f}', flush=True)

    # permutation control for the classifier change
    rng = np.random.default_rng(0)
    apt = w4['apt']
    diffs = {'12m': [], 'jan': [], 'jul': []}
    for i in range(20):
        pp = p14.copy()
        for ap_ in np.unique(apt):
            k = np.flatnonzero(apt == ap_); pp[k] = pp[rng.permutation(k)]
        f = base14.copy(); f[Pidx] = v11[Pidx] + (mixP(pp, pb_old) - mixP(p_cur, pb_old))
        for s, k in (('12m', np.ones(len(y), bool)), ('jan', month == 1), ('jul', month == 7)):
            diffs[s].append(rmse(y[k], f[k]) - rmse(y[k], tfm_cur[k]))
    real = {s: rmse(y[k], tfm_real[k]) - rmse(y[k], tfm_cur[k])
            for s, k in (('12m', np.ones(len(y), bool)), ('jan', month == 1), ('jul', month == 7))}
    print('PERMUTATION CONTROL (p14 shuffled within airport among P rows, 20 draws; vs p_cur):')
    for s in diffs:
        print(f'  {s}: real {real[s]:+.3f} | permuted mean {np.mean(diffs[s]):+.3f} '
              f'[{np.percentile(diffs[s], 2.5):+.3f},{np.percentile(diffs[s], 97.5):+.3f}] | '
              f'share of draws at least as good as real {np.mean(np.array(diffs[s]) <= real[s]):.2f}')
    out['perm'] = {'real': real, 'perm_mean': {s: float(np.mean(v)) for s, v in diffs.items()}}
    json.dump(out, open(SCRATCH / 'discipline.json', 'w'), indent=1, default=float)
    print(f'peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
