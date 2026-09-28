"""Arm O: the seven adsb2 members with seeds 0-5 instead of 0-2 (the seq pool fixed), in the shipped structure.

Measured 2026-09-28 (dev log 09-28 s15): every member gains from seed averaging (member alone, held-out, 1 seed -> 3 seeds:
seq -3.7, w3 -3.0, nn -2.2, lintree -1.9, xgboost -1.5, mix/perapt/routed -0.7..-0.9). Variance arithmetic says 3 -> 6
seeds buys about a quarter of that per member [E]; the blend effect is expected small (-0.05 .. -0.2 s [E]).

Both sides: per-route NNLS on the screening months (3, 6, 9, 12) outside P over the 8 members, route-D corrector x0.5,
zone P = v24's branch (TabICL p, pb = 0.5 W2 mix bulk + 0.5 pooled seq bulk) -- identical on both sides. The seq member is
the shipped pool (the 9 settings of arm M, seeds 0..S-1 with S = --seq-seeds: 3 = v24, 6 = v26 if arm N passed).
    old  adsb2 members seeds 0-2  (= the shipped version; its weights must reproduce --ref)
    new  adsb2 members seeds 0-5

PRE-REGISTERED RULE (written 2026-09-28 before any seed-3..5 member fold exists): adopt if, on all flights,
  R1 held-out (8 months) AND scored (Jan + Jul) improve;  R2 held-out airport x local-day block bootstrap 95% CI < 0;
  R3 January AND July each improve;  R4 (pool form) the members with seeds 3-5 ONLY are within +0.10 s held-out of
  seeds 0-2 (the new seeds are of the same quality);  R5 without the 100 most-improved rows, held-out and scored improve.

    python experiments/seq/member_seeds_eval.py --seq-seeds 3 --ref swap_seqsw_ep30__...   (v24 as the reference)
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import block_bootstrap

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
SCREEN = [3, 6, 9, 12]


def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            a = np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}')
            if len(a) != (month == m).sum():
                sys.exit(f'FATAL: {folder.name}/{fam} seed {s} month {m}: {len(a)} rows')
            o[month == m] += a / len(seeds)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-seeds', type=int, default=3, help='seeds per pooled seq setting on BOTH sides (3 = v24, 6 = v26)')
    ap.add_argument('--ref', required=True, help='cache/seq/<ref>.json whose weights_new the old side must reproduce')
    ap.add_argument('--members', default='adsb2', help='member campaign under cache/cv/ (dry runs only: another folder)')
    a = ap.parse_args()
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    A2, W2 = CACHE / 'cv' / a.members, CACHE / 'cv' / 'w2'
    miss = [(f, s, m) for f in FAMS for s in range(6) for m in range(1, 13) if not (A2 / f'{f}__seed{s:02d}__m{m:02d}.npy').exists()]
    if miss:
        sys.exit(f'FATAL: {len(miss)} adsb2 member folds missing for seeds 0-5, e.g. {miss[:4]}')
    sseeds = list(range(a.seq_seeds))
    seq = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', sseeds, month) for v in POOL], axis=0)
    seqb = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', sseeds, month, '.bulk.npy') for v in POOL], axis=0)
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    pbP = 0.5 * oof(W2, 'mix', [0, 1, 2], month, '.bulk.npy')[Pidx] + 0.5 * seqb[Pidx]
    rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
    if len(rw) != len(y) or not np.allclose(rw.y.to_numpy(float), y, equal_nan=True):
        sys.exit('FATAL: rows_training.parquet is not row-aligned with base.npz')
    block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
    apt = rw.apt.astype(str).to_numpy()
    Ar, DnP = ~D & ~P, D & ~P
    fit = np.isin(month, SCREEN) & ~P
    fams8 = FAMS + ['seq']

    def serve(mseeds):
        M = {f: oof(A2, f, mseeds, month) for f in FAMS}; M['seq'] = seq
        W = {}
        for key, g in (('routes A/B/C', ~D), ('route D', D)):
            w, _ = nnls(np.column_stack([M[f] for f in fams8])[fit & g], y[fit & g])
            W[key] = dict(zip(fams8, np.round(w, 4).tolist()))
        f = v11.copy()
        for g, key in ((Ar, 'routes A/B/C'), (DnP, 'route D')):
            bl = sum(W[key][m] * M[m] for m in fams8)
            f[g] = np.clip(bl[g] + (0.5 * (v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        f[Pidx] = v11[Pidx] + (mixP(p14, pbP) - mixP(p_cur, pb_old))
        return f, W

    fo, Wo = serve([0, 1, 2])
    fn, Wn = serve([0, 1, 2, 3, 4, 5])
    ref = json.load(open(CACHE / 'seq' / f'{a.ref}.json'))['weights_new']
    dw = max(abs(Wo[k][f] - ref[k].get(f, 0.0)) for k in Wo for f in fams8)
    print(f'check: old-side weights vs {a.ref} max |diff| {dw:.4f}' + ('  <- MISMATCH' if dw > 1e-3 else ''))
    print('new weights: ' + ' | '.join(f'{k}: ' + ' '.join(f'{f} {w:.3f}' for f, w in Wn[k].items() if w) for k in Wn))
    held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
    r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    print(f'old (shipped structure): held-out {r(fo, held):.3f}  scored {r(fo, S):.3f}')
    res = {sc: block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=1000)
           for sc, k in (('held-out', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
    per_m = [r(fn, month == m) - r(fo, month == m) for m in range(1, 13)]
    print('members seeds 0-5 vs 0-2, all flights, block bootstrap: ' + '  '.join(
        f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items())
        + f'  months better {sum(v < 0 for v in per_m)}/12')
    f345, _ = serve([3, 4, 5])
    r4 = r(f345, held) - r(fo, held)
    print(f'R4 (pool form): members seeds 3-5 only vs seeds 0-2, held-out {r4:+.3f} (must be <= +0.10)')
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(len(y), bool); keep[np.argsort(gain)[::-1][:100]] = False
    d100 = {sc: r(fn, k & keep) - r(fo, k & keep) for sc, k in (('held-out', held), ('scored', S))}
    print(f'without the 100 most-improved rows: held-out {d100["held-out"]:+.3f}  scored {d100["scored"]:+.3f}')
    print('per airport (all flights, 12 months): ' + '  '.join(f'{x} {r(fn, apt == x) - r(fo, apt == x):+.2f}' for x in sorted(set(apt))))
    R = dict(R1=res['held-out']['point'] < 0 and res['scored']['point'] < 0, R2=res['held-out']['hi'] < 0,
             R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=r4 <= 0.10,
             R5=d100['held-out'] < 0 and d100['scored'] < 0)
    verdict = 'PASS' if all(R.values()) else 'FAIL'
    print('RULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R.items()) + f'  => {verdict}')
    json.dump({'members_seeds': [0, 1, 2, 3, 4, 5], 'seq_seeds': sseeds, 'weights_new': Wn, 'weights_old': Wo, 'rule': R,
               'verdict': verdict, 'result': res, 'r4': r4, 'drop100': d100},
              open(CACHE / 'seq' / (f'members_s012345__seq{a.seq_seeds}.json' if a.members == 'adsb2' else '_dryrun_members.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
