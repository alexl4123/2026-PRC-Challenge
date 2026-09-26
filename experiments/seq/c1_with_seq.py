"""C1 in the setting that ships: the C1 members + seq against the adsb2 members + seq (the v20 structure).

The pre-registered C1 rule (experiments/adsb/a3_eval.py --old adsb2 --new c1 --block) compares 7 members with 7.
Since v19 the blend has an 8th member, seq, which already reads part of the neighbour / disruption signal the C1
de-icing share columns carry, so C1's value must also hold with seq in the blend before it can ship.

Both sides: per-route NNLS on the screening months (3, 6, 9, 12) outside P over the 8 members, route-D corrector
x0.5 on route D outside P, zone P = the v20 branch (TabICL p, pb = 0.5 W2 mix + 0.5 seq bulk) -- identical on both
sides, so only A/B/C and route D outside P differ. Old side with adsb2 members reproduces the shipped v19/v20 weights
(checked). Members seeds 0-2 on both sides; seq seeds 0-2 on both sides.

PRE-REGISTERED RULE (written 2026-09-26 BEFORE either C1 evaluation was run): ship C1 (as members of the 8-member
blend) only if, on all flights,
  R1  held-out (8 months) AND scored (Jan + Jul) improve (point estimates);
  R2  the held-out airport x local-day block bootstrap 95% interval (1,000 draws) lies below 0;
  R3  January AND July each improve;
  R4  every member seed alone (old adsb2 seed s vs C1 seed s, seq fixed) improves held-out;
  R5  with the 100 most-improved rows removed, held-out and scored still improve.
Shipping additionally requires the 7-vs-7 rule (a3_eval) to PASS.

    python experiments/seq/c1_with_seq.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import block_bootstrap

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
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
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    A2, C1, SQ, W2 = (CACHE / 'cv' / x for x in ('adsb2', 'c1', 'seq', 'w2'))
    miss = [(f, s) for f in FAMS for s in range(3) for m in range(1, 13)
            if not (C1 / f'{f}__seed{s:02d}__m{m:02d}.npy').exists()]
    if miss:
        sys.exit(f'FATAL: C1 incomplete, {len(miss)} folds missing, e.g. {miss[:4]}')
    seq = oof(SQ, 'seq', [0, 1, 2], month)
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    pbP = 0.5 * oof(W2, 'mix', [0, 1, 2], month, '.bulk.npy')[Pidx] + 0.5 * oof(SQ, 'seq', [0, 1, 2], month, '.bulk.npy')[Pidx]
    rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
    if len(rw) != len(y) or not np.allclose(rw.y.to_numpy(float), y, equal_nan=True):
        sys.exit('FATAL: rows_training.parquet is not row-aligned with base.npz')
    block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
    apt = rw.apt.astype(str).to_numpy()
    A, DnP = ~D & ~P, D & ~P
    fit = np.isin(month, SCREEN) & ~P
    fams8 = FAMS + ['seq']

    def weights(M):
        W = {}
        for key, g in (('routes A/B/C', ~D), ('route D', D)):
            w, _ = nnls(np.column_stack([M[f] for f in fams8])[fit & g], y[fit & g])
            W[key] = dict(zip(fams8, np.round(w, 4).tolist()))
        return W

    def structure(M, W):
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            bl = sum(W[key][m] * M[m] for m in fams8)
            f[g] = np.clip(bl[g] + (0.5 * (v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        f[Pidx] = v11[Pidx] + (mixP(p14, pbP) - mixP(p_cur, pb_old))
        return f

    def side(folder, seeds):
        M = {f: oof(folder, f, seeds, month) for f in FAMS}; M['seq'] = seq
        W = weights(M)
        return structure(M, W), W

    fo, Wo = side(A2, [0, 1, 2])
    fn, Wn = side(C1, [0, 1, 2])
    shipped = json.load(open(CACHE / 'seq' / 'seq_eval_seq_adsb2_corr0.5.json'))['weights']
    dw = max(abs(Wo[k][f] - shipped[k].get(f, 0.0)) for k in Wo for f in fams8)
    print(f'check: old-side weights vs shipped v19/v20 weights max |diff| {dw:.4f}' + ('  <- MISMATCH' if dw > 1e-3 else ''))
    for nm, W in (('adsb2 + seq', Wo), ('C1 + seq', Wn)):
        for key in W:
            print(f'  {nm:11s} {key:12s}: ' + '  '.join(f'{m} {v:.3f}' for m, v in W[key].items() if v))
    held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
    r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    print(f'\nold (v20 structure): held-out {r(fo, held):.3f}  scored {r(fo, S):.3f}')
    res = {sc: block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=1000)
           for sc, k in (('held-out', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
    per_m = [r(fn, month == m) - r(fo, month == m) for m in range(1, 13)]
    print('C1 + seq vs adsb2 + seq, all flights, block bootstrap: ' + '  '.join(
        f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items())
        + f'  months better {sum(v < 0 for v in per_m)}/12')
    per_seed = {}
    for s in range(3):
        f_o, _ = side(A2, [s]); f_n, _ = side(C1, [s])
        per_seed[s] = r(f_n, held) - r(f_o, held)
    print('per member seed held-out: ' + '  '.join(f'{s} {v:+.3f}' for s, v in per_seed.items()))
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(len(y), bool); keep[np.argsort(gain)[::-1][:100]] = False
    d100 = {sc: r(fn, k & keep) - r(fo, k & keep) for sc, k in (('held-out', held), ('scored', S))}
    share = gain[~keep].sum() / gain.sum() if gain.sum() > 0 else float('nan')
    print(f'top-100 share {share:.2f}; without them: held-out {d100["held-out"]:+.3f}  scored {d100["scored"]:+.3f}')
    print('per airport (all flights, 12 months): ' + '  '.join(
        f'{x} {r(fn, apt == x) - r(fo, apt == x):+.2f}' for x in sorted(set(apt))))
    R = dict(R1=res['held-out']['point'] < 0 and res['scored']['point'] < 0, R2=res['held-out']['hi'] < 0,
             R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=all(v < 0 for v in per_seed.values()),
             R5=d100['held-out'] < 0 and d100['scored'] < 0)
    verdict = 'PASS' if all(R.values()) else 'FAIL'
    print('RULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R.items()) + f'  => {verdict}')
    json.dump({'weights_new': Wn, 'weights_old': Wo, 'rule': R, 'verdict': verdict, 'result': res,
               'per_seed': per_seed, 'drop100': d100}, open(CACHE / 'seq' / 'c1_with_seq.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
