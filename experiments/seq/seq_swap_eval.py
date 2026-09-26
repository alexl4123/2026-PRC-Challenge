"""Swap the shipped seq member for a variant (e.g. 15 epochs) everywhere v20 uses it, and judge it on held-out months.

v20 uses seq twice: as the 8th blend member outside P (NNLS weight ~0.4) and as half of zone P's real-taxi branch
(pb = 0.5 W2 mix + 0.5 seq bulk). A variant ships only as a whole, so both uses are swapped:
    old  adsb2 members + seq (cache/cv/seq, seeds 0-2)          = the v20 structure (reproduces the shipped weights)
    new  adsb2 members + variant (cache/cv/<new>, seeds 0-2)     NNLS refitted on the screening months, as always
Route-D corrector x0.5 on route D outside P on both sides. EQUAL seed counts on both sides (3 vs 3): the settings
screen's bar compared 2 seeds with a 3-seed ensemble, which is biased (dev log 09-26 s35).

PRE-REGISTERED RULE (written 2026-09-26 23:5x CEST, BEFORE any full-campaign fold of the variant exists; the screen
only used the screening months 3/6/9/12, which are not judged here): adopt the variant if, on all flights,
  R1  held-out (8 months) AND scored (Jan + Jul) improve (point estimates);
  R2  the held-out airport x local-day block bootstrap 95% interval (1,000 draws) lies below 0;
  R3  January AND July each improve;
  R4  every seed alone (old seq seed s vs variant seed s, members fixed) improves held-out;
  R5  with the 100 most-improved rows removed, held-out and scored still improve.

    python experiments/seq/seq_swap_eval.py --new seqsw_ep15
"""
import argparse, json, os, sys
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True, help='campaign folder of the variant under cache/cv/')
    a = ap.parse_args()
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    A2, SQ, NEW, W2 = CACHE / 'cv' / 'adsb2', CACHE / 'cv' / 'seq', CACHE / 'cv' / a.new, CACHE / 'cv' / 'w2'
    miss = [(s, m) for s in range(3) for m in range(1, 13) if not (NEW / f'seq__seed{s:02d}__m{m:02d}.npy').exists()]
    if miss:
        sys.exit(f'FATAL: {a.new} incomplete, {len(miss)} of 36 folds missing, e.g. {miss[:4]}')
    print(f'variant settings: {json.load(open(NEW / "_config.json"))}')
    M = {f: oof(A2, f, [0, 1, 2], month) for f in FAMS}
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    w2mix = oof(W2, 'mix', [0, 1, 2], month, '.bulk.npy')[Pidx]
    rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
    if len(rw) != len(y) or not np.allclose(rw.y.to_numpy(float), y, equal_nan=True):
        sys.exit('FATAL: rows_training.parquet is not row-aligned with base.npz')
    block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
    apt = rw.apt.astype(str).to_numpy()
    A, DnP = ~D & ~P, D & ~P
    fit = np.isin(month, SCREEN) & ~P
    fams8 = FAMS + ['seq']

    def serve(folder, seeds):
        Mx = dict(M, seq=oof(folder, 'seq', seeds, month))
        W = {}
        for key, g in (('routes A/B/C', ~D), ('route D', D)):
            w, _ = nnls(np.column_stack([Mx[f] for f in fams8])[fit & g], y[fit & g])
            W[key] = dict(zip(fams8, np.round(w, 4).tolist()))
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            bl = sum(W[key][m] * Mx[m] for m in fams8)
            f[g] = np.clip(bl[g] + (0.5 * (v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        pbP = 0.5 * w2mix + 0.5 * oof(folder, 'seq', seeds, month, '.bulk.npy')[Pidx]
        f[Pidx] = v11[Pidx] + (mixP(p14, pbP) - mixP(p_cur, pb_old))
        return f, W, Mx['seq']

    fo, Wo, so_ = serve(SQ, [0, 1, 2])
    fn, Wn, sn_ = serve(NEW, [0, 1, 2])
    shipped = json.load(open(CACHE / 'seq' / 'seq_eval_seq_adsb2_corr0.5.json'))['weights']
    dw = max(abs(Wo[k][f] - shipped[k].get(f, 0.0)) for k in Wo for f in fams8)
    print(f'check: old-side weights vs shipped v19/v20 weights max |diff| {dw:.4f}' + ('  <- MISMATCH' if dw > 1e-3 else ''))
    print('new weights: ' + ' | '.join(f'{k}: ' + ' '.join(f'{f} {w:.3f}' for f, w in Wn[k].items() if w) for k in Wn))
    held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
    r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    oP = held & ~P
    print(f'seq member alone, held-out outside P: shipped {r(so_, oP):.3f}  variant {r(sn_, oP):.3f}')
    print(f'old (v20 structure): held-out {r(fo, held):.3f}  scored {r(fo, S):.3f}')
    res = {sc: block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=1000)
           for sc, k in (('held-out', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
    per_m = [r(fn, month == m) - r(fo, month == m) for m in range(1, 13)]
    print('variant vs shipped seq, all flights, block bootstrap: ' + '  '.join(
        f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items())
        + f'  months better {sum(v < 0 for v in per_m)}/12')
    per_seed = {s: r(serve(NEW, [s])[0], held) - r(serve(SQ, [s])[0], held) for s in range(3)}
    print('per seed held-out: ' + '  '.join(f'{s} {v:+.3f}' for s, v in per_seed.items()))
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(len(y), bool); keep[np.argsort(gain)[::-1][:100]] = False
    d100 = {sc: r(fn, k & keep) - r(fo, k & keep) for sc, k in (('held-out', held), ('scored', S))}
    print(f'without the 100 most-improved rows: held-out {d100["held-out"]:+.3f}  scored {d100["scored"]:+.3f}')
    print('per airport (all flights, 12 months): ' + '  '.join(
        f'{x} {r(fn, apt == x) - r(fo, apt == x):+.2f}' for x in sorted(set(apt))))
    R = dict(R1=res['held-out']['point'] < 0 and res['scored']['point'] < 0, R2=res['held-out']['hi'] < 0,
             R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=all(v < 0 for v in per_seed.values()),
             R5=d100['held-out'] < 0 and d100['scored'] < 0)
    verdict = 'PASS' if all(R.values()) else 'FAIL'
    print('RULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R.items()) + f'  => {verdict}')
    json.dump({'new': a.new, 'weights_new': Wn, 'weights_old': Wo, 'rule': R, 'verdict': verdict, 'result': res,
               'per_seed': per_seed, 'drop100': d100}, open(CACHE / 'seq' / f'swap_{a.new}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
