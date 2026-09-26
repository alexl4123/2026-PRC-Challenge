"""A3 evaluation: the adsb.lol columns INSIDE all seven members vs the same members without them.

Members, 12-fold leave-one-month-out mixtures, seed-matched:
  old  cache/cv/w2/<fam>__seedNN__mMM.npy     mix nn perapt lintree routed xgboost w3   (= what v16 ships)
  new  cache/cv/adsb/<fam>__seedNN__mMM.npy   the same families with the adsb.lol columns (w2_task --adsb)
Each side gets its own per-route NNLS blend fitted on the screening months (3, 6, 9, 12) outside P, exactly as
the shipped weights were fitted; with old members this reproduces cache/w2/w3_weights.json. Both are served in the
v16 structure (route-D corrector kept on route D outside P; inside P the v13/v14 TabICL p with the W2 bulk), so
the comparison is on ALL flights. A second arm also swaps the real-taxi branch inside P for the ADS-B bulk
(0.5 mix + 0.5 nn .bulk files) -- reported, decided separately.

PRE-REGISTERED RULE (written 2026-09-25 before any ADS-B fold existed): ship the new blend if, on all flights,
it beats the old blend on BOTH the 8 held-out months and the scored months with the held-out paired-bootstrap
interval below 0, AND January and July EACH improve on their own (point estimates) -- the per-month requirement
added after the participants' Discord (validation discipline). Also reported: per-airport deltas and the share of
the gain carried by the 100 most-improved rows (a gain living in < 100 rows is flagged).

C1 RULE (written 2026-09-26 before any C1 fold existed; campaign cache/cv/c1 = arm-2 columns + 8 de-icing share
columns in all members + nn NaN sentinel), judged against arm 2 with the AIRPORT-DAY BLOCK bootstrap:
    python experiments/adsb/a3_eval.py --old adsb2 --new c1 --block
adopt if held-out AND scored improve, the held-out block CI is entirely < 0, January AND July each improve, the
seeds are matched (>= 2 of 3), and the 100 most-improved rows carry <= 50% of the SSE reduction.

    python experiments/adsb/a3_eval.py              # seeds complete in both campaigns
    python experiments/adsb/a3_eval.py --seeds 0    # early read
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'w2'))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import paired_bootstrap, block_bootstrap
from w2_eval import oof, complete, SCREEN

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
OLD, NEW = CACHE / 'cv' / 'w2', CACHE / 'cv' / 'adsb'


def bulk(folder, fam, seeds, month):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}.bulk.npy') / len(seeds)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--old', default='w2', help='comparator campaign (arm 2 is judged against arm 1: --old adsb --new adsb2)')
    ap.add_argument('--new', default='adsb')
    ap.add_argument('--block', action='store_true',
                    help='airport x local-day block bootstrap (cache/drivers_0926/rows_training.parquet) instead of rows')
    a = ap.parse_args()
    global OLD, NEW
    OLD, NEW = CACHE / 'cv' / a.old, CACHE / 'cv' / a.new
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    seeds = a.seeds if a.seeds is not None else [s for s in range(3) if all(complete(NEW, f, s) and complete(OLD, f, s) for f in FAMS)]
    miss = [(f, s) for f in FAMS for s in seeds if not (complete(NEW, f, s) and complete(OLD, f, s))]
    if not seeds or miss:
        sys.exit(f'FATAL: no usable seed set; incomplete (family, seed): {miss[:8]}')
    print(f'seeds (matched): {seeds}')
    Mo = {f: oof(OLD, f, seeds, month) for f in FAMS}
    Mn = {f: oof(NEW, f, seeds, month) for f in FAMS}

    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'off_sched'])
    apt = dd.APT.astype(str).to_numpy(); so = dd.off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    pb_w2 = 0.5 * bulk(OLD, 'mix', seeds, month)[Pidx] + 0.5 * bulk(OLD, 'nn', seeds, month)[Pidx]
    pb_ad = 0.5 * bulk(NEW, 'mix', seeds, month)[Pidx] + 0.5 * bulk(NEW, 'nn', seeds, month)[Pidx]
    A, DnP = ~D & ~P, D & ~P
    fit = np.isin(month, SCREEN) & ~P

    def weights(M):
        W = {}
        for key, g in (('routes A/B/C', ~D), ('route D', D)):
            X = np.column_stack([M[f] for f in FAMS])
            w, _ = nnls(X[fit & g], y[fit & g])
            W[key] = dict(zip(FAMS, np.round(w, 4).tolist()))
        return W

    def structure(M, W, pbP):
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            bl = sum(W[key][m] * M[m] for m in FAMS)
            f[g] = np.clip(bl[g] + ((v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        f[Pidx] = v11[Pidx] + (mixP(p14, pbP) - mixP(p_cur, pb_old))
        return f

    Wo, Wn = weights(Mo), weights(Mn)
    for nm, W in (('old', Wo), ('new', Wn)):
        for key in W:
            print(f'  {nm} weights {key:12s}: ' + '  '.join(f'{m} {v:.3f}' for m, v in W[key].items()))
    fo = structure(Mo, Wo, pb_w2)
    fn = structure(Mn, Wn, pb_w2)
    fnP = structure(Mn, Wn, pb_ad)

    if a.block:
        rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
        if len(rw) != len(y) or not np.allclose(rw.y.to_numpy(float), y, equal_nan=True):
            sys.exit('FATAL: rows_training.parquet is not row-aligned with base.npz')
        blk = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
        boot = lambda k, f: block_bootstrap(y[k], f[k], fo[k], blk[k], n_boot=1000)
    else:
        boot = lambda k, f: paired_bootstrap(y[k], f[k], fo[k], n_boot=500)
    print(f'bootstrap: {"airport-day BLOCKS" if a.block else "rows"}')
    held, S = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
    r = lambda f, k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
    print(f'\nold (v16 structure, seeds {seeds}): held-out {r(fo, held):.2f}  scored {r(fo, S):.2f}')
    for nm, f in (('NEW blend (A/B/C + D)  [decisive]', fn), ('NEW blend + ADS-B bulk in P [info]', fnP)):
        out = []
        for scope, k in (('held-out', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7)):
            bb = boot(k, f)
            out.append(f'{scope} {bb["diff"]:+.2f} [{bb["lo"]:+.2f},{bb["hi"]:+.2f}]')
        per = [r(f, month == m) - r(fo, month == m) for m in range(1, 13)]
        print(f'  {nm}: ' + '  '.join(out) + f'  months better {sum(v < 0 for v in per)}/12')
    gain = (y - fo) ** 2 - (y - fn) ** 2
    if gain.sum() > 0:
        top = np.sort(gain)[::-1][:100].sum() / gain.sum()
        print(f'\nshare of the SSE reduction carried by the 100 most-improved rows: {top:.2f}' + ('  <- FLAG' if top > 0.5 else ''))
    else:                                       # no net reduction: a share is undefined (was printed as ~1e17)
        print('\nshare of the SSE reduction carried by the 100 most-improved rows: n/a (the new blend has no net SSE reduction)')
    print('per airport (all flights, 12 months): ' + '  '.join(
        f'{x} {r(fn, apt == x) - r(fo, apt == x):+.1f}' for x in sorted(set(apt))))
    json.dump({'members': FAMS, 'seeds': seeds, 'weights': Wn}, open(CACHE / 'w2' / f'{a.new}_weights.json', 'w'), indent=1)
    print(f'saved cache/w2/{a.new}_weights.json')


if __name__ == '__main__':
    main()
