"""Arms U and U+T -- disagreement-tercile blend weights on routes A/B/C (frozen evaluation before building v26/v27).

FOUND post hoc in the 2026-10-01 next-steps workflow (reports/2026-10-01_next_steps_plan.md; ~18 variants screened on
the v24 OOF that day; screens in experiments/plan_1001/). This script FREEZES both designs and their gates before any
file is built (2026-10-02). Nothing here is tuned after it is run.

DESIGN (both arms; v24 = the reference):
  s_i   = standard deviation across the 27 seq-pool BULK predictions of row i (9 settings x seeds 0-2; 2025: the fold
          .bulk.npy files, 2026: the finals cache/seq_<V>_final/seq__seed0{0,1,2}.npy).
  bins  = terciles of s with the FIXED cut points CUTS below (the 2025 OOF tercile cut points over the screening
          months outside P, i.e. 1/3 and 2/3 quantiles); never re-estimated on 2026.
  U     routes A/B/C outside P: per-bin NNLS weights over the 8 v24 members (7 adsb2 members at seeds 0-2 + the 27-fit seq
          pool mean), fitted on the screening months 3/6/9/12 outside P on routes A/B/C (= rows with an NM record);
          route D, zone P, the route-D corrector x0.5 and everything else exactly as v24.
  U+T   U + lambda_bin * (seq10 - seqnc10) on routes A/B/C, lambda_bin = least squares of the U residual on that
          difference over the same fit rows (seq10/seqnc10 = cache/cv/seq 10-epoch seq and its no-context twin,
          seeds 0-2; in production their finals cache/seq_final/seq*__seed0{0,1,2}.npy).
GATES (each arm separately, all flights, vs v24):
  R1 held-out (8 months) AND scored (Jan+Jul) better;  R2 held-out airport x local-day block bootstrap 95% CI < 0;
  R3 January AND July each better;  R4 every member seed alone (members AND seq pool at seed s only, both sides refitted)
  better held-out;  R5 held-out gain survives removing the 100 most-improved HELD-OUT rows;
  X1 leave-one-month-out fit (both sides LOMO) better in >= 10 of 12 months;  X2 Jan->Jul and Jul->Jan single-month
  cross-fits both < 0.  REPORTED, not gated: R5 on scored rows (U: +0.008 in the screens), per airport.
  Label-free 2026 gates (exp81 --arm-u): each bin's share of 2026 A/B/C rows within 10 points of 33%; the v26-v24 shift per
  airport x month within [-6, +6] s (the 2025 analogue of past swaps stayed within about +-5).
USER DECISION (2026-10-02): both arms may be uploaded as separate versions; pre-committed choice (dev log 10-02): if
  only the last/nominated file counts, U+T replaces U only if its live score is better by > 0.3 s (the noise level of a
  single live delta), otherwise U. Under best-of scoring nothing is chosen.

    python experiments/seq/arm_u_eval.py            # writes cache/seq/arm_u.json (weights, lambdas, cut points, gates)
"""
import json, os, sys
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
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o


def main():
    V = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
    y, month, P, D, f24, corr = V['y'], V['month'], V['P'], V['D'], V['f'], V['corr']
    A = ~D & ~P
    rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
    assert np.allclose(rw.y.to_numpy(float), y, equal_nan=True)
    apt = rw.apt.astype(str).to_numpy()
    block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
    s1 = np.zeros(len(y)); s2 = np.zeros(len(y))
    for v in POOL:
        for s in (0, 1, 2):
            x = oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], month, '.bulk.npy'); s1 += x; s2 += x * x
    sd = np.sqrt(np.maximum(s2 / 27 - (s1 / 27) ** 2, 0)); del s1, s2
    fitS = np.isin(month, SCREEN)
    CUTS = np.quantile(sd[fitS & ~P], [1 / 3, 2 / 3])
    b = np.digitize(sd, CUTS)
    seqp = {s: np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], month) for v in POOL], axis=0) for s in (0, 1, 2)}
    memb = {s: {f: oof(CACHE / 'cv' / 'adsb2', f, [s], month) for f in FAMS} for s in (0, 1, 2)}
    X3 = np.column_stack([np.mean([memb[s][f] for s in (0, 1, 2)], axis=0) for f in FAMS] + [np.mean(list(seqp.values()), axis=0)])
    dl = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2], month) - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2], month)
    held, S = ~fitS, np.isin(month, [1, 7])
    r = lambda g, k: float(np.sqrt(np.mean((y[k] - g[k]) ** 2)))
    names = FAMS + ['seq']

    def design(X, fitmask, kind):
        """f24 everywhere except A/B/C outside P, which gets the shared (kind='shared') or tercile weights."""
        fm = fitmask & ~P & ~D
        out = f24.copy(); W, L = {}, {}
        if kind == 'shared':
            w, _ = nnls(X[fm], y[fm]); out[A] = np.clip(X[A] @ w, 0, None); return out, {'all': w.tolist()}, {}
        for c in (0, 1, 2):
            k = fm & (b == c); w, _ = nnls(X[k], y[k]); base = X @ w
            lam = float(np.dot(dl[k], (y - base)[k]) / np.dot(dl[k], dl[k])) if kind == 'UT' else 0.0
            sel = A & (b == c); out[sel] = np.clip(base[sel] + lam * dl[sel], 0, None)
            W[c] = dict(zip(names, np.round(w, 4).tolist())); L[c] = lam
        return out, W, L

    sh, _, _ = design(X3, fitS, 'shared')
    print(f'check: shared refit vs v24, held-out {r(sh, held) - r(f24, held):+.4f} (must be ~0) | cut points {CUTS.round(2).tolist()} | '
          f'2025 A/B/C bin shares ' + ' '.join(f'{np.mean(b[A] == c):.3f}' for c in (0, 1, 2)))
    out = {'cuts': CUTS.tolist(), 'members': names}
    for arm in ('U', 'UT'):
        fn, W, L = design(X3, fitS, arm)
        res = {sc: block_bootstrap(y[k], fn[k], f24[k], block[k], n_boot=1000)
               for sc, k in (('held', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
        gain = (y - f24) ** 2 - (y - fn) ** 2
        hi = np.where(held)[0]; keep = np.ones(len(y), bool); keep[hi[np.argsort(gain[held])[::-1][:100]]] = False
        si = np.where(S)[0]; keep_s = np.ones(len(y), bool); keep_s[si[np.argsort(gain[S])[::-1][:100]]] = False
        r5h, r5s = r(fn, held & keep) - r(f24, held & keep), r(fn, S & keep_s) - r(f24, S & keep_s)
        per_seed = {}
        for s in (0, 1, 2):
            Xs = np.column_stack([memb[s][f] for f in FAMS] + [seqp[s]])
            o_s, _, _ = design(Xs, fitS, 'shared'); n_s, _, _ = design(Xs, fitS, arm)
            per_seed[s] = r(n_s, held) - r(o_s, held)
        lomo_o, lomo_n = f24.copy(), f24.copy()
        for m in range(1, 13):
            so_, _, _ = design(X3, month != m, 'shared'); sn_, _, _ = design(X3, month != m, arm)
            lomo_o[month == m], lomo_n[month == m] = so_[month == m], sn_[month == m]
        lomo_better = sum(r(lomo_n, month == m) < r(lomo_o, month == m) for m in range(1, 13))
        xf = {}
        for fm_, jm in ((1, 7), (7, 1)):
            so_, _, _ = design(X3, month == fm_, 'shared'); sn_, _, _ = design(X3, month == fm_, arm)
            xf[f'{fm_}->{jm}'] = r(sn_, month == jm) - r(so_, month == jm)
        G = dict(R1=res['held']['point'] < 0 and res['scored']['point'] < 0, R2=res['held']['hi'] < 0,
                 R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=all(v < 0 for v in per_seed.values()),
                 R5=r5h < 0, X1=lomo_better >= 10, X2=all(v < 0 for v in xf.values()))
        verdict = 'PASS' if all(G.values()) else 'FAIL'
        print(f'\n== {arm}: ' + '  '.join(f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items()))
        print(f'   per seed held {" ".join(f"{v:+.3f}" for v in per_seed.values())} | w/o top-100 held rows {r5h:+.3f} | '
              f'(reported) w/o top-100 scored rows {r5s:+.3f} | LOMO months better {lomo_better}/12 | cross-fits '
              + ' '.join(f'{k} {v:+.3f}' for k, v in xf.items()))
        print('   per airport held: ' + ' '.join(f'{a} {r(fn, held & (apt == a)) - r(f24, held & (apt == a)):+.2f}' for a in sorted(set(apt))))
        print('   weights per bin: ' + ' | '.join(f'{c}: ' + ' '.join(f'{k} {v:.3f}' for k, v in W[c].items() if v) + (f' lam {L[c]:+.3f}' if arm == 'UT' else '') for c in (0, 1, 2)))
        print(f'   GATES: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in G.items()) + f'  => {verdict}')
        out[arm] = dict(weights={str(c): W[c] for c in W}, lambdas={str(c): L[c] for c in L}, gates=G, verdict=verdict,
                        result=res, per_seed=per_seed, r5_held=r5h, r5_scored=r5s, lomo_better=lomo_better, crossfit=xf)
    json.dump(out, open(CACHE / 'seq' / 'arm_u.json', 'w'), indent=1)
    print('\nsaved cache/seq/arm_u.json')


if __name__ == '__main__':
    main()
