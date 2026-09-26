"""Zone P: the seq bulk in place of the W2 network bulk in the real-taxi branch (v19 structure).

Zone P (route D, off_sched > 7200 s; 4,944 rows in 2025, 1,720 in the 2026 ranking set) serves
    p * off_sched + (1 - p) * pb,        p = the v13/v14 TabICL probability,
    pb = 0.5 * W2 mix bulk + 0.5 * W2 nn bulk          (v15-era, no ADS-B -- the oldest part of the pipeline)
Outside P, v19 replaced the nn member with seq (nn weight 0 on both routes). The candidate makes the same single swap
inside P:
    pb_new = 0.5 * W2 mix bulk + 0.5 * seq bulk        (seq .bulk.npy, mean of its seeds)
Baseline and candidate are both served in the v19 structure (8-member blend outside P, route-D corrector x0.5), so
only P rows differ; RMSEs are on ALL flights.

PRE-REGISTERED RULE (written 2026-09-26 before this comparison was run): adopt if
  R1  held-out (8 months) AND scored (Jan + Jul) improve (point estimates);
  R2  the held-out airport-day block bootstrap 95% interval (1,000 draws) lies below 0;
  R3  January AND July each improve;
  R4  every seq seed alone (pb with that seed's bulk) also improves held-out;
  R5  with the 100 most-improved rows removed, held-out and scored still improve.
One arm only; no alternative mixes are tried after seeing the result.

    python experiments/seq/seq_p_branch.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np, pandas as pd
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
    S = json.load(open(CACHE / 'seq' / 'seq_eval_seq_adsb2_corr0.5.json'))
    W8, seeds = S['weights'], S['seq_seeds']
    A2, SQ, W2 = CACHE / 'cv' / 'adsb2', CACHE / 'cv' / 'seq', CACHE / 'cv' / 'w2'
    M = {f: oof(A2, f, [0, 1, 2], month) for f in FAMS}
    M['seq'] = oof(SQ, 'seq', seeds, month)
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    if not np.array_equal(np.sort(Pidx), np.flatnonzero(P)):
        sys.exit('FATAL: w4 Pidx is not zone P')
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'off_sched', 'MVT_TIME_UTC_mvt'])
    apt = dd.APT.astype(str).to_numpy(); so = dd.off_sched.to_numpy()[Pidx]
    block = np.char.add(apt, dd.MVT_TIME_UTC_mvt.dt.strftime('%m%d').to_numpy().astype(str))
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    w2mix = oof(W2, 'mix', [0, 1, 2], month, '.bulk.npy')[Pidx]
    w2nn = oof(W2, 'nn', [0, 1, 2], month, '.bulk.npy')[Pidx]
    seqb = {s: oof(SQ, 'seq', [s], month, '.bulk.npy')[Pidx] for s in seeds}
    seqb_all = np.mean([seqb[s] for s in seeds], axis=0)

    A, DnP = ~D & ~P, D & ~P

    def structure(pbP):
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            bl = sum(W8[key][m] * M[m] for m in W8[key])
            f[g] = np.clip(bl[g] + (0.5 * (v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        f[Pidx] = v11[Pidx] + (mixP(p14, pbP) - mixP(p_cur, pb_old))
        return f

    fo = structure(0.5 * w2mix + 0.5 * w2nn)
    fn = structure(0.5 * w2mix + 0.5 * seqb_all)
    held, Sc = ~np.isin(month, SCREEN), np.isin(month, [1, 7])
    r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    print(f'baseline (v19 structure): held-out {r(fo, held):.3f}  scored {r(fo, Sc):.3f}   | P rows {P.sum():,}, '
          f'seq seeds {seeds}')
    yP = y[Pidx]; kb = yP <= 7200
    print(f'bulk regressors on P bulk rows ({kb.sum():,}): W2 mix {np.sqrt(np.mean((yP[kb] - w2mix[kb]) ** 2)):.1f}  '
          f'W2 nn {np.sqrt(np.mean((yP[kb] - w2nn[kb]) ** 2)):.1f}  seq {np.sqrt(np.mean((yP[kb] - seqb_all[kb]) ** 2)):.1f}')
    res = {}
    for sc, k in (('held-out', held), ('scored', Sc), ('Jan', month == 1), ('Jul', month == 7)):
        res[sc] = block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=1000)
    print('candidate - baseline, all flights, block bootstrap: ' + '  '.join(
        f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items()))
    per_seed = {}
    for s in seeds:
        fs = structure(0.5 * w2mix + 0.5 * seqb[s])
        per_seed[s] = r(fs, held) - r(fo, held)
    print('per seed held-out: ' + '  '.join(f'{s} {v:+.3f}' for s, v in per_seed.items()))
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(len(y), bool); keep[np.argsort(gain)[::-1][:100]] = False
    d100 = {sc: r(fn, k & keep) - r(fo, k & keep) for sc, k in (('held-out', held), ('scored', Sc))}
    print(f'without the 100 most-improved rows: held-out {d100["held-out"]:+.3f}  scored {d100["scored"]:+.3f}')
    print('per airport inside P (RMSE on P rows, 12 months): ' + '  '.join(
        f'{x} {r(fn, P & (apt == x)) - r(fo, P & (apt == x)):+.1f} (n {int((P & (apt == x)).sum())})'
        for x in sorted(set(apt[P]))))
    R = dict(R1=res['held-out']['point'] < 0 and res['scored']['point'] < 0, R2=res['held-out']['hi'] < 0,
             R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=all(v < 0 for v in per_seed.values()),
             R5=d100['held-out'] < 0 and d100['scored'] < 0)
    verdict = 'PASS' if all(R.values()) else 'FAIL'
    print('RULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R.items()) + f'  => {verdict}')
    json.dump({'rule': R, 'verdict': verdict, 'result': res, 'per_seed': per_seed, 'drop100': d100},
              open(CACHE / 'seq' / 'seq_p_branch.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
