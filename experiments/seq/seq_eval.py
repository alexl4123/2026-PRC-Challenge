"""Bet D evaluation: the 'seq' member (cluster/seq_task.py) as an 8th member of the v17 blend.

Members, 12-fold leave-one-month-out mixtures:
  old  cache/cv/adsb2/<fam>__seedNN__mMM.npy   mix nn perapt lintree routed xgboost w3  (= what v17 ships), seeds 0-2
  new  the same seven + cache/cv/seq/seq__seedNN__mMM.npy                               (seq seeds as available)
Each side gets its own per-route NNLS blend (no intercept) fitted on the screening months (3, 6, 9, 12) outside P,
exactly as experiments/adsb/a3_eval.py fitted the shipped weights; with the old members this reproduces
cache/w2/adsb2_weights.json and cache/w2/v17s_oof.npy (checked below). Both are served in the v17 structure
(experiments/w2/v17_anatomy.py): route-D corrector (v10 - v9) on route D outside P (x --corr-scale; 1.0 = v17,
0.5 = v18), inside P the v14 TabICL p with the W2 bulk -- so the comparison is on ALL flights.

PRE-REGISTERED RULE (written 2026-09-26 BEFORE any seq fold exists; the project rule of the brief): adopt seq as an
8th member only if, on ALL flights,
  R1  the new blend beats the old on BOTH the 8 held-out months and the scored months (Jan + Jul), point estimates;
  R2  the held-out airport-day BLOCK bootstrap 95% interval (1,000 draws, airport x UTC day) lies below 0;
  R3  January AND July each improve on their own (point estimates);
  R4  >= 2 seq seeds are complete; the decision uses their mean, AND each seed alone also improves held-out;
  R5  the gain is not carried by < 100 rows: with the 100 most-improved rows removed, held-out and scored still < 0.
Also reported: per-airport deltas, months better /12, seq's NNLS weight, seq alone, the residual correlation of seq
with each member, and the share of the gain carried by the top-100 rows.

INFO ONLY (not the decision; added after the subset test, where ctx and its no-context twin with two FREE weights
gave -4.83 on top of v17 against -1.87 for ctx alone): when the ablation twin seqnc (seq_task --no-context) is
complete for the same seeds, (a) seq vs seqnc alone on held-out bulk rows = the full-data context effect, and (b) the
new blend + lambda * (seq - seqnc), one lambda per route fitted by least squares on the screening months outside P,
judged with the same R1-R5. Both mixtures share the tail classifier, so seq - seqnc = (1 - p)(bulk_seq - bulk_seqnc).

    python experiments/seq/seq_eval.py                 # seq seeds complete
    python experiments/seq/seq_eval.py --corr-scale 0.5
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
SCREEN, SCORED = [3, 6, 9, 12], [1, 7]
ADSB2, SEQ = CACHE / 'cv' / 'adsb2', CACHE / 'cv' / 'seq'


def complete(folder, fam, s):
    return all((folder / f'{fam}__seed{s:02d}__m{m:02d}.npy').exists() for m in range(1, 13))


def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            a = np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}')
            if len(a) != (month == m).sum():
                sys.exit(f'FATAL: {folder.name}/{fam} seed {s} month {m}: {len(a)} rows, expected {(month == m).sum()}')
            o[month == m] += a / len(seeds)
    return o


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def block_boot(y, a, b, block, n_boot=1000, seed=0):
    """RMSE(a) - RMSE(b) and its 95% interval, resampling whole blocks (airport x day) with replacement."""
    da, db = (y - a) ** 2, (y - b) ** 2
    _, inv = np.unique(block, return_inverse=True)
    SA, SB, N = np.bincount(inv, da), np.bincount(inv, db), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed); nb = len(N)
    out = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        n = w @ N
        out[i] = np.sqrt(w @ SA / n) - np.sqrt(w @ SB / n)
    return rmse(y, a) - rmse(y, b), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help='seq seeds (default: all complete)')
    ap.add_argument('--corr-scale', type=float, default=1.0, help='route-D corrector scale (1.0 = v17, 0.5 = v18)')
    ap.add_argument('--campaign', default='seq', help='cache/cv/<campaign>, or an absolute folder (dry runs)')
    ap.add_argument('--members', default='adsb2', help='the seven members campaign (adsb2 = v17; c1 after C1 lands)')
    a = ap.parse_args()
    global SEQ, ADSB2
    SEQ = Path(a.campaign) if os.path.isabs(a.campaign) else CACHE / 'cv' / a.campaign
    ADSB2 = CACHE / 'cv' / a.members
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    mseeds = [0, 1, 2]
    miss = [(f, s) for f in FAMS for s in mseeds if not complete(ADSB2, f, s)]
    if miss:
        sys.exit(f'FATAL: {a.members} members incomplete: {miss[:8]}')
    seeds = a.seeds if a.seeds is not None else [s for s in range(10) if complete(SEQ, 'seq', s)]
    if not seeds or not all(complete(SEQ, 'seq', s) for s in seeds):
        sys.exit(f'FATAL: no complete seq seed in {SEQ} (requested {a.seeds})')
    print(f'members: {a.members} seeds {mseeds}; seq ({SEQ}) seeds {seeds}; route-D corrector x{a.corr_scale}')

    M = {f: oof(ADSB2, f, mseeds, month) for f in FAMS}
    seq_s = {s: oof(SEQ, 'seq', [s], month) for s in seeds}
    seq_bulk = oof(SEQ, 'seq', seeds, month, '.bulk.npy')

    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'off_sched', 'MVT_TIME_UTC_mvt'])
    apt = dd.APT.astype(str).to_numpy(); so = dd.off_sched.to_numpy()[Pidx]
    block = np.char.add(apt, dd.MVT_TIME_UTC_mvt.dt.strftime('%m%d').to_numpy().astype(str))
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    w2 = CACHE / 'cv' / 'w2'                                  # v17 keeps v15's W2 bulk inside P
    pb_w2 = 0.5 * oof(w2, 'mix', mseeds, month, '.bulk.npy')[Pidx] + 0.5 * oof(w2, 'nn', mseeds, month, '.bulk.npy')[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    A, DnP = ~D & ~P, D & ~P
    fit = np.isin(month, SCREEN) & ~P

    def weights(Mx, fams):
        W = {}
        for key, g in (('routes A/B/C', ~D), ('route D', D)):
            w, _ = nnls(np.column_stack([Mx[f] for f in fams])[fit & g], y[fit & g])
            W[key] = dict(zip(fams, np.round(w, 4).tolist()))
        return W

    def structure(Mx, W, fams):
        f = v11.copy()
        for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
            bl = sum(W[key][m] * Mx[m] for m in fams)
            f[g] = np.clip(bl[g] + (a.corr_scale * (v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
        f[Pidx] = v11[Pidx] + (mixP(p14, pb_w2) - mixP(p_cur, pb_old))
        return f

    Wo = weights(M, FAMS)
    fo = structure(M, Wo, FAMS)
    ref = CACHE / 'w2' / 'v17s_oof.npy'
    if a.corr_scale == 1.0 and a.members == 'adsb2' and ref.exists():
        dv = np.abs(fo - np.load(ref).astype(float)).max()
        print(f'check: old structure vs cache/w2/v17s_oof.npy max |diff| {dv:.3f} s' + ('  <- MISMATCH' if dv > 0.5 else ''))
    fams8 = FAMS + ['seq']
    Mn = dict(M, seq=np.mean([seq_s[s] for s in seeds], axis=0))
    Wn = weights(Mn, fams8)
    fn = structure(Mn, Wn, fams8)
    for nm, W in (('old', Wo), ('new', Wn)):
        for key in W:
            print(f'  {nm} weights {key:12s}: ' + '  '.join(f'{m} {v:.3f}' for m, v in W[key].items()))

    held, S = ~np.isin(month, SCREEN), np.isin(month, SCORED)
    print(f'\nold (v17 structure): held-out {rmse(y[held], fo[held]):.2f}  scored {rmse(y[S], fo[S]):.2f}')
    oP = ~P & held
    print('members alone, held-out outside P: ' + '  '.join(f'{f} {rmse(y[oP], Mn[f][oP]):.1f}' for f in fams8))
    kb = oP & (y <= 7200)
    print('bulk regressors on held-out bulk rows outside P: seq ' + f'{rmse(y[kb], seq_bulk[kb]):.1f}  ' + '  '.join(
        f'{f} {rmse(y[kb], oof(ADSB2, f, mseeds, month, ".bulk.npy")[kb]):.1f}' for f in FAMS if f != 'routed'))
    e_seq = y - Mn['seq']
    print('residual correlation with seq (held-out outside P): ' + '  '.join(
        f'{f} {np.corrcoef(e_seq[oP], (y - M[f])[oP])[0, 1]:.3f}' for f in FAMS))

    res = {}
    for scope, k in (('held-out', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7)):
        res[scope] = block_boot(y[k], fn[k], fo[k], block[k])
    per = [rmse(y[month == m], fn[month == m]) - rmse(y[month == m], fo[month == m]) for m in range(1, 13)]
    print('\nNEW (7 + seq) vs OLD, all flights, airport-day block bootstrap: ' + '  '.join(
        f'{s} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]' for s, v in res.items()) + f'  months better {sum(v < 0 for v in per)}/12')
    per_seed = {}
    for s in seeds:
        Ms = dict(M, seq=seq_s[s]); fs = structure(Ms, weights(Ms, fams8), fams8)
        per_seed[s] = (rmse(y[held], fs[held]) - rmse(y[held], fo[held]), rmse(y[S], fs[S]) - rmse(y[S], fo[S]))
        print(f'  seq seed {s} alone in the blend: held-out {per_seed[s][0]:+.2f}  scored {per_seed[s][1]:+.2f}')
    gain = (y - fo) ** 2 - (y - fn) ** 2
    top = np.argsort(gain)[::-1][:100]
    share = gain[top].sum() / gain.sum() if gain.sum() > 0 else float("nan")      # undefined without a net gain
    keep = np.ones(len(y), bool); keep[top] = False
    d100 = {sc: rmse(y[k & keep], fn[k & keep]) - rmse(y[k & keep], fo[k & keep]) for sc, k in (('held-out', held), ('scored', S))}
    print(f'share of the SSE reduction carried by the 100 most-improved rows: {share:.2f}; without them: '
          f'held-out {d100["held-out"]:+.2f}, scored {d100["scored"]:+.2f}')
    print('per airport (all flights, 12 months): ' + '  '.join(
        f'{x} {rmse(y[apt == x], fn[apt == x]) - rmse(y[apt == x], fo[apt == x]):+.2f}' for x in sorted(set(apt))))

    info = None
    if all(complete(SEQ, 'seqnc', s_) for s_ in seeds):
        nc_s = {s_: oof(SEQ, 'seqnc', [s_], month) for s_ in seeds}
        nc_bulk = oof(SEQ, 'seqnc', seeds, month, '.bulk.npy')
        print(f'\nINFO context effect, seq - seqnc bulk regressors on held-out bulk rows outside P: '
              f'{"%+.2f [%+.2f,%+.2f]" % block_boot(y[kb], seq_bulk[kb], nc_bulk[kb], block[kb])}; per airport: ' +
              '  '.join(f'{x} {rmse(y[kb & (apt == x)], seq_bulk[kb & (apt == x)]) - rmse(y[kb & (apt == x)], nc_bulk[kb & (apt == x)]):+.2f}'
                        for x in sorted(set(apt))))

        def delta_arm(seq_list):
            dl = np.mean([seq_s[s_] - nc_s[s_] for s_ in seq_list], axis=0)
            f2 = fn_by[tuple(seq_list)].copy(); lam = {}
            for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
                k = fit & g
                lam[key] = float(dl[k] @ (y[k] - f2[k]) / max(dl[k] @ dl[k], 1e-9))
                f2[g] = np.clip(f2[g] + lam[key] * dl[g], 0, None)
            return f2, lam
        fn_by = {tuple(seeds): fn}
        for s_ in seeds:
            Ms = dict(M, seq=seq_s[s_]); fn_by[(s_,)] = structure(Ms, weights(Ms, fams8), fams8)
        f2, lam = delta_arm(seeds)
        r2 = {sc: block_boot(y[k], f2[k], fo[k], block[k]) for sc, k in
              (('held-out', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
        g2 = (y - fo) ** 2 - (y - f2) ** 2
        top2 = np.argsort(g2)[::-1][:100]; keep2 = np.ones(len(y), bool); keep2[top2] = False
        d2 = {sc: rmse(y[k & keep2], f2[k & keep2]) - rmse(y[k & keep2], fo[k & keep2]) for sc, k in (('held-out', held), ('scored', S))}
        ps2 = {s_: rmse(y[held], delta_arm([s_])[0][held]) - rmse(y[held], fo[held]) for s_ in seeds}
        R2i = dict(R1=r2['held-out'][0] < 0 and r2['scored'][0] < 0, R2=r2['held-out'][2] < 0,
                   R3=r2['Jan'][0] < 0 and r2['Jul'][0] < 0, R4=len(seeds) >= 2 and all(v < 0 for v in ps2.values()),
                   R5=d2['held-out'] < 0 and d2['scored'] < 0)
        R2i = {k: bool(v) for k, v in R2i.items()}
        print(f'INFO new blend + lambda (seq - seqnc), lambda {lam}: ' + '  '.join(
            f'{sc} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]' for sc, v in r2.items()) + f'  per seed held-out {ps2}; '
              f'without top-100 rows {d2}; rules ' + ' '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R2i.items()))
        info = {'lambda': lam, 'result': {k: list(v) for k, v in r2.items()}, 'per_seed': ps2, 'drop100': d2, 'rule': R2i}

    R = dict(R1=res['held-out'][0] < 0 and res['scored'][0] < 0, R2=res['held-out'][2] < 0,
             R3=res['Jan'][0] < 0 and res['Jul'][0] < 0,
             R4=len(seeds) >= 2 and all(v[0] < 0 for v in per_seed.values()),
             R5=d100['held-out'] < 0 and d100['scored'] < 0)
    R = {k: bool(v) for k, v in R.items()}
    verdict = 'PASS -> adopt seq as the 8th member' if all(R.values()) else 'FAIL -> do not adopt'
    print(f'\nRULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R.items()) + f'  => {verdict}')
    out = (CACHE / 'seq' if not os.path.isabs(a.campaign) else SEQ) / f'seq_eval_{a.campaign if not os.path.isabs(a.campaign) else "dry"}_{a.members}_corr{a.corr_scale:g}.json'
    out.parent.mkdir(exist_ok=True)
    json.dump({'members': fams8, 'member_seeds': mseeds, 'seq_seeds': seeds, 'weights': Wn, 'weights_old': Wo,
               'result': {k: list(v) for k, v in res.items()}, 'per_month': per, 'per_seed': per_seed,
               'top100_share': share, 'drop100': d100, 'rule': R, 'verdict': verdict, 'info_delta_arm': info},
              open(out, 'w'), indent=1)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
