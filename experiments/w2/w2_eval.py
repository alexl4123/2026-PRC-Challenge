"""W2 evaluation: the v11 signals inside the members vs the shipped structure (blend + stack).

Inputs (12-fold leave-one-month-out mixtures, one file per family x seed x month):
  old  cache/cv/study/<fam>__seedNN__mMM.npy  -- the members of the shipped family blend
  new  cache/cv/w2/<fam>__seedNN__mMM.npy     -- the same members with the v11 columns inside
                                                (cluster/w2_task.py; network also at emb cap 16)
Seeds are MATCHED: only seeds complete (all 12 months) for every family in BOTH campaigns are
averaged, so a 1-seed W2 read is compared with the old campaign's same seed.

Structures compared outside P (P = no NM record and > 2 h behind schedule, which keeps the tail
specialist and is not touched by W2):
  shipped  old per-route blend with the SHIPPED weights (cache/exp66_weights_perroute_all.json)
           + the stack arm (c) correction (cache/v11/oof_c.npy) -- what v12/v13 serve on A/B/C
  old      the same blend without the stack
  new      per-route NNLS blend of the W2 members, weights fitted on the screening months
           (3, 6, 9, 12) outside P exactly as exp66 fitted the shipped ones
Judged on the 8 held-out months and on the scored months (1, 7), paired bootstrap.

PRE-REGISTERED DECISION (written 2026-09-24, before any W2 fold existed): W2 replaces the
shipped blend + stack on routes A/B/C if `new` beats `shipped` there on BOTH the 8 held-out
months and the scored months, with the held-out interval below 0. Route D is reported, not
decided here: production serves it differently (the v10 route-D corrector), and its 2026
behaviour is known not to follow 2025 CV.

    python experiments/w2/w2_eval.py            # every seed complete in both campaigns
    python experiments/w2/w2_eval.py --seeds 0  # early read after the seed-0 tasks
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.optimize import nnls

from prc.io import load, peak_rss_gb
from prc.config import TARGET, TAIL_CAP, CACHE
from prc.validation import paired_bootstrap
from prc.models.routed import route_of

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost']
SCREEN, SCORED = [3, 6, 9, 12], [1, 7]
OLD, NEW = CACHE / 'cv' / 'study', CACHE / 'cv' / 'w2'
OUT = CACHE / 'w2'


def complete(folder, fam, s):
    return all((folder / f'{fam}__seed{s:02d}__m{m:02d}.npy').exists() for m in range(1, 13))


def oof(folder, fam, seeds, month):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            a = np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}.npy')
            if len(a) != (month == m).sum():
                sys.exit(f'FATAL: {folder.name}/{fam} seed {s} month {m}: {len(a)} rows, expected {(month == m).sum()}')
            o[month == m] += a / len(seeds)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--new-dir', default=None, help='self-test: point `new` at another campaign (e.g. study)')
    a = ap.parse_args()
    global NEW
    if a.new_dir:
        NEW = CACHE / 'cv' / a.new_dir

    d = load('training', phase='DEP', columns=[TARGET, 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'])
    y = d[TARGET].astype(float).to_numpy()
    month = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    D = route_of((d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds().to_numpy()) == 'D'
    so = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    P = D & np.isfinite(so) & (so > TAIL_CAP)
    del d

    seeds = a.seeds if a.seeds is not None else [s for s in range(3)
                                                  if all(complete(NEW, f, s) and complete(OLD, f, s) for f in FAMS)]
    missing = [(f, s) for f in FAMS for s in seeds if not (complete(NEW, f, s) and complete(OLD, f, s))]
    if not seeds or missing:
        sys.exit(f'FATAL: no usable seed set (requested {a.seeds}); incomplete (family, seed): {missing[:8]}')
    print(f'seeds (matched in both campaigns): {seeds}')
    Xo = np.column_stack([oof(OLD, f, seeds, month) for f in FAMS])
    Xn = np.column_stack([oof(NEW, f, seeds, month) for f in FAMS])
    c = np.nan_to_num(np.load(CACHE / 'v11' / 'oof_c.npy').astype(float))
    c[P] = 0.0

    W = json.load(open(CACHE / 'exp66_weights_perroute_all.json'))
    assert W['members'] == FAMS or sorted(W['members']) == sorted(FAMS)
    old = np.empty(len(y)); new = np.empty(len(y)); wnew = {}
    fit = np.isin(month, SCREEN) & ~P
    for gname, g in (('route D', D), ('routes A/B/C', ~D)):
        wo = np.array([W['weights'][gname][f] for f in FAMS])
        old[g] = Xo[g] @ wo
        wn, _ = nnls(Xn[fit & g], y[fit & g])
        new[g] = Xn[g] @ wn
        wnew[gname] = dict(zip(FAMS, np.round(wn, 4).tolist()))
        print(f'  new weights {gname:12s}: ' + '  '.join(f'{f} {w:.3f}' for f, w in zip(FAMS, wn)) + f'  (sum {wn.sum():.3f})')
    shipped = old + c

    held = ~np.isin(month, SCREEN) & ~P
    scored = np.isin(month, SCORED) & ~P

    def cmp(nm, pa, pb, g):
        out = []
        for scope, k in (('8 held-out', held & g), ('scored', scored & g)):
            b = paired_bootstrap(y[k], pa[k], pb[k], n_boot=500)
            out.append(f'{scope} {b["diff"]:+.3f} [{b["lo"]:+.3f},{b["hi"]:+.3f}]')
        per = [np.sqrt(np.mean((y[k] - pa[k]) ** 2)) - np.sqrt(np.mean((y[k] - pb[k]) ** 2))
               for m in range(1, 13) for k in [(month == m) & ~P & g]]
        print(f'  {nm:34s} ' + '   '.join(out) + f'   months better {sum(v < 0 for v in per)}/12')
        return out

    print('\nPER MEMBER, new minus old (seed-matched), all routes outside P:')
    for j, f in enumerate(FAMS):
        cmp(f, Xn[:, j], Xo[:, j], np.ones(len(y), bool))

    ABC = ~D
    print('\nROUTES A/B/C outside P (the decision):')
    cmp('new blend - old blend', new, old, ABC)
    cmp('new blend - SHIPPED (old + stack)', new, shipped, ABC)
    cmp('new blend + stack - shipped (info)', new + c, shipped, ABC)
    print('\nROUTE D outside P (reported only):')
    cmp('new blend - old blend', new, old, D)
    cmp('new blend - old blend + stack', new, shipped, D)
    r = lambda p, k: np.sqrt(np.mean((y[k] - p[k]) ** 2))
    print(f'\nRMSE A/B/C outside P, 8 held-out months: shipped {r(shipped, held & ABC):.2f}  old {r(old, held & ABC):.2f}  '
          f'new {r(new, held & ABC):.2f} | scored: {r(shipped, scored & ABC):.2f} / {r(old, scored & ABC):.2f} / {r(new, scored & ABC):.2f}')

    if a.new_dir:
        print('self-test run: nothing saved'); return
    OUT.mkdir(exist_ok=True)
    json.dump({'members': FAMS, 'seeds': seeds, 'weights': wnew}, open(OUT / 'w2_weights.json', 'w'), indent=1)
    np.savez(OUT / 'w2_eval.npz', new=new, old=old, shipped=shipped, seeds=seeds)
    print(f'saved {OUT}/w2_weights.json, w2_eval.npz | peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
