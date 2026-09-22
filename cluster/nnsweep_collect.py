"""Score the hyperparameter sweep: accuracy AND decorrelation, per configuration.

    python cluster/nnsweep_collect.py --campaign nnsweep --group A

WHY NOT cluster/collect.py

collect.py assembles TWELVE months into one OOF vector and skips any arm that is
short a fold, which is exactly right for the `confirm` group and useless for a
4-fold screen. It also has no notion of a reference model, so it cannot report
the number this sweep exists to produce. Use collect.py for the 12-fold
confirmation (`python cluster/collect.py <campaign>`); use this for screening.

WHAT IT REPORTS, AND WHY THAT SET

  rmse            the full mixture over the screening folds -- what the model
                  would actually score. Reported, but NOT ranked on: the ~183
                  tail rows in these folds carry a squared error of order 1e8
                  each, are identical across arms (the tail is held fixed), and
                  so add a large shared constant that compresses every real
                  difference. Ranking on it would mostly be ranking on noise.
  bulk            the mixture over rows with y <= 7200. The honest headline.
  nn_bulk         the network alone on bulk rows -- the quantity the sweep
                  varies, and the one with the cleanest signal.
  rho             corr(residual_nn, residual_lgb) on bulk rows.
  blend           the 50/50 blend of the network and the production LightGBM.

  The blend is computed on the bulk predictions and is IDENTICAL to blending the
  two finished mixtures, because the mixture p*e_tail + (1-p)*pred_bulk is affine
  in pred_bulk with weights shared by both members.

THE ARITHMETIC THAT DRIVES THE DECISION. Two models with the same residual sd
sigma and residual correlation rho blend 50/50 to sigma*sqrt((1+rho)/2). At the
measured rho = 0.956 that is a 1.1% gain; at rho = 0.90, 2.5%; at rho = 0.80,
5.1%. On a ~287 s baseline that is ~1 s, ~7 s and ~15 s. So an arm that is 3 s
WORSE solo but drops rho to 0.90 beats an arm that is 2 s better solo at 0.956.
The table therefore prints both, plus `pred50`, the sigma*sqrt((1+rho)/2)
prediction, next to the measured blend -- when they disagree it is because the
two members have unequal variance, and the measured one is the truth.

EVERY COMPARISON IS AGAINST THE `base` ARM OF THIS CAMPAIGN, not against the
286.66 from the `nn` campaign. See cluster/nnsweep_task.py for why those two
numbers are not comparable.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from cluster.nnsweep_configs import CONFIGS, group
from prc.config import CACHE, TAIL_CAP
from prc.validation import paired_bootstrap, rmse


def load_fixed(fd, months):
    """Pooled y, off_sched, p_tail, tail_mean and the LightGBM reference."""
    y, os_, pt, tm, lgb, mon = [], [], [], [], [], []
    for m in months:
        man = fd / f'manifest__m{m:02d}.json'
        if not man.exists():
            sys.exit(f'no prep for month {m} under {fd} -- run the fixed stage')
        meta = json.loads(man.read_text())
        va = np.load(fd / f'va__m{m:02d}.npy')
        yall = np.load(fd / f'yall__m{m:02d}.npy')
        y.append(yall[va])
        os_.append(np.load(fd / f'offsched__m{m:02d}.npy'))
        pt.append(np.load(fd / f'ptail__m{m:02d}.npy'))
        tm.append(np.full(int(va.sum()), float(np.load(fd / f'tailmean__m{m:02d}.npy')[0])))
        mon.append(np.full(int(va.sum()), m))
        p = fd / f'lgbbulk__m{m:02d}.npy'
        lgb.append(np.load(p).astype(np.float64) if p.exists() else None)
        if meta.get('subsample'):
            print(f'  *** WARNING: month {m} was prepped with --subsample '
                  f'{meta["subsample"]}. These are SMOKE-TEST numbers. ***')
    return (np.concatenate(y), np.concatenate(os_), np.concatenate(pt),
            np.concatenate(tm), np.concatenate(mon),
            np.concatenate(lgb) if all(v is not None for v in lgb) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--group', default='A')
    ap.add_argument('--base', default='base', help='the arm every comparison is paired against')
    ap.add_argument('--boot', type=int, default=500)
    ap.add_argument('--per-fold', action='store_true', help='also print a per-fold table')
    ap.add_argument('--out-root', default=None,
                    help='campaign root, if the run did not use cache/cv')
    a = ap.parse_args()

    g = group(a.group)
    months = list(g['months'])
    seeds = list(g['seeds'])
    d = (Path(a.out_root) if a.out_root else CACHE / 'cv') / a.campaign
    fd, bd, md = d / '_fixed', d / '_bulk', d / '_meta'
    if not md.exists():
        sys.exit(f'no results under {md}')

    y, offs, ptail, tailm, mon, p_lgb = load_fixed(fd, months)
    bulk = y <= TAIL_CAP
    print(f'\ncampaign {a.campaign!r} group {a.group!r}: months {months}, seeds {seeds}, '
          f'{len(y):,} rows ({int((~bulk).sum())} tail)')
    if p_lgb is None:
        print('  *** no LightGBM reference cached (prep run with --skip-lgbref): '
              'residual correlation and blends are NOT available ***')

    def mix(pb):
        et = np.where(np.isfinite(offs) & (offs > TAIL_CAP), offs, tailm)
        return np.clip(ptail * et + (1 - ptail) * np.asarray(pb, float), 0, None)

    # ---- assemble every arm's pooled bulk prediction -----------------------
    arms, incomplete, metas = {}, {}, defaultdict(list)
    for cfg in g['configs']:
        for s in seeds:
            parts, miss = [], []
            for m in months:
                p = bd / f'{cfg}__seed{s:02d}__m{m:02d}.npy'
                if p.exists():
                    parts.append(np.load(p).astype(np.float64))
                    mp = md / f'{cfg}__seed{s:02d}__m{m:02d}.json'
                    if mp.exists():
                        metas[cfg].append(json.loads(mp.read_text()))
                else:
                    miss.append(m)
            if miss:
                incomplete[f'{cfg}/seed{s:02d}'] = miss
                continue
            v = np.concatenate(parts)
            if len(v) != len(y):
                incomplete[f'{cfg}/seed{s:02d}'] = f'{len(v):,} preds for {len(y):,} rows'
                continue
            arms.setdefault(cfg, {})[s] = v
    for k, v in sorted(incomplete.items()):
        print(f'  INCOMPLETE {k}: missing {v}')
    if not arms:
        sys.exit('no complete arms')

    # A config edited after some arms had already run would leave two different
    # models under one name. Each unit recorded the parameters it actually used.
    for cfg, recs in metas.items():
        want = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in CONFIGS[cfg].items() if k != 'note'}
        for r in recs:
            if r.get('params') and r['params'] != want:
                print(f'  *** {cfg}: a result was produced with parameters that '
                      f'differ from the current registry. The registry was edited '
                      f'mid-campaign; these numbers are not comparable. ***')
                break

    per_seed = {k: v for k, v in arms.items() if len(v) > 1}
    arms = {k: np.mean(list(v.values()), axis=0) for k, v in arms.items()}
    print(f'  {len(arms)} complete arm(s) of {len(g["configs"])}\n')

    # ---- the table ---------------------------------------------------------
    yb = y[bulk]
    rows = []
    for cfg, pb in arms.items():
        r = {'cfg': cfg, 'nn_bulk': rmse(yb, pb[bulk])}
        mx = mix(pb)
        r['rmse'] = rmse(y, mx)
        r['bulk'] = rmse(yb, mx[bulk])
        if p_lgb is not None:
            rn, rl = yb - pb[bulk], yb - p_lgb[bulk]
            r['rho'] = float(np.corrcoef(rn, rl)[0, 1])
            bl = 0.5 * (pb + p_lgb)
            r['blend'] = rmse(yb, bl[bulk])
            r['blend_mix'] = rmse(y, mix(bl))
            # sigma*sqrt((1+rho)/2) with sigma taken as the WORSE member, the
            # form quoted in the log. Equal-variance approximation on purpose:
            # where it misses the measured blend, the members differ in spread.
            sg = max(r['nn_bulk'], rmse(yb, p_lgb[bulk]))
            r['pred50'] = sg * np.sqrt((1 + r['rho']) / 2)
        m0 = metas.get(cfg, [{}])[0]
        r['params'] = m0.get('n_params')
        r['secs'] = int(np.mean([x.get('fit_secs', 0) for x in metas.get(cfg, [{}])]))
        r['note'] = CONFIGS[cfg].get('note', '')
        rows.append(r)

    key = 'blend' if p_lgb is not None else 'nn_bulk'
    rows.sort(key=lambda r: r[key])
    hdr = (f'{"config":10s} {"nn_bulk":>8s} {"rho":>6s} {"blend":>8s} {"pred50":>8s} '
           f'{"bulk":>8s} {"rmse":>8s} {"params":>10s} {"fit_s":>6s}  note')
    print('=' * 130)
    print(f'RANKED BY {key.upper()}  (the network alone on bulk rows; '
          f'blend = 50/50 with production LightGBM)')
    print('=' * 130)
    print(hdr)
    for r in rows:
        print(f'{r["cfg"]:10s} {r["nn_bulk"]:8.2f} '
              f'{r.get("rho", float("nan")):6.3f} {r.get("blend", float("nan")):8.2f} '
              f'{r.get("pred50", float("nan")):8.2f} {r["bulk"]:8.2f} {r["rmse"]:8.2f} '
              f'{(r["params"] or 0):10,d} {r["secs"]:6d}  {r["note"][:44]}')
    if p_lgb is not None:
        print(f'{"[lightgbm]":10s} {rmse(yb, p_lgb[bulk]):8.2f} {1.0:6.3f} '
              f'{"-":>8s} {"-":>8s} {rmse(yb, mix(p_lgb)[bulk]):8.2f} '
              f'{rmse(y, mix(p_lgb)):8.2f} {"-":>10s} {"-":>6s}  '
              f'the production bulk regressor -- the blend partner')

    # ---- paired bootstraps against the control -----------------------------
    if a.base not in arms:
        print(f'\nbase arm {a.base!r} incomplete -- no paired comparisons')
        return
    pb0 = arms[a.base]
    print('\n' + '=' * 130)
    print(f'PAIRED BOOTSTRAP vs {a.base!r}, bulk rows, n_boot={a.boot}. '
          f'Negative = better. A CI that crosses 0 means NOTHING WAS MEASURED.')
    print('=' * 130)
    print(f'{"config":10s} {"solo diff":>22s} {"P(worse)":>9s} | {"blend diff":>22s} '
          f'{"P(worse)":>9s}')
    for r in rows:
        if r['cfg'] == a.base:
            continue
        pb = arms[r['cfg']]
        s = paired_bootstrap(yb, pb[bulk], pb0[bulk], n_boot=a.boot)
        line = (f'{r["cfg"]:10s} {s["diff"]:+7.2f} +- {s["sd"]:4.2f} '
                f'[{s["lo"]:+6.2f},{s["hi"]:+6.2f}] {s["p_a_worse"] * 100:8.1f}%')
        if p_lgb is not None:
            b = paired_bootstrap(yb, (0.5 * (pb + p_lgb))[bulk],
                                 (0.5 * (pb0 + p_lgb))[bulk], n_boot=a.boot)
            line += (f' | {b["diff"]:+7.2f} +- {b["sd"]:4.2f} '
                     f'[{b["lo"]:+6.2f},{b["hi"]:+6.2f}] {b["p_a_worse"] * 100:8.1f}%')
        print(line)

    # ---- the noise floor ---------------------------------------------------
    if per_seed:
        print('\n' + '=' * 130)
        print('SEED SPREAD -- the same configuration, different initialisations. '
              'THE NOISE FLOOR.')
        print('The paired bootstrap above resamples ROWS and is blind to this, so '
              'any ranking difference')
        print('smaller than this spread is not a difference.')
        print('=' * 130)
        for cfg, bysd in sorted(per_seed.items()):
            vals = {sd: rmse(yb, v[bulk]) for sd, v in sorted(bysd.items())}
            txt = '  '.join(f'seed{sd:02d} {v:.2f}' for sd, v in vals.items())
            arr = np.array(list(vals.values()))
            print(f'  {cfg:10s} {txt}   spread {arr.max() - arr.min():.2f}  '
                  f'sd {arr.std(ddof=1):.2f}  mean {arr.mean():.2f}  '
                  f'{len(arr)}-seed average {rmse(yb, arms[cfg][bulk]):.2f}')

    # ---- what the blend would be worth -------------------------------------
    if p_lgb is not None:
        print('\n' + '=' * 130)
        print('DECORRELATION: is an arm worth keeping for the blend even if it '
              'loses solo?')
        print('=' * 130)
        r0 = next(r for r in rows if r['cfg'] == a.base)
        print(f'  control {a.base!r}: solo {r0["nn_bulk"]:.2f}, rho {r0["rho"]:.3f}, '
              f'blend {r0["blend"]:.2f}')
        for r in rows:
            if r['cfg'] == a.base:
                continue
            ds, dr, db = (r['nn_bulk'] - r0['nn_bulk'], r['rho'] - r0['rho'],
                          r['blend'] - r0['blend'])
            tag = ''
            if ds > 0.2 and db < -0.2:
                tag = '  <- WORSE SOLO, BETTER BLENDED: the decorrelation case'
            elif ds < -0.2 and db > 0.2:
                tag = '  <- better solo but worse blended; do not ship it alone'
            print(f'  {r["cfg"]:10s} solo {ds:+6.2f}  rho {dr:+6.3f}  '
                  f'blend {db:+6.2f}{tag}')

    # ---- learning curves ---------------------------------------------------
    print('\n' + '=' * 130)
    print('LEARNING CURVES -- last three epochs of VALIDATION bulk RMSE (the per-'
          'epoch train_loss in the')
    print('meta files is NOT comparable across arms: `huber` optimises a '
          'different objective). A curve still falling at the end')
    print('means the budget binds; one that has turned up means it does not.')
    print('=' * 130)
    for r in rows:
        cs = [x for x in metas.get(r['cfg'], []) if x.get('curve')]
        if not cs:
            continue
        # average the curve across folds; they have the same length by construction
        L = min(len(x['curve']) for x in cs)
        vals = np.mean([[e['val_bulk_rmse'] for e in x['curve'][:L]] for x in cs], axis=0)
        tail = '  '.join(f'{v:.2f}' for v in vals[-3:])
        print(f'  {r["cfg"]:10s} {L:3d} epochs  first {vals[0]:8.2f}  last three {tail}'
              f'   delta(last-3 -> last) {vals[-1] - vals[max(0, L - 3)]:+6.2f}')

    if a.per_fold:
        print('\n' + '=' * 130)
        print('PER FOLD, network alone on bulk rows')
        print('=' * 130)
        print(f'{"config":10s}' + ''.join(f'{"m%02d" % m:>10s}' for m in months))
        for r in rows:
            pb = arms[r['cfg']]
            cells = ''.join(
                f'{rmse(y[bulk & (mon == m)], pb[bulk & (mon == m)]):10.2f}'
                for m in months)
            print(f'{r["cfg"]:10s}{cells}')


if __name__ == '__main__':
    main()
