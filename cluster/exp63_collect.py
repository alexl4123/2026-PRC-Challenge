"""EXP63 step 2 collector: 12-fold OOF of each option against the v6 baseline.

Reads the per-(option, fold) units written by cluster/exp63_task.py, attaches the
UNCHANGED v6 tail branch (pt, et) reconstructed from exp62's caches, and runs the
paired bootstrap against the v6 baseline -- which is REUSED, never refitted:

    required  cache/exp62_pb.npy      v6 bulk regressor OOF, seed 0   float32 [2,085,047]
    required  cache/exp62_ptg.npy     v6 global tail classifier OOF   float32 [2,085,047]
    required  cache/exp62_spec_P.npy  specialist P(tail) on P rows    [2, 4,944]
                                      row 0 = v6 TailSpecialist, row 1 = TailSpecialistX
    optional  cache/exp62_oof.npy     rows v5, a_v6, b_v6x -- used to CHECK the
                                      reconstruction to 0.01 s
All four are produced by experiments/exp62_cv.py (the exp62 agent's step 2):
phase A writes pb/ptg after all 12 folds, phase B writes spec_P/oof seconds later.

Hypothesis, gate and the facts behind them: see the cluster/exp63_task.py docstring.

DECISION RULE (fixed before any result is seen): an option is adopted only if
its 12-month paired bootstrap (option - v6) has a 95% CI entirely below zero.
If the null-control arm N is present, its difference is printed alongside: an
option whose effect is not clearly larger than N's is refit noise.
Scored months (1, 7) and inside/outside P are REPORTED, never decisive.

USAGE (run locally after pulling cache/exp63/units/ back from the cluster)
  python cluster/exp63_collect.py --options A:p95:pooled,B:p95:pooled,N
  python cluster/exp63_collect.py --smoke          # plumbing test on smoke units
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, CACHE, AIRPORTS
from prc.io import load, peak_rss_gb
from prc.validation import paired_bootstrap, rmse
from cluster.exp63_task import (N_DEP, UNIT_DIR, SMOKE_DIR, SMOKE_MONTHS, parse_option,
                                unit_path)

pd.set_option('display.width', 220)

EXP62_REQUIRED = {
    'exp62_pb.npy': 'v6 bulk-regressor OOF (phase A of exp62_cv.py)',
    'exp62_ptg.npy': 'v6 global tail-classifier OOF (phase A of exp62_cv.py)',
    'exp62_spec_P.npy': 'specialist OOF inside P (phase B of exp62_cv.py)',
}
PRODUCER = 'experiments/exp62_cv.py -- the exp62 agent\'s 12-fold step 2'
SCORED = (1, 7)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def require_exp62(d):
    """Fail loudly, before anything expensive, if exp62's caches are absent."""
    missing = [f for f in EXP62_REQUIRED if not (d / f).exists()]
    if missing:
        lines = [f'FATAL: exp63 reuses the v6 baseline from exp62 and cannot run without it.',
                 f'  missing in {d}:']
        lines += [f'    {f:18s} {EXP62_REQUIRED[f]}' for f in missing]
        lines += [f'  producer: {PRODUCER}.',
                  '  Wait for the exp62 run to finish (it writes all three at the end of',
                  '  its phase B; check with: ls -la cache/exp62_*.npy cache/exp62_folds/).',
                  '  Only if NO exp62 run is in progress, start one with:',
                  '    /home/thinklex/prc2026-venv/bin/python experiments/exp62_cv.py']
        sys.exit('\n'.join(lines))


def load_meta():
    """Light load (5 columns) of what the report needs, in DEP row order."""
    d = load('training', phase='DEP',
             columns=[TARGET, 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'])
    y = d[TARGET].astype(float).to_numpy()
    if len(y) != N_DEP:
        sys.exit(f'FATAL: {len(y):,} DEP rows, expected {N_DEP:,}')
    month = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    apt = np.asarray(d.APT.astype(str), dtype=object)
    off_sched = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy(float)
    miss = d.AOBT_3_flt.isna().to_numpy()
    # P exactly as prc/models/tail_specialist.population (not imported: that file
    # is the exp62 agent's work in progress).
    inP = miss & np.isfinite(off_sched) & (off_sched > TAIL_CAP)
    return dict(y=y, month=month, apt=apt, off_sched=off_sched, inP=inP)


def _load_vec(path, n):
    v = np.load(path)
    if v.shape != (n,):
        sys.exit(f'FATAL: {path.name} has shape {v.shape}, expected ({n},) -- produced '
                 f'against a different training cache?')
    return v.astype(float)


def anchor_et(y, month, off_sched):
    """v6 anchor, exp53/exp62 convention: off_sched inside the anchor zone, else
    the tail mean of the TRAINING months of that fold."""
    et = np.full(len(y), np.nan)
    for m in range(1, 13):
        vm = month == m
        tail_mean = y[(month != m) & (y > TAIL_CAP)].mean()
        os_ = off_sched[vm]
        et[vm] = np.where(np.isfinite(os_) & (os_ > TAIL_CAP), os_, tail_mean)
    return et


def mixture(pt, et, pbulk):
    return np.clip(pt * et + (1 - pt) * pbulk, 0, None)


def v6_branch(d, meta, spec_row=0):
    """pb, pt, et and the v6 baseline prediction, all length N_DEP."""
    require_exp62(d)
    pb = _load_vec(d / 'exp62_pb.npy', N_DEP)
    ptg = _load_vec(d / 'exp62_ptg.npy', N_DEP)
    spec = np.load(d / 'exp62_spec_P.npy')
    Pidx = np.where(meta['inP'])[0]
    if spec.ndim != 2 or spec.shape[1] != len(Pidx) or spec_row >= spec.shape[0]:
        sys.exit(f'FATAL: exp62_spec_P.npy shape {spec.shape}, expected (>={spec_row+1}, '
                 f'{len(Pidx)}) -- P rows no longer match exp62\'s')
    if not (np.isfinite(pb).all() and np.isfinite(ptg).all()):
        sys.exit('FATAL: exp62_pb/ptg contain non-finite values -- exp62 phase A incomplete')
    pt = ptg.copy()
    s = spec[spec_row].astype(float)
    k = np.isfinite(s)
    pt[Pidx[k]] = s[k]
    et = anchor_et(meta['y'], meta['month'], meta['off_sched'])
    v6 = mixture(pt, et, pb)
    ref = d / 'exp62_oof.npy'
    if ref.exists():
        r = np.load(ref)
        if r.ndim == 2 and r.shape[1] == N_DEP and r.shape[0] > 1 + spec_row:
            dmax = float(np.nanmax(np.abs(r[1 + spec_row] - v6)))
            if dmax > 0.01:
                sys.exit(f'FATAL: reconstructed v6 differs from exp62_oof.npy row '
                         f'{1+spec_row} by up to {dmax:.3f} s -- the tail branch is not '
                         f'the one exp62 scored; stop and investigate')
            print(f'  v6 reconstruction matches exp62_oof.npy row {1+spec_row} '
                  f'(max |diff| {dmax:.2e} s)', flush=True)
    else:
        print('  NOTE: exp62_oof.npy absent -- reconstruction not cross-checked', flush=True)
    return pb, pt, et, v6


def make_smoke_standins(d, meta):
    """SMOKE ONLY. Write stand-in exp62 caches with the right names, shapes and
    dtypes so the loading, alignment and cross-check code paths run before exp62
    has finished. The numbers are crude out-of-fold per-airport statistics and
    mean NOTHING. Never written to cache/ itself."""
    d.mkdir(parents=True, exist_ok=True)
    y, month, apt, inP = meta['y'], meta['month'], meta['apt'], meta['inP']
    pb = np.zeros(N_DEP, np.float32)
    ptg = np.zeros(N_DEP, np.float32)
    for m in range(1, 13):
        vm, tm = month == m, month != m
        for a in AIRPORTS:
            s = vm & (apt == a)
            t = tm & (apt == a)
            pb[s] = np.median(y[t & (y <= TAIL_CAP)])
            ptg[s] = (y[t] > TAIL_CAP).mean()
    Pidx = np.where(inP)[0]
    spec = np.vstack([np.clip(ptg[Pidx] * 50, 0, 1), np.clip(ptg[Pidx] * 60, 0, 1)])
    np.save(d / 'exp62_pb.npy', pb)
    np.save(d / 'exp62_ptg.npy', ptg)
    np.save(d / 'exp62_spec_P.npy', spec)
    et = anchor_et(y, month, meta['off_sched'])
    rows = []
    for r in (None, 0, 1):
        pt = ptg.astype(float).copy()
        if r is not None:
            pt[Pidx] = spec[r]
        rows.append(mixture(pt, et, pb.astype(float)))
    np.save(d / 'exp62_oof.npy', np.vstack(rows))
    print(f'  SMOKE: wrote SYNTHETIC exp62 stand-ins to {d}', flush=True)


def load_units(unit_dir, opt, seed, months):
    """Full-length arrays (NaN where absent) for pbulk and q, plus months found."""
    pbulk = np.full(N_DEP, np.nan)
    q = np.full(N_DEP, np.nan)
    got, metas = [], {}
    for m in months:
        f = unit_path(unit_dir, opt, seed, m)
        if not f.exists():
            continue
        z = np.load(f)
        idx = z['idx']
        pbulk[idx] = z['pbulk']
        if 'q' in z.files:
            q[idx] = z['q']
        metas[m] = json.loads(str(z['meta']))
        got.append(m)
    return pbulk, q, got, metas


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def pb_line(label, y, a, b, n_boot, seed=0):
    r = paired_bootstrap(y, a, b, n_boot=n_boot, seed=seed)
    print(f'  {label:46s} {r["diff"]:+8.3f} +- {r["sd"]:6.3f}  CI [{r["lo"]:+8.3f}, '
          f'{r["hi"]:+8.3f}]  P(worse) {100*r["p_a_worse"]:5.1f}%  n={len(y):,}', flush=True)
    return r


def report(meta, ev, preds, n_boot, out_csv):
    y, month, apt, inP = meta['y'], meta['month'], meta['apt'], meta['inP']
    arms = [k for k in preds if k != 'v6']
    rows = []

    print('\n' + '=' * 110)
    print(f'EXP63 STEP 2 -- {int(ev.sum()):,} rows, months {sorted(set(month[ev].tolist()))}')
    print('=' * 110)
    for k, p in preds.items():
        s = ev
        print(f'  {k:22s} RMSE {rmse(y[s], p[s]):8.3f} | outside P {rmse(y[s & ~inP], p[s & ~inP]):8.3f}'
              f' | inside P {rmse(y[s & inP], p[s & inP]) if (s & inP).any() else float("nan"):9.1f}'
              f' | bulk rows {rmse(y[s & (y <= TAIL_CAP)], p[s & (y <= TAIL_CAP)]):7.2f}')

    scopes = [('12-month', ev), ('outside P', ev & ~inP), ('inside P', ev & inP),
              ('scored 1+7', ev & np.isin(month, SCORED)),
              ('scored 1+7 outside P', ev & np.isin(month, SCORED) & ~inP)]
    scopes += [(f'month {m:02d}', ev & (month == m)) for m in SCORED]
    print(f'\nPAIRED BOOTSTRAP, arm minus v6 (negative = arm better; {n_boot} resamples)')
    res = {}
    for k in arms:
        print(f' {k}')
        for label, s in scopes:
            if s.sum() < 2:
                continue
            r = pb_line(label, y[s], preds[k][s], preds['v6'][s], n_boot)
            res[(k, label)] = r
            rows.append(dict(arm=k, scope=label, n=int(s.sum()),
                             rmse_arm=rmse(y[s], preds[k][s]), rmse_v6=rmse(y[s], preds['v6'][s]),
                             **r))

    print('\nPER MONTH  (RMSE; arm - v6)')
    pm = pd.DataFrame({k: pd.Series({m: rmse(y[ev & (month == m)], p[ev & (month == m)])
                                     for m in sorted(set(month[ev].tolist()))})
                       for k, p in preds.items()})
    for k in arms:
        pm[f'{k} - v6'] = pm[k] - pm['v6']
    print(pm.round(2).to_string())

    print('\nPER AIRPORT, outside P, all evaluated months (arm - v6, paired, '
          f'{max(200, n_boot // 2)} resamples)')
    for k in arms:
        for a in AIRPORTS:
            s = ev & ~inP & (apt == a)
            if s.sum() < 2:
                continue
            r = paired_bootstrap(y[s], preds[k][s], preds['v6'][s], n_boot=max(200, n_boot // 2))
            rows.append(dict(arm=k, scope=f'apt {a} outside P', n=int(s.sum()),
                             rmse_arm=rmse(y[s], preds[k][s]), rmse_v6=rmse(y[s], preds['v6'][s]), **r))
            print(f'  {k:22s} {a}  v6 {rmse(y[s], preds["v6"][s]):7.2f} -> {rmse(y[s], preds[k][s]):7.2f}'
                  f'   {r["diff"]:+7.3f} CI [{r["lo"]:+7.3f}, {r["hi"]:+7.3f}]')

    print('\nPER TARGET BAND (flat reference bands of dev.log s13): SSE share and RMSE')
    bands = [('y <= 1800', y <= 1800), ('1800 < y <= 7200', (y > 1800) & (y <= TAIL_CAP)),
             ('y > 7200', y > TAIL_CAP)]
    tot = {k: ((y[ev] - p[ev]) ** 2).sum() for k, p in preds.items()}
    for label, bm in bands:
        s = ev & bm
        line = f'  {label:18s} n={int(s.sum()):>9,}'
        for k, p in preds.items():
            e = ((y[s] - p[s]) ** 2).sum()
            line += f' | {k}: RMSE {np.sqrt(e / max(1, s.sum())):8.1f} SSE {100*e/tot[k]:5.1f}%'
        print(line)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'\n  wrote {out_csv}')

    print('\n' + '=' * 110)
    nctl = res.get(('N', '12-month'))
    for k in arms:
        if k == 'N':
            continue
        r = res.get((k, '12-month'))
        if r is None:
            continue
        adopt = r['hi'] < 0
        verdict = 'ADOPT-ELIGIBLE (12-month CI clears zero)' if adopt else 'DO NOT ADOPT (CI crosses zero)'
        print(f'DECISION {k}: 12-month {r["diff"]:+.3f} +- {r["sd"]:.3f}, CI [{r["lo"]:+.3f}, '
              f'{r["hi"]:+.3f}] -> {verdict}')
        if nctl is not None:
            print(f'   null control N (refit noise): {nctl["diff"]:+.3f}, CI [{nctl["lo"]:+.3f}, '
                  f'{nctl["hi"]:+.3f}]'
                  + ('  <-- the effect is NOT clearly larger than refit noise'
                     if adopt and abs(r['diff']) < 2 * abs(nctl['diff']) else ''))
            s = ev
            rn = paired_bootstrap(y[s], preds[k][s], preds['N'][s], n_boot=n_boot)
            print(f'   {k} minus N (both refitted in this pipeline): {rn["diff"]:+.3f} +- '
                  f'{rn["sd"]:.3f}, CI [{rn["lo"]:+.3f}, {rn["hi"]:+.3f}]  (supplementary)')
        rs = res.get((k, 'scored 1+7'))
        if rs is not None:
            print(f'   scored 1+7 (reported, not decisive): {rs["diff"]:+.3f} +- {rs["sd"]:.3f}')
    if nctl is not None and (nctl['hi'] < 0 or nctl['lo'] > 0):
        print('WARNING: the null control N alone clears zero -- the row bootstrap understates '
              'refit noise; treat every CI above as too narrow.')
    print('=' * 110)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--options', default='A:p95:pooled')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--spec-row', type=int, default=0,
                    help='0 = v6 TailSpecialist (default), 1 = TailSpecialistX if exp62 adopts it')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()

    opts = [parse_option(s) for s in a.options.split(',') if s.strip()]
    if a.smoke:
        if a.options == 'A:p95:pooled':
            opts = [parse_option(s) for s in ('A:p95:pooled', 'B:p95:ns', 'N')]
        exp62_dir, unit_dir = SMOKE_DIR / 'exp62_standin', SMOKE_DIR / 'units'
        months, n_boot = list(SMOKE_MONTHS), 50
        out_npy, out_csv = SMOKE_DIR / 'exp63_oof.npy', SMOKE_DIR / 'exp63_report.csv'
    else:
        exp62_dir, unit_dir = CACHE, UNIT_DIR
        months, n_boot = list(range(1, 13)), a.n_boot
        out_npy, out_csv = CACHE / 'exp63_oof.npy', CACHE / 'exp63_report.csv'
        require_exp62(exp62_dir)          # before any loading

    meta = load_meta()
    if a.smoke and not (exp62_dir / 'exp62_spec_P.npy').exists():
        make_smoke_standins(exp62_dir, meta)
    pb, pt, et, v6 = v6_branch(exp62_dir, meta, a.spec_row)
    print(f'[{time.time()-t0:.0f}s] v6 branch loaded; 12-month v6 RMSE {rmse(meta["y"], v6):.3f}',
          flush=True)

    preds = {'v6': v6}
    missing = []
    for i, o in enumerate(opts):
        pbulk, q, got, metas = load_units(unit_dir, o, a.seed, months)
        miss = sorted(set(months) - set(got))
        if miss:
            missing.append((i, o, miss))
            continue
        cover = np.isin(meta['month'], months)
        if not a.smoke and not np.isfinite(pbulk[cover]).all():
            sys.exit(f'FATAL: {o.tag}: units present but rows uncovered -- stale units?')
        preds[o.tag] = mixture(pt, et, pbulk)
        secs = [mm.get('seconds', np.nan) for mm in metas.values()]
        rss = [mm.get('peak_rss_gb', np.nan) for mm in metas.values()]
        print(f'  {o.tag}: {len(got)} folds | per-fold wall {np.nanmin(secs):.0f}-{np.nanmax(secs):.0f}s '
              f'| peak RSS {np.nanmax(rss):.2f} GB', flush=True)
    if missing:
        lines = ['FATAL: incomplete units -- nothing scored. Missing (option, month):']
        idx = []
        for i, o, miss in missing:
            lines.append(f'  {o.text:16s} months {miss}')
            idx += [i * 12 + (m - 1) for m in miss]
        lines.append('  resubmit only those array indices (same option order as the original sbatch):')
        lines.append(f'    sbatch --array={",".join(map(str, idx))}%40 cluster/exp63_array.sbatch '
                     + ' '.join(o.text for o in opts))
        sys.exit('\n'.join(lines))

    ev = np.isin(meta['month'], months)
    for k, p in preds.items():
        ev &= np.isfinite(p)
    names = list(preds)
    np.save(out_npy, np.vstack([preds[k] for k in names]).astype(np.float32))
    out_npy.with_suffix('.json').write_text(json.dumps({'rows': names, 'spec_row': a.spec_row,
                                                        'seed': a.seed}))
    print(f'  saved {out_npy} (rows {names})', flush=True)

    report(meta, ev, preds, n_boot, out_csv)
    print(f'\npeak RSS {peak_rss_gb():.2f} GB   total {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
