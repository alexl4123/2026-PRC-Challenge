"""EXP63 step 1 -- SCREEN the three-way target treatment (normal / delayed / artefact).

HYPOTHESIS
----------
The band 1800 < y <= 7200 is 3.53% of rows but 24.4% of squared error (37% of
everything inside "bulk"), at RMSE 464-3,151 s against ~190 s for ordinary taxis.
v6 pools it with the 96% ordinary rows under one L2 regressor. Splitting the bulk
into NORMAL (y <= b_apt) and DELAYED (b_apt < y <= 7200) and recombining as a
soft mixture

    E[y|x] = pt*et + (1-pt) * [ (1-q)*m_n + q*m_d ],   q = P(y > b_apt | y <= 7200, x)

may estimate the same conditional mean better. The v6 tail branch (pt, et) is
kept EXACTLY (from exp62's caches); the artefact band is not touched.

WHY A SCREEN FIRST -- the measured facts that make this doubtful
  * The mixture targets the SAME E[y | y <= 7200, x] the bulk regressor already
    estimates. It can only win on estimation, i.e. if q ranks delayed flights
    better than the regressor implicitly does. So the key number is not q's AUC
    but its AUC WITHIN (airport x decile of the v6 bulk prediction pb):
    discrimination the regressor does not already have.
  * The (1-q) algebra: at a band rate ~3.5%, a calibrated but non-discriminating
    q buys about nothing. Only discrimination pays.
  * The LOO airport x day x hour residual ceiling is -4.9 s: the delayed state is
    per-flight, not shared, so congestion-state features have no headroom here.
  * The NM proxy off_aobt3 does not see long taxis at most airports (within 5 min
    of truth for 1.4% of >30-min flights at LIRF-Jul, 0.0% at EDDF).
  * "Genuine 30-60 min flights are under-predicted by +462 s at LIRF-Jul" is a
    residual binned on the TRUE y. Even a perfect conditional-mean estimator shows
    that pattern (regression to the mean), so it is not by itself evidence of a
    fixable bias. This screen measures the fixable part.

WHAT IS MEASURED (leave-one-month-out; every boundary, encoding and model is
fitted on the 11 training months only)
  options = boundary {flat1800, p95, p97} x region {pooled, ns}
    boundary b_apt : per-airport percentile of y <= 7200 of the training months
                     (flat1800 = the original proposal; 1800 s is the 99.24th pct at
                     LSZH but the 89.41st at EGLL, so it mostly encodes airport)
    region         : pooled = one q classifier; ns = north {EDDF EDDM EHAM LSZH
                     LFPG EGLL} / south {LIRF LEBL LEMD LTFM}. 'season' (EGLL moved
                     to the summer-heavy group) exists in cluster/exp63_task.REGIONS
                     but is not in the default list.
  1. q classifier, out of fold, rows OUTSIDE P with y <= 7200: AUC (pooled,
     within airport, within airport x pb decile), AUC of pb itself, Brier, Brier
     skill vs the airport's training-month rate, log loss, calibration (ECE and
     deciles), overall and per airport.
  2. ORACLE CEILING: band membership known, q := 1{y > b_apt}, with the in-fold
     m_n/m_d. The most exp63 could ever buy. All months, scored 1+7, in/out of P.
  3. REALISTIC mixture (q from the classifier) against a MATCHED two-way baseline
     (one regressor on y <= 7200 at the same capacity), paired bootstrap.
  Components use the cheap pre-exp50 regressor (255 leaves / lr 0.05 / 350
  rounds) for BOTH arms so the comparison is matched and ~4x cheaper than v6's.
  q uses the v6 classifier config on all 72 features; m_d uses 63 leaves with
  in-fold early stopping. These are identical to step 2 (same functions).

GATE (fixed in advance)
  Choose boundary/region on SCREENING MONTHS 3, 6, 9, 12 ONLY -- never 1 or 7.
  Step 2 runs only if some option's realistic mixture beats the matched baseline
  there with a paired-bootstrap 95% CI entirely below zero. The chosen option is
  then CONFIRMED on the other eight months (reported, incl. 1 and 7 separately).
  Step 2 then tests A (mixture) and B (q as a stacked feature) for that option,
  plus the null control N, against the real v6 baseline.

NEGATIVE RESULT MEANS
  small oracle ceiling           -> even perfect band knowledge buys little; stop.
  big ceiling, AUC|pb ~ 0.5      -> the band is not discriminable per flight beyond
                                    what the regressor already knows; stop (B too).
  big ceiling, AUC|pb >> 0.5 but no realistic gain -> m_d/q estimation is the
                                    limit; B (stacking) is the one variant left.

INPUTS: cache/exp62_{pb,ptg,spec_P}.npy (+ exp62_oof.npy for a cross-check),
produced by experiments/exp62_cv.py. Checked FIRST; missing -> exits naming them.

RUN (after exp62 has finished; ~45 min for the 4 screening months, ~2.2 h for 12)
  systemd-inhibit --what=idle:sleep:handle-lid-switch --why=exp63 \\
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \\
    /home/thinklex/prc2026-venv/bin/python experiments/exp63_screen.py --months 3,6,9,12 \\
    2>&1 | tee cache/exp63_screen_a.log
  ... then the same with --months 1-12 (finished months are skipped) > exp63_screen.log
  --phase units : fit only (no exp62 needed) | --phase report : report only
  --check-inputs: verify prerequisites and exit | --smoke : plumbing test (<3 min)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from prc.config import TAIL_CAP, CACHE, AIRPORTS
from prc.io import peak_rss_gb
from prc.validation import paired_bootstrap, rmse
from cluster.exp63_task import (BOUNDARIES, REGIONS, BUDGET, SMOKE_DIR, SMOKE_MONTHS,
                                assemble, subsample, boundaries, row_boundary, reg_params,
                                fit_reg, fit_q, fit_md, parse_months, save_npz, log)
from cluster.exp63_collect import (require_exp62, load_meta, v6_branch, mixture,
                                   make_smoke_standins)

pd.set_option('display.width', 230)
pd.set_option('display.max_columns', 40)
SCREEN_MONTHS = (3, 6, 9, 12)
SCORED = (1, 7)
SCREEN_DIR = CACHE / 'exp63_screen'


# ----------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------
def p_base(d, m):
    return d / f'base__m{m:02d}.npz'


def p_nd(d, b, m):
    return d / f'nd-{b}__m{m:02d}.npz'


def p_q(d, b, r, m):
    return d / f'q-{b}-{r}__m{m:02d}.npz'


def unit_list(d, months, bnds, regs):
    out = []
    for m in months:
        out.append(p_base(d, m))
        for b in bnds:
            out.append(p_nd(d, b, m))
            out += [p_q(d, b, r, m) for r in regs]
    return out


def fit_units(d, months, bnds, regs, threads, budget, smoke, t0):
    todo = [p for p in unit_list(d, months, bnds, regs) if not p.exists()]
    if not todo:
        log(t0, 'all units present, nothing to fit')
        return
    log(t0, f'{len(todo)} unit(s) to fit, threads {threads}')
    ctx = assemble()
    if smoke:
        ctx = subsample(ctx)
    log(t0, f'features assembled: {ctx.X.shape}')
    y, FULL = ctx.y, ctx.FULL
    for m in months:                      # month-major: a crash leaves whole months
        val = ctx.month == m
        train = ~val
        idx = ctx.gidx[val]
        if not p_base(d, m).exists():
            ts = time.time()
            p, r = reg_params('screen', 0, threads, budget)
            pb2 = fit_reg(ctx, FULL, train & (y <= TAIL_CAP), val, p, r)
            save_npz(p_base(d, m), idx, {'pb2': pb2},
                     dict(month=m, seconds=time.time() - ts, peak_rss_gb=peak_rss_gb()))
            log(t0, f'{p_base(d, m).name} ({time.time()-ts:.0f}s)')
        for b in bnds:
            bmap = boundaries(y, ctx.apt, train, b)
            b_row = row_boundary(ctx.apt, bmap)
            if not p_nd(d, b, m).exists():
                ts = time.time()
                p, r = reg_params('screen', 0, threads, budget)
                mn = fit_reg(ctx, FULL, train & (y <= b_row), val, p, r)
                t_mn = time.time() - ts
                md, mdinfo = fit_md(ctx, FULL, b_row, train, val, m, 0, threads, budget)
                save_npz(p_nd(d, b, m), idx, {'mn': mn, 'md': md},
                         dict(month=m, boundary=bmap, m_d=mdinfo, seconds_mn=t_mn,
                              seconds=time.time() - ts, peak_rss_gb=peak_rss_gb()))
                log(t0, f'{p_nd(d, b, m).name}: m_n {t_mn:.0f}s, m_d rounds {mdinfo["best"]}'
                        f'->{mdinfo["used"]} on {mdinfo["n"]:,} rows ({time.time()-ts:.0f}s)')
            for rg in regs:
                if p_q(d, b, rg, m).exists():
                    continue
                ts = time.time()
                q, qinfo = fit_q(ctx, FULL, b_row, train, val, rg, 0, threads, budget)
                save_npz(p_q(d, b, rg, m), idx, {'q': q},
                         dict(month=m, boundary=bmap, q_info=qinfo,
                              seconds=time.time() - ts, peak_rss_gb=peak_rss_gb()))
                log(t0, f'{p_q(d, b, rg, m).name}: {qinfo} ({time.time()-ts:.0f}s)')
    del ctx


def read_unit(path, arrays, n):
    z = np.load(path)
    out = {}
    for k in arrays:
        v = np.full(n, np.nan)
        v[z['idx']] = z[k]
        out[k] = v
    return out, json.loads(str(z['meta'])), z['idx']


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def auc(lab, s):
    lab = np.asarray(lab, bool)
    n1 = int(lab.sum()); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strat_auc(lab, s, strata):
    """P(random positive outranks random negative FROM THE SAME STRATUM)."""
    num = den = 0.0
    for g in np.unique(strata):
        k = strata == g
        n1 = int(lab[k].sum()); n0 = int(k.sum()) - n1
        if n1 and n0:
            num += auc(lab[k], s[k]) * n1 * n0
            den += n1 * n0
    return num / den if den else np.nan


def deciles_within(values, groups):
    out = np.zeros(len(values), np.int64)
    for g in np.unique(groups):
        k = groups == g
        out[k] = np.minimum(9, (rankdata(values[k]) - 1) * 10 // k.sum()).astype(np.int64)
    return out


def calib_table(lab, q, bins=10):
    edges = np.unique(np.quantile(q, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, q, side='right') - 1, 0, max(0, len(edges) - 2))
    t = (pd.DataFrame({'b': b, 'q': q, 'y': lab.astype(float)}).groupby('b')
         .agg(n=('y', 'size'), mean_q=('q', 'mean'), rate=('y', 'mean')))
    ece = float((t.n / t.n.sum() * (t.mean_q - t.rate).abs()).sum())
    return t, ece


def clf_metrics(lab, q, ref, pb, aptcode):
    """aptcode: integer airport codes (index into AIRPORTS)."""
    qc = np.clip(q, 1e-6, 1 - 1e-6)
    brier = float(np.mean((q - lab) ** 2))
    brier_ref = float(np.mean((ref - lab) ** 2))
    strata = aptcode * 10 + deciles_within(pb, aptcode)
    return dict(n=len(lab), rate=float(lab.mean()), mean_q=float(q.mean()),
                AUC=auc(lab, q), AUC_in_apt=strat_auc(lab, q, aptcode),
                AUC_given_pb=strat_auc(lab, q, strata),
                AUC_pb_in_apt=strat_auc(lab, pb, aptcode),
                brier=brier, brier_apt_rate=brier_ref,
                BSS_vs_apt_rate=1 - brier / brier_ref if brier_ref > 0 else np.nan,
                logloss=float(-np.mean(lab * np.log(qc) + (1 - lab) * np.log(1 - qc))))


def boot(y, a, b, n_boot):
    return paired_bootstrap(y, a, b, n_boot=n_boot)


def fmt(r):
    return (f'{r["diff"]:+8.3f} +- {r["sd"]:6.3f} CI [{r["lo"]:+8.3f}, {r["hi"]:+8.3f}] '
            f'P(worse) {100*r["p_a_worse"]:5.1f}%')


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def report(d, exp62_dir, months, bnds, regs, spec_row, n_boot, out_prefix, t0):
    meta = load_meta()
    y, month, apt, inP = meta['y'], meta['month'], meta['apt'], meta['inP']
    n = len(y)
    pb, pt, et, v6 = v6_branch(exp62_dir, meta, spec_row)

    ok = [m for m in months if all(p.exists() for p in unit_list(d, [m], bnds, regs))]
    if not ok:
        sys.exit(f'FATAL: no month has a complete set of units under {d}; run --phase units')
    log(t0, f'report on months {ok} (missing: {sorted(set(months) - set(ok))})')

    pb2 = np.full(n, np.nan)
    mn, md, brow, ref = {}, {}, {}, {}
    q = {}
    for b in bnds:
        mn[b], md[b], brow[b], ref[b] = (np.full(n, np.nan) for _ in range(4))
    for m in ok:
        u, _, _ = read_unit(p_base(d, m), ['pb2'], n)
        k = np.isfinite(u['pb2']); pb2[k] = u['pb2'][k]
        tm = (month != m) & (y <= TAIL_CAP)
        for b in bnds:
            u, mt, idx = read_unit(p_nd(d, b, m), ['mn', 'md'], n)
            mn[b][idx], md[b][idx] = u['mn'][idx], u['md'][idx]
            bmap = mt['boundary']
            brow[b][idx] = row_boundary(apt[idx], bmap)
            # reference forecast for the Brier skill score: the airport's
            # training-month band rate under this fold's boundary
            for a in AIRPORTS:
                s = tm & (apt == a)
                ref[b][idx[apt[idx] == a]] = float((y[s] > bmap[a]).mean())
            for rg in regs:
                uq, mq, _ = read_unit(p_q(d, b, rg, m), ['q'], n)
                if mq['boundary'] != bmap:
                    sys.exit(f'FATAL: boundary mismatch between nd-{b} and q-{b}-{rg} for m{m}')
                q.setdefault((b, rg), np.full(n, np.nan))[idx] = uq['q'][idx]

    ev = np.isin(month, ok) & np.isfinite(pb2)
    for b in bnds:
        ev &= np.isfinite(mn[b]) & np.isfinite(md[b])
    for k in q:
        ev &= np.isfinite(q[k])
    scr = ev & np.isin(month, SCREEN_MONTHS)
    rest = ev & ~np.isin(month, SCREEN_MONTHS)
    base2 = mixture(pt, et, pb2)
    print(f'\nrows evaluated {int(ev.sum()):,} | screening months present '
          f'{sorted(set(month[scr].tolist()))} | other months {sorted(set(month[rest].tolist()))}')
    print(f'band rates by boundary (rows y <= 7200, outside P): ' + ', '.join(
        f'{b} {np.mean(y[ev & ~inP & (y <= TAIL_CAP)] > brow[b][ev & ~inP & (y <= TAIL_CAP)]):.4f}'
        for b in bnds))

    # ------------------------------------------------------------------ 1
    print('\n' + '=' * 118)
    print('1. DELAYED-BAND CLASSIFIER q, out of fold, rows OUTSIDE P with y <= 7200')
    print('   AUC_in_apt = within-airport AUC; AUC_given_pb = within airport x decile of the v6')
    print('   bulk prediction pb (discrimination pb does NOT already have); AUC_pb_in_apt = how')
    print('   well pb alone ranks the band; BSS = Brier skill vs the airport training-month rate')
    print('=' * 118)
    C = ev & ~inP & (y <= TAIL_CAP)
    acode = pd.Categorical(apt, categories=AIRPORTS).codes.astype(np.int64)
    rows, prow, cal = [], [], []
    for (b, rg), qq in q.items():
        lab = (y[C] > brow[b][C]).astype(float)
        r = clf_metrics(lab, qq[C], ref[b][C], pb[C], acode[C])
        t, ece = calib_table(lab, qq[C])
        r.update(option=f'{b}/{rg}', ECE=ece)
        rows.append(r)
        cal.append(t.assign(option=f'{b}/{rg}').reset_index())
        for a in AIRPORTS:
            s = apt[C] == a
            if s.sum() and 0 < lab[s].sum() < s.sum():
                ra = clf_metrics(lab[s], qq[C][s], ref[b][C][s], pb[C][s], acode[C][s])
                ra.update(option=f'{b}/{rg}', airport=a)
                prow.append(ra)
        for sm, lbl in ((SCORED, 'scored 1+7'),):
            s = np.isin(month[C], sm)
            if s.sum() and 0 < lab[s].sum() < s.sum():
                rs = clf_metrics(lab[s], qq[C][s], ref[b][C][s], pb[C][s], acode[C][s])
                rs.update(option=f'{b}/{rg}', airport=lbl)
                prow.append(rs)
    T = pd.DataFrame(rows).set_index('option')
    print(T[['n', 'rate', 'mean_q', 'AUC', 'AUC_in_apt', 'AUC_given_pb', 'AUC_pb_in_apt',
             'brier', 'BSS_vs_apt_rate', 'logloss', 'ECE']].round(4).to_string())
    PA = pd.DataFrame(prow)
    PA.to_csv(f'{out_prefix}_clf_airport.csv', index=False)
    T.to_csv(f'{out_prefix}_clf.csv')
    pd.concat(cal).to_csv(f'{out_prefix}_calib.csv', index=False)
    if len(PA):
        for col in ('AUC_in_apt', 'AUC_given_pb', 'BSS_vs_apt_rate'):
            print(f'\n  per airport: {col}')
            print(PA.pivot(index='airport', columns='option', values=col).round(3).to_string())

    # ------------------------------------------------------------------ 2
    print('\n' + '=' * 118)
    print('2. ORACLE CEILING (band membership known) vs the matched two-way baseline '
          '(same capacity) and vs v6 (exp62)')
    print('=' * 118)
    mix_rows = []
    scopes = [('all evaluated months', ev), ('scored 1+7', ev & np.isin(month, SCORED)),
              ('screening 3,6,9,12', scr), ('outside P', ev & ~inP), ('inside P', ev & inP),
              ('scored 1+7 outside P', ev & np.isin(month, SCORED) & ~inP)]
    for b in bnds:
        orc = mixture(pt, et, np.where(y > brow[b], md[b], mn[b]))
        for label, s in scopes:
            if not s.any():
                continue
            r2, ro, rv = rmse(y[s], base2[s]), rmse(y[s], orc[s]), rmse(y[s], v6[s])
            mix_rows.append(dict(kind='oracle', option=b, scope=label, n=int(s.sum()),
                                 rmse_base2=r2, rmse_arm=ro, rmse_v6=rv, diff_vs_base2=ro - r2))
            print(f'  {b:9s} {label:22s} n={int(s.sum()):>9,}  base2 {r2:8.2f}  oracle {ro:8.2f}'
                  f'  ({ro-r2:+8.2f})   v6 {rv:8.2f}  oracle-v6 {ro-rv:+8.2f}')

    # ------------------------------------------------------------------ 3
    print('\n' + '=' * 118)
    print('3. REALISTIC MIXTURE (q from the classifier) vs matched two-way baseline -- '
          'SCREENING MONTHS 3,6,9,12 ONLY')
    print(f'   paired bootstrap, {n_boot} resamples, arm minus baseline (negative = better)')
    print('=' * 118)
    real, gate = {}, {}
    for (b, rg), qq in q.items():
        real[(b, rg)] = mixture(pt, et, (1 - qq) * mn[b] + qq * md[b])
    if not scr.any():
        print('  NOT EVALUABLE: no screening month has complete units.')
    else:
        for (b, rg), pr in real.items():
            orc = mixture(pt, et, np.where(y > brow[b], md[b], mn[b]))
            r_all = boot(y[scr], pr[scr], base2[scr], n_boot)
            so = scr & ~inP
            r_out = boot(y[so], pr[so], base2[so], n_boot)
            r_v6 = boot(y[scr], pr[scr], v6[scr], n_boot)
            ceil = rmse(y[scr], base2[scr]) - rmse(y[scr], orc[scr])
            got = rmse(y[scr], base2[scr]) - rmse(y[scr], pr[scr])
            gate[(b, rg)] = r_all
            print(f'  {b:9s}/{rg:7s} all       {fmt(r_all)}   captured {100*got/ceil if ceil else np.nan:5.1f}% of oracle')
            print(f'  {"":17s} outside P {fmt(r_out)}')
            print(f'  {"":17s} vs v6     {fmt(r_v6)}   (capacity-mismatched: cheap components vs v6)')
            for tag, r, s in (('screen all', r_all, scr), ('screen outside P', r_out, so),
                              ('screen vs v6', r_v6, scr)):
                mix_rows.append(dict(kind='realistic', option=f'{b}/{rg}', scope=tag,
                                     n=int(s.sum()), diff_vs_base2=r['diff'], **r))

    # band trade on screening months: where the realistic mixture gains and loses
    if scr.any():
        print('\n  WHERE IT MOVES (screening months; bands by each option\'s own boundary):')
        print('  change in SSE vs baseline, as % of the baseline\'s total SSE; mean q per band')
        tot = ((y[scr] - base2[scr]) ** 2).sum()
        for (b, rg), pr in real.items():
            parts = []
            for lbl, bm in (('normal', y <= brow[b]), ('delayed', (y > brow[b]) & (y <= TAIL_CAP)),
                            ('artefact', y > TAIL_CAP)):
                s = scr & bm
                dsse = ((y[s] - pr[s]) ** 2).sum() - ((y[s] - base2[s]) ** 2).sum()
                mq = np.nanmean(q[(b, rg)][s]) if s.any() else np.nan
                parts.append(f'{lbl} {100*dsse/tot:+6.2f}% (q {mq:.3f})')
            print(f'  {b:9s}/{rg:7s} ' + ' | '.join(parts))
    pd.DataFrame(mix_rows).to_csv(f'{out_prefix}_mix.csv', index=False)

    # ------------------------------------------------------------------ 4
    print('\n' + '=' * 118)
    passing = {k: r for k, r in gate.items() if r['hi'] < 0}
    if gate:
        pool = passing or gate
        best = min(pool, key=lambda k: pool[k]['diff'])
    else:
        best = None
    if best is None:
        print('4. GATE: NOT EVALUABLE (no screening months). Run the screening months first.')
    else:
        b, rg = best
        print(f'4. GATE on screening months: {"PASS" if passing else "FAIL"} -- '
              f'{len(passing)} of {len(gate)} options have a CI entirely below zero.')
        print(f'   {"selected" if passing else "best (gate failed, reported only)"}: '
              f'{b}/{rg}  {fmt(gate[best])}')
        print('=' * 118)
        pr = real[best]
        print(f'   CONFIRMATION of {b}/{rg} on the months NOT used to choose it:')
        conf = {}
        for label, s in (('other months, all', rest), ('other months, outside P', rest & ~inP),
                         ('scored 1+7', ev & np.isin(month, SCORED)),
                         ('month 01', ev & (month == 1)), ('month 07', ev & (month == 7))):
            if s.sum() > 1:
                conf[label] = boot(y[s], pr[s], base2[s], n_boot)
                print(f'     {label:26s} {fmt(conf[label])}  n={int(s.sum()):,}')
        rc = conf.get('other months, all')
        if passing and (rc is None or rc['diff'] >= 0):
            print('   WARNING: the screening pass does NOT replicate on the other months. With '
                  f'{len(gate)} options screened on 4 months a false pass is plausible;')
            print('   do not start step 2 without discussing this result first.')
        print(f'\n   PER AIRPORT, {b}/{rg}, outside P (RMSE baseline -> realistic):')
        for a in AIRPORTS:
            parts = []
            for label, s in (('screen', scr), ('other', rest), ('scored', ev & np.isin(month, SCORED))):
                s = s & ~inP & (apt == a)
                if s.any():
                    parts.append(f'{label} {rmse(y[s], base2[s]):7.1f}->{rmse(y[s], pr[s]):7.1f}')
            print(f'     {a}  ' + '  '.join(parts))
        t = pd.concat(cal)
        print(f'\n   calibration of q for {b}/{rg} (deciles of q; mean q vs observed band rate):')
        print(t[t.option == f'{b}/{rg}'].drop(columns='option').round(4).to_string(index=False))
        print('\n' + '=' * 118)
        if passing:
            opt = f'{b}:{rg}'
            print('NEXT: step 2 for the selected option (A = mixture, B = stacked q, N = null control):')
            print(f'  cluster: sbatch --array=0-35%40 cluster/exp63_array.sbatch A:{opt} B:{opt} N')
            print(f'  laptop : /home/thinklex/prc2026-venv/bin/python experiments/exp63_cv.py '
                  f'--options A:{opt},B:{opt},N')
        else:
            print('STOP: no option clears the gate. exp63 is negative at screening; do not run step 2.')
            print('  Read section 1 (AUC_given_pb) and section 2 (oracle) to say WHY in the log.')
    print('=' * 118)


def main():
    ap = argparse.ArgumentParser(description='EXP63 step 1 screen')
    ap.add_argument('--months', default='1-12')
    ap.add_argument('--boundaries', default=','.join(BOUNDARIES))
    ap.add_argument('--regions', default='pooled,ns')
    ap.add_argument('--phase', choices=('all', 'units', 'report'), default='all')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--spec-row', type=int, default=0)
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--check-inputs', action='store_true')
    a = ap.parse_args()
    t0 = time.time()

    bnds = [b for b in a.boundaries.split(',') if b]
    regs = [r for r in a.regions.split(',') if r]
    for b in bnds:
        if b not in BOUNDARIES:
            sys.exit(f'unknown boundary {b}')
    for r in regs:
        if r not in REGIONS:
            sys.exit(f'unknown region {r}')

    if a.check_inputs:
        require_exp62(CACHE)
        for f in ('training.parquet', 'external/atfm_daily.csv'):
            if not (CACHE / f).exists():
                sys.exit(f'FATAL: cache/{f} missing')
        print('inputs OK: exp62 caches and training data present')
        return

    if a.smoke:
        months, n_boot, budget = list(SMOKE_MONTHS), 50, BUDGET['smoke']
        d, exp62_dir = SMOKE_DIR / 'screen', SMOKE_DIR / 'exp62_standin'
        out_prefix, threads = str(SMOKE_DIR / 'exp63_screen'), 4
        if a.phase in ('all', 'units'):
            import shutil                 # a smoke must refit, never skip stale units
            shutil.rmtree(d, ignore_errors=True)
    else:
        months, n_boot, budget = parse_months(a.months), a.n_boot, BUDGET['full']
        d, exp62_dir = SCREEN_DIR, CACHE
        out_prefix, threads = str(CACHE / 'exp63_screen'), a.threads
        if a.phase in ('all', 'report'):
            require_exp62(exp62_dir)      # fail before hours of fitting, not after
    d.mkdir(parents=True, exist_ok=True)

    if a.phase in ('all', 'units'):
        fit_units(d, months, bnds, regs, threads, budget, a.smoke, t0)
    if a.phase in ('all', 'report'):
        if a.smoke and not (exp62_dir / 'exp62_spec_P.npy').exists():
            make_smoke_standins(exp62_dir, load_meta())
        report(d, exp62_dir, months, bnds, regs, a.spec_row, n_boot, out_prefix, t0)
    log(t0, f'done; peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
