"""routed_0926 step 2: the 2026 side of the route-D corrector question, label-free, plus the labelled 2025 analogue.

No 2026 label exists or is used. The 2026 evidence is the flight's own adsb.lol trajectory (features_ranking.parquet)
and, for E2, the NM off-block proxies (MVT - AOBT_3) of route-A flights on the same airport-day -- inputs, not labels.

======================================================================================================================
PRE-REGISTERED (written 2026-09-26 together with r01_cv.py, BEFORE any candidate was scored)
----------------------------------------------------------------------------------------------------------------------
Candidates and scopes as in r01_cv.py: a shipped (v17), b drop, c half, d refit; S1 route D outside P, S2 zone P & R.
"Covered" row: ADS-B off-block present, 0 < adsb_taxi < 7200 (adsb_taxi = adsb_off_best = MVT - ADS-B off-block).
ADS-B pattern cell: airport x source (rest / push) x time at rest before moving (adsb_stat_s: na, <120, 120-600,
  600-1200, >=1200 s) x first sighting far from the stand (adsb_d_stand_first: na, <=300 m, >300 m) [x season].
Implied labels:
  E1  adsb_taxi + mean(y - adsb_taxi) over 2025 covered rows (all routes, y <= 7200) in the same pattern cell and
      season (winter = Dec/Jan/Feb); fallbacks when a cell has < 30 rows: drop season -> (airport, source) -> source.
      For 2025 rows the offsets are leave-one-month-out.
  E2  adsb_taxi + mean(off_aobt3 - adsb_taxi) over ROUTE-A covered flights of the SAME airport and local date and
      pattern (>= 5 rows; else airport-date-source >= 5; else missing) + the 2025 route-A proxy-to-label correction
      mean(y - off_aobt3) in the (airport, source, rest bin, far) cell (>= 30 rows, else airport-source; LOMO in 2025).
      E2 measures the ADS-B -> off-block offset on the same day, i.e. inside the 2026 event itself.
R2 (per scope S, candidate X; all four needed):
  R2a pseudo-MSE_E1(X) < pseudo-MSE_E1(a) on the covered 2026 corrector rows of S
  R2b pseudo-MSE_E1(X) <= pseudo-MSE_E1(a) on the covered EHAM-January 2026 rows of S (skipped if < 20 rows)
  R2c validity gate: on the covered 2025 corrector rows of S, the E1 (LOMO) pseudo-delta of X vs a has the same sign
      as the TRUE-label delta
  R2d on EHAM January, E2's pseudo-delta (X vs a) has the same sign as E1's (skipped if E2 covers < 20 rows)
Final decision = R1 (r01_cv.py) AND R2; ties between passing candidates -> lowest 2026 pseudo-MSE_E1 on S.
======================================================================================================================

Also reported (not in the rule): the labelled 2025 analogue -- corrector rows with the de-icing / remote-hold pattern
(rest source, > 600 s at rest, first seen > 300 m from the stand; drivers_0926/d08 definition), per scope, season and
airport -- mean residual and RMSE per candidate; and the same pattern's 2026 rows with each candidate's mean served
value next to E1 / E2.

    python experiments/routed_0926/r02_2026.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, AIRPORTS, ID
from prc.io import peak_rss_gb

OUT = CACHE / 'routed_0926'
APTS = sorted(AIRPORTS)
ACOLS = ['adsb_off_best', 'adsb_best_src', 'adsb_stat_s', 'adsb_d_stand_first', 'adsb_stop_s', 'adsb_hold_s']
pd.set_option('display.width', 250)


def adsb(which, ids):
    a = pd.read_parquet(CACHE / 'adsb_lol' / f'features_{which}.parquet', columns=[ID] + ACOLS).set_index(ID)
    if not a.index.is_unique:
        sys.exit(f'FATAL: duplicate ids in features_{which}')
    return a.reindex(ids)


def codes(apt, month, a):
    taxi = a.adsb_off_best.to_numpy(float)
    cov = np.isfinite(taxi) & (taxi > 0) & (taxi < 7200)
    src = np.where(a.adsb_best_src.to_numpy(float) == 2, 1, 0)
    s = a.adsb_stat_s.to_numpy(float)
    stat = np.select([~np.isfinite(s), s < 120, s < 600, s < 1200], [0, 1, 2, 3], 4)
    d = a.adsb_d_stand_first.to_numpy(float)
    far = np.select([~np.isfinite(d), d <= 300], [0, 1], 2)
    ac = pd.Categorical(apt, categories=APTS).codes.astype(np.int64)
    win = np.isin(month, [12, 1, 2]).astype(np.int64)
    rr = (a.adsb_best_src.to_numpy(float) == 1) & (s > 600) & (d > 300)
    return dict(taxi=taxi, cov=cov, apt=ac, src=src, stat=stat, far=far, win=win, rr=rr)


def key(c, lv, date=None):
    k = np.zeros(len(c['apt']), np.int64)
    for f in lv:
        v = date if f == 'date' else c[f]
        k = k * (100000000 if f == 'date' else 16) + v
    return k


def hier_mean(train_mask, target, c, levels, min_n, query_mask, month=None, lomo=False, date=None, c_q=None, date_q=None):
    """Cell mean of `target` over train rows, looked up for query rows with fallbacks; LOMO if asked (same frame)."""
    c_q = c if c_q is None else c_q; date_q = date if date_q is None else date_q
    out = np.full(len(c_q['apt']), np.nan)
    for lv in levels:
        kt = key(c, lv, date)[train_mask]; kq = key(c_q, lv, date_q)
        uk, inv = np.unique(kt, return_inverse=True)
        pos = np.searchsorted(uk, kq); pos = np.clip(pos, 0, len(uk) - 1); hit = uk[pos] == kq
        if lomo:
            mt = month[train_mask].astype(np.int64)
            S = np.bincount(inv * 13 + mt, weights=target[train_mask], minlength=len(uk) * 13).reshape(len(uk), 13)
            N = np.bincount(inv * 13 + mt, minlength=len(uk) * 13).reshape(len(uk), 13).astype(float)
            St, Nt = S.sum(1), N.sum(1); mq = month.astype(np.int64)
            s = St[pos] - S[pos, mq]; n = Nt[pos] - N[pos, mq]
        else:
            s = np.bincount(inv, weights=target[train_mask], minlength=len(uk))[pos]
            n = np.bincount(inv, minlength=len(uk))[pos].astype(float)
        ok = hit & (n >= min_n) & np.isnan(out) & query_mask
        out[ok] = s[ok] / n[ok]
    return out


def main():
    t0 = time.time()
    z5 = np.load(OUT / 'arrays_2025.npz', allow_pickle=True); z6 = np.load(OUT / 'arrays_2026.npz', allow_pickle=True)
    cn5 = np.load(OUT / 'r01_oof.npz')['corr_new']; r6 = np.load(OUT / 'refit_2026.npz', allow_pickle=True)
    if not np.array_equal(r6['ids'], z6['ids']):
        sys.exit('FATAL: refit_2026 order')
    cn6 = r6['corr_new']
    d5 = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=[ID, 'off_aobt3', 'y'])
    d6 = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=[ID, 'off_aobt3'])
    if not np.allclose(d5.y.to_numpy(), z5['y']) or not np.array_equal(d6[ID].to_numpy(), z6['ids']):
        sys.exit('FATAL: order mismatch')
    date6 = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_ranking.parquet', columns=['date_loc']).date_loc.to_numpy().astype(np.int64)
    y, m5 = z5['y'], z5['month']; m6 = z6['month']
    c5 = codes(z5['apt'], m5, adsb('training', d5[ID].to_numpy()))
    c6 = codes(z6['apt'], m6, adsb('ranking', d6[ID].to_numpy()))
    date5 = z5['date_loc'].astype(np.int64)
    print(f'coverage (0 < adsb_taxi < 7200): 2025 all {c5["cov"].mean():.3f}, corrector rows {c5["cov"][z5["R"]].mean():.3f}; '
          f'2026 all {c6["cov"].mean():.3f}, corrector rows {c6["cov"][z6["R"]].mean():.3f}')

    # ---- implied labels ---------------------------------------------------------------------------------------------
    L1 = [('apt', 'src', 'stat', 'far', 'win'), ('apt', 'src', 'stat', 'far'), ('apt', 'src'), ('src',)]
    tr5 = c5['cov'] & (y <= 7200); tgt5 = np.where(c5['cov'], y - np.nan_to_num(c5['taxi']), 0.0)
    E1_5 = c5['taxi'] + hier_mean(tr5, tgt5, c5, L1, 30, c5['cov'], month=m5, lomo=True)
    E1_6 = c6['taxi'] + hier_mean(tr5, tgt5, c5, L1, 30, c6['cov'], c_q=c6)
    oa5, oa6 = d5.off_aobt3.to_numpy(float), d6.off_aobt3.to_numpy(float)
    A5 = c5['cov'] & np.isfinite(oa5) & (oa5 > 0) & (oa5 <= 7200); A6 = c6['cov'] & np.isfinite(oa6) & (oa6 > 0) & (oa6 <= 7200)
    L2 = [('apt', 'date', 'src', 'stat', 'far'), ('apt', 'date', 'src')]
    Lk = [('apt', 'src', 'stat', 'far'), ('apt', 'src')]
    k5 = hier_mean(A5 & (y <= 7200), np.where(A5, y - np.nan_to_num(oa5), 0.0), c5, Lk, 30, c5['cov'], month=m5, lomo=True)
    k6 = hier_mean(A5 & (y <= 7200), np.where(A5, y - np.nan_to_num(oa5), 0.0), c5, Lk, 30, c6['cov'], c_q=c6)
    off5 = hier_mean(A5, np.where(A5, oa5 - np.nan_to_num(c5['taxi']), 0.0), c5, L2, 5, c5['cov'], date=date5)
    off6 = hier_mean(A6, np.where(A6, oa6 - np.nan_to_num(c6['taxi']), 0.0), c6, L2, 5, c6['cov'], date=date6)
    E2_5 = c5['taxi'] + off5 + k5; E2_6 = c6['taxi'] + off6 + k6

    # ---- candidates ---------------------------------------------------------------------------------------------
    def cands(z, cnew, S):
        base, co, v17 = z['base'], z['corr_old'], z['v17']
        out = {'a shipped': v17}
        for nm, c in (('b drop', 0 * co), ('c half', 0.5 * co), ('d refit', cnew)):
            f = v17.copy(); f[S] = np.clip(base[S] + c[S], 0, None); out[nm] = f
        return out

    names = ['a shipped', 'b drop', 'c half', 'd refit']
    verdict = {}
    for sname, S5, S6 in (('S1', z5['R'] & ~z5['P'], z6['R'] & ~z6['P']), ('S2', z5['R'] & z5['P'], z6['R'] & z6['P'])):
        F5 = cands(z5, cn5, S5); F6 = cands(z6, cn6, S6)
        print(f'\n===================== scope {sname}: 2025 rows {S5.sum():,}, 2026 rows {S6.sum():,} =====================')
        # --- validity of the implied labels on 2025 (true labels available)
        v = S5 & c5['cov'] & np.isfinite(E1_5)
        e = E1_5[v] - y[v]
        print(f'E1 on 2025 covered corrector rows ({v.sum():,}): mean(E1 - y) {e.mean():+.0f}, median {np.median(e):+.0f}, '
              f'RMSE {np.sqrt((e ** 2).mean()):.0f} (a: RMSE {np.sqrt(((F5["a shipped"] - y)[v] ** 2).mean()):.0f}); '
              f'corr(E1 error, corr_old) {np.corrcoef(e, z5["corr_old"][v])[0, 1]:+.3f}')
        v2 = S5 & c5['cov'] & np.isfinite(E2_5)
        e2 = E2_5[v2] - y[v2]
        print(f'E2 on 2025 covered corrector rows ({v2.sum():,}): mean(E2 - y) {e2.mean():+.0f}, median {np.median(e2):+.0f}, '
              f'RMSE {np.sqrt((e2 ** 2).mean()):.0f}; corr(E2 error, corr_old) {np.corrcoef(e2, z5["corr_old"][v2])[0, 1]:+.3f}')
        print(f"  {'2025 validity (covered rows)':34s}{'true dRMSE':>11s}{'E1 pseudo':>11s}{'E2 pseudo':>11s}  sign ok (E1)")
        gate = {}
        for x in names[1:]:
            tr = np.sqrt(((y - F5[x])[v] ** 2).mean()) - np.sqrt(((y - F5['a shipped'])[v] ** 2).mean())
            p1 = np.sqrt(((E1_5 - F5[x])[v] ** 2).mean()) - np.sqrt(((E1_5 - F5['a shipped'])[v] ** 2).mean())
            p2 = np.sqrt(((E2_5 - F5[x])[v2] ** 2).mean()) - np.sqrt(((E2_5 - F5['a shipped'])[v2] ** 2).mean())
            gate[x] = np.sign(tr) == np.sign(p1)
            print(f'  {x:34s}{tr:+11.1f}{p1:+11.1f}{p2:+11.1f}  {gate[x]}')
        # winter EHAM 2025 subset (the closest labelled analogue), information only
        vw = v & (z5['apt'] == 'EHAM') & np.isin(m5, [12, 1, 2])
        if vw.sum() >= 20:
            print(f'  2025 EHAM winter covered corrector rows ({vw.sum()}): ' + '  '.join(
                f'{x}: true {np.sqrt(((y - F5[x])[vw] ** 2).mean()):.0f} / E1 {np.sqrt(((E1_5 - F5[x])[vw] ** 2).mean()):.0f}' for x in names))

        # --- 2026: pseudo-MSE per candidate
        w = S6 & c6['cov'] & np.isfinite(E1_6)
        ej = w & (z6['apt'] == 'EHAM') & (m6 == 1)
        w2 = S6 & c6['cov'] & np.isfinite(E2_6); ej2 = w2 & (z6['apt'] == 'EHAM') & (m6 == 1)
        print(f'\n2026 covered corrector rows: {w.sum():,} of {S6.sum():,} (E1); EHAM-Jan {ej.sum()}; E2 {w2.sum():,}, EHAM-Jan {ej2.sum()}')
        pm = lambda L, F, k: np.sqrt(((L - F) ** 2)[k].mean()) if k.sum() else np.nan
        print(f"  {'cand':10s}{'E1 RMSE all':>12s}{'E1 EHAM-Jan':>12s}{'E2 all':>9s}{'E2 EHAM-Jan':>12s}{'mean shift vs b (EHAM-Jan)':>28s}")
        res = {}
        for x in names:
            res[x] = (pm(E1_6, F6[x], w), pm(E1_6, F6[x], ej), pm(E2_6, F6[x], w2), pm(E2_6, F6[x], ej2))
            print(f'  {x:10s}{res[x][0]:12.1f}{res[x][1]:12.1f}{res[x][2]:9.1f}{res[x][3]:12.1f}'
                  f'{(F6[x] - F6["b drop"])[ej].mean() if ej.sum() else np.nan:+28.0f}')
        # per airport-month cells: implied gap vs corrector shift
        rows = []
        for a_ in APTS:
            for mm in (1, 7):
                k = w & (z6['apt'] == a_) & (m6 == mm)
                if k.sum() < 10:
                    continue
                k2 = k & np.isfinite(E2_6)
                rows.append(dict(cell=f'{a_}-{mm}', n_cov=int(k.sum()), n_rows=int((S6 & (z6['apt'] == a_) & (m6 == mm)).sum()),
                                 base=F6['b drop'][k].mean(), E1=E1_6[k].mean(), E2=E2_6[k2].mean() if k2.sum() >= 10 else np.nan,
                                 gap_E1=(E1_6 - F6['b drop'])[k].mean(), corr_old=z6['corr_old'][k].mean(), corr_new=cn6[k].mean(),
                                 **{f'dMSE_{x[0]}': ((E1_6 - F6[x]) ** 2 - (E1_6 - F6['a shipped']) ** 2)[k].sum() / 1e6 for x in names[1:]}))
        t = pd.DataFrame(rows).set_index('cell')
        print('\n2026 per airport-month (covered rows; means in s; dMSE_x = pseudo SSE change vs a under E1, 1e6 s^2):')
        print(t.round(1).to_string())
        t.to_csv(OUT / f'r02_cells_{sname}.csv')
        # --- R2 verdict
        verdict[sname] = {}
        for x in names[1:]:
            r2a = res[x][0] < res['a shipped'][0]
            r2b = True if ej.sum() < 20 else res[x][1] <= res['a shipped'][1]
            r2c = bool(gate[x])
            if ej2.sum() < 20 or ej.sum() < 20:
                r2d = True
            else:
                r2d = np.sign(res[x][3] - res['a shipped'][3]) == np.sign(res[x][1] - res['a shipped'][1])
            verdict[sname][x] = (r2a, r2b, r2c, r2d)
            print(f'  R2 {sname} {x}: {"PASS" if all(verdict[sname][x]) else "FAIL"}  (R2a {r2a}, R2b {r2b}, R2c {r2c}, R2d {r2d})')

        # --- labelled 2025 analogue: the remote-rest pattern
        print('\n2025 corrector rows with the remote-rest pattern (labelled): mean residual y - pred and RMSE, per candidate')
        for nm, k in (('all', S5 & c5['rr']), ('winter', S5 & c5['rr'] & np.isin(m5, [12, 1, 2])),
                      ('EHAM', S5 & c5['rr'] & (z5['apt'] == 'EHAM')), ('EHAM winter', S5 & c5['rr'] & (z5['apt'] == 'EHAM') & np.isin(m5, [12, 1, 2])),
                      ('not rr, covered', S5 & c5['cov'] & ~c5['rr'])):
            if k.sum() < 5:
                print(f'  {nm:16s} n={k.sum()}'); continue
            print(f'  {nm:16s} n={k.sum():5d} mean y {y[k].mean():6.0f} adsb_taxi {np.nanmean(c5["taxi"][k]):6.0f} | ' + ' '.join(
                f'{x[0]}: {np.mean(y[k] - F5[x][k]):+5.0f}/{np.sqrt(np.mean((y[k] - F5[x][k]) ** 2)):5.0f}' for x in names))
        print('2026 rows with the same pattern: mean served per candidate vs implied')
        for nm, k in (('all', S6 & c6['rr']), ('EHAM Jan', S6 & c6['rr'] & (z6['apt'] == 'EHAM') & (m6 == 1)),
                      ('other Jan', S6 & c6['rr'] & (z6['apt'] != 'EHAM') & (m6 == 1)), ('Jul', S6 & c6['rr'] & (m6 == 7))):
            if k.sum() < 5:
                print(f'  {nm:10s} n={k.sum()}'); continue
            kk2 = k & np.isfinite(E2_6)
            print(f'  {nm:10s} n={k.sum():4d} adsb_taxi {np.nanmean(c6["taxi"][k]):5.0f} E1 {np.nanmean(E1_6[k]):5.0f} '
                  f'E2 {np.nanmean(E2_6[kk2]) if kk2.sum() else np.nan:5.0f} (n {kk2.sum()}) | ' +
                  ' '.join(f'{x[0]} {F6[x][k].mean():5.0f}' for x in names) +
                  f' | median v17 - adsb_taxi {np.median(F6["a shipped"][k] - c6["taxi"][k]):+.0f}, median corr_old {np.median(z6["corr_old"][k]):+.0f}')

    # the review's figure: EHAM January zone-P rows with an ADS-B off-block (363 in the review)
    k = z6['P'] & (z6['apt'] == 'EHAM') & (m6 == 1) & np.isfinite(c6['taxi'])
    print(f'\nreview check: EHAM Jan zone-P rows with an ADS-B off-block: {k.sum()}; median v17 - adsb_taxi '
          f'{np.median(z6["v17"][k] - c6["taxi"][k]):+.0f}; median v17 {np.median(z6["v17"][k]):.0f}; median corr_old {np.median(z6["corr_old"][k]):+.0f}; '
          f'corrector active on {int((z6["R"] & k).sum())} of them')
    np.savez(OUT / 'r02_implied.npz', E1_5=E1_5, E2_5=E2_5, E1_6=E1_6, E2_6=E2_6)
    print(f'\npeak RSS {peak_rss_gb():.2f} GB | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
