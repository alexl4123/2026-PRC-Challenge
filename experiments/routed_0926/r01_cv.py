"""routed_0926 step 1: should the v10 route-D corrector be kept, shrunk, dropped or re-fitted on the v17 base? 2025 CV side.

======================================================================================================================
PRE-REGISTERED (written 2026-09-26 BEFORE any candidate below was scored; r00 only rebuilt and identity-checked arrays)
----------------------------------------------------------------------------------------------------------------------
Known beforehand (so this is not a blind test): in W2-era CV, route D without the corrector was better than with it
(D0 -1.03 vs D1 -0.70 against v14, 09-25 dev log), and the corrector's live value (-5.0 s at v10) came from the
January-2026 Amsterdam disruption, which month-wise 2025 CV cannot see. 2025 CV therefore can NOT justify dropping it
on its own; the 2026 side (r02_2026.py, label-free) carries the decisive evidence.

Candidates, on the rows the corrector acts on (R = m11d mask: route D & (outside P  or  exp62 spec p < 0.1)):
  a  shipped   v17 structure: base + corr_old                                  (COMPARATOR for every test)
  b  drop      base
  c  half      base + 0.5 * corr_old
  d  refit     base + corr_new; corr_new = the m11d design unchanged (LightGBM, 15 leaves, 300 rounds, same params,
               features [base, off_sched, APT, UTC hour, hpT, hpT75, hpTn, arrT, dshT, hpT_rel, hpT75_rel, arrT_rel],
               residual target y - base, fit rows R & y <= 7200) with the CURRENT base in place of v9;
               12-fold leave-one-month-out: month m is corrected by a model fitted on the other 11 months only.
               (2026: fitted on all 12 months.)
  base = v17 without the corrector (route D outside P: adsb2 blend; zone P: v17's P branch). All clipped at 0.
Scopes, decided SEPARATELY (the other scope keeps the shipped corrector):
  S1  route D outside P (17,526 rows in 2025)   -- primary, the question asked
  S2  zone P rows where the corrector acts (4,316 rows in 2025; this is where the EHAM Jan-2026 de-icing rows sit)
Metrics: all-flights RMSE delta (candidate - a), uncapped, on 12 months, scored months {1,7}, January, July, and per
  airport; 95% intervals from (i) the project's paired ROW bootstrap (500 resamples, seed 0, same resampling as
  prc.validation.paired_bootstrap) and (ii) an AIRPORT-DAY BLOCK bootstrap (2,000 resamples of airport x local-date
  blocks). Months better x/12 reported.

RULE (per scope; candidate X in {b, c, d}):
  R1 (2025 CV, necessary)   Delta(X - a) <= 0 on January AND on July AND on 12 months (point estimates, all flights).
  R2 (2026 label-free, necessary; defined and computed in r02_2026.py): on the 2026 corrector rows of the scope that
      have an ADS-B off-block, X is closer than a to the ADS-B-implied label (pseudo-MSE lower on the scope AND not
      higher on the EHAM-January cell), the implied label passes its 2025 validity gate, and the independent same-day
      route-A calibration (E2) agrees in sign on EHAM January.
  Change the scope from a to X only if X passes R1 AND R2. If several pass, take the one with the lowest 2026
  pseudo-MSE on the scope. Otherwise keep a (no v18 change for that scope). "b drop" is never adopted on R1 alone.
======================================================================================================================

Also (not part of the rule): a DESIGN CHECK that re-running this implementation with v9~ as the base reproduces the
shipped cache/major/m11d_oof_trailD.npy, so "refit" really is the same design.

Outputs: cache/routed_0926/r01_cv.csv (every delta), r01_oof.npz (corr_new 2025), refit_2026.npz (corr_new 2026)

    python experiments/routed_0926/r01_cv.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE, AIRPORTS
from prc.io import peak_rss_gb

OUT = CACHE / 'routed_0926'
STATE = ['hpT', 'hpT75', 'hpTn', 'arrT', 'dshT', 'hpT_rel', 'hpT75_rel', 'arrT_rel']
FEATS = ['v9', 'off_sched', 'APT', 'hour'] + STATE          # m11d ARMS['trail_D']; 'v9' = the base being corrected
PAR = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20,
           num_threads=4, verbose=-1, seed=0)
NR = 300
APTS = sorted(AIRPORTS)
N_ROW, N_BLOCK = 500, 2000


def frame(base, off_sched, apt, hour, st):
    X = pd.DataFrame({'v9': base, 'off_sched': off_sched, 'APT': pd.Categorical(apt, categories=APTS), 'hour': hour})
    for c in STATE:
        X[c] = st[c].to_numpy()
    return X


def lomo(X, y, base, month, R):
    fitm = R & (y <= 7200); r = y - base; c = np.zeros(len(y))
    for m in range(1, 13):
        tr = fitm & (month != m); te = R & (month == m)
        c[te] = lgb.train(PAR, lgb.Dataset(X.loc[tr, FEATS], r[tr]), NR).predict(X.loc[te, FEATS])
    return c


def row_boot(e2a, e2x, seed=0):
    """Same resampling as prc.validation.paired_bootstrap(y, x, a, 500, 0), shared across candidates."""
    rng = np.random.default_rng(seed); n = len(e2a); out = np.empty((N_ROW, len(e2x)))
    for i in range(N_ROW):
        idx = rng.integers(0, n, n); ra = np.sqrt(e2a[idx].mean())
        for j, e in enumerate(e2x):
            out[i, j] = np.sqrt(e[idx].mean()) - ra
    return out


def block_boot(key, e2a, e2x, seed=1):
    codes, inv = np.unique(key, return_inverse=True); B = len(codes)
    nb = np.bincount(inv, minlength=B).astype(float)
    sa = np.bincount(inv, weights=e2a, minlength=B)
    sx = [np.bincount(inv, weights=e, minlength=B) for e in e2x]
    rng = np.random.default_rng(seed); out = np.empty((N_BLOCK, len(e2x)))
    for i in range(N_BLOCK):
        w = np.bincount(rng.integers(0, B, B), minlength=B)
        n = w @ nb; ra = np.sqrt((w @ sa) / n)
        for j, s in enumerate(sx):
            out[i, j] = np.sqrt((w @ s) / n) - ra
    return out, B


def main():
    t0 = time.time()
    z = np.load(OUT / 'arrays_2025.npz', allow_pickle=True)
    y, month, P, D, R, base, corr_old, v17 = (z[k] for k in ('y', 'month', 'P', 'D', 'R', 'base', 'corr_old', 'v17'))
    apt, hour, off_sched, date_loc = z['apt'], z['hour'], z['off_sched'], z['date_loc']
    st = pd.read_parquet(CACHE / 'major' / 'm11_state_training.parquet', columns=STATE)
    n = len(y)

    # ---- design check: this implementation on v9~ must reproduce the shipped corrector ----------------------------
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    c9 = lomo(frame(v9, off_sched, apt, hour, st), y, v9, month, R)
    shipped = np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float) - v9
    dd = np.abs(c9[R] - shipped[R])
    print(f'DESIGN CHECK (v9~ base): |re-run - shipped m11d corrector| on R: max {dd.max():.3f} s, mean {dd.mean():.4f} s, '
          f'corr {np.corrcoef(c9[R], shipped[R])[0, 1]:.5f}')

    # ---- refit on the current base ---------------------------------------------------------------------------------
    corr_new = lomo(frame(base, off_sched, apt, hour, st), y, base, month, R)
    np.savez(OUT / 'r01_oof.npz', corr_new=corr_new)
    for nm, k in (('S1 D outside P', R & ~P), ('S2 P & R', R & P)):
        print(f'{nm}: rows {k.sum():,} | corr_old mean {corr_old[k].mean():+.1f} sd {corr_old[k].std():.1f} | corr_new mean '
              f'{corr_new[k].mean():+.1f} sd {corr_new[k].std():.1f} | corr(old,new) {np.corrcoef(corr_old[k], corr_new[k])[0, 1]:.3f}')
    fr = frame(base, off_sched, apt, hour, st); fitm = R & (y <= 7200)
    bst = lgb.train(PAR, lgb.Dataset(fr.loc[fitm, FEATS], (y - base)[fitm]), NR)
    r6 = np.load(OUT / 'arrays_2026.npz', allow_pickle=True)
    st6 = pd.read_parquet(CACHE / 'major' / 'm11_state_ranking.parquet', columns=STATE)
    X6 = frame(r6['base'], r6['off_sched'], r6['apt'], r6['hour'], st6)
    c6 = np.zeros(len(r6['base'])); R6 = r6['R']
    c6[R6] = bst.predict(X6.loc[R6, FEATS])
    np.savez(OUT / 'refit_2026.npz', ids=r6['ids'], corr_new=c6)
    print(f'2026 refit corrector on R ({R6.sum():,} rows): mean {c6[R6].mean():+.1f}, sd {c6[R6].std():.1f}; '
          f'old corrector mean {r6["corr_old"][R6].mean():+.1f}, sd {r6["corr_old"][R6].std():.1f}')

    # ---- candidates per scope --------------------------------------------------------------------------------------
    key = np.char.add(apt.astype(str), date_loc.astype(str))
    rows = []
    for sname, S in (('S1', R & ~P), ('S2', R & P)):
        cand = {}
        for cn, c in (('b drop', 0 * corr_old), ('c half', 0.5 * corr_old), ('d refit', corr_new)):
            f = v17.copy(); f[S] = np.clip(base[S] + c[S], 0, None); cand[cn] = f
        names = list(cand)
        e2a = (y - v17) ** 2; e2x = [(y - cand[c]) ** 2 for c in names]
        print(f'\n===== scope {sname} ({S.sum():,} rows in 2025): all-flights RMSE delta vs a (shipped) =====')
        print(f"  {'scope':14s}{'cand':9s}{'delta':>8s}{'row CI':>20s}{'block CI':>20s}{'blocks':>8s}{'scope-row RMSE a->X':>24s}")
        scopes = [('12m', np.ones(n, bool)), ('scored', np.isin(month, [1, 7])), ('Jan', month == 1), ('Jul', month == 7)]
        scopes += [(f'{a} 12m', apt == a) for a in APTS] + [(f'{a} scored', (apt == a) & np.isin(month, [1, 7])) for a in APTS]
        for scn, k in scopes:
            ra = np.sqrt(e2a[k].mean())
            rb = row_boot(e2a[k], [e[k] for e in e2x])
            bb, B = block_boot(key[k], e2a[k], [e[k] for e in e2x])
            for j, c in enumerate(names):
                dlt = np.sqrt(e2x[j][k].mean()) - ra
                ks = k & S
                sa = np.sqrt(e2a[ks].mean()) if ks.any() else np.nan; sx = np.sqrt(e2x[j][ks].mean()) if ks.any() else np.nan
                rlo, rhi = np.percentile(rb[:, j], [2.5, 97.5]); blo, bhi = np.percentile(bb[:, j], [2.5, 97.5])
                rows.append(dict(scope_rows=sname, scope=scn, cand=c, delta=dlt, row_lo=rlo, row_hi=rhi, block_lo=blo,
                                 block_hi=bhi, n_blocks=B, n_scope_rows=int(ks.sum()), rmse_scope_a=sa, rmse_scope_x=sx))
                print(f'  {scn:14s}{c:9s}{dlt:+8.3f}   [{rlo:+.3f},{rhi:+.3f}]   [{blo:+.3f},{bhi:+.3f}]{B:8d}'
                      f'{ks.sum():8d} {sa:7.0f} -> {sx:5.0f}')
        mb = {c: sum(np.sqrt(e2x[j][month == m].mean()) < np.sqrt(e2a[month == m].mean()) for m in range(1, 13)) for j, c in enumerate(names)}
        pm = {c: ' '.join(f'{m}:{np.sqrt(e2x[j][month == m].mean()) - np.sqrt(e2a[month == m].mean()):+.2f}' for m in range(1, 13))
              for j, c in enumerate(names)}
        for c in names:
            print(f'  months better {c}: {mb[c]}/12 | {pm[c]}')
        # R1 verdict
        df = pd.DataFrame([r for r in rows if r['scope_rows'] == sname]).set_index(['scope', 'cand'])
        for c in names:
            ok = all(df.loc[(s, c), 'delta'] <= 0 for s in ('12m', 'Jan', 'Jul'))
            print(f'  R1 {sname} {c}: {"PASS" if ok else "FAIL"} (12m {df.loc[("12m", c), "delta"]:+.3f}, Jan {df.loc[("Jan", c), "delta"]:+.3f}, '
                  f'Jul {df.loc[("Jul", c), "delta"]:+.3f})')
    pd.DataFrame(rows).to_csv(OUT / 'r01_cv.csv', index=False)
    print(f'\npeak RSS {peak_rss_gb():.2f} GB | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
