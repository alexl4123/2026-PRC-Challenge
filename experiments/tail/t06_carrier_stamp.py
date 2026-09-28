"""Carrier stamping propensity on route D (no NM record), from a participant's hint (Discord 2026-09-28: "the echo of
the planned time in the off-block is mostly a carrier thing when there's no NM match, and it's pretty stable month to
month"). Two pre-registered tests in the v24 structure (experiments/tail/v24_oof.py), all flights, 2025.

CARRIER = the 3-letter airline prefix of FLIGHT_mvt (the flight-plan operator field is empty on every route-D row).
STAMP   = on route D with take-off > 30 min after schedule (off_sched > 1800 s), the block was written at the schedule:
          |BLOCK - SCHED| <= 120 s  (= |off_sched - y| <= 120).
Features (label-derived from 2025 TRAINING rows only, month-OUT-OF-FOLD: a row of month m sees the other 11 months;
the ranking set would see all of 2025): car_apt_rate / car_apt_n (carrier at this airport), car_rate / car_n (carrier,
all airports), smoothed (s + 10 * prior) / (n + 10) with the airport's (resp. overall) out-of-fold rate as prior.

(a) ROME ZONE P: Rome's TabICL tail classifier (v13: the 38 exp62 columns; reference p = cache/tfm_f2/base.npy, F2's
    identical protocol) + the 4 carrier columns, leave-one-month-out over Rome's 627 P rows, seeds 0-2.
(b) ROUTE-D CORRECTOR REFRESH (route D outside P): replace v24's 0.5 * (old v10 corrector) by g, a LightGBM regressor of
    r = y - blendD trained month-out-of-fold on route-D-outside-P rows with y <= 43,200 s (day-scale artefacts excluded
    from TRAINING only), features: the 4 carrier columns, off_sched, scheduled UTC hour, airport, blendD; seeds 0-2.

PRE-REGISTERED RULE (written 2026-09-28 before either arm was run), each arm separately, all flights, vs v24:
  R1 12 months AND Jan+Jul improve; R2 12-month airport x local-day block bootstrap 95% interval < 0;
  R3 January AND July each improve; R4 every seed alone improves 12 months; R5 without the 100 most-improved rows,
  12 months and Jan+Jul still improve.

    HF_HUB_OFFLINE=1 python experiments/tail/t06_carrier_stamp.py --threads 3
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from prc.config import CACHE, ID
from prc.io import load
from prc.validation import block_bootstrap
from v24_oof import build

OUT = CACHE / 'tail'
KEYS = ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month')
CCOLS = ['car_apt_rate', 'car_apt_n', 'car_rate', 'car_n']


def carrier_features(dd, y, so, month, D):
    """Month-out-of-fold carrier stamping rates for every 2025 departure (NaN-free; route-D population)."""
    raw = load('training', phase='DEP', columns=[ID, 'FLIGHT_mvt']).set_index(ID).reindex(dd.MVT_ID_mvt.to_numpy())
    car = raw.FLIGHT_mvt.astype(str).str.upper().str.extract(r'^([A-Z]{3})')[0].fillna('???').to_numpy()
    apt = dd.APT.astype(str).to_numpy()
    pop = D & (so > 1800) & np.isfinite(y)
    st = (np.abs(so - y) <= 120) & pop
    F = {c: np.full(len(y), np.nan) for c in CCOLS}
    for m in range(1, 13):
        tr = pop & (month != m); tg = month == m
        p_all = st[tr].sum() / max(tr.sum(), 1)
        t = pd.DataFrame({'car': car[tr], 'apt': apt[tr], 's': st[tr]})
        ga = t.groupby(['car', 'apt']).s.agg(['sum', 'size']); gc = t.groupby('car').s.agg(['sum', 'size'])
        pa = t.groupby('apt').s.mean()
        q = pd.DataFrame({'car': car[tg], 'apt': apt[tg]})
        ja = q.join(ga, on=['car', 'apt']); jc = q.join(gc, on='car')
        prior_a = q.apt.map(pa).fillna(p_all).to_numpy()
        sa, na = ja['sum'].fillna(0).to_numpy(), ja['size'].fillna(0).to_numpy()
        sc, nc = jc['sum'].fillna(0).to_numpy(), jc['size'].fillna(0).to_numpy()
        F['car_apt_rate'][tg] = (sa + 10 * prior_a) / (na + 10); F['car_apt_n'][tg] = na
        F['car_rate'][tg] = (sc + 10 * p_all) / (nc + 10); F['car_n'][tg] = nc
    return pd.DataFrame(F), car


def verdict(y, fn, fo, month, block, per_seed_f):
    r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    A, S = np.ones(len(y), bool), np.isin(month, [1, 7])
    res = {sc: block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=1000)
           for sc, k in (('12m', A), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
    per_seed = [r(f, A) - r(fo, A) for f in per_seed_f]
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(len(y), bool); keep[np.argsort(gain)[::-1][:100]] = False
    d100 = {sc: r(fn, k & keep) - r(fo, k & keep) for sc, k in (('12m', A), ('scored', S))}
    print('  vs v24, all flights, block bootstrap: ' + '  '.join(
        f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items()))
    print('  per seed 12m: ' + ' '.join(f'{v:+.3f}' for v in per_seed)
          + f' | without the 100 most-improved rows: 12m {d100["12m"]:+.3f} scored {d100["scored"]:+.3f}')
    R = dict(R1=res['12m']['point'] < 0 and res['scored']['point'] < 0, R2=res['12m']['hi'] < 0,
             R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=all(v < 0 for v in per_seed),
             R5=d100['12m'] < 0 and d100['scored'] < 0)
    print('  RULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R.items())
          + f'  => {"PASS" if all(R.values()) else "FAIL"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    B = build()
    y, month, P, D, f24, so = B['y'], B['month'], B['P'], B['D'], B['f'], B['so']
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'SCHED_TIME_UTC_mvt'])
    C, car = carrier_features(dd, y, so, month, D)
    rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
    assert np.allclose(rw.y.to_numpy(float), y, equal_nan=True)
    block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
    pop = D & (so > 1800)
    print(f'route-D stamping population (off_sched > 1800): {pop.sum():,} rows, stamped {np.mean(np.abs(so[pop] - y[pop]) <= 120):.3f}; '
          f'carriers {len(set(car[pop]))} | {time.time() - t0:.0f}s', flush=True)

    # ---------- (a) Rome zone P: TabICL + carrier columns ----------
    Pidx, rome, p14, pbP, P_rest = B['Pidx'], B['rome'], B['p14'], B['pbP'], B['P_rest']
    cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    num = [c for c in cand.columns if c not in KEYS and cand[c].dtype != object]
    X = np.hstack([cand[num].astype(float).to_numpy(), C.iloc[Pidx][CCOLS].to_numpy(float)])
    t = (y[Pidx] > 7200).astype(int); pm = cand.month.to_numpy()
    fs = OUT / 't06a_seeds.npy'
    if fs.exists():
        ps = np.load(fs)
    else:
        import torch
        torch.set_num_threads(a.threads)
        from tabicl import TabICLClassifier
        ps = np.full((3, len(t)), np.nan)
        for m in range(1, 13):
            tr, te = rome & (pm != m), rome & (pm == m)
            for s in range(3):
                clf = TabICLClassifier(device='cpu', random_state=s, n_jobs=a.threads, batch_size=1)
                clf.fit(X[tr], t[tr]); ps[s, te] = clf.predict_proba(X[te])[:, 1]
        np.save(fs, ps)
    p_new, p_base = ps.mean(0), np.load(CACHE / 'tfm_f2' / 'base.npy')
    mixP = lambda p: np.clip(p * so[Pidx] + (1 - p) * pbP, 0, None)
    def serve_p(p_rome):
        f = f24.copy(); f[Pidx] = P_rest + mixP(np.where(rome, p_rome, p14)); return f
    print(f'\n(a) Rome P ({rome.sum()} rows, {t[rome].sum()} tails): Brier base {np.mean((p_base[rome] - t[rome]) ** 2):.4f} '
          f'AUC {roc_auc_score(t[rome], p_base[rome]):.3f} | + carrier {np.mean((p_new[rome] - t[rome]) ** 2):.4f} '
          f'AUC {roc_auc_score(t[rome], p_new[rome]):.3f} | {time.time() - t0:.0f}s', flush=True)
    verdict(y, serve_p(p_new), serve_p(p_base), month, block, [serve_p(q) for q in ps])

    # ---------- (b) route-D corrector refresh ----------
    DnP = D & ~P
    blendD = B['blendD']
    Xd = C[CCOLS].copy()
    Xd['off_sched'] = so; Xd['sched_hour'] = dd.SCHED_TIME_UTC_mvt.dt.hour.to_numpy()
    Xd['apt'] = pd.Categorical(dd.APT.astype(str)); Xd['blendD'] = blendD
    r = y - blendD
    gs = np.zeros((3, len(y)))
    for s in range(3):
        for m in range(1, 13):
            tr = DnP & (month != m) & (y <= 43200); te = DnP & (month == m)
            mdl = lgb.train(dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
                                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                                 seed=s, verbose=-1, num_threads=a.threads),
                            lgb.Dataset(Xd.loc[tr], r[tr], categorical_feature=['apt']), num_boost_round=300)
            gs[s, te] = mdl.predict(Xd.loc[te])
    def serve_g(g):
        f = f24.copy(); f[DnP] = np.clip(blendD[DnP] + g[DnP], 0, None); return f
    fg = serve_g(gs.mean(0))
    rr = lambda f, k: np.sqrt(np.mean((y[k] - f[k]) ** 2))
    k = DnP
    print(f'\n(b) route D outside P ({k.sum():,} rows): RMSE on those rows -- no corrector {rr(serve_g(np.zeros(len(y))), k):.1f} | '
          f'v24 (0.5 x old) {rr(f24, k):.1f} | old x1.0 {rr(serve_g(B["corr"]), k):.1f} | new g {rr(fg, k):.1f} | {time.time() - t0:.0f}s')
    verdict(y, fg, f24, month, block, [serve_g(g) for g in gs])
    np.savez(OUT / 't06_results.npz', p_new=p_new, g=gs.mean(0))


if __name__ == '__main__':
    main()
