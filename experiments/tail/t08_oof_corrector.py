"""Bet (2026-10-06, user: "go ahead with your bets"): a residual corrector on the v28-structure out-of-fold predictions.

DESIGN (frozen before the first run):
  rows      routes A/B/C outside zone P (rows with an NM record; 2026: 339,551 ranking rows). Route D and P untouched.
  target    r = clip(y - f27, -3600, 3600), f27 = the v27-structure OOF before the LOBT window (cache/w2/v28s_oof.npz).
  features  (all label-free, available for 2026) the 8 member OOF mixtures (mix nn perapt lintree routed xgboost w3, seq pool
            mean), f27, the seq-pool disagreement sd and the seq context difference dl, off_lobt, off_sched, off_aobt3, the
            position of f27 inside its LOBT window (f27 - lo) / (hi - lo), airport (categorical), scheduled UTC hour, adsb.lol
            flags (has ground points, off-block estimate present).
  model     LightGBM regression, num_leaves 31, learning_rate 0.03, 400 rounds, min_data_in_leaf 500, feature_fraction 0.8,
            bagging 0.8/1, lambda_l2 10; leave-one-month-out (the month being corrected is never in the corrector's training);
            seeds 0, 1, 2 averaged. No tuning; one configuration.
  output    f = window(f27 + g) on the A/B/C rows (the LOBT window applied after the correction, as in v28).
RULE (vs v28 structure, all flights): R1 held-out (8 months) AND scored (Jan+Jul) better; R2 held-out airport x local-day
  block bootstrap 95% CI < 0; R3 January AND July each better; R4 every corrector seed alone better held-out; R5 held-out gain
  survives removing the 100 most-improved held-out rows; X1 >= 10 of 12 months better; X2 single-month cross-fits (corrector
  trained on January only -> judged on July, and July -> January) both < 0. PASS -> v29 built by assemble.py and uploaded
  (user's standing permission for passing bets, 10-06). FAIL -> nothing is uploaded.

    python experiments/tail/t08_oof_corrector.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE, ID
from prc.io import load
from prc.validation import block_bootstrap

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
PARAMS = dict(objective='regression', num_leaves=31, learning_rate=0.03, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=6)
ROUNDS = 400


def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o


def features(Z, apt, sched_hour, adsb_ground, adsb_off, members):
    """Corrector design matrix (same function for 2025 OOF and 2026 production)."""
    lo, hi = np.maximum(0, Z['off_lobt'] - 3606), Z['off_lobt'] + 3606
    X = pd.DataFrame({f'm_{k}': v for k, v in members.items()})
    X['f27'] = Z['f27']; X['sd'] = Z['sd']; X['dl'] = Z['dl']
    X['off_lobt'] = Z['off_lobt']; X['off_sched'] = Z['off_sched']; X['off_aobt3'] = Z['off_aobt3']
    X['win_pos'] = (Z['f27'] - lo) / np.maximum(hi - lo, 1.0)
    X['apt'] = pd.Categorical(apt); X['sched_hour'] = sched_hour
    X['adsb_ground'] = adsb_ground; X['adsb_off'] = adsb_off
    return X


def window(f, off_lobt):
    m = np.isfinite(off_lobt); out = f.copy()
    out[m] = np.minimum(np.maximum(f[m], np.maximum(0, off_lobt[m] - 3606)), off_lobt[m] + 3606)
    return out


def main():
    t0 = time.time()
    Z = dict(np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True))
    y, month, P, D, f28 = Z['y'], Z['month'], Z['P'], Z['D'], Z['f']
    A = ~D & ~P
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'off_sched', 'off_aobt3'])
    Z['off_sched'], Z['off_aobt3'] = dd.off_sched.to_numpy(float), dd.off_aobt3.to_numpy(float)
    tr = load('training', phase='DEP', columns=[ID, 'SCHED_TIME_UTC_mvt'])
    assert np.array_equal(tr[ID].to_numpy(), dd.MVT_ID_mvt.to_numpy())
    a = pd.read_parquet(CACHE / 'adsb_lol' / 'features_training.parquet', columns=['MVT_ID_mvt', 'adsb_has_ground', 'adsb_off_best'])
    a = a.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(dd.MVT_ID_mvt.to_numpy())
    members = {f: oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month) for f in FAMS}
    members['seq'] = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2], month) for v in POOL], axis=0)
    X = features(Z, dd.APT.astype(str).to_numpy(), tr.SCHED_TIME_UTC_mvt.dt.hour.to_numpy(),
                 (a.adsb_has_ground == 1).to_numpy(float), a.adsb_off_best.notna().to_numpy(float), members)
    r = np.clip(y - Z['f27'], -3600, 3600)
    print(f'design {X.shape}, A/B/C outside P rows {A.sum():,} | {time.time() - t0:.0f}s', flush=True)

    def fit_predict(train_rows, pred_rows, seed):
        mdl = lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(X.loc[train_rows], r[train_rows], categorical_feature=['apt']), ROUNDS)
        return mdl.predict(X.loc[pred_rows])

    G = np.zeros((3, len(y)))
    for s in range(3):
        for m in range(1, 13):
            te = A & (month == m)
            G[s, te] = fit_predict(A & (month != m), te, s)
        print(f'  seed {s} done | {time.time() - t0:.0f}s', flush=True)
    serve = lambda g: np.where(A, window(Z['f27'] + g, Z['off_lobt']), f28)
    fn = serve(G.mean(0))
    rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
    assert np.allclose(rw.y.to_numpy(float), y, equal_nan=True)
    block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
    held, S = ~np.isin(month, [3, 6, 9, 12]), np.isin(month, [1, 7])
    rm = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    res = {sc: block_bootstrap(y[k], fn[k], f28[k], block[k], n_boot=1000)
           for sc, k in (('held', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
    per_seed = [rm(serve(G[s]), held) - rm(f28, held) for s in range(3)]
    gain = (y - f28) ** 2 - (y - fn) ** 2
    hi_ = np.where(held)[0]; keep = np.ones(len(y), bool); keep[hi_[np.argsort(gain[held])[::-1][:100]]] = False
    r5 = rm(fn, held & keep) - rm(f28, held & keep)
    months_better = sum(rm(fn, month == m) < rm(f28, month == m) for m in range(1, 13))
    xf = {}
    for fm_, jm in ((1, 7), (7, 1)):
        g1 = np.zeros(len(y)); te = A & (month == jm)
        g1[te] = fit_predict(A & (month == fm_), te, 0)
        f1 = serve(g1); xf[f'{fm_}->{jm}'] = rm(f1, month == jm) - rm(f28, month == jm)
    G_ = dict(R1=res['held']['point'] < 0 and res['scored']['point'] < 0, R2=res['held']['hi'] < 0,
              R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=all(v < 0 for v in per_seed), R5=r5 < 0,
              X1=months_better >= 10, X2=all(v < 0 for v in xf.values()))
    print('corrector vs v28 structure: ' + '  '.join(f'{k} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for k, v in res.items()))
    print(f'  per seed held {" ".join(f"{v:+.3f}" for v in per_seed)} | w/o top-100 held rows {r5:+.3f} | months better '
          f'{months_better}/12 | cross-fits ' + ' '.join(f'{k} {v:+.3f}' for k, v in xf.items()))
    apt = dd.APT.astype(str).to_numpy()
    print('  per airport held: ' + ' '.join(f'{x} {rm(fn, held & (apt == x)) - rm(f28, held & (apt == x)):+.2f}' for x in sorted(set(apt))))
    print(f'  correction g on A/B/C: mean {G.mean(0)[A].mean():+.2f} sd {G.mean(0)[A].std():.1f} p99 |g| {np.percentile(np.abs(G.mean(0)[A]), 99):.0f}')
    verdict = 'PASS' if all(G_.values()) else 'FAIL'
    print('  RULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in G_.items()) + f'  => {verdict} | {time.time() - t0:.0f}s')
    np.savez(CACHE / 'tail' / 't08_corrector.npz', g=G.mean(0), verdict=verdict)


if __name__ == '__main__':
    main()
