"""Rome's zone-P classifier (TabICL, v13) + the label-free ARRIVAL stamping regime (t01), judged in the v21 structure.

Rome P = LIRF departures without an NM record and > 2 h behind schedule (627 rows in 2025, 181 in the 2026 ranking
set). The served value is p * off_sched + (1 - p) * pb; the tail anchor is exact for stamped rows, so the whole
headroom is p (oracle -10.8 s 12 months / -16.2 s Jan+Jul). Better learners all converge at Brier ~0.13 (W4, F1),
and F2's stand-level stamp / turnover columns failed their rule. t01 (2026-09-27) measured that Rome's airport-wide
arrival stamping regime adds information to a stamp classifier (zone P: AUC .824 -> .834, Brier .159 -> .155).

Arms (leave-one-month-out over Rome's P rows, TabICL, context = the other months' Rome P rows, seeds 0-2):
  base    the 38 exp62 columns = v13               (reused: cache/tfm_f2/base.npy, F2's identical protocol)
  regime  base + the 9 t01 regime columns (arr_st_1h, arr_n_1h, arr_st_3h, arr_n_3h, arr_st_day, arr_st_stand_day,
          arr_n_stand_day, arr_st_op_day, arr_n_op_day)
Both are served in the v21 structure (cache/w2/v21s_oof.npy with Rome's P rows re-mixed), all flights.

PRE-REGISTERED RULE (written 2026-09-27 before the regime arm was fitted): `regime` replaces Rome's p only if
  R1  12 months AND Jan+Jul improve;  R2  12-month airport x local-day block bootstrap 95% interval < 0;
  R3  January AND July each improve;  R4  every seed alone improves 12 months;
  R5  without the 100 most-improved rows, 12 months and Jan+Jul still improve.

    HF_HUB_OFFLINE=1 python experiments/tail/t03_rome_P_regime.py --threads 3
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from prc.config import CACHE, ID
from prc.validation import block_bootstrap
from t01_rome_stamp import regime

OUT = CACHE / 'tail'
KEYS = ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month')
RCOLS = ['arr_st_1h', 'arr_n_1h', 'arr_st_3h', 'arr_n_3h', 'arr_st_day', 'arr_st_stand_day', 'arr_n_stand_day',
         'arr_st_op_day', 'arr_n_op_day']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from tabicl import TabICLClassifier
    t0 = time.time()
    d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'month', 'y', 'off_sched', 'off_aobt3'])
    y, so, mon = d.y.to_numpy(), d.off_sched.to_numpy(), d.month.to_numpy()
    Pidx = np.flatnonzero(~np.isfinite(d.off_aobt3.to_numpy()) & (so > 7200))
    cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    assert len(cand) == len(Pidx) and np.array_equal(cand.month.to_numpy(), mon[Pidx])
    num = [c for c in cand.columns if c not in KEYS and cand[c].dtype != object]
    rome = cand.apt.astype(str).to_numpy() == 'LIRF'
    t = (y[Pidx] > 7200).astype(int); pm = cand.month.to_numpy()
    R = regime('training').set_index(ID).reindex(d.MVT_ID_mvt.to_numpy()[Pidx])
    if R.loc[rome, RCOLS[0]].isna().any():
        sys.exit('FATAL: regime features missing on Rome P rows')
    X = np.hstack([cand[num].astype(float).to_numpy(), R[RCOLS].astype(float).to_numpy()])
    print(f'Rome P rows {rome.sum()}, tails {t[rome].sum()}, features {X.shape[1]} ({len(num)} base + {len(RCOLS)} regime)', flush=True)
    f_seeds = OUT / 't03_regime_seeds.npy'
    if f_seeds.exists():
        ps = np.load(f_seeds)
    else:
        ps = np.full((3, len(t)), np.nan)
        for m in range(1, 13):
            tr, te = rome & (pm != m), rome & (pm == m)
            for s in range(3):
                clf = TabICLClassifier(device='cpu', random_state=s, n_jobs=a.threads, batch_size=1)
                clf.fit(X[tr], t[tr]); ps[s, te] = clf.predict_proba(X[te])[:, 1]
            print(f'  month {m:2d} done, {time.time() - t0:.0f} s', flush=True)
        np.save(f_seeds, ps)
    p_reg = ps.mean(0)
    p_base = np.load(CACHE / 'tfm_f2' / 'base.npy')
    print(f'Brier base {np.mean((p_base[rome] - t[rome]) ** 2):.4f} AUC {roc_auc_score(t[rome], p_base[rome]):.3f} | '
          f'regime {np.mean((p_reg[rome] - t[rome]) ** 2):.4f} AUC {roc_auc_score(t[rome], p_reg[rome]):.3f} | per seed Brier '
          + ' '.join(f'{np.mean((q[rome] - t[rome]) ** 2):.4f}' for q in ps))

    # v21 structure: re-mix Rome's P rows around the v21 OOF (the non-branch part is kept)
    f21 = np.load(CACHE / 'w2' / 'v21s_oof.npy').astype(float)
    b = np.load(CACHE / 'v11' / 'base.npz'); month = b['month']
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    assert np.array_equal(w4['Pidx'], Pidx) and np.array_equal(w4['apt'] == 'LIRF', rome)
    p_ship = np.where(rome, w4['p_own'], w4['p_pool'])
    def oof(c, fam):
        o = np.zeros(len(y))
        for s in range(3):
            for mm in range(1, 13):
                o[month == mm] += np.load(CACHE / 'cv' / c / f'{fam}__seed{s:02d}__m{mm:02d}.bulk.npy') / 3
        return o
    pb = (0.5 * oof('w2', 'mix') + 0.5 * oof('seqsw_ep15', 'seq'))[Pidx]
    mixP = lambda p: np.clip(p * so[Pidx] + (1 - p) * pb, 0, None)
    rest = f21[Pidx] - mixP(p_ship)
    def serve(p_rome):
        f = f21.copy(); f[Pidx] = rest + mixP(np.where(rome, p_rome, p_ship)); return f
    fo, fn = serve(p_base), serve(p_reg)
    print(f'check: v21 with F2-base p at Rome vs shipped v21: 12m RMSE {np.sqrt(np.mean((y - fo) ** 2)):.3f} vs '
          f'{np.sqrt(np.mean((y - f21) ** 2)):.3f}')
    rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'y'])
    assert np.allclose(rw.y.to_numpy(float), y, equal_nan=True)
    block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
    r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))
    A, S = np.ones(len(y), bool), np.isin(month, [1, 7])
    res = {sc: block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=1000)
           for sc, k in (('12m', A), ('scored', S), ('Jan', month == 1), ('Jul', month == 7))}
    print('regime vs base, all flights, v21 structure, block bootstrap: ' + '  '.join(
        f'{sc} {v["point"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]' for sc, v in res.items()))
    per_seed = [r(serve(q), A) - r(fo, A) for q in ps]
    print('per seed 12m: ' + ' '.join(f'{v:+.3f}' for v in per_seed))
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(len(y), bool); keep[np.argsort(gain)[::-1][:100]] = False
    d100 = {sc: r(fn, k & keep) - r(fo, k & keep) for sc, k in (('12m', A), ('scored', S))}
    print(f'without the 100 most-improved rows: 12m {d100["12m"]:+.3f}  scored {d100["scored"]:+.3f}')
    per_m = [r(fn, month == m) - r(fo, month == m) for m in range(1, 13)]
    print('per month: ' + ' '.join(f'{m}:{v:+.2f}' for m, v in zip(range(1, 13), per_m)))
    R_ = dict(R1=res['12m']['point'] < 0 and res['scored']['point'] < 0, R2=res['12m']['hi'] < 0,
              R3=res['Jan']['point'] < 0 and res['Jul']['point'] < 0, R4=all(v < 0 for v in per_seed),
              R5=d100['12m'] < 0 and d100['scored'] < 0)
    print('RULE: ' + '  '.join(f'{k} {"ok" if v else "FAIL"}' for k, v in R_.items())
          + f'  => {"PASS" if all(R_.values()) else "FAIL"} | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
