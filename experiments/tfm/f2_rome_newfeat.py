"""F2 -- give Rome's tail-zone TabICL the v11 feature-stack signals it has never seen.

W4 and F1 (2026-09-24) showed that a better LEARNER does not help Rome's P rows any more: in-context
TabICL, pooled context and three fine-tuning variants all land at Brier ~0.130. The remaining
headroom (oracle -10.8 / -16.2 s) needs new INFORMATION. The v13 classifier sees the 38 exp62
columns; the v11 stack (experiments/v11/v11_features.py) built more, month-isolated, from the
provided files only:
  stamp     pa_stamped, pa_inb_m_sibt, pa_ldg_m_sibt -- was the PREVIOUS ARRIVAL at our stand
            in-blocked at its schedule (the airport's stamping habit, right at this stand)
  clocks    off_iobt, lobt_m_iobt, eobt_m_lobt, eobt_m_sched, iobt_m_sched (flight-plan clocks; the
            AOBT_3-based ones are empty in P by definition and left out)
  turnover  BACKWARD si_before_sched, si_same_typ, sd_prev_to
            LOOK-AHEAD sl_to, sl_after_sched, sl_same_typ, sl_same_op, sn_minus_to, sn_same_op,
            sn_same_typ, sa_after_to, sd_next_to (next aircraft on our stand: bounds when we left)
Look-ahead inside the provided ranking file is allowed (user, 2026-09-24).

Arms (leave-one-month-out on Rome's 627 P rows, context = other months' Rome P rows, 3 seeds):
  base      the 38 exp62 columns                         (= v13)
  back      base + stamp + clocks + BACKWARD turnover     (reported for information)
  full      base + stamp + clocks + all turnover          (THE candidate)
PRE-REGISTERED RULE: `full` replaces v13's p at Rome only if it beats `base` on both 12 months
and the scored months, with the 12-month paired-bootstrap interval below 0. `back` is reported,
not selectable, so the rule is not a pick-the-best over two arms.

usage: HF_HUB_OFFLINE=1 systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
           nice -n 5 python experiments/tfm/f2_rome_newfeat.py --threads 3
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from prc.config import CACHE
from prc.validation import paired_bootstrap

OUT = CACHE / 'tfm_f2'
KEYS = ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month')
STAMP = ['pa_stamped', 'pa_inb_m_sibt', 'pa_ldg_m_sibt']
CLOCKS = ['off_iobt', 'lobt_m_iobt', 'eobt_m_lobt', 'eobt_m_sched', 'iobt_m_sched']
BACK = ['si_before_sched', 'si_same_typ', 'sd_prev_to']
AHEAD = ['sl_to', 'sl_after_sched', 'sl_same_typ', 'sl_same_op', 'sn_minus_to', 'sn_same_op',
         'sn_same_typ', 'sa_after_to', 'sd_next_to']
ARMS = {'base': [], 'back': STAMP + CLOCKS + BACK, 'full': STAMP + CLOCKS + BACK + AHEAD}


def load():
    M = CACHE / 'major'
    d = pd.read_parquet(M / 'dep_training.parquet', columns=['month', 'y', 'off_sched', 'off_aobt3'])
    y, so, mon = d.y.to_numpy(), d.off_sched.to_numpy(), d.month.to_numpy()
    Pidx = np.flatnonzero(~np.isfinite(d.off_aobt3.to_numpy()) & (so > 7200))
    cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    assert len(cand) == len(Pidx) and np.array_equal(cand.month.to_numpy(), mon[Pidx])
    f = pd.read_parquet(CACHE / 'v11' / 'feat_training.parquet', columns=['month', 'off_sched'] + ARMS['full'])
    assert np.array_equal(f.month.to_numpy(), mon) and np.allclose(f.off_sched.to_numpy(), so, equal_nan=True), \
        'v11 features are not in the dep_training row order'
    num = [c for c in cand.columns if c not in KEYS and cand[c].dtype != object]
    Xb = cand[num].astype(float).to_numpy()
    Xn = f.iloc[Pidx][ARMS['full']].astype(float).to_numpy()
    rome = cand.apt.astype(str).to_numpy() == 'LIRF'
    return dict(y=y, so=so, mon=mon, Pidx=Pidx, rome=rome, pm=cand.month.to_numpy(), Xb=Xb, Xn=Xn,
                t=(y[Pidx] > 7200).astype(int), p_cur=np.load(CACHE / 'exp62_spec_P.npy')[1],
                v9=np.load(M / 'v9s_oof.npy'), pb=np.load(CACHE / 'exp62_pb.npy').astype(float))


def X_of(D, arm):
    k = len(ARMS[arm])
    return D['Xb'] if k == 0 else np.hstack([D['Xb'], D['Xn'][:, :k]])   # ARMS nest: back is a prefix of full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from tabicl import TabICLClassifier
    D = load()
    rome, t, pm = D['rome'], D['t'], D['pm']
    OUT.mkdir(parents=True, exist_ok=True)
    cov = np.isfinite(D['Xn'][rome]).mean(0)
    print('coverage of the new features on Rome P rows (2025): '
          + ', '.join(f'{c} {v:.2f}' for c, v in zip(ARMS['full'], cov)), flush=True)
    P = {}
    t0 = time.time()
    for arm in ARMS:
        f = OUT / f'{arm}.npy'
        if f.exists():
            P[arm] = np.load(f); continue
        X = X_of(D, arm)
        p = np.full(len(t), np.nan)
        for m in range(1, 13):
            tr, te = rome & (pm != m), rome & (pm == m)
            ps = []
            for s in a.seeds:
                clf = TabICLClassifier(device=a.device, random_state=s, n_jobs=a.threads, batch_size=1)
                clf.fit(X[tr], t[tr]); ps.append(clf.predict_proba(X[te])[:, 1])
            p[te] = np.mean(ps, axis=0)
        np.save(f, p); P[arm] = p
        print(f'{arm}: {X.shape[1]} features, {time.time() - t0:.0f} s', flush=True)

    y, so, mon, Pidx = D['y'], D['so'], D['mon'], D['Pidx']
    S = np.isin(mon, [1, 7])

    def full(p):
        o = D['v9'].copy()
        pp = np.where(rome, p, D['p_cur'])
        o[Pidx] = np.clip(pp * so[Pidx] + (1 - pp) * D['pb'][Pidx], 0, None)
        return o

    print(f'\nRome P rows {rome.sum()}, tails {t[rome].sum()}, seeds {a.seeds}')
    for arm, p in P.items():
        print(f'  {arm:5s} Brier {np.mean((p[rome] - t[rome]) ** 2):.4f}  AUC {roc_auc_score(t[rome], p[rome]):.3f}')
    ref = full(P['base'])
    print('\nRMSE change over ALL 2025 departures vs base (= v13), paired bootstrap 95%:')
    for arm in ('back', 'full'):
        f = full(P[arm]); out = []
        for scope, mm in (('12m', np.ones(len(y), bool)), ('scored', S)):
            b = paired_bootstrap(y[mm], f[mm], ref[mm])
            out.append(f'{scope} {b["diff"]:+.2f} [{b["lo"]:+.2f},{b["hi"]:+.2f}]')
        per = [np.sqrt(np.mean((y[mon == m] - f[mon == m]) ** 2)) - np.sqrt(np.mean((y[mon == m] - ref[mon == m]) ** 2))
               for m in range(1, 13)]
        print(f'  {arm:5s} ' + '   '.join(out) + f'   months better {sum(v < 0 for v in per)}/12, worst {max(per):+.2f}')
    np.savez(OUT / 'f2_oof.npz', Pidx=Pidx, rome=rome, **{f'p_{k}': v for k, v in P.items()})


if __name__ == '__main__':
    main()
