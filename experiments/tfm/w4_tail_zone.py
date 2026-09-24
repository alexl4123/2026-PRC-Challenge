"""W4 -- TabICL as the tail-zone classifier at every airport, and pooled context for Rome.

P = no NM off-block record (AOBT_3 missing) and off_sched > 7200 s: 4,944 departures in 2025.
tail = y > 7200 s, i.e. the airport stamped the block time at the schedule. The served value is
p * off_sched + (1 - p) * bulk, and p comes from the v7 specialist (TabICL at Rome since v13).

CEILING, MEASURED FIRST (2026-09-24, v9~ base): only 37 of the 4,317 non-Rome P rows are tails.
A PERFECT classifier on all nine other airports gains -0.93 s (12 months) / -2.38 s (scored);
Rome alone -10.76 / -16.21. So this script answers two questions:
  1. does TabICL beat the specialist outside Rome (bounded by the ceiling above)?
  2. does Rome's classifier improve when the context also holds the other airports' P rows?

Arms, leave-one-month-out on the P rows of 2025 (context = the other 11 months):
  cur   production v7 specialist p                         (cache/exp62_spec_P.npy[1])
  own   TabICL, context = this airport's P rows            (what v13 does at Rome; skipped where
                                                            the context holds no tail)
  pool  TabICL, context = all airports' P rows, plus a one-hot airport code
Features: the 38 numeric exp62 candidate columns (no target encodings, nothing to nest).

PRE-REGISTERED DECISION (written before any result):
  outside Rome: ship `pool` on ALL non-Rome P rows only if the non-Rome total is better on both
                12 months and the scored months, with the 12-month interval below 0. The
                per-airport rows are diagnostics, not a menu: picking airports on the same
                out-of-fold numbers would overfit.
  Rome:         replace v13's `own` with `pool` only under the same rule, judged against `own`.

Resumable and idempotent: every (arm, month, seed) is saved under cache/tfm_w4/parts/ and skipped
when present, so a killed run continues where it stopped. The report is rebuilt from the parts.

usage (laptop):  HF_HUB_OFFLINE=1 systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
                     nice -n 5 python experiments/tfm/w4_tail_zone.py --threads 3
       (GPU):    python experiments/tfm/w4_tail_zone.py --device cuda --batch-size 8 --seeds 0 1 2
       report only, from saved parts:  --report
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from prc.config import CACHE
from prc.validation import paired_bootstrap

OUT = CACHE / 'tfm_w4'
PARTS = OUT / 'parts'
KEYS = ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month')


def load():
    M = CACHE / 'major'
    d = pd.read_parquet(M / 'dep_training.parquet', columns=['month', 'y', 'off_sched', 'off_aobt3'])
    y, so, mon = d.y.to_numpy(), d.off_sched.to_numpy(), d.month.to_numpy()
    Pidx = np.flatnonzero(~np.isfinite(d.off_aobt3.to_numpy()) & (so > 7200))
    cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    if len(cand) != len(Pidx) or not np.array_equal(cand.month.to_numpy(), mon[Pidx]):
        sys.exit('FATAL: exp62_cand_training.parquet is not aligned with the P rows of dep_training')
    num = [c for c in cand.columns if c not in KEYS and cand[c].dtype != object]
    apt = cand.apt.astype(str).to_numpy()
    onehot = pd.get_dummies(pd.Series(apt), prefix='apt').astype(float)
    X = cand[num].astype(float).to_numpy()
    Xpool = np.hstack([X, onehot.to_numpy()])
    return dict(y=y, so=so, mon=mon, Pidx=Pidx, apt=apt, pm=cand.month.to_numpy(), X=X, Xpool=Xpool,
                t=(y[Pidx] > 7200).astype(int), p_cur=np.load(CACHE / 'exp62_spec_P.npy')[1],
                v9=np.load(M / 'v9s_oof.npy'), pb=np.load(CACHE / 'exp62_pb.npy').astype(float), nfeat=len(num))


def fit_predict(Xtr, ttr, Xte, seed, device, threads, batch_size):
    from tabicl import TabICLClassifier
    # batch_size = ensemble members processed at once; 8 (the default) needed > 6 GB on CPU with a
    # 4,300-row pooled context (cgroup OOM 2026-09-24). Same predictions, less memory at a time.
    clf = TabICLClassifier(device=device, random_state=seed, n_jobs=threads, batch_size=batch_size)
    clf.fit(Xtr, ttr)
    return clf.predict_proba(Xte)[:, 1]


def run(D, arms, seeds, device, threads, batch_size):
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for seed in seeds:
        for m in range(1, 13):
            te = D['pm'] == m
            for arm in arms:
                f = PARTS / f'{arm}_m{m:02d}_s{seed}.npy'
                if f.exists():
                    continue
                p = np.full(te.sum(), np.nan)
                if arm == 'pool':
                    tr = ~te
                    p = fit_predict(D['Xpool'][tr], D['t'][tr], D['Xpool'][te], seed, device, threads, batch_size)
                else:                                   # own: one fit per airport
                    for ap in np.unique(D['apt'][te]):
                        tr = (D['apt'] == ap) & ~te
                        q = te & (D['apt'] == ap)
                        if D['t'][tr].sum() == 0:       # no tail in the context: cannot learn one
                            continue
                        p[(D['apt'][te] == ap)] = fit_predict(D['X'][tr], D['t'][tr], D['X'][q], seed, device, threads, batch_size)
                np.save(f, p)
                print(f'seed {seed} month {m:2d} {arm:4s}: {te.sum()} rows  {time.time() - t0:6.0f} s', flush=True)


def gather(D, arm, seeds):
    """Mean over the seeds that are complete for all 12 months; None if none is."""
    ps = []
    for seed in seeds:
        files = [PARTS / f'{arm}_m{m:02d}_s{seed}.npy' for m in range(1, 13)]
        if not all(f.exists() for f in files):
            continue
        p = np.full(len(D['Pidx']), np.nan)
        for m, f in zip(range(1, 13), files):
            p[D['pm'] == m] = np.load(f)
        ps.append(p)
    return (np.mean(ps, axis=0), len(ps)) if ps else (None, 0)


def report(D, seeds):
    y, so, mon, Pidx, t, apt = D['y'], D['so'], D['mon'], D['Pidx'], D['t'], D['apt']
    cur = D['p_cur']
    S = np.isin(mon, [1, 7])

    def full(p):
        o = D['v9'].copy()
        o[Pidx] = np.clip(p * so[Pidx] + (1 - p) * D['pb'][Pidx], 0, None)
        return o

    def rmse_d(f, base, mm):
        return np.sqrt(np.mean((y[mm] - f[mm]) ** 2)) - np.sqrt(np.mean((y[mm] - base[mm]) ** 2))

    arms = {'cur': (cur, 0)}
    for a in ('own', 'pool'):
        p, k = gather(D, a, seeds)
        if p is not None:
            arms[a] = (np.where(np.isfinite(p), p, cur), k)   # own: airports without a tail keep cur
    print(f'\nP rows {len(Pidx)}, tails {t.sum()} | arms: ' + ', '.join(f'{a} ({k} seeds)' if a != 'cur' else a for a, (_, k) in arms.items()))

    print(f"\n{'airport':8s}{'n':>6s}{'tails':>6s}  " + ''.join(f'{a + " Brier":>12s}{a + " AUC":>10s}' for a in arms)
          + ''.join(f'{"d12m " + a:>11s}{"dscr " + a:>11s}' for a in arms if a != 'cur'))
    base = full(cur)
    for ap in sorted(set(apt), key=lambda s: -(apt == s).sum()):
        k = apt == ap
        row = f'{ap:8s}{k.sum():6d}{t[k].sum():6d}  '
        for a, (p, _) in arms.items():
            auc = roc_auc_score(t[k], p[k]) if 0 < t[k].sum() < k.sum() else float('nan')
            row += f'{np.mean((p[k] - t[k]) ** 2):12.4f}{auc:10.3f}'
        for a, (p, _) in arms.items():
            if a == 'cur':
                continue
            f = full(np.where(k, p, cur))
            row += f'{rmse_d(f, base, slice(None)):+11.2f}{rmse_d(f, base, S):+11.2f}'
        print(row)

    print('\nRMSE change over ALL 2025 departures (negative = better), paired bootstrap 95%:')
    nl = apt != 'LIRF'
    tests = []
    if 'pool' in arms:
        tests.append(('pool on non-Rome P  vs cur', np.where(nl, arms['pool'][0], cur), cur))
    if 'own' in arms:
        tests.append(('own on non-Rome P   vs cur', np.where(nl, arms['own'][0], cur), cur))
        own_rome = np.where(~nl, arms['own'][0], cur)
        tests.append(('own on Rome (=v13)  vs cur', own_rome, cur))
        if 'pool' in arms:
            tests.append(('pool on Rome        vs own', np.where(~nl, arms['pool'][0], cur), own_rome))
            tests.append(('pool everywhere     vs own@Rome', arms['pool'][0], own_rome))
    for nm, pa, pbase in tests:
        fa, fb = full(pa), full(pbase)
        out = []
        for scope, mm in (('12m', np.ones(len(y), bool)), ('scored', S)):
            b = paired_bootstrap(y[mm], fa[mm], fb[mm])
            out.append(f'{scope} {b["diff"]:+.2f} [{b["lo"]:+.2f},{b["hi"]:+.2f}]')
        per = [rmse_d(fa, fb, mon == m) for m in range(1, 13)]
        print(f'  {nm:31s} ' + '   '.join(out) + f'   months better {sum(v < 0 for v in per)}/12')
    np.savez(OUT / 'w4_oof.npz', Pidx=Pidx, apt=apt, month=D['pm'], t=t,
             **{f'p_{a}': p for a, (p, _) in arms.items()})
    print(f'\nsaved {OUT / "w4_oof.npz"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['own', 'pool'], choices=['own', 'pool'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1, help='ensemble members per forward pass (memory)')
    ap.add_argument('--report', action='store_true', help='only rebuild the report from saved parts')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    D = load()
    print(f'P rows {len(D["Pidx"])}, features {D["nfeat"]} (+{D["Xpool"].shape[1] - D["nfeat"]} airport one-hot in pool)')
    if not a.report:
        run(D, a.arms, a.seeds, a.device, a.threads, a.batch_size)
    report(D, a.seeds)


if __name__ == '__main__':
    main()
