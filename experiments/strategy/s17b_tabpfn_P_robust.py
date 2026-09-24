"""s17b -- robustness of the s17 finding (TabPFN v2 as the LIRF-P classifier).

1. Row concentration: per-row SSE change vs the current specialist; drop the top-k rows by |dSSE|.
2. Seed / ensemble-size replicate: random_state 1 and 2, n_estimators 16 (vs s17: default state, 8).
3. Where it moves: mean p by month for LIRF Jan / Jul, false alarms vs tails.
4. Shrinkage: p = (1-lam) p_cur + lam p_tfm for lam in 0.25..1 (a pre-registered family; judged on
   12 months, reported on the scored months).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from prc.config import CACHE
from prc.validation import paired_bootstrap

torch.set_num_threads(3)
M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy(); mon = d.month.to_numpy()
P = ~np.isfinite(oa) & (so > 7200); Pidx = np.flatnonzero(P)
v9 = np.load(M / 'v9s_oof.npy'); pb = np.load(CACHE / 'exp62_pb.npy').astype(float)
p_cur = np.load(CACHE / 'exp62_spec_P.npy')[1]
cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
num = [c for c in cand.columns if c not in ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month') and cand[c].dtype != object]
X = cand[num].astype(float).to_numpy()
lirf = cand.apt.astype(str).to_numpy() == 'LIRF'; cm = cand.month.to_numpy()
t = (y[Pidx] > 7200).astype(int)
yP = y[Pidx]; etP = so[Pidx]; pbP = pb[Pidx]; S = np.isin(mon, [1, 7]); al = np.ones(len(y), bool)


def full(p):
    out = v9.copy(); out[Pidx] = np.clip(p * etP + (1 - p) * pbP, 0, None); return out


base = full(p_cur)
p0 = np.load(CACHE / 'strategy' / 's17_p_tfm_lirf.npy')
pt0 = np.where(lirf, p0, p_cur)

ONLY_REP = os.environ.get('ONLY_REP') == '1'
print('=== 1. concentration (s17 run, TabPFN replaces p on LIRF P rows)')
f = full(pt0)
dsse = (y - f) ** 2 - (y - base) ** 2
order = np.argsort(-np.abs(dsse))
for k in ([] if ONLY_REP else [0, 1, 3, 5, 10, 20]):
    keep = np.ones(len(y), bool); keep[order[:k]] = False
    for scope, mm in [('12m', al), ('scored', S)]:
        m2 = mm & keep
        dd = np.sqrt(np.mean((y[m2] - f[m2]) ** 2)) - np.sqrt(np.mean((y[m2] - base[m2]) ** 2))
        print(f'  drop top {k:2d} |dSSE| rows: {scope:6s} d {dd:+.3f}', end='')
    print()
top = order[:10]
print('  top-10 rows: y, off_sched, p_cur, p_tfm, dSSE(1e8), month')
for i in ([] if ONLY_REP else top):
    j = np.searchsorted(Pidx, i)
    print(f'    y {y[i]:8.0f}  off_sched {so[i]:8.0f}  p_cur {p_cur[j]:.3f}  p_tfm {pt0[j]:.3f}  dSSE {dsse[i]/1e8:+7.2f}  m{mon[i]}')
tot_gain = dsse.sum(); print(f'  total dSSE {tot_gain/1e8:+.2f}e8; share from rows with dSSE<0: {dsse[dsse<0].sum()/1e8:+.2f}e8, >0: {dsse[dsse>0].sum()/1e8:+.2f}e8')

print('\n=== 3. where it moves (LIRF P rows): mean p, current vs TabPFN')
for m in ([] if ONLY_REP else [1, 7]):
    k = lirf & (cm == m)
    print(f'  month {m}: n {k.sum()}  tails {t[k].sum()}  mean p_cur {p_cur[k].mean():.3f}  mean p_tfm {p0[k].mean():.3f}  '
          f'| on tails {p_cur[k & (t==1)].mean():.3f} -> {p0[k & (t==1)].mean():.3f} | on non-tails {p_cur[k & (t==0)].mean():.3f} -> {p0[k & (t==0)].mean():.3f}')
k = lirf
print(f'  all months: tails {p_cur[k & (t==1)].mean():.3f} -> {p0[k & (t==1)].mean():.3f}; non-tails {p_cur[k & (t==0)].mean():.3f} -> {p0[k & (t==0)].mean():.3f}')

print('\n=== 4. shrinkage lam (p = (1-lam) p_cur + lam p_tfm)')
for lam in ([] if ONLY_REP else [0.25, 0.5, 0.75, 1.0]):
    p = np.where(lirf, (1 - lam) * p_cur + lam * p0, p_cur); f = full(p)
    b12 = paired_bootstrap(y, f, base); bS = paired_bootstrap(y[S], f[S], base[S])
    print(f'  lam {lam:.2f}: 12m {b12["diff"]:+.3f} [{b12["lo"]:+.3f},{b12["hi"]:+.3f}]  scored {bS["diff"]:+.3f} [{bS["lo"]:+.3f},{bS["hi"]:+.3f}]')

print('\n=== 2. replicates (random_state, n_estimators=16)')
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
for rs in [1, 2]:
    t0 = time.time(); pr = np.full(len(Pidx), np.nan)
    for m in range(1, 13):
        tr = lirf & (cm != m); te = lirf & (cm == m)
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', n_estimators=8, random_state=rs,
                                                         ignore_pretraining_limits=True)
        clf.fit(X[tr], t[tr]); pr[te] = clf.predict_proba(X[te])[:, 1]
    p = np.where(lirf, pr, p_cur); f = full(p)
    b12 = paired_bootstrap(y, f, base); bS = paired_bootstrap(y[S], f[S], base[S])
    permo = [np.sqrt(np.mean((y[mon == mm] - f[mon == mm]) ** 2)) - np.sqrt(np.mean((y[mon == mm] - base[mon == mm]) ** 2)) for mm in range(1, 13)]
    print(f'  rs {rs}: Brier {np.mean((pr[lirf]-t[lirf])**2):.4f} AUC {roc_auc_score(t[lirf], pr[lirf]):.3f} | 12m {b12["diff"]:+.3f} '
          f'[{b12["lo"]:+.3f},{b12["hi"]:+.3f}]  scored {bS["diff"]:+.3f} [{bS["lo"]:+.3f},{bS["hi"]:+.3f}]  months better {sum(v<0 for v in permo)}/12  '
          f'corr with s17 p {np.corrcoef(pr[lirf], p0[lirf])[0,1]:.3f}  ({time.time()-t0:.0f} s)', flush=True)
    np.save(CACHE / 'strategy' / f's17b_p_tfm_lirf_rs{rs}.npy', pr)
