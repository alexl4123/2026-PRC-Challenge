"""s19 -- (copy of s17 with TabICL, BSD-3 licence) a tabular foundation model where it actually fits: the small-data P classifier at LIRF.

P = {AOBT_3 missing, off_sched > 7200}: 4,944 training rows, 627 at LIRF (413 tails), and LIRF
holds 86% of P's remaining squared error (s10).  That is TabPFN's home ground (<10k rows,
classification), unlike the 2M-row bulk regressor (s04).

LOMO over the 12 months on the LIRF P rows: TabPFN v2 classifier (context = LIRF P rows of the
other 11 months) on the numeric exp62 candidate features (no target encodings -> nothing to nest),
vs the production specialist p (exp62 arm b = v7).  Arms on LIRF P rows: current p, TabPFN p,
their average.  Reports Brier / AUC and the full-set RMSE effect (s10 machinery, paired bootstrap).
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
assert len(cand) == len(Pidx) and np.array_equal(cand.month.to_numpy(), mon[Pidx])
num = [c for c in cand.columns if c not in ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month') and cand[c].dtype != object]
X = cand[num].astype(float).to_numpy()
lirf = cand.apt.astype(str).to_numpy() == 'LIRF'
t = (y[Pidx] > 7200).astype(int)
from tabicl import TabICLClassifier

p_tfm = np.full(len(Pidx), np.nan)
t0 = time.time()
for m in range(1, 13):
    tr = lirf & (cand.month.to_numpy() != m); te = lirf & (cand.month.to_numpy() == m)
    if te.sum() == 0:
        continue
    clf = TabICLClassifier(device='cpu', random_state=0)
    clf.fit(X[tr], t[tr])
    p_tfm[te] = clf.predict_proba(X[te])[:, 1]
    print(f'month {m}: train {tr.sum()} test {te.sum()}  {time.time()-t0:.0f} s', flush=True)
k = lirf
print(f'\nLIRF P rows {k.sum()}, tails {t[k].sum()}')
for nm, p in [('current specialist (v7)', p_cur), ('TabICL (BSD-3)', np.where(k, p_tfm, p_cur)),
              ('average', np.where(k, 0.5 * p_cur + 0.5 * p_tfm, p_cur))]:
    print(f'  {nm:26s} Brier {np.mean((p[k]-t[k])**2):.4f}  AUC {roc_auc_score(t[k], p[k]):.3f}')
yP = y[Pidx]; etP = so[Pidx]; pbP = pb[Pidx]; S = np.isin(mon, [1, 7])


def full(p):
    out = v9.copy(); out[Pidx] = np.clip(p * etP + (1 - p) * pbP, 0, None); return out


base = full(p_cur)
for nm, p in [('TabICL on LIRF P', np.where(k, p_tfm, p_cur)), ('average on LIRF P', np.where(k, 0.5 * p_cur + 0.5 * p_tfm, p_cur))]:
    f = full(p)
    for scope, mm in [('12m', np.ones(len(y), bool)), ('scored', S)]:
        b = paired_bootstrap(y[mm], f[mm], base[mm])
        print(f'  {nm:22s} {scope:6s} d {b["diff"]:+.3f} [{b["lo"]:+.3f},{b["hi"]:+.3f}]')
    permo = [np.sqrt(np.mean((y[mon == mm] - f[mon == mm]) ** 2)) - np.sqrt(np.mean((y[mon == mm] - base[mon == mm]) ** 2)) for mm in range(1, 13)]
    print(f'      months better {sum(v < 0 for v in permo)}/12; worst month {max(permo):+.2f}, best {min(permo):+.2f}')
np.save(CACHE / 'strategy' / 's19_p_ticl_lirf.npy', p_tfm)
