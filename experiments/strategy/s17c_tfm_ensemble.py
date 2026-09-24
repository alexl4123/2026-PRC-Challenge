"""s17c -- ensemble of the LIRF-P foundation-model probabilities already computed (no refit):
TabPFN seeds (s17, s17b rs1, rs2) and TabICL (s19).  Same evaluation as s17."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from prc.config import CACHE
from prc.validation import paired_bootstrap
M = CACHE / 'major'; ST = CACHE / 'strategy'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
y = d.y.to_numpy(); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy(); mon = d.month.to_numpy()
P = ~np.isfinite(oa) & (so > 7200); Pidx = np.flatnonzero(P)
v9 = np.load(M / 'v9s_oof.npy'); pb = np.load(CACHE / 'exp62_pb.npy').astype(float)
p_cur = np.load(CACHE / 'exp62_spec_P.npy')[1]
cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
lirf = cand.apt.astype(str).to_numpy() == 'LIRF'; t = (y[Pidx] > 7200).astype(int)
etP = so[Pidx]; pbP = pb[Pidx]; S = np.isin(mon, [1, 7]); al = np.ones(len(y), bool)
full = lambda p: np.where(P, 0, v9) + 0 if False else _full(p)
def _full(p):
    out = v9.copy(); out[Pidx] = np.clip(p * etP + (1 - p) * pbP, 0, None); return out
base = _full(p_cur)
tp = [np.load(ST / 's17_p_tfm_lirf.npy'), np.load(ST / 's17b_p_tfm_lirf_rs1.npy'), np.load(ST / 's17b_p_tfm_lirf_rs2.npy')]
ti = np.load(ST / 's19_p_ticl_lirf.npy')
arms = {'TabPFN 3-seed mean': np.mean(tp, 0), 'TabICL': ti, 'TabPFN3 + TabICL (50/50)': 0.5 * np.mean(tp, 0) + 0.5 * ti,
        'v7 + TabPFN3 + TabICL (1/3 each)': (p_cur + np.mean(tp, 0) + ti) / 3}
k = lirf
print(f'current v7: Brier {np.mean((p_cur[k]-t[k])**2):.4f} AUC {roc_auc_score(t[k], p_cur[k]):.3f}')
for nm, pa in arms.items():
    p = np.where(lirf, pa, p_cur); f = _full(p)
    b12 = paired_bootstrap(y, f, base); bS = paired_bootstrap(y[S], f[S], base[S])
    permo = [np.sqrt(np.mean((y[mon == mm] - f[mon == mm]) ** 2)) - np.sqrt(np.mean((y[mon == mm] - base[mon == mm]) ** 2)) for mm in range(1, 13)]
    print(f'{nm:34s} Brier {np.mean((p[k]-t[k])**2):.4f} AUC {roc_auc_score(t[k], p[k]):.3f} | 12m {b12["diff"]:+.3f} [{b12["lo"]:+.3f},{b12["hi"]:+.3f}] '
          f'scored {bS["diff"]:+.3f} [{bS["lo"]:+.3f},{bS["hi"]:+.3f}] | months better {sum(v < 0 for v in permo)}/12, worst {max(permo):+.2f}')
print('corr TabPFN3 vs TabICL on LIRF P:', round(np.corrcoef(np.mean(tp, 0)[k], ti[k])[0, 1], 3))
