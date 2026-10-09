"""s21 -- production arrays for W0: a tabular foundation model as the LIRF-P classifier.

Fits on ALL 627 LIRF P rows of 2025 (the exp62 numeric candidate features, no target encodings),
predicts the 181 LIRF P rows of the 2026 ranking file.  Writes ONLY to cache/strategy/:
    tfm_P_lirf_ranking.npz : pos (DIRECT row index into the ranking DEP order -- verified 2026-09-24; NOT a position among the 1,720 P rows of
                             cache/exp62_cand_ranking.parquet), p_tabpfn (mean of 3 random states),
                             p_tabicl (if tabicl is importable), apt, month
Assembly (main session, NOT done here): in cache/final/spec_x_ranking.npz replace p_spec on the LIRF
P rows by p (lambda = 1) or 0.5*(p_spec + p) (lambda = 0.5), then rebuild the submission with the
existing exp64/exp65/exp67 assembly.  Only the 181 LIRF P rows change.

usage: PYTHONPATH=<scratch>/tfm python s21_tfm_P_final.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, torch
from prc.config import CACHE

torch.set_num_threads(3)
M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['month', 'y', 'off_sched', 'off_aobt3'])
P = ~np.isfinite(d.off_aobt3.to_numpy()) & (d.off_sched.to_numpy() > 7200)
yP = d.y.to_numpy()[P]
tr = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
rk0 = pd.read_parquet(CACHE / 'exp62_cand_ranking.parquet'); rk_rows = rk0.index.to_numpy()   # row positions in ranking DEP load order
rk = rk0.reset_index(drop=True)
assert len(tr) == P.sum()
num = [c for c in tr.columns if c not in ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month') and tr[c].dtype != object]
assert all(c in rk.columns for c in num)
ltr = tr.apt.astype(str).to_numpy() == 'LIRF'; lrk = rk.apt.astype(str).to_numpy() == 'LIRF'
Xtr = tr.loc[ltr, num].astype(float).to_numpy(); ttr = (yP[ltr] > 7200).astype(int)
Xrk = rk.loc[lrk, num].astype(float).to_numpy()
print(f'train LIRF P {ltr.sum()} (tails {ttr.sum()}), ranking LIRF P {lrk.sum()}; features {len(num)}')
cov_tr = np.isfinite(Xtr).mean(0); cov_rk = np.isfinite(Xrk).mean(0)
bad = [(c, round(a, 3), round(b, 3)) for c, a, b in zip(num, cov_tr, cov_rk) if abs(a - b) > 0.10]
print('coverage gaps > 10 pp (train vs ranking):', bad if bad else 'none')
out = dict(pos=rk_rows[lrk], apt=rk.apt.astype(str).to_numpy()[lrk], month=rk.month.to_numpy()[lrk])
try:                    # p_tabpfn is diagnostic only (the builders use p_tabicl); optional since the final phase
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    ps = []
    for rs in [0, 1, 2]:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', n_estimators=8, random_state=rs,
                                                         ignore_pretraining_limits=True)
        clf.fit(Xtr, ttr); ps.append(clf.predict_proba(Xrk)[:, 1])
    out['p_tabpfn'] = np.mean(ps, 0)
except ImportError as e:
    print('tabpfn skipped (diagnostic only):', e)
try:
    from tabicl import TabICLClassifier
    ps = []
    for rs in [0, 1, 2]:
        clf = TabICLClassifier(device='cpu', random_state=rs); clf.fit(Xtr, ttr); ps.append(clf.predict_proba(Xrk)[:, 1])
    out['p_tabicl'] = np.mean(ps, 0)
except Exception as e:
    print('tabicl skipped:', type(e).__name__, e)
for k in [k for k in out if k.startswith('p_')]:
    for m in np.unique(out['month']):
        mm = out['month'] == m
        print(f'{k}: month {m}: n {mm.sum()}  mean p {out[k][mm].mean():.3f}  p>0.5 share {np.mean(out[k][mm] > 0.5):.3f}')
OUT = os.environ.get('S21_OUT') or CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz'   # S21_OUT: regression-check target
np.savez(OUT, **out)
print('saved', OUT)
