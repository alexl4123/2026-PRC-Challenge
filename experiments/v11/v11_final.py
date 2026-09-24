"""v11 step 5 -- production stacker fits for the ranking set (no submission is built here).

Fits the stacker on ALL 12 training months outside P (y <= 7200, target r = y - base clipped at
+-3000), with training-row encodings leave-own-month-out and ranking encodings from all 12 months
(same shrinkage W_TE = 20). Ranking features come from ranking.parquet alone (v11_features.py,
each month from its own events). The correction does NOT use the base prediction as an input, so
the main session adds it to v10 directly:  pred_v11 = v10 + corr  outside P,  v10 inside P.

Writes cache/final/stack_<arm>_ranking.npz: ids (ranking DEP order == lgb_components ids),
in_P (bool), corr (float64, NaN inside P).
    python experiments/v11/v11_final.py b c [a]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.io import peak_rss_gb
from prc.config import CACHE, ID, TAIL_CAP
from v11_common import ARMS, CATS, CLIP, ROUNDS, PARAMS, key_month_sums, te_final_train, te_final_apply

t0 = time.time()
OUT = CACHE / 'v11'; FIN = CACHE / 'final'
b = np.load(OUT / 'base.npz'); base, y, month, P = b['base'], b['y'], b['month'], b['P']
lab = np.load(OUT / 'labels_training.npz'); assert np.array_equal(lab['y'], y)
Ftr = pd.read_parquet(OUT / 'feat_training.parquet'); Frk = pd.read_parquet(OUT / 'feat_ranking.parquet')
comp = np.load(FIN / 'lgb_components.npz', allow_pickle=True)
ids, in_P = comp['ids'], comp['in_P'].astype(bool)
assert np.array_equal(Frk[ID].to_numpy(), ids), 'ranking feature order != lgb_components ids'
P_rk = (Frk.aobt_missing.to_numpy() == 1) & (Frk.off_sched.to_numpy() > TAIL_CAP)
assert np.array_equal(P_rk, in_P), f'in_P mismatch: {int((P_rk != in_P).sum())} rows'
r = y - base; rc = np.clip(r, -CLIP, CLIP)
src_r = ~P & (y <= 7200)
sm, sl = lab['stamp_mask'], lab['stamp_lab']
nk = {k: int(max(Ftr[k].max(), Frk[k].max())) + 1 for k in ['k_op', 'k_stand', 'k_ades']}
enc = {}
for nm, key, mask, val in (('te_op_st', 'k_op', sm, sl), ('te_stand_st', 'k_stand', sm, sl), ('te_ades_r', 'k_ades', src_r, rc)):
    S, N = key_month_sums(Ftr[key].to_numpy(), month, mask, val, nk[key])
    prior = float(S.sum() / N.sum()) if nm.endswith('_st') else 0.0
    enc[nm] = (key, S, N, prior)
tr = np.where(~P & (y <= 7200))[0]
oof_c = {}
for arm in sys.argv[1:]:
    cols, tes = ARMS[arm]
    assert 'pb' not in cols, 'production arms must not use the base prediction'
    Xtr = Ftr[cols].iloc[tr].reset_index(drop=True); Xrk = Frk[cols].copy()
    for nm in tes:
        key, S, N, prior = enc[nm]
        Xtr[nm] = te_final_train(S, N, Ftr[key].to_numpy()[tr], month[tr], prior).astype('float32')
        Xrk[nm] = te_final_apply(S, N, Frk[key].to_numpy(), prior).astype('float32')
    feat = list(cols) + list(tes)
    assert list(Xtr[feat].dtypes) == list(Xrk[feat].dtypes)
    mdl = lgb.train(PARAMS, lgb.Dataset(Xtr[feat], rc[tr], categorical_feature=[c for c in CATS if c in feat]), ROUNDS)
    corr = np.full(len(ids), np.nan)
    corr[~in_P] = mdl.predict(Xrk.loc[~in_P, feat])
    np.savez(FIN / f'stack_{arm}_ranking.npz', ids=ids, in_P=in_P, corr=corr)
    mdl.save_model(str(OUT / f'stack_{arm}_final.txt'))
    # parity: ranking correction vs the 12-fold OOF correction of the same months in 2025
    oof = np.load(OUT / f'oof_{arm}.npy').astype(float)
    print(f'\n[{arm}] {len(feat)} features, trained on {len(tr):,} rows; saved cache/final/stack_{arm}_ranking.npz')
    print('   correction distribution (s):      mean     sd     p1    p50    p99   |corr|>300 share')
    for nm, v in (('2025 Jan OOF', oof[~P & (month == 1)]), ('2026 Jan rank', corr[~in_P & (Frk.month.to_numpy() == 1)]),
                  ('2025 Jul OOF', oof[~P & (month == 7)]), ('2026 Jul rank', corr[~in_P & (Frk.month.to_numpy() == 7)])):
        print(f'   {nm:14s} {v.mean():+8.1f} {v.std():6.1f} {np.percentile(v, 1):+6.0f} {np.percentile(v, 50):+6.0f} {np.percentile(v, 99):+6.0f}   {np.mean(np.abs(v) > 300):.4f}')
    print(f'   finite outside P: {np.isfinite(corr[~in_P]).all()} | NaN inside P: {np.isnan(corr[in_P]).all()} ({in_P.sum()} rows)')
print(f'\npeak RSS {peak_rss_gb():.2f} GB | {time.time()-t0:.0f}s')
