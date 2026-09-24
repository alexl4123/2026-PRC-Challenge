"""v11 step 3 -- twelve-fold OOF of the residual stacker on top of the v10 family blend, OUTSIDE P.

target r = y - base, clipped at +-3000; trained on outside-P rows with y <= 7200 of the other
11 months (ALL of them, no subsample); predicts every outside-P row of the held-out month.
Encodings are nested (rows of month k in fold m exclude months k and m). Same folds, seed and
LightGBM parameters for every arm (v11_common.py).
    python experiments/v11/v11_stack.py <arm>      arms: null a_pb a b c
Writes cache/v11/oof_<arm>.npy (NaN inside P).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.io import peak_rss_gb
from prc.config import CACHE
from v11_common import ARMS, CATS, CLIP, ROUNDS, PARAMS, key_month_sums, te_oof

arm = sys.argv[1]
t0 = time.time()
OUT = CACHE / 'v11'
cols, tes = ARMS[arm]
b = np.load(OUT / 'base.npz'); base, y, month, P = b['base'], b['y'], b['month'], b['P']
lab = np.load(OUT / 'labels_training.npz'); assert np.array_equal(lab['y'], y), 'label order mismatch'
F = pd.read_parquet(OUT / 'feat_training.parquet', columns=[c for c in cols if c != 'pb'] + ['k_op', 'k_stand', 'k_ades', 'month'])
assert np.array_equal(F.month.to_numpy(), month)
X = F[[c for c in cols if c != 'pb']].copy()
if 'pb' in cols:
    X['pb'] = base.astype('float32')
r = y - base
rc = np.clip(r, -CLIP, CLIP)
src_r = ~P & (y <= 7200)
enc = {}
if tes:
    nk = {k: int(max(F[k].max(), 0)) + 1 for k in ['k_op', 'k_stand', 'k_ades']}
    sm, sl = lab['stamp_mask'], lab['stamp_lab']
    for nm, key, mask, val in (('te_op_st', 'k_op', sm, sl), ('te_stand_st', 'k_stand', sm, sl), ('te_ades_r', 'k_ades', src_r, rc)):
        codes = F[key].to_numpy()
        S, N = key_month_sums(codes, month, mask, val, nk[key])
        enc[nm] = (codes, S, N, nm.endswith('_st'))
feat = list(X.columns) + list(enc)
corr = np.full(len(y), np.nan)
ntr_last = 0
for m in range(1, 13):
    tf = time.time()
    tr = np.where(~P & (y <= 7200) & (month != m))[0]; va = np.where(~P & (month == m))[0]
    Xm = X.iloc[np.concatenate([tr, va])].reset_index(drop=True)
    mm = month[np.concatenate([tr, va])]
    for nm, (codes, S, N, is_rate) in enc.items():
        if is_rate:   # prior = stamping rate of the training months of this fold
            prior = (S.sum() - S[:, m].sum()) / (N.sum() - N[:, m].sum())
        else:
            prior = 0.0
        Xm[nm] = te_oof(S, N, codes[np.concatenate([tr, va])], mm, m, prior).astype('float32')
    ds = lgb.Dataset(Xm.iloc[:len(tr)][feat], rc[tr], categorical_feature=[c for c in CATS if c in feat], free_raw_data=True)
    mdl = lgb.train(PARAMS, ds, ROUNDS)
    corr[va] = mdl.predict(Xm.iloc[len(tr):][feat])
    ntr_last = len(tr)
    if m == 7:
        imp = pd.Series(mdl.feature_importance('gain'), index=feat); imp = imp / imp.sum()
        print('  gain share (fold 7):', imp.sort_values(ascending=False).round(3).head(12).to_dict(), flush=True)
    print(f'  fold {m:02d}: train {len(tr):,} val {len(va):,}  {time.time()-tf:.0f}s  peak {peak_rss_gb():.2f} GB', flush=True)
np.save(OUT / f'oof_{arm}.npy', corr.astype('float32'))
nP = ~P
print(f'{arm}: {len(feat)} features; outside-P RMSE base {np.sqrt(np.mean(r[nP]**2)):.3f} -> {np.sqrt(np.mean((r[nP]-corr[nP])**2)):.3f} '
      f'| total {time.time()-t0:.0f}s | peak RSS {peak_rss_gb():.2f} GB')
