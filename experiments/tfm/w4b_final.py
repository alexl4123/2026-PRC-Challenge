"""W4b -- production arrays for the W4 `pool` arm on the NON-ROME P rows of the ranking file.

W4 (w4_tail_zone.py, 2026-09-24) passed its pre-registered rule for the non-Rome P rows only:
    pool on non-Rome P vs cur: 12m -0.10 [-0.17,-0.04], scored -0.17 [-0.39,+0.02], 8/12 months
(Rome keeps v13's per-airport TabICL: pool on Rome was +1.09 scored, 5/12 months.)

Fits TabICL on ALL 4,944 P rows of 2025 (38 numeric exp62 features + airport one-hot, as in W4)
and predicts every non-Rome P row of the ranking file. Mean of 3 random states. Writes
cache/tfm_w4/pool_nonrome_ranking.npz with pos (DIRECT row index into the ranking DEP order, as
in tfm_P_lirf_ranking.npz), apt, month, p_pool. Assembly happens in the submission builder.

usage: HF_HUB_OFFLINE=1 systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
           nice -n 5 python experiments/tfm/w4b_final.py --threads 3
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE

KEYS = ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=1)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from tabicl import TabICLClassifier

    d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['month', 'y', 'off_sched', 'off_aobt3'])
    P = ~np.isfinite(d.off_aobt3.to_numpy()) & (d.off_sched.to_numpy() > 7200)
    t = (d.y.to_numpy()[P] > 7200).astype(int)
    tr = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    rk0 = pd.read_parquet(CACHE / 'exp62_cand_ranking.parquet')
    rk_rows = rk0.index.to_numpy()                       # DIRECT row positions in the ranking DEP order
    rk = rk0.reset_index(drop=True)
    assert len(tr) == P.sum()
    num = [c for c in tr.columns if c not in KEYS and tr[c].dtype != object]
    assert all(c in rk.columns for c in num), 'ranking candidates lack a training feature'
    apts = sorted(set(tr.apt.astype(str)))
    assert set(rk.apt.astype(str)) <= set(apts), 'ranking has an airport unseen in training P'

    def mat(df):
        oh = np.stack([(df.apt.astype(str).to_numpy() == x).astype(float) for x in apts], axis=1)
        return np.hstack([df[num].astype(float).to_numpy(), oh])

    Xtr, Xrk = mat(tr), mat(rk)
    q = rk.apt.astype(str).to_numpy() != 'LIRF'
    print(f'context: all {len(tr)} P rows ({t.sum()} tails), {Xtr.shape[1]} features; '
          f'query: {q.sum()} non-Rome ranking P rows', flush=True)
    cov_tr, cov_rk = np.isfinite(Xtr).mean(0), np.isfinite(Xrk[q]).mean(0)
    bad = [(c, round(x, 3), round(y, 3)) for c, x, y in zip(num, cov_tr, cov_rk) if abs(x - y) > 0.10]
    print('feature coverage gaps > 10 pp (train vs ranking):', bad if bad else 'none', flush=True)
    ps = []
    for rs in (0, 1, 2):
        t0 = time.time()
        clf = TabICLClassifier(device=a.device, random_state=rs, n_jobs=a.threads, batch_size=a.batch_size)
        clf.fit(Xtr, t)
        ps.append(clf.predict_proba(Xrk[q])[:, 1])
        print(f'random_state {rs}: {time.time() - t0:.0f} s', flush=True)
    p = np.mean(ps, axis=0)
    out = dict(pos=rk_rows[q], apt=rk.apt.astype(str).to_numpy()[q], month=rk.month.to_numpy()[q], p_pool=p)
    for x in sorted(set(out['apt'])):
        k = out['apt'] == x
        print(f'  {x}: n {k.sum():4d}  mean p {p[k].mean():.4f}  max {p[k].max():.3f}  p>0.5 {int((p[k] > 0.5).sum())}')
    np.savez(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', **out)
    print('saved cache/tfm_w4/pool_nonrome_ranking.npz')


if __name__ == '__main__':
    main()
