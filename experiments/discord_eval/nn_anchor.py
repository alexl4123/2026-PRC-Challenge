"""Idea 3 (a participant's suggestion on the challenge Discord): does the W2 NETWORK gain from predicting the target anchored on
the NM off-block time, r = y - off_aobt3, instead of y?

The trees already failed this (IDEAS 'Rejected on evidence': prc/models/anchored.py, anchored on
ALL rows incl. the AOBT-missing branch). Here the anchor is applied only where it is defined and
sane: route A (0 < off_aobt3 <= 7200). Laptop-sized screen, same inputs as the production 'nn'
member (cluster/w2_task.py: W2 columns, numeric_design, embed_nn.fit_predict, emb cap 16):

  Y_all  target y, trained on a 15% subsample of ALL bulk training rows (production-like)
  Y_A    target y, trained on the route-A part of the same subsample
  R_A    target y - off_aobt3 (+ a constant 10,000 so the network's final clip at 0 never binds),
         trained on the same route-A rows; the anchor is added back at prediction time
Held-out months 1 and 7 (the scored months), seeds 0 and 1, 4 epochs, CPU, 6 threads.
Evaluated on held-out route-A bulk rows (y <= 7200): RMSE, paired bootstrap, and the 50/50 average
with Y_all (does the anchored network add diversity even if it is not better alone?).

    python experiments/discord_eval/nn_anchor.py [--seeds 2 3 4 --out nn_anchor_s234.json]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cluster.w2_features import assemble_w2, columns, W2_CATS
from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.models.altfam import numeric_design
from prc.models.embed_nn import fit_predict
from prc.validation import paired_bootstrap
from common import SCRATCH

C = 10_000.0
FRAC, EPOCHS, THREADS = 0.15, 4, 6


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--out', default='nn_anchor.json')
    args = ap.parse_args()
    t0 = time.time()
    X, y, month, groups = assemble_w2()
    cols, _ = columns(groups)
    a3 = X['off_aobt3'].to_numpy(float)
    routeA = np.isfinite(a3) & (a3 > 0) & (a3 <= TAIL_CAP)
    print(f'assembled {X.shape} in {time.time() - t0:.0f} s, peak {peak_rss_gb():.2f} GB; route A {routeA.mean():.3f}', flush=True)
    res = {}
    for m in (1, 7):
        trn_all = (month != m) & (y <= TAIL_CAP)
        codes, cards, Z = numeric_design(X[cols], cols, fit_mask=trn_all, cats=W2_CATS)[:3]
        ev = (month == m) & routeA & (y <= TAIL_CAP)
        for seed in args.seeds:
            keep = np.random.default_rng(100 * m + seed).random(len(y)) < FRAC
            tr_all, tr_A = trn_all & keep, trn_all & keep & routeA
            t1 = time.time()
            q = lambda msg: None
            p_yall = fit_predict(codes, Z, y, tr_all, ev, epochs=EPOCHS, seed=seed, threads=THREADS, emb_cap=16, log=q)
            p_ya = fit_predict(codes, Z, y, tr_A, ev, epochs=EPOCHS, seed=seed, threads=THREADS, emb_cap=16, log=q)
            r = np.where(routeA, y - a3 + C, 0.0)
            p_ra = fit_predict(codes, Z, r, tr_A, ev, epochs=EPOCHS, seed=seed, threads=THREADS, emb_cap=16, log=q) - C + a3[ev]
            p_ra = np.clip(p_ra, 0, None)
            yy = y[ev]
            rm = lambda p: float(np.sqrt(np.mean((yy - p) ** 2)))
            mix = 0.5 * p_ra + 0.5 * p_yall
            out = {'n_eval': int(ev.sum()), 'n_train_all': int(tr_all.sum()), 'n_train_A': int(tr_A.sum()),
                   'rmse_Y_all': rm(p_yall), 'rmse_Y_A': rm(p_ya), 'rmse_R_A': rm(p_ra), 'rmse_mix_RA_Yall': rm(mix),
                   'rmse_mix_YA_Yall': rm(0.5 * p_ya + 0.5 * p_yall),
                   'resid_corr_RA_Yall': float(np.corrcoef(yy - p_ra, yy - p_yall)[0, 1]),
                   'resid_corr_YA_Yall': float(np.corrcoef(yy - p_ya, yy - p_yall)[0, 1])}
            for nm, a_, b_ in (('RA_vs_YA', p_ra, p_ya), ('RA_vs_Yall', p_ra, p_yall), ('YA_vs_Yall', p_ya, p_yall)):
                bb = paired_bootstrap(yy, a_, b_, n_boot=300)
                out[nm] = (bb['diff'], bb['lo'], bb['hi'])
            # where does it differ: large-anchor rows (off_aobt3 > 3600) vs the rest
            big = a3[ev] > 3600
            out['big_rows'] = int(big.sum())
            out['big_rmse'] = {k: float(np.sqrt(np.mean((yy[big] - p[big]) ** 2))) for k, p in
                               (('Y_all', p_yall), ('Y_A', p_ya), ('R_A', p_ra))}
            res[f'm{m}_s{seed}'] = out
            print(f'month {m} seed {seed} ({time.time() - t1:.0f} s): ' + json.dumps(out), flush=True)
    json.dump(res, open(SCRATCH / args.out, 'w'), indent=1)
    print(f'peak RSS {peak_rss_gb():.2f} GB, total {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
