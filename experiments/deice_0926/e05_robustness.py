"""De-icing / disruption indicator, step 5 (POST HOC -- written after reading e02, so nothing here changes the
pre-registered verdict; it only explains it).

Why: in e02 most of the cand - control SSE gain sat at LTFM (no ADS-B coverage at all, so every candidate column
is constant there: counts 0, shares NaN) and on rows whose hour share is NaN. A gain on rows where the candidate
carries no information can only be re-fit variation of the residual model. This step measures that directly:

  seed replicates   control and cand re-fitted with LightGBM seed 1 and 2 (same rows, same folds):
                    control_sK - control = the pure re-fit noise of a residual model with identical inputs;
                    cand_sK - control_sK = the candidate effect at another seed
  ablations (seed 0)  cand_shares = cand without the two covered-count columns (dz_ncov_h, dz_ncov_d):
                      the de-icing signal alone;  cand_ncov = only those two count columns (a traffic x coverage
                      count, not a de-icing signal)
  SSE split           for every comparison: the SSE change on "informative" rows (hour remote-rest share > 5%)
                      vs on rows where the share is NaN (no coverage in the hour) -- the latter is re-fit noise

    python experiments/deice_0926/e05_robustness.py
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from e02_residual import load_B, block_bootstrap, CAND, OUT       # sets DISCORD_SCRATCH, imports the harness
import common
from common import residual_arms, rmse, concentration
from prc.io import peak_rss_gb


def main():
    t0 = time.time()
    B = load_B(); ind = B['ind']
    y, month, blk = B['y'], B['month'], B['block']
    S = np.isin(month, [1, 7])
    preds = residual_arms(B, {'control': None, 'cand': ind[CAND]}, tag='deice')          # cached from e02
    shares = [c for c in CAND if 'ncov' not in c]
    preds.update(residual_arms(B, {'cand_shares': ind[shares], 'cand_ncov': ind[['dz_ncov_h', 'dz_ncov_d']]}, tag='deice'))
    for s in (1, 2):
        common.PARAMS['seed'] = s
        p = residual_arms(B, {'control': None, 'cand': ind[CAND]}, tag=f'deice_seed{s}')
        preds[f'control_s{s}'] = p['control']; preds[f'cand_s{s}'] = p['cand']
    common.PARAMS['seed'] = 0
    sh = ind.dz_rr_sh_h.to_numpy()
    info = sh > .05
    nan_ = np.isnan(sh)
    apt = B['apt']
    pairs = [('control_s1', 'control'), ('control_s2', 'control'), ('control_s2', 'control_s1'),
             ('cand', 'control'), ('cand_s1', 'control_s1'), ('cand_s2', 'control_s2'),
             ('cand_shares', 'control'), ('cand_ncov', 'control'), ('cand', 'cand_shares')]
    out = {}
    print(f'{"a - b":28s} {"12m":>7s} {"scored":>7s} {"Jan":>7s} {"Jul":>7s} {"blk12m lo..hi":>17s} {"blkSc lo..hi":>17s} '
          f'{"drop100 12m/sc":>15s} | dSSE: info rows | NaN-share rows | LTFM   (1e6 s^2)', flush=True)
    for a, b in pairs:
        pa, pb = preds[a], preds[b]
        d = (y - pa) ** 2 - (y - pb) ** 2
        r = dict(d12=rmse(y, pa) - rmse(y, pb), dS=rmse(y[S], pa[S]) - rmse(y[S], pb[S]),
                 dJ=rmse(y[month == 1], pa[month == 1]) - rmse(y[month == 1], pb[month == 1]),
                 dL=rmse(y[month == 7], pa[month == 7]) - rmse(y[month == 7], pb[month == 7]),
                 b12=block_bootstrap(y, pa, pb, blk, 1000), bS=block_bootstrap(y[S], pa[S], pb[S], blk[S], 1000),
                 drop12=concentration(y, pa, pb)[0], dropS=concentration(y[S], pa[S], pb[S])[0],
                 sse_info=float(d[info].sum()), sse_nan=float(d[nan_].sum()), sse_ltfm=float(d[apt == 'LTFM'].sum()),
                 sse_all=float(d.sum()), sse_info_jan=float(d[info & (month == 1)].sum()),
                 months_better=int(sum(rmse(y[month == m], pa[month == m]) < rmse(y[month == m], pb[month == m]) for m in range(1, 13))))
        out[f'{a} - {b}'] = r
        print(f'{a + " - " + b:28s} {r["d12"]:+7.3f} {r["dS"]:+7.3f} {r["dJ"]:+7.3f} {r["dL"]:+7.3f} '
              f'[{r["b12"][1]:+.3f},{r["b12"][2]:+.3f}] [{r["bS"][1]:+.3f},{r["bS"][2]:+.3f}] '
              f'{r["drop12"]:+.2f}/{r["dropS"]:+.2f} | {r["sse_info"] / 1e6:+8.1f} | {r["sse_nan"] / 1e6:+8.1f} | '
              f'{r["sse_ltfm"] / 1e6:+8.1f}  (all {r["sse_all"] / 1e6:+.1f}; info Jan {r["sse_info_jan"] / 1e6:+.1f}; '
              f'{r["months_better"]}/12)', flush=True)
    # informative rows only: block bootstrap of the RMSE change restricted to hot hours (share > 5%)
    for a, b in (('cand', 'control'), ('cand_s1', 'control_s1'), ('cand_s2', 'control_s2'), ('cand_shares', 'control'),
                 ('control_s1', 'control')):
        k = info
        bb = block_bootstrap(y[k], preds[a][k], preds[b][k], blk[k], 1000)
        kj = info & (month == 1)
        bj = block_bootstrap(y[kj], preds[a][kj], preds[b][kj], blk[kj], 1000)
        out[f'{a} - {b} | info rows'] = dict(all=bb, jan=bj, n=int(k.sum()), n_jan=int(kj.sum()))
        print(f'  hot-hour rows (share > 5%, n={int(k.sum()):,}): {a} - {b}: {bb[0]:+.2f} [{bb[1]:+.2f},{bb[2]:+.2f}] '
              f'over {bb[3]} airport-days;  January only (n={int(kj.sum()):,}): {bj[0]:+.2f} [{bj[1]:+.2f},{bj[2]:+.2f}]', flush=True)
    json.dump(out, open(OUT / 'e05_results.json', 'w'), indent=1, default=float)
    print(f'peak RSS {peak_rss_gb():.2f} GB, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
