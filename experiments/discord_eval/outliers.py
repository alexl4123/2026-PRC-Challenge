"""Idea 10 (lbragado, Discord l.689): "force the submission's outliers to look like the training
outliers". Concrete, label-free-at-serving readings, priced on the v15-structure 2025 OOF
(leave-one-month-out: the held-out month's own predictions + the OTHER months' labels only):

  QM_full(q0=0)   per airport, map each held-out prediction's within-month rank to the training
                  months' label quantile at the same level (full quantile mapping)
  QM_tail(q0)     the same, applied only to predictions above their month's q0 quantile
                  (q0 = 0.99, 0.995, 0.999): the upper tail of the submission is made to have the
                  training label tail's shape
  CNT(7200)       tail-count matching: per airport, the top-k predictions of the held-out month,
                  k = round(training rate of y > 7200 x n), are raised to at least 7200 s... and
                  the rest capped below 7200 -- the "same number of outliers" reading

    python experiments/discord_eval/outliers.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load_base, compare, fmt, rmse, SCRATCH


def qmap(pred, month, apt, y, q0):
    out = pred.copy()
    for m in range(1, 13):
        for a in np.unique(apt):
            te = np.flatnonzero((month == m) & (apt == a))
            tr = (month != m) & (apt == a) & np.isfinite(y)
            if len(te) == 0:
                continue
            lvl = (np.argsort(np.argsort(pred[te])) + 0.5) / len(te)
            sel = lvl >= q0
            out[te[sel]] = np.quantile(y[tr], lvl[sel])
    return out


def cntmatch(pred, month, apt, y, thr=7200):
    out = pred.copy()
    for m in range(1, 13):
        for a in np.unique(apt):
            te = np.flatnonzero((month == m) & (apt == a))
            tr = (month != m) & (apt == a) & np.isfinite(y)
            k = int(round((y[tr] > thr).mean() * len(te)))
            o = te[np.argsort(-pred[te])]
            out[o[:k]] = np.maximum(pred[o[:k]], thr)
            out[o[k:]] = np.minimum(pred[o[k:]], thr)
    return out


def main():
    B = load_base()
    y, month, base = B['y'], B['month'], B['base']
    apt = B['ctx'].APT.astype(str).to_numpy()
    S = np.isin(month, [1, 7])
    # how different are the prediction and label tails? (per-month, all airports)
    for q in (0.99, 0.995, 0.999):
        print(f'q{q}: prediction {np.quantile(base, q):8.0f}  label {np.quantile(y, q):8.0f}')
    print(f'share > 7200: prediction {np.mean(base > 7200):.5f}  label {np.mean(y > 7200):.5f}')
    res = {}
    for name, p in (('QM_full', qmap(base, month, apt, y, 0.0)),
                    ('QM_tail_0.99', qmap(base, month, apt, y, 0.99)),
                    ('QM_tail_0.995', qmap(base, month, apt, y, 0.995)),
                    ('QM_tail_0.999', qmap(base, month, apt, y, 0.999)),
                    ('CNT_7200', cntmatch(base, month, apt, y))):
        r = compare(B, p, base, n_boot=300); res[name] = r
        print(f'{name:14s} vs v15s: {fmt(r)}', flush=True)
    json.dump(res, open(SCRATCH / 'outliers.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
