"""Figure for reports/2026-09-26_adsb_v2.md: on the flights ADS-B v2 adds (no production off-block), the median
|BLOCK-implied error| per airport of the back-calculated estimate, AOBT_3, and the residual model with the production
ADS-B columns (screen arm OLD, leave-one-day-out) -- i.e. why the estimator adds nothing: the model is already better.

    python experiments/adsb/adsb_v2_figures.py --pre cache/adsb_lol/match/v2_measure_final --v2 <features_v2 file> --out <png>

Palette: dataviz reference categorical slots 1-3 (light), validated all-pairs; marker shapes as secondary encoding.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adsb_match_measure import load_deps  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', type=Path, required=True)
    ap.add_argument('--v2', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    pr = pd.read_parquet(f'{a.pre}_preds.parquet')
    v2 = pd.read_parquet(a.v2, columns=['MVT_ID_mvt', 'adsb2_off_bc_stand', 'adsb2_off_best2'])
    prod = pd.read_parquet(REPO_ROOT / 'cache' / 'adsb_lol' / 'features_training.parquet',
                           columns=['MVT_ID_mvt', 'adsb_off_best'])
    m = pr.merge(v2, on='MVT_ID_mvt').merge(prod, on='MVT_ID_mvt', how='left')
    days = sorted(pd.read_csv(f'{a.pre}_screen_days.csv').day)
    dep = load_deps(days, 'training')[['MVT_ID_mvt', 'APT', 'mvt_s', 'block_s', 'aobt3_s']]
    m = m.merge(dep, on='MVT_ID_mvt')
    m['APT'] = m.APT.astype(str)
    add = m.adsb_off_best.isna() & m.adsb2_off_bc_stand.notna() & (m.block_s < m.mvt_s) & (m.y <= 7200) & m.aobt3_s.notna()
    q = m[add]
    rows = []
    for apt, g in q.groupby('APT'):
        if len(g) < 100:
            continue
        rows.append(dict(APT=apt, n=len(g), bc=np.median(np.abs(g.adsb2_off_bc_stand - g.y)),
                         aobt3=np.median(np.abs(g.mvt_s - g.aobt3_s - g.y)),
                         model=np.median(np.abs(g.pred_OLD - g.y))))
    r = pd.DataFrame(rows).sort_values('model')
    r.to_csv(f'{a.pre}_figure_data.csv', index=False)
    print(r.round(0).to_string(index=False))
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    y = np.arange(len(r))
    for yy, (_, rr) in zip(y, r.iterrows()):
        lo, hi = min(rr.bc, rr.aobt3, rr.model), max(rr.bc, rr.aobt3, rr.model)
        ax.plot([lo, hi], [yy, yy], color=GRID, lw=2, zorder=1, solid_capstyle='round')
    specs = [('bc', 'v2 back-calculated off-block', 'o'), ('aobt3', 'AOBT_3', 's'),
             ('model', 'model on production ADS-B columns', 'D')]
    for (col, lab, mk), c in zip(specs, SERIES):
        ax.scatter(r[col], y, s=64, marker=mk, color=c, edgecolor=SURF, linewidth=2, zorder=3, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{a_} (n={n:,})' for a_, n in zip(r.APT, r.n)], color=INK, fontsize=9)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, None)
    ax.set_xlabel('median |error| of the taxi time implied (s), flights v2 adds (no production off-block)',
                  color=INK2, fontsize=9)
    ax.set_title('Flights v2 adds: the existing model already beats the new estimate',
                 color=INK, fontsize=10.5, loc='left', pad=26)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8, labelcolor=INK2,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=SURF)
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
