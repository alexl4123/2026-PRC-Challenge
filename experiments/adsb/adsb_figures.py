"""Figures for reports/2026-09-25_adsb_lol_feasibility.md (reads the CSVs written by adsb_match_measure.py).

    python experiments/adsb/adsb_figures.py OUTDIR  PERIOD=cache/adsb_lol/match/<file>.coverage.csv ...
           --acc cache/adsb_lol/match/<training file>.summary.csv

Fig 1: per airport, share of departures with ADS-B ground points before take-off and with an
       off-block estimate (pushback / stand rest), one bar per period.
Fig 2: per airport, median and p90 |off-block - BLOCK| for the ADS-B estimate vs AOBT_3 on the same flights.
Palette: dataviz reference categorical slots 1-4 (light mode), text in neutral inks.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
APTS = ['EHAM', 'EDDF', 'LSZH', 'LEBL', 'EDDM', 'EGLL', 'LEMD', 'LIRF', 'LFPG', 'LTFM']


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_coverage(periods, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True, facecolor=SURF)
    k = len(periods)
    h = 0.8 / k
    y = np.arange(len(APTS))
    for ax, col, title in ((axes[0], 'ground_pct', 'ADS-B ground points before take-off'),
                           (axes[1], 'best_pct', 'Off-block observed (pushback or stand rest)')):
        style(ax)
        for i, (name, df) in enumerate(periods):
            v = df.reindex(APTS)[col].to_numpy()
            ax.barh(y + (i - (k - 1) / 2) * h, v, height=h * 0.9, color=SERIES[i], label=name,
                    edgecolor=SURF, linewidth=1)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlim(0, 100)
        ax.set_xlabel('% of departures', color=INK2, fontsize=9)
    axes[0].set_yticks(y, APTS)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=9, loc='lower right', labelcolor=INK2)
    fig.suptitle('adsb.lol coverage per airport and period (sample days)', color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_accuracy(acc, out):
    a = acc.reindex([x for x in APTS if x in acc.index and acc.loc[x, 'n_best'] >= 10])
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURF)
    style(ax)
    y = np.arange(len(a))
    for off, (med, p90, name, c) in zip((-0.15, 0.15), (('best_abs_med', 'best_abs_p90', 'ADS-B off-block', SERIES[0]),
                                                        ('aobt3_abs_med_same', 'aobt3_abs_p90_same', 'AOBT_3 (NM)', SERIES[1]))):
        ax.hlines(y + off, a[med], a[p90], color=c, lw=2)
        ax.plot(a[med], y + off, 'o', ms=8, color=c, mec=SURF, mew=2, label=f'{name} (dot = median, line end = p90)')
    ax.set_yticks(y, [f'{x} (n={int(a.loc[x, "n_best"])})' for x in a.index])
    ax.invert_yaxis()
    ax.set_xlabel('|estimate - BLOCK_TIME| (s), same flights', color=INK2, fontsize=9)
    ax.set_title('Off-block accuracy where ADS-B observed it (July 2025 sample days)', color=INK, fontsize=11,
                 loc='left', pad=44)
    ax.legend(frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=1, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', type=Path)
    ap.add_argument('periods', nargs='+', help='NAME=path/to/coverage.csv')
    ap.add_argument('--acc', type=Path)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    periods = [(p.split('=', 1)[0], pd.read_csv(p.split('=', 1)[1], index_col=0)) for p in a.periods]
    fig_coverage(periods, a.outdir / '2026-09-25_adsb_coverage.png')
    if a.acc:
        fig_accuracy(pd.read_csv(a.acc, index_col=0), a.outdir / '2026-09-25_adsb_accuracy.png')
    print('written to', a.outdir)


if __name__ == '__main__':
    main()
