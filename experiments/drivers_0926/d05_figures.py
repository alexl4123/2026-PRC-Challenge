"""Drivers 09-26, step 5: figures for reports/2026-09-26_rmse_drivers_review.md (reads the CSV/NPY/JSON written by
d02-d04; recomputes only the zero-spike histogram from rows_training.parquet).

Palette: the validated categorical order blue, orange, aqua, yellow, magenta, green, violet, red (CVD-checked on
adjacent pairs); one-hue blue ramp for magnitudes; every chart with >= 2 series has a legend or direct labels.

    python experiments/drivers_0926/d05_figures.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from prc.config import CACHE

IN = CACHE / 'drivers_0926'
REP = CACHE.parent.parent / 'reports'
PRE = '2026-09-26_rmse_'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GREY = '#8a8984'; INK = '#0b0b0b'; INK2 = '#52514e'
BLUES = LinearSegmentedColormap.from_list('b', ['#f4f8fe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titleweight': 'bold', 'axes.titlesize': 11.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.7, 'axes.axisbelow': True})
MECH_LABEL = {'ord30': 'ordinary, label <= 30 min', 'ord60': 'ordinary, 30-60 min', 'ord120': 'ordinary, 1-2 h',
              'schedB': 'bulk with BLOCK = SCHED signature', 'Pfalse': 'zone P, real (late) pushback',
              'sched2h': 'zone P / >2 h, BLOCK = SCHED', 'early': 'Rome: BLOCK 1-2 h before SCHED',
              'long': 'other > 2 h', 'day': 'day-scale label (BLOCK a day early)'}
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']


def save(fig, name):
    fig.savefig(REP / f'{PRE}{name}.png'); plt.close(fig); print('wrote', REP / f'{PRE}{name}.png')


def f1_mechanism():
    order = ['day', 'ord30', 'ord60', 'ord120', 'schedB', 'Pfalse', 'sched2h', 'long', 'early']
    a = pd.read_csv(IN / 't1_mech_12m.csv', index_col=0).reindex(order); b = pd.read_csv(IN / 't1_mech_scored.csv', index_col=0).reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    yy = np.arange(len(order)); h = 0.38
    ax.barh(yy - h / 2, a.sse_pct, h, color=C[0], label='12 months 2025 (RMSE 234.6)')
    ax.barh(yy + h / 2, b.sse_pct, h, color=C[1], label='Jan + Jul 2025 (RMSE 296.9)')
    for i, m in enumerate(order):
        ax.text(max(a.sse_pct[m], b.sse_pct[m]) + 0.6, i, f'{int(a.rows[m]):,} rows', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels([MECH_LABEL[m] for m in order]); ax.invert_yaxis()
    ax.set_xlabel('share of squared error (%)'); ax.set_xlim(0, 48); ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title('15 day-scale label artefacts carry 27% of the 12-month squared error (7 rows: 40% of Jan+Jul)', loc='left')
    save(fig, 'mechanism')


def f2_concentration():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, (sc, lab) in enumerate((('12m', '12 months 2025'), ('scored', 'Jan + Jul 2025'))):
        x = np.load(IN / f'conc_{sc}_x.npy'); c = np.load(IN / f'conc_{sc}.npy')
        ax.plot(100 * x, 100 * c, color=C[i], lw=2, label=lab)
    s = json.load(open(IN / 'summary.json'))['concentration']
    for sc, i, dy in (('12m', 0, -7), ('scored', 1, 5)):
        for n in (10, 1000):
            frac = n / (2085047 if sc == '12m' else 344419) * 100
            ax.plot(frac, 100 * s[sc][f'top{n}'], 'o', ms=7, color=C[i], mec='white', mew=1.5)
            ax.annotate(f'top {n:,} rows: {100 * s[sc][f"top{n}"]:.0f}%', (frac, 100 * s[sc][f'top{n}']),
                        xytext=(8, dy), textcoords='offset points', fontsize=8.5, color=INK2)
    ax.set_xscale('log'); ax.set_xlim(2e-4, 100); ax.set_ylim(0, 101)
    ax.set_xlabel('share of flights, largest errors first (%, log scale)'); ax.set_ylabel('cumulative share of squared error (%)')
    ax.legend(loc='lower right', frameon=False)
    ax.set_title('0.1% of flights carry half of the 12-month error; 10 rows, 42% of Jan+Jul', loc='left')
    save(fig, 'concentration')


def f3_heatmap():
    order = ['ord30', 'ord60', 'ord120', 'schedB', 'Pfalse', 'sched2h', 'early', 'long', 'day']
    t = pd.read_csv(IN / 't2c_apt_mech_12m.csv', index_col=0)
    t[['apt', 'mech']] = t.index.to_series().str.split('|', expand=True).to_numpy()
    M = t.pivot(index='apt', columns='mech', values='sse_pct').reindex(index=AIRPORTS, columns=order).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(M.to_numpy(), cmap=BLUES, vmin=0, vmax=15, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if v >= 0.3:
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8.5, color='white' if v > 7 else INK)
    SHORT = {'ord30': 'ordinary\n<= 30 min', 'ord60': 'ordinary\n30-60 min', 'ord120': 'ordinary\n1-2 h', 'schedB': 'bulk\nBLOCK=\nSCHED', 'Pfalse': 'zone P\nreal push', 'sched2h': '> 2 h\nBLOCK=\nSCHED', 'early': 'Rome\nBLOCK 1-2 h\nearly', 'long': 'other\n> 2 h', 'day': 'day-scale\nlabel'}
    ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[m] for m in order], fontsize=8.5)
    ax.set_yticks(range(len(AIRPORTS))); ax.set_yticklabels(AIRPORTS); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.08); cb.set_label('% of 12-month squared error'); cb.outline.set_visible(False)
    tot = M.sum(axis=1)
    for i, a in enumerate(AIRPORTS):
        ax.text(len(order) - 0.4, i, f'{tot[a]:.1f}%', va='center', fontsize=8.5, color=INK2, fontweight='bold')
    ax.text(len(order) - 0.4, -0.75, 'airport\ntotal', fontsize=8, color=INK2)
    ax.set_title('12-month squared error by airport x mechanism: Rome everywhere; Istanbul, Paris, Heathrow in ordinary flights', loc='left', pad=22)
    save(fig, 'airport_mechanism')


def f4_adsb():
    t = pd.read_csv(IN / 't3_adsb_apt.csv')
    apts = ['EDDM', 'LSZH', 'EHAM', 'EDDF', 'EGLL', 'LEMD', 'LEBL', 'LIRF', 'LFPG', 'LTFM']
    st = ['off-block seen', 'ground only', 'no ground']
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)
    for k, s in enumerate(st):
        ax = axs[k]; d = t[t.adsb == s].set_index('apt').reindex(apts)
        yy = np.arange(len(apts))
        for i, a in enumerate(apts):
            if np.isfinite(d.rmse15.get(a, np.nan)):
                ax.plot([d.rmse15[a], d.rmse17[a]], [i, i], color='#c9c8c2', lw=2, zorder=1)
        ax.scatter(d.rmse15, yy, s=46, color=C[1], label='v15 structure (no ADS-B)', zorder=2, edgecolor='white', linewidth=1)
        ax.scatter(d.rmse17, yy, s=46, color=C[0], label='v17 structure (ADS-B inside)', zorder=3, edgecolor='white', linewidth=1)
        for i, a in enumerate(apts):
            if np.isfinite(d.rows.get(a, np.nan)) and d.rows[a] > 0:
                ax.text(385, i, (f'{int(d.rows[a]) / 1000:.0f}k' if d.rows[a] >= 1000 else f'{int(d.rows[a])}'), va='center', fontsize=8, color=INK2)
        ax.set_title(s, loc='left', fontsize=10.5); ax.set_xlim(40, 400); ax.set_xlabel('RMSE (s), ordinary A/B/C rows <= 2 h')
        ax.set_yticks(yy); ax.set_yticklabels(apts); ax.grid(axis='y', visible=False)
    axs[0].invert_yaxis(); fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Where adsb.lol sees the pushback, v17 error falls 70-85% -- except at Rome, where labels follow the recorder, not the aircraft',
                 x=0.01, ha='left', fontweight='bold', fontsize=11.5)
    fig.tight_layout(); save(fig, 'adsb_effect')


def f5_split():
    s = json.load(open(IN / 'summary.json'))['split']
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    parts = [('day', 'day-scale label artefacts (no pre-takeoff signal)', C[7]),
             ('recording', 'recording artefacts (stamping, zone-P classes)', C[3]),
             ('long', 'other > 2 h', C[4]),
             ('ord_reach', 'ordinary: above the ADS-B-observed floor', C[0]),
             ('ord_floor', 'ordinary: at the floor (RMSE 66 s)', C[2])]
    for i, sc in enumerate(('12m', 'scored')):
        v = s[sc]; vals = {'day': v['day_pct'], 'recording': v['recording_pct'], 'long': v['long_other_pct'],
                          'ord_reach': v['ordinary_pct'] - v['ordinary_floor4_pct'], 'ord_floor': v['ordinary_floor4_pct']}
        left = 0
        for key, lab, col in parts:
            ax.barh(i, vals[key], left=left, color=col, edgecolor='white', linewidth=2, label=lab if i == 0 else None, height=0.62)
            if vals[key] > 4:
                ax.text(left + vals[key] / 2, i, f'{vals[key]:.0f}%', ha='center', va='center', fontsize=9, color='white' if key in ('day', 'ord_reach') else INK)
            left += vals[key]
    ax.set_yticks([0, 1]); ax.set_yticklabels(['12 months 2025', 'Jan + Jul 2025']); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel('share of squared error (%)'); ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=8.5)
    ax.set_title('About a quarter (12m) to two fifths (Jan+Jul) of the error is unreachable; ordinary flights hold most of the rest', loc='left')
    save(fig, 'reachable_split')


def f6_mix():
    t = pd.read_csv(IN / 't12_adsb_mix.csv', index_col=[0, 1])
    st = [('off-block seen', C[0]), ('ground only', C[2]), ('no ground', '#b5b4ad')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for k, mo in enumerate((1, 7)):
        ax = axs[k]; yy = np.arange(len(AIRPORTS)); h = 0.38
        for j, yr in enumerate(('2025', '2026')):
            left = np.zeros(len(AIRPORTS))
            for s, col in st:
                v = np.array([t.loc[(a, mo), f'{yr} {s}'] if (a, mo) in t.index else 0 for a in AIRPORTS])
                ax.barh(yy + (j - 0.5) * h, v, h * 0.92, left=left, color=col, edgecolor='white', linewidth=1,
                        alpha=1.0 if yr == '2026' else 0.55, label=f'{s}' if (k == 0 and yr == '2026') else None)
                left += v
            for i in range(len(AIRPORTS)):
                ax.text(101, yy[i] + (j - 0.5) * h, yr, va='center', fontsize=7, color=INK2)
        ax.set_yticks(yy); ax.set_yticklabels(AIRPORTS); ax.set_xlim(0, 110); ax.grid(axis='y', visible=False)
        ax.set_title('January' if mo == 1 else 'July', loc='left', fontsize=10.5); ax.set_xlabel('% of departures')
    axs[0].invert_yaxis(); axs[0].legend(loc='lower left', bbox_to_anchor=(0, -0.3), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle('2026 is better covered by adsb.lol than 2025 (pale = 2025), except Rome in July (0%) and Frankfurt; Istanbul stays dark',
                 x=0.01, ha='left', fontweight='bold', fontsize=11.5)
    fig.tight_layout(); save(fig, 'adsb_mix_2025_2026')


def f7_expected():
    t = pd.read_csv(IN / 't10b_expected_2026_airports.csv', index_col=0).reindex(AIRPORTS)
    t = t.sort_values('sse_pct', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.4)); yy = np.arange(len(t)); h = 0.38
    ax.barh(yy + h / 2, t.sse_pct25, h, color=C[1], label='2025 Jan+Jul, measured')
    ax.barh(yy - h / 2, t.sse_pct, h, color=C[0], label='2026 Jan+Jul, expected (2025 cell error x 2026 mix)')
    for i, a in enumerate(t.index):
        ax.text(max(t.sse_pct[a], t.sse_pct25[a]) + 0.5, i, f'RMSE {t.rmse25[a]:.0f} -> {t.rmse[a]:.0f}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels(t.index); ax.set_xlabel('share of squared error, day-scale artefacts excluded (%)')
    ax.set_xlim(0, 50); ax.grid(axis='y', visible=False); ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    ax.set_title('On the 2026 mix Rome is expected to carry ~40% of the error, Istanbul 20%, Paris 13%', loc='left')
    save(fig, 'expected_2026')


def f8_zero_spike():
    r = pd.read_parquet(IN / 'rows_training.parquet', columns=['apt', 'y', 'off_sched'])
    d = (r.y - r.off_sched).to_numpy(); m0 = (r.off_sched.to_numpy() > 1200) & (r.y.to_numpy() <= 7200) & np.isfinite(d)
    fig, ax = plt.subplots(figsize=(8, 3.8)); bins = np.arange(-600, 601, 60) - 30
    for i, (lab, sel) in enumerate((('Rome LIRF', r.apt == 'LIRF'), ('Barcelona LEBL', r.apt == 'LEBL'),
                                     ('Heathrow EGLL', r.apt == 'EGLL'), ('other 7 airports', ~r.apt.isin(['LIRF', 'LEBL', 'EGLL'])))):
        m = m0 & sel.to_numpy()
        hgt, _ = np.histogram(d[m], bins=bins)
        ax.plot(bins[:-1] / 60 + 0.5, 100 * hgt / m.sum(), '-o', color=C[i], lw=2, ms=4, label=lab)
    ax.set_xlabel('SCHED - BLOCK (minutes)'); ax.set_ylabel('% of delayed flights (off_sched > 20 min)')
    ax.legend(frameon=False); ax.set_xlim(-10.5, 10.5)
    ax.set_title('Rome writes BLOCK = SCHED on 11% of its delayed departures (excess over neighbouring minutes)', loc='left')
    save(fig, 'zero_spike')


def f9_days_hours():
    gd = pd.read_csv(IN / 't6_airport_days.csv')
    h = pd.read_csv(IN / 't5_hour.csv', index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axs[0]
    ss = np.sort(gd.sse.to_numpy())[::-1]; cc = np.cumsum(ss) / ss.sum(); x = np.arange(1, len(ss) + 1) / len(ss) * 100
    ax.plot(x, 100 * cc, color=C[0], lw=2, label='airport-days, worst first')
    xe = np.geomspace(0.03, 100, 200); ax.plot(xe, xe, color='#b5b4ad', lw=1, ls='--', label='equal share')
    for p in (1, 5, 10):
        ax.plot(p, 100 * cc[int(p / 100 * len(ss)) - 1], 'o', color=C[0], mec='white', mew=1.5, ms=7)
        ax.annotate(f'worst {p}%: {100 * cc[int(p / 100 * len(ss)) - 1]:.0f}%', (p, 100 * cc[int(p / 100 * len(ss)) - 1]),
                    xytext=(10, -6), textcoords='offset points', fontsize=8.5, color=INK2)
    ax.set_xscale('log'); ax.set_xlim(0.03, 100); ax.set_ylim(0, 101)
    ax.set_xlabel('share of the 3,653 airport-days (%, log)'); ax.set_ylabel('cumulative share of ordinary-flight SSE (%)')
    ax.legend(frameon=False, loc='upper left', fontsize=8.5); ax.set_title('Disruption days: worst 5% of days = 23% of ordinary error', loc='left', fontsize=10.5)
    ax = axs[1]
    ax.bar(h.index, h.rmse, color=C[0], width=0.8)
    ax2c = h.row_pct
    for hh in h.index:
        if h.row_pct[hh] < 1.5:
            ax.bar(hh, h.rmse[hh], color='#b5b4ad', width=0.8)
    ax.set_xlabel('local hour of take-off'); ax.set_ylabel('RMSE (s), ordinary flights'); ax.set_xticks(range(0, 24, 3))
    ax.grid(axis='x', visible=False)
    ax.set_title('Night hours (grey, < 1.5% of rows each) are worse but small', loc='left', fontsize=10.5)
    fig.tight_layout(); save(fig, 'days_hours')


if __name__ == '__main__':
    for fn in (f1_mechanism, f2_concentration, f3_heatmap, f4_adsb, f5_split, f6_mix, f7_expected, f8_zero_spike, f9_days_hours):
        fn()
