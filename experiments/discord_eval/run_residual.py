"""Residual harness for the Discord ideas 2, 4, 5, 6 on top of the v15-structure 2025 OOF.

Arms (each = context columns + the listed candidate columns; see common.py for the protocol):
  control    APT, off_aobt3, off_sched, base (v15 OOF) only
  wx_td      + wx_tmpc, wx_dwpc                            (what the members already hold)
  wx_spread  + wx_tmpc, wx_dwpc, spread, frost flags, 12 h minimum temperature / spread  (idea 4)
  wake       + own / previous-departure wake category, pair, required separation, gap, heavies
               in the previous 10 min on the runway                                    (idea 5)
  backlog    + overdue / gate-hold / EOBT-overdue counts at our SCHED, AOBT_3, takeoff (idea 6)
  te_srw     + nested out-of-month (APT, STAND, RUNWAY) target encoding and its log count (idea 2)
  te_all     + te_srw, te_srw - te_rw, te_rw, te_stand, log count                    (idea 2)
  adsb_dist  + straight-line ADS-B stand -> lift-off-point distance (NaN for ~72% of rows) (idea 2b)

    python experiments/discord_eval/run_residual.py --build-only
    python experiments/discord_eval/run_residual.py [--arms control wake ...]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import load_base, residual_arms, report_arms, compare, fmt, SCRATCH
from prc.io import peak_rss_gb
import features as F

ALL = ['control', 'wx_td', 'wx_spread', 'wake', 'backlog', 'te_srw', 'te_all', 'adsb_dist']


def perm_within_month(month, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(month))
    for m in np.unique(month):
        k = np.flatnonzero(month == m); idx[k] = rng.permutation(k)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=ALL)
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--save', default='residual_results.json')
    a = ap.parse_args()
    t0 = time.time()
    B = load_base()
    print(f'base loaded {time.time() - t0:.0f} s, peak {peak_rss_gb():.2f} GB', flush=True)
    wx = F.weather_features(); print(f'weather {wx.shape}, spread NaN {wx.wx_spread.isna().mean():.3f}', flush=True)
    wk = F.wake_features(); print(f'wake {wk.shape}, prev NaN {wk.wk_prev.isna().mean():.3f}', flush=True)
    bl = F.backlog_features(); print(f'backlog {bl.shape}\n{bl.describe().T[["mean", "50%", "max"]]}', flush=True)
    ad = F.adsb_distance(); print(f'adsb distance coverage {ad.dist_stand_liftoff.notna().mean():.3f}', flush=True)
    te = F.StandTE(B['y'], B['month'])
    print(f'features built {time.time() - t0:.0f} s, peak {peak_rss_gb():.2f} GB', flush=True)
    if a.build_only:
        t = te.fold(7)
        print(t.describe().T)
        return
    te_cols = ['te_srw', 'te_srw_logn']
    spec = {'control': None,
            'wx_td': wx[['wx_tmpc', 'wx_dwpc']],
            'wx_spread': wx,
            'wake': wk,
            'backlog': bl,
            'te_srw': lambda m: te.fold(m)[te_cols],
            'te_all': lambda m: te.fold(m),
            'adsb_dist': ad,
            # ablations / controls added after the first pass
            'wake_base': wk[['wk_own', 'wk_gap_prev']],             # own category + gap only (no sequence)
            'wake_perm': wk.iloc[perm_within_month(B['month'])].reset_index(drop=True),   # null: shuffled rows
            'backlog_sched': bl[[c for c in bl.columns if c.endswith('_at_sched')]],
            'backlog_a3': bl[[c for c in bl.columns if c.endswith('_at_a3') or c.endswith('_at_mvt')]],
            'wake_backlog': pd.concat([wk, bl], axis=1),
            'wake_backlog_sched': pd.concat([wk, bl[[c for c in bl.columns if c.endswith('_at_sched')]]], axis=1)}
    arms = {k: spec[k] for k in (['control'] + [x for x in a.arms if x != 'control'])}
    preds = residual_arms(B, arms, tag='res')
    print(f'\nRESULTS (all flights; negative = arm better) -- {time.time() - t0:.0f} s', flush=True)
    report_arms(B, preds, save=a.save)
    for a_, b_ in (('wx_spread', 'wx_td'), ('wake', 'wake_base'), ('wake_backlog', 'backlog'), ('wake_backlog', 'wake'),
                   ('wake_backlog_sched', 'wake'), ('wake_backlog_sched', 'wake_backlog')):
        if a_ in preds and b_ in preds:
            print(f"  {a_} vs {b_:14s}: {fmt(compare(B, preds[a_], preds[b_]))}", flush=True)
    print(f'peak RSS {peak_rss_gb():.2f} GB, total {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
