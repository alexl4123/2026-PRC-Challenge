"""W2 feature assembly: the members' 72 columns plus the v11 feature-stack signals.

Until v13 the v11 signals only reached the submission through a residual booster on a frozen
blend (stack arm c, -5.7 s over 12 months). W2 moves them INTO the members, so each model can
use them in interaction with everything else. Added to cluster/cv_task.assemble()'s columns:

  clocks    nm_delay, off_iobt, lobt_m_iobt, eobt_m_lobt, a3_m_eobt, a3_m_lobt, eobt_m_sched,
            iobt_m_sched                        (NM flight-plan clock differences)
  stamp     pa_stamped, pa_inb_m_sibt, pa_ldg_m_sibt
            (was the previous arrival at our stand in-blocked at its scheduled time)
  turnover  si_before_sched, si_same_typ, sd_prev_to                          (backward)
            sl_to, sl_after_sched, sl_same_typ, sl_same_op, sn_minus_to, sn_same_op,
            sn_same_typ, sa_after_to, sd_next_to                            (look-ahead)
  ident     ADES_mvt as a categorical (from the raw column, so training and ranking share one
            vocabulary; the v11 parquet only holds per-file codes)

The numeric columns are read from cache/v11/feat_{training,ranking}.parquet, which
experiments/v11/v11_features.py built MONTH BY MONTH from the provided files only (no label,
no cross-month events). They are joined by row order and the MVT_ID of every row is asserted
equal, so a stale or reordered parquet stops the task instead of silently misaligning.
Nothing here writes to cache/v11/.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import TARGET, ID, CACHE
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.disruption import attach as attach_atfm
from prc.io import load
from prc.weather import download, prepare, attach as attach_wx

CLOCKS = ['nm_delay', 'off_iobt', 'lobt_m_iobt', 'eobt_m_lobt', 'a3_m_eobt', 'a3_m_lobt',
          'eobt_m_sched', 'iobt_m_sched']
STAMP = ['pa_stamped', 'pa_inb_m_sibt', 'pa_ldg_m_sibt']
TURNOVER = ['si_before_sched', 'si_same_typ', 'sd_prev_to', 'sl_to', 'sl_after_sched', 'sl_same_typ',
            'sl_same_op', 'sn_minus_to', 'sn_same_op', 'sn_same_typ', 'sa_after_to', 'sd_next_to']
V11_NUM = CLOCKS + STAMP + TURNOVER
W2_CATS = CATS + ['ADES_mvt']
# adsb.lol ADS-B features (experiments/adsb/adsb_build_features.py), keyed by MVT_ID_mvt; approved 2026-09-25.
# NaN where the flight was not seen; adsb_day_ok = 0 where the archive day was not processed.
ADSB_FILE = CACHE / 'adsb_lol' / 'features_{which}.parquet'
# C1 (2026-09-26): de-icing / disruption shares of OTHER departures (experiments/deice_0926/e01_build_indicators.py,
# look-ahead within the file, label-free), keyed by MVT_ID_mvt: the 7 share columns the de-icing report kept
# (reports/2026-09-26_deice_indicator.md s6.3) + dz_taxi_h, the mean adsb.lol taxi of the others within +-60 min.
DEICE_FILE = CACHE / 'deice_0926' / 'ind_{which}.parquet'
DEICE_COLS = ['dz_rr_sh_h', 'dz_rr_n_h', 'dz_ls_sh_h', 'dz_rest_h', 'dz_rr_sh_d', 'dz_rr_n_d', 'dz_ls_sh_d', 'dz_taxi_h']


def _one(which, adsb=False, extra=False, deice=False):
    """cv_task.assemble()'s build for `which`, plus the v11 columns (and, with adsb=True, the adsb.lol
    columns as group 'adsb'). Returns (X, meta, groups)."""
    full = load(which)
    cong = build_cong(full)
    d = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    del full
    X = build_base(d)
    X['aobt_missing'] = d.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = d.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    base = list(X.columns)
    wx = attach_wx(d, prepare(download()))
    atfm = attach_atfm(d)
    groups = {'base': base, 'cong': list(cong.columns), 'wx': list(wx.columns), 'atfm': list(atfm.columns)}
    X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index), atfm.set_axis(X.index)], axis=1)

    f = pd.read_parquet(CACHE / 'v11' / f'feat_{which}.parquet', columns=['MVT_ID_mvt'] + V11_NUM)
    if len(f) != len(d) or not np.array_equal(f.MVT_ID_mvt.to_numpy(), d[ID].to_numpy()):
        sys.exit(f'FATAL: cache/v11/feat_{which}.parquet is not row-aligned with the {which} departures '
                 f'({len(f):,} vs {len(d):,} rows) -- rebuild it with experiments/v11/v11_features.py')
    for c in V11_NUM:
        X[c] = f[c].astype('float32').to_numpy()
    X['ADES_mvt'] = d.ADES_mvt.astype(object).to_numpy()
    groups['v11'] = V11_NUM + ['ADES_mvt']
    if adsb:
        path = Path(str(ADSB_FILE).format(which=which))
        if not path.exists():
            sys.exit(f'FATAL: {path} missing -- run the adsb.lol stage-2 job and rsync it first')
        a = pd.read_parquet(path)
        a = a.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(d[ID].to_numpy())
        cols = [c for c in a.columns if c.startswith('adsb_')]
        if a['adsb_day_ok'].isna().any():
            sys.exit(f'FATAL: {path} does not cover every {which} departure ({int(a.adsb_day_ok.isna().sum()):,} missing ids)')
        for c in cols:
            X[c] = a[c].astype('float32').to_numpy()
        groups['adsb'] = cols
        print(f'adsb.lol columns: {len(cols)}; {which} rows with the archive day processed '
              f'{int((a.adsb_day_ok == 1).sum()):,} of {len(a):,}, with an off-block estimate '
              f'{int(a.adsb_off_best.notna().sum()):,}', flush=True)
    if extra:                                   # A3 arm 2: wake + backlog-at-schedule + gap_prev_to 6 h cap
        from cluster import w2_extra
        e = w2_extra.build(d, which)
        for c in w2_extra.EXTRA_COLS:
            X[c] = e[c].to_numpy()
        if 'gap_prev_to' in X.columns:
            g = X['gap_prev_to'].to_numpy(dtype=float)
            X['gap_prev_to'] = np.where(g > w2_extra.MAX_GAP, np.nan, g)
        groups['extra'] = list(w2_extra.EXTRA_COLS)
    if deice:                                   # C1: de-icing / disruption shares (DEICE_COLS)
        path = Path(str(DEICE_FILE).format(which=which))
        if not path.exists():
            sys.exit(f'FATAL: {path} missing -- rsync cache/deice_0926/ind_*.parquet from the laptop first')
        e = pd.read_parquet(path, columns=['MVT_ID_mvt'] + DEICE_COLS).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
        n_miss = int((~pd.Index(d[ID].to_numpy()).isin(e.index)).sum())
        if n_miss:
            sys.exit(f'FATAL: {path} misses {n_miss:,} {which} departures -- rebuild it with e01_build_indicators.py')
        e = e.reindex(d[ID].to_numpy())[DEICE_COLS].astype('float32').set_axis(X.index)
        X = pd.concat([X, e], axis=1)
        groups['deice'] = list(DEICE_COLS)
    meta = pd.DataFrame({ID: d[ID].to_numpy(), 'month': d.MVT_TIME_UTC_mvt.dt.month.to_numpy()})
    if TARGET in d.columns:
        meta[TARGET] = d[TARGET].astype(float).to_numpy()
    return X, meta, groups


def _align_ades(train_X, *others):
    vocab = pd.Index(sorted(pd.Series(train_X['ADES_mvt']).dropna().unique()))
    for f in (train_X,) + others:
        f['ADES_mvt'] = pd.Categorical(f['ADES_mvt'], categories=vocab)


def assemble_w2(adsb=False, extra=False, deice=False):
    """Training only (the CV tasks). Same return shape as cluster.cv_task.assemble()."""
    X, meta, groups = _one('training', adsb=adsb, extra=extra, deice=deice)
    X, = align_categories(X)
    _align_ades(X)
    return X, meta[TARGET].to_numpy(), meta['month'].to_numpy(), groups


def assemble_w2_pair(adsb=False, extra=False, deice=False):
    """Training AND ranking with one shared vocabulary per categorical (the final fits)."""
    Xtr, mtr, groups = _one('training', adsb=adsb, extra=extra, deice=deice)
    Xrk, mrk, _ = _one('ranking', adsb=adsb, extra=extra, deice=deice)
    Xtr, Xrk = align_categories(Xtr, Xrk)
    _align_ades(Xtr, Xrk)
    if list(Xtr.columns) != list(Xrk.columns):
        sys.exit('FATAL: training and ranking W2 frames have different columns')
    return Xtr, mtr, Xrk, mrk, groups


def columns(groups):
    """(bulk regressor columns, tail classifier columns). The classifier keeps the 18 base
    columns on purpose (EXP15: extra columns make the 0.02%-positive classifier noisier)."""
    cols = [c for g in ('base', 'cong', 'wx', 'atfm', 'v11', 'adsb', 'extra', 'deice') if g in groups for c in groups[g]]
    return cols, list(groups['base'])
