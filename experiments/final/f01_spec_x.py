"""f01 -- final phase: TailSpecialistX probabilities on the ranking P rows (spec_x_ranking.npz).

In the initial phase cache/final/spec_x_ranking.npz came from exp64_submit_v7.py (which also refits the 5 LightGBM
seeds and compares against the shipped v6/v7 files) followed by exp64_export_spec.py. Neither shipped file exists
for final_ranking.parquet, and the 5 seeds are exp65 --stage components' job, so this script does only the
specialist part of exp64, with exp64's code verbatim: the same feature build, the same candidates and target
encodings, the same TailSpecialistX(extra=FAMILIES['all'], threads=4) fit on all 2025 P rows.

It also writes cache/exp62_cand_ranking.parquet exactly as exp62_screen.py does (the TabICL scripts s21/w4b read it).

REGRESSION CHECK (initial mode): without PRC_RANKING=final it writes cache/final/spec_x_ranking_f01check.npz and
compares it with the shipped cache/final/spec_x_ranking.npz (ids, in_P exactly; p_spec to 1e-12).

    systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 nice -n 5 python experiments/final/f01_spec_x.py
    PRC_RANKING=final systemd-run ... python experiments/final/f01_spec_x.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP, ID, CACHE, FINAL
from prc.features import build as build_base, align_categories
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.models.tail_specialist import TailSpecialistX, SPEC_FEATURES, population
from prc.p_features import load_set, candidates, te_features, FAMILIES

EXTRA = FAMILIES['all']
SHIPPED = CACHE / 'final' / 'spec_x_ranking.npz'
OUT = SHIPPED if FINAL else CACHE / 'final' / 'spec_x_ranking_f01check.npz'
t0 = time.time()


def log(msg):
    print(f'[{time.time()-t0:6.0f}s | RSS {peak_rss_gb():.2f} GB] {msg}', flush=True)


if len(EXTRA) != 42:
    sys.exit(f'FATAL: FAMILIES["all"] has {len(EXTRA)} features, exp62 arm b had 42')
if FINAL and OUT.exists():
    sys.exit(f'FATAL: {OUT} already exists -- move it aside deliberately before rebuilding')


def assemble(which, extra_cols):                  # exp64_submit_v7.assemble, verbatim
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
    base_cols = list(X.columns)
    wx = attach_wx(d, prepare(download()))
    atfm = attach_atfm(d)
    X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
                   atfm.set_axis(X.index)], axis=1)
    return d[extra_cols], X, base_cols


# ---- P-row features (exp64) ---------------------------------------------------
drk, depm_rk, _ = load_set('ranking')
cand_rk, _, P_rk_c = candidates(drk, depm_rk)
del drk
frk = CACHE / 'exp62_cand_ranking.parquet'
if FINAL:
    if frk.exists():
        sys.exit(f'FATAL: {frk} exists in the final cache -- it must be built from final_ranking.parquet here')
    cand_rk.to_parquet(frk)
else:
    if not cand_rk.equals(pd.read_parquet(frk)):
        sys.exit('FATAL: ranking candidates differ from the shipped cache/exp62_cand_ranking.parquet')
    log('regression: ranking candidates == shipped exp62_cand_ranking.parquet')
cand_tr = pd.read_parquet(CACHE / 'exp62_cand_training.parquet')
pool = pd.read_parquet(CACHE / 'exp62_pool_training.parquet')
log(f'candidates: ranking P {len(cand_rk):,}, training P {len(cand_tr):,}')

tr_meta, Xtr, BASE = assemble('training', [TARGET])
rk_meta, Xrk, _ = assemble('ranking', [ID])
Xtr, Xrk = align_categories(Xtr, Xrk)
FULL = list(Xtr.columns)
if (len(FULL), len(BASE)) != (72, 18):
    sys.exit(f'FATAL: feature counts {len(FULL)}/{len(BASE)} differ from v6 (72/18)')
y = tr_meta[TARGET].astype(float).to_numpy()
apt_tr, apt_rk = Xtr['APT'].astype(str).to_numpy(), Xrk['APT'].astype(str).to_numpy()
os_tr, os_rk = Xtr['off_sched'].to_numpy(), Xrk['off_sched'].to_numpy()
P_tr = population(os_tr, Xtr['aobt_missing'].to_numpy().astype(bool))
P_rk = population(os_rk, Xrk['aobt_missing'].to_numpy().astype(bool))
if not np.array_equal(np.where(P_tr)[0], cand_tr.index.to_numpy()):
    sys.exit('FATAL: training candidates do not match P on this load')
if not (np.array_equal(np.where(P_rk)[0], cand_rk.index.to_numpy()) and np.array_equal(P_rk, P_rk_c)):
    sys.exit('FATAL: ranking candidates do not match P on this load')
log(f'P: training {int(P_tr.sum()):,} | ranking {int(P_rk.sum()):,} of {len(Xrk):,}')

itr, irk = cand_tr.index.to_numpy(), cand_rk.index.to_numpy()
ctr, crk = cand_tr.reset_index(drop=True), cand_rk.reset_index(drop=True)
tP = (y[itr] > TAIL_CAP).astype(int)
pool_art = (np.abs(y[pool['pos'].to_numpy()] - pool.off_sched.to_numpy()) < 60).astype(float)
te_tr, te_rk = te_features(ctr, tP, pool, pool_art, crk)
raw = [c for c in EXTRA if not c.startswith('te_')]
XPtr = pd.concat([Xtr.loc[itr, SPEC_FEATURES].reset_index(drop=True), ctr[raw],
                  te_tr.reset_index(drop=True)], axis=1)
XPrk = pd.concat([Xrk.loc[irk, SPEC_FEATURES].reset_index(drop=True), crk[raw],
                  te_rk.reset_index(drop=True)], axis=1)
ones_tr, ones_rk = np.ones(len(itr), bool), np.ones(len(irk), bool)
spec7 = TailSpecialistX(extra=EXTRA, threads=4).fit(XPtr, y[itr], apt_tr[itr], os_tr[itr], ones_tr)
p7 = np.full(len(Xrk), np.nan)
p7[irk] = spec7.predict(XPrk, apt_rk[irk], os_rk[irk], ones_rk)[0]
log(f'TailSpecialistX p inside P: median {np.nanmedian(p7):.4f} mean {np.nanmean(p7):.4f} max {np.nanmax(p7):.4f}')

ids = rk_meta[ID].to_numpy(np.float64)
if pd.Index(ids).duplicated().any() or not np.array_equal(np.isfinite(p7), P_rk):
    sys.exit('FATAL: duplicate ids or NaN pattern != P')
np.savez(OUT, ids=ids, in_P=P_rk, p_spec=p7.astype(np.float64))
log(f'wrote {OUT}: {len(ids):,} rows, {int(P_rk.sum()):,} in P')

if not FINAL:
    z = np.load(SHIPPED)
    ok = (np.array_equal(z['ids'], ids) and np.array_equal(z['in_P'], P_rk)
          and np.allclose(z['p_spec'], p7, rtol=0, atol=1e-12, equal_nan=True))
    d = np.nanmax(np.abs(z['p_spec'] - p7)) if len(z['ids']) == len(ids) else np.nan
    print(f'REGRESSION vs shipped spec_x_ranking.npz: {"PASS" if ok else "FAIL"} (max |dp| {d:.3g})')
    sys.exit(0 if ok else 1)
