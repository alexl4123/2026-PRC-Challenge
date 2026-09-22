"""EXP34 -- the multi-anchor mixture, against the shipped model.

See prc/models/tail_anchor.py for the mechanism. Holds the expensive bulk
regressor FIXED (the cached component from exp31) so the paired bootstrap
isolates the mixture weights.

  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
    systemd-inhibit --what=idle:sleep:handle-lid-switch \
    /home/thinklex/prc2026-venv/bin/python experiments/exp34_anchor_mixture.py 1 3 7
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.validation import decompose, paired_bootstrap
from prc.models.tail_harness import TAILDIR, mixture
from prc.models.tail_anchor import (assemble, anchor_label, anchor_mixture,
                                    fit_anchor_clf, CLASSES, DIRECT, ANCHORS)
pd.set_option('display.width', 230)

MONTHS = [int(a) for a in sys.argv[1:]] or list(range(1, 13))
SEED, THREADS = 0, 4

t0 = time.time()
X, meta, offs, dist, base_cols = assemble()
ext_cols = list(X.columns)
y = meta['y'].to_numpy(); month = meta['month'].to_numpy()
os_ = meta['off_sched'].to_numpy(); oa = meta['off_aobt3'].to_numpy()
apt = meta['apt'].to_numpy(); miss = meta['aobt_missing'].to_numpy()
tail = y > TAIL_CAP
lab = anchor_label(dist)
sel = np.isin(month, MONTHS)
print(f'{len(y):,} rows, {len(ext_cols)} features ({len(base_cols)} base). '
      f'eval months {MONTHS}: {sel.sum():,} rows / {int(tail[sel].sum())} tail '
      f'({time.time()-t0:.0f}s)')
print('  anchor label distribution: ' + '  '.join(
    f'{CLASSES[i]}={(lab==i).mean()*100:.2f}%' for i in range(len(CLASSES))))
print('  on tail rows:              ' + '  '.join(
    f'{CLASSES[i]}={(lab[tail]==i).mean()*100:.1f}%' for i in range(len(CLASSES))))


def run_anchor(name, cols, force=False):
    P = np.full((len(y), len(CLASSES)), np.nan)
    for m in MONTHS:
        f = TAILDIR / f'{name}__seed{SEED:02d}__m{m:02d}.npy'
        va = month == m
        if f.exists() and not force:
            P[va] = np.load(f)
            print(f'    {name} m{m:02d}: cached', flush=True)
            continue
        t = time.time()
        mdl = fit_anchor_clf(X, lab, ~va, cols, seed=SEED, threads=THREADS)
        v = mdl.predict(X.loc[va, cols])
        np.save(f, v.astype(np.float32)); P[va] = v
        print(f'    {name} m{m:02d}: fitted ({time.time()-t:.0f}s)', flush=True)
    return P


print('\n--- anchor classifier, extended features ---')
W_ext = run_anchor('anchor_ext', ext_cols)
print('--- anchor classifier, base features only ---')
W_base = run_anchor('anchor_base', base_cols)

rb = np.full(len(y), np.nan)
pT = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    rb[va] = np.load(TAILDIR / f'bulk_reg__seed{SEED:02d}__m{m:02d}.npy')
    pT[va] = np.load(TAILDIR / f'clf_T__seed{SEED:02d}__m{m:02d}.npy')
e_tail = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    tm = y[~va][y[~va] > TAIL_CAP].mean()
    e_tail[va] = np.where(np.isfinite(os_[va]) & (os_[va] > TAIL_CAP), os_[va], tm)

preds = {
    'BASE   shipped mixture':      mixture(pT, e_tail, rb),
    'AM     anchor mix (ext)':     anchor_mixture(W_ext, offs, rb),
    'AMb    anchor mix (base ft)': anchor_mixture(W_base, offs, rb),
}
# ablation: which anchors carry the gain
for keep in [['SCHED'], ['SCHED', 'LOBT'], ['SCHED', 'EOBT_1']]:
    Wk = W_ext.copy()
    for a in DIRECT:
        if a not in keep:
            Wk[:, CLASSES.index(a)] = 0.0
    preds[f'AM-    anchors {"+".join(keep)}'] = anchor_mixture(Wk, offs, rb)
# additive: shipped mixture PLUS the anchor correction for the non-SCHED clocks
Wn = W_ext.copy()
Wn[:, CLASSES.index('SCHED')] = 0.0
preds['AM+    shipped + LOBT/EOBT'] = np.clip(
    anchor_mixture(Wn, offs, preds['BASE   shipped mixture']), 0, None)

print('\n' + '=' * 115)
print(f'RESULTS, months {MONTHS}')
print('=' * 115)
rows = [dict(model=k, **{kk: vv for kk, vv in decompose(y[sel], p[sel]).items()})
        for k, p in preds.items()]
res = pd.DataFrame(rows).sort_values('rmse')
print(res[['model', 'rmse', 'rmse_bulk', 'rmse_tail', 'contrib_bulk',
           'contrib_tail']].round(2).to_string(index=False))
base = preds['BASE   shipped mixture']
print('\n--- paired bootstrap vs BASE (negative = better) ---')
for k, p in preds.items():
    if k.startswith('BASE'):
        continue
    r = paired_bootstrap(y[sel], p[sel], base[sel])
    print(f'  {k:30s} {r["diff"]:+9.2f} +- {r["sd"]:6.2f}  CI [{r["lo"]:+8.1f}, '
          f'{r["hi"]:+8.1f}]  P(worse)={r["p_a_worse"]*100:5.1f}%')

print('\n' + '=' * 115)
print('SSE BY MECHANISM (the budget: where every squared second lives)')
print('=' * 115)
cats = {
    'TAIL S  miss & os>cap':   sel & tail & miss & (os_ > TAIL_CAP),
    'TAIL Sb miss & os<=cap':  sel & tail & miss & ~(os_ > TAIL_CAP),
    'TAIL L  pres & oa>cap':   sel & tail & ~miss & (oa > TAIL_CAP),
    'TAIL X  pres & oa<=cap':  sel & tail & ~miss & ~(oa > TAIL_CAP),
    'FP   LIRF miss os>cap':   sel & ~tail & miss & (os_ > TAIL_CAP) & (apt == 'LIRF'),
    'FP   other miss os>cap':  sel & ~tail & miss & (os_ > TAIL_CAP) & (apt != 'LIRF'),
    'BULK pres & os>cap':      sel & ~tail & ~miss & (os_ > TAIL_CAP),
    'BULK rest':               sel & ~tail & ~(miss & (os_ > TAIL_CAP)) & ~(~miss & (os_ > TAIL_CAP)),
}
tab = []
for k, s in cats.items():
    row = {'group': k, 'n': int(s.sum())}
    for mk, p in preds.items():
        row[mk.split()[0]] = np.sqrt(((y[s] - p[s]) ** 2).sum() / sel.sum())
    tab.append(row)
tot = {'group': 'TOTAL (RMSE)', 'n': int(sel.sum())}
for mk, p in preds.items():
    tot[mk.split()[0]] = np.sqrt(np.mean((y[sel] - p[sel]) ** 2))
tab.append(tot)
print('  each cell = that group\'s contribution to overall RMSE (sqrt(SSE/n_all))')
print(pd.DataFrame(tab).round(1).to_string(index=False))

print('\n--- anchor classifier quality on category X (pres & oa<=cap & os>cap) ---')
xp = sel & ~miss & (oa <= TAIL_CAP) & (os_ > TAIL_CAP)
print(f'  n={int(xp.sum()):,}  tail={int(tail[xp].sum())} ({tail[xp].mean()*100:.2f}%)')
for nm, W in [('ext', W_ext), ('base', W_base)]:
    wS = W[:, CLASSES.index('SCHED')]; wL = W[:, CLASSES.index('LOBT')]
    wE = W[:, CLASSES.index('EOBT_1')]; wA = W[:, CLASSES.index('AOBT_3')]
    wdir = wS + wL + wE
    print(f'    {nm:5s}: mean w(SCHED)={wS[xp].mean():.4f} w(LOBT)={wL[xp].mean():.4f} '
          f'w(EOBT)={wE[xp].mean():.4f} w(AOBT)={wA[xp].mean():.4f}')
    print(f'           on the {int(tail[xp].sum())} TAIL rows there: w_direct='
          f'{wdir[xp & tail].mean():.4f}   on bulk rows: {wdir[xp & ~tail].mean():.4f}')
    tr_lab = lab[xp]
    print(f'           true label mix there: ' + '  '.join(
        f'{CLASSES[i]}={(tr_lab==i).mean()*100:.1f}%' for i in range(len(CLASSES))))

np.save(TAILDIR / 'exp34_preds.npy', np.vstack([preds[k] for k in preds]).astype(np.float32))
open(TAILDIR / 'exp34_keys.txt', 'w').write('\n'.join(preds))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
