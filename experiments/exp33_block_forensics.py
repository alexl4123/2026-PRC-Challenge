"""EXP33 -- what is BLOCK_TIME actually stamped at?

EXP32 split the 584 tail rows into four mechanisms and priced each one by the
contribution it makes to total RMSE if it is left mispredicted at ~1100s:

    S   AOBT_3 missing, off_sched > 7200    450 rows   contrib 327   anchor off_sched -> 5,176
    Sb  AOBT_3 missing, off_sched <= 7200    31 rows   contrib 100   NO anchor found  -> 25,625
    L   AOBT_3 present, off_aobt3 > 7200      6 rows   contrib  62   anchor off_aobt3 -> 2,257
    X   AOBT_3 present, off_aobt3 <= 7200    97 rows   contrib  54   anchor off_sched -> 2,200

X is the surprise: AOBT_3 exists and says the taxi was ~1,055s, but the label
says ~8,208s and off_sched ~9,004s. So BLOCK was stamped at something that is
NOT the real off-block and NOT exactly SCHED either (delta ~ 800s, outside the
60s artifact window). Sb is the other surprise: BLOCK lands BEFORE SCHED by up
to a day.

Hypothesis: BLOCK_TIME is copied from whichever off-block estimate the airport
had to hand -- SCHED, EOBT_1, LOBT, IOBT -- and the tail is a mixture over WHICH
anchor was used. This measures that directly: for every row, the offset from
BLOCK to each candidate, and which candidate is closest.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import NEEDED
from prc.io import load, peak_rss_gb
pd.set_option('display.width', 240)

t0 = time.time()
cols = sorted(set(NEEDED + [TARGET, 'BLOCK_TIME_UTC_mvt', 'IOBT_flt', 'ARVT_1_flt',
                            'ARVT_3_flt']))
tr = load('training', phase='DEP', columns=cols)
y = tr[TARGET].astype(float).to_numpy()
apt = tr.APT.astype(str).to_numpy()
B = tr.BLOCK_TIME_UTC_mvt
M = tr.MVT_TIME_UTC_mvt
tail = y > TAIL_CAP
miss = tr.AOBT_3_flt.isna().to_numpy()
os_ = (M - tr.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
oa = (M - tr.AOBT_3_flt).dt.total_seconds().to_numpy()

CAND = {
    'SCHED':       tr.SCHED_TIME_UTC_mvt,
    'EOBT_1':      tr.EOBT_1_flt,
    'LOBT':        tr.LOBT_flt,
    'IOBT':        tr.IOBT_flt,
    'AOBT_3':      tr.AOBT_3_flt,
    'SCHED-1d':    tr.SCHED_TIME_UTC_mvt - pd.Timedelta(days=1),
    'SCHED+1d':    tr.SCHED_TIME_UTC_mvt + pd.Timedelta(days=1),
}
off = {k: (B - v).dt.total_seconds().to_numpy() for k, v in CAND.items()}
# offset from MVT to each candidate: this is what a PREDICTION would use, and it
# is computable at scoring time for every candidate except AOBT_3-derived ones.
mvt_off = {k: (M - v).dt.total_seconds().to_numpy() for k, v in CAND.items()}

cats = {
    'S  miss & os>cap':      miss & (os_ > TAIL_CAP),
    'Sb miss & os<=cap':     miss & ~(os_ > TAIL_CAP),
    'L  pres & oa>cap':      ~miss & (oa > TAIL_CAP),
    'X  pres & oa<=cap':     ~miss & ~(oa > TAIL_CAP),
}

print('=' * 120)
print('1. |BLOCK - candidate| ON TAIL ROWS, BY MECHANISM  (median seconds; "hit%" = within 120s)')
print('=' * 120)
rows = []
for k, s in cats.items():
    st = s & tail
    if not st.any():
        continue
    r = {'cat': k, 'n': int(st.sum())}
    for c, v in off.items():
        d = np.abs(v[st])
        r[f'{c}_med'] = float(np.nanmedian(d))
        r[f'{c}_hit%'] = float(np.nanmean(d <= 120) * 100)
    rows.append(r)
d = pd.DataFrame(rows)
print(d[['cat', 'n'] + [f'{c}_med' for c in CAND]].round(0).to_string(index=False))
print()
print(d[['cat', 'n'] + [f'{c}_hit%' for c in CAND]].round(1).to_string(index=False))

print('\n--- same, for ALL rows (the base rate a classifier competes against) ---')
rows = []
for k, s in cats.items():
    r = {'cat': k, 'n': int(s.sum())}
    for c, v in off.items():
        r[f'{c}_hit%'] = float(np.nanmean(np.abs(v[s]) <= 120) * 100)
    rows.append(r)
print(pd.DataFrame(rows).round(2).to_string(index=False))

print('\n' + '=' * 120)
print('2. WHICH CANDIDATE IS CLOSEST?  (argmin over candidates, tail rows only)')
print('=' * 120)
names = list(CAND)
Dm = np.vstack([np.abs(off[c]) for c in names])
Dm = np.where(np.isfinite(Dm), Dm, np.inf)
best_i = Dm.argmin(axis=0)
best_d = Dm.min(axis=0)
lab = np.array(names)[best_i]
lab = np.where(np.isfinite(best_d) & (best_d <= 300), lab, 'NONE<=300s')
for k, s in cats.items():
    st = s & tail
    if not st.any():
        continue
    print(f'  {k:22s} n={int(st.sum()):4d}  {dict(pd.Series(lab[st]).value_counts())}')
print(f'  {"ALL tail":22s} n={int(tail.sum()):4d}  {dict(pd.Series(lab[tail]).value_counts())}')
print(f'  {"ALL rows":22s} n={len(y):,}  '
      f'{dict((pd.Series(lab).value_counts(normalize=True)*100).round(2))}')

print('\n' + '=' * 120)
print('3. CATEGORY X IN DETAIL  (97 tail rows: AOBT_3 present, off_aobt3 <= 7200)')
print('=' * 120)
Xt = cats['X  pres & oa<=cap'] & tail
sub = pd.DataFrame({c: off[c][Xt] for c in CAND})
sub['y'] = y[Xt]; sub['apt'] = apt[Xt]; sub['off_sched'] = os_[Xt]; sub['off_aobt3'] = oa[Xt]
print(sub.describe().round(0).to_string())
print('\n  first 15 rows:')
print(sub.head(15).round(0).to_string(index=False))
print('\n  RMSE of each candidate used as the prediction for these 97 rows:')
for c in CAND:
    v = mvt_off[c][Xt]
    v = np.where(np.isfinite(v), v, 1100.0)
    print(f'    MVT - {c:10s}: {np.sqrt(np.mean((y[Xt]-v)**2)):9.1f}')

print('\n' + '=' * 120)
print('4. CATEGORY Sb IN DETAIL  (31 tail rows: AOBT_3 missing, off_sched <= 7200)')
print('=' * 120)
St = cats['Sb miss & os<=cap'] & tail
sub = pd.DataFrame({'y': y[St], 'off_sched': os_[St], 'apt': apt[St],
                    'B_minus_SCHED': off['SCHED'][St],
                    'y_minus_os': y[St] - os_[St],
                    'mvt': M[St].dt.strftime('%Y-%m-%d %H:%M'),
                    'block': B[St].dt.strftime('%Y-%m-%d %H:%M'),
                    'sched': tr.SCHED_TIME_UTC_mvt[St].dt.strftime('%Y-%m-%d %H:%M')})
print(sub.sort_values('y').round(0).to_string(index=False))

print('\n' + '=' * 120)
print('5. A BETTER TARGET?  event A2 = BLOCK matches SCHED within 120s, by population')
print('=' * 120)
A2 = np.abs(off['SCHED']) <= 120
print(f'  base rate {A2.mean()*100:.3f}%   P(A2|tail)={A2[tail].mean()*100:.1f}%  '
      f'P(tail|A2)={tail[A2].mean()*100:.4f}%')
for nm, s in [('miss', miss), ('present', ~miss)]:
    for lo, hi in [(-np.inf, 7200), (7200, 14400), (14400, np.inf)]:
        sel = s & (os_ >= lo) & (os_ < hi)
        if sel.sum() == 0:
            continue
        print(f'    {nm:8s} off_sched [{lo:8.0f},{hi:8.0f}) n={int(sel.sum()):9,}  '
              f'P(A2)={A2[sel].mean()*100:7.3f}%  P(tail)={tail[sel].mean()*100:7.4f}%  '
              f'n_tail={int(tail[sel].sum()):4d}')

print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
