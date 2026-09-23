"""Export v7's TailSpecialistX probabilities on the ranking set for reuse (v8 builder).

Reads what exp64_submit_v7.py already computed -- no refit, no LightGBM:
    cache/exp64_p6_p7.npy           row 1 = TailSpecialistX p(tail), ranking DEP order,
                                    NaN outside P  (row 0 = the v6 specialist)
    cache/exp64_pred_v6recipe_v7.npy row 1 = v7 5-seed prediction, same order
Writes cache/final/spec_x_ranking.npz with keys
    ids     MVT_ID_mvt, ranking DEP order = the array exp64 passed to prc.submit.build
    in_P    bool, P = {AOBT_3_flt missing} AND {off_sched > 7200}
    p_spec  float64, TailSpecialistX probability, NaN outside P

ALIGNMENT PROOF: rint(clip(v7 prediction)) mapped through `ids` must reproduce the
on-disk outputs/understanding-accordion_v7.parquet on every one of the 344,841
rows; any misordering of ids would break that, so the script refuses to write.

Run after exp64_submit_v7.py has finished:
    /home/thinklex/prc2026-venv/bin/python experiments/exp64_export_spec.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import ID, CACHE, OUTPUTS
from prc.io import load
from prc.models.tail_specialist import population

N_RK = 344_841
OUT = CACHE / 'final' / 'spec_x_ranking.npz'

d = load('ranking', phase='DEP',
         columns=[ID, 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'])
ids = d[ID].to_numpy(np.float64)
off_sched = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
in_P = population(off_sched, d.AOBT_3_flt.isna().to_numpy())
p_spec = np.load(CACHE / 'exp64_p6_p7.npy')[1].astype(np.float64)
pred7 = np.load(CACHE / 'exp64_pred_v6recipe_v7.npy')[1]

problems = []
if not (len(ids) == len(p_spec) == len(pred7) == N_RK):
    problems.append(f'lengths ids {len(ids)} p {len(p_spec)} pred {len(pred7)}, expected {N_RK}')
if pd.Index(ids).duplicated().any():
    problems.append('duplicate ids')
if not np.array_equal(np.isfinite(p_spec), in_P):
    problems.append(f'NaN pattern of p_spec does not equal the P mask '
                    f'({int(np.isfinite(p_spec).sum())} finite vs {int(in_P.sum())} in P)')
v7 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v7.parquet').set_index(ID).TAXITIME_SEC_mvt
disk = v7.reindex(ids).to_numpy()
mine = np.rint(np.clip(pred7, 0.0, 200000.0))
if np.isnan(disk).any() or not np.array_equal(disk.astype(np.int64), mine.astype(np.int64)):
    problems.append('ids order does not reproduce the on-disk v7 parquet')
if problems:
    sys.exit('FATAL, nothing written:\n  ' + '\n  '.join(problems))

OUT.parent.mkdir(exist_ok=True)
np.savez(OUT, ids=ids, in_P=in_P, p_spec=p_spec)
z = np.load(OUT)
print(f'wrote {OUT}: ids {z["ids"].shape} {z["ids"].dtype}, in_P {int(z["in_P"].sum())} True '
      f'({z["in_P"].dtype}), p_spec finite {int(np.isfinite(z["p_spec"]).sum())} '
      f'({z["p_spec"].dtype}) mean {np.nanmean(z["p_spec"]):.4f}')
print('alignment proof: rint(clip(v7 pred)) via ids == on-disk v7 parquet on all '
      f'{N_RK:,} rows')
