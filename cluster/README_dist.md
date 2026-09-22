# Cluster campaigns for the EXP20-EXP23 architectures

Companion to `README.md`, which covers access, the conda environment and how the
data gets to `~/prc2026/cache`. Nothing here changes any of that. These
campaigns use `dist_task.py` / `dist_array.sbatch` / `dist_collect.py`, which sit
beside the existing `cv_task.py` / `cv_array.sbatch` / `collect.py` rather than
replacing them.

## Why this needs the cluster more than the L2 campaigns do

LightGBM grows **one tree per class per boosting iteration**. A K=47 softmax at
300 rounds is 14,100 trees against the 350 the L2 regressor fits, and one fold
measured **1,213 s on the laptop** at 6 threads against ~35 s for the incumbent
mixture. A single 12-fold LOMO pass is therefore ~3 h locally and minutes as an
array. Config selection for this architecture is only affordable on the cluster.

## The campaigns

| campaign | what it asks | tasks |
|---|---|---|
| `dist` | bins x capacity for the distributional model | 72 |
| `arch` | mixture vs anchored-residual bulk vs anchored alone | 36 |
| `seeds` | 10 seeds of the best distributional config | 120 |

Ask `dist_task.py` for the array bound rather than counting by hand — the
campaign definitions live in Python and the `--array` line has to match them:

    python cluster/dist_task.py --campaign dist --size
    # campaign dist: 6 configs x 1 seeds x 12 months = 72 tasks  ->  --array=0-71

## Exact commands

    ssh copperhead 'cd ~/prc2026 && git pull'
    ssh copperhead 'cd ~/prc2026 && python cluster/dist_task.py --campaign dist --size'
    ssh copperhead 'cd ~/prc2026 && sbatch --array=0-71%30 cluster/dist_array.sbatch dist'
    ssh copperhead 'cd ~/prc2026 && sbatch --array=0-35%30 cluster/dist_array.sbatch arch'
    ssh copperhead 'squeue -u $USER'
    ssh copperhead 'cd ~/prc2026 && conda activate prc2026 && python cluster/dist_collect.py dist'

Run `seeds` only after `dist` has produced a config worth ten seeds:

    ssh copperhead 'cd ~/prc2026 && sbatch --array=0-119%30 cluster/dist_array.sbatch seeds'
    ssh copperhead 'cd ~/prc2026 && conda activate prc2026 && \
        python cluster/dist_collect.py seeds --seed-average'

## What healthy output looks like

* `sbatch` prints `Submitted batch job <id>`.
* `squeue` shows `PD` then `R`; `%30` runs thirty at a time.
* Each `dist` task prints one line and writes one `.npz`:

      dist_nt1_l63__seed00__m07.npz: n=190,713 variants=9 bestRMSE 518.32
      {'tail_mass_mean': 0.000777, 'tail_rate': 0.000629} (1100s)

  `variants=9` is the point of the `.npz`: the softmax is fitted once and the
  nine ways of integrating it are evaluated for free. `arch` tasks print
  `variants=1` and finish in ~2 min.
* `tail_mass_mean` next to `tail_rate` is the calibration check. A ratio far
  from 1 means the softmax is mis-stating how much probability belongs above
  7,200 s, which is the failure that shows up as tail RMSE. Measured so far:
  0.90 at n_tail=10, 1.24 at n_tail=1.
* `dist_collect.py` refuses to score an incomplete OOF set and names the missing
  months, so a partially failed array is obvious rather than silently wrong.
* Tasks are idempotent — an existing `.npz` is skipped — so a failed array can
  be resubmitted without redoing finished work.
* `dist_collect.py` prints a `verdict` column. `ns` means the 95% paired
  interval against the EXP13 mixture crosses zero. **Most rows will say `ns` or
  `worse`, and that is the expected result** given EXP20; a `BETTER` row is the
  only thing worth acting on.

## Resource notes

`dist_array.sbatch` asks for 16G and 45 min where `cv_array.sbatch` asks for 12G
and 25 min. A K=47 softmax holds a `num_data x num_class` score matrix plus
gradients and hessians — about 1.4 GB on top of the feature matrix at 1.9M rows
— and 12G is not enough. The `arch` campaign is over-provisioned by these
settings; that is cheaper than maintaining a second sbatch file.
