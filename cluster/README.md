# Running on the TU Wien KR cluster

## Why the cluster

One leave-one-month-out CV run is ~20-30 min on the laptop, and it is serial.
The work that is now bottlenecked on that:

| job | laptop | cluster |
|---|---|---|
| seed averaging, 20 seeds x 12 folds = 240 fits | ~3.7 h | minutes, as an array |
| feature ablation, 8 configs x 12 folds | ~1.5 h | minutes |
| per-airport models, 10 x 12 folds | ~2 h | minutes |

Each individual fit is ~55 s and needs ~5 GB, so the natural unit of work is one
(config, seed, fold) triple. That also matches the local admin guidance in
`cluster_hardware.txt`: many small timeout-bounded tasks, not one long job.

## Access

The cluster is reached as `abeiser@vivaldi.kr.tuwien.ac.at` using a dedicated
key at `~/.keyfiles/copperhead.key`. Adding this to `~/.ssh/config` means the
key never has to be passed explicitly, and rsync/scp pick it up too:

    Host vivaldi
        HostName vivaldi.kr.tuwien.ac.at
        User abeiser
        IdentityFile ~/.keyfiles/copperhead.key
        IdentitiesOnly yes

The key must be mode 600 or ssh refuses to use it (`chmod 600
~/.keyfiles/copperhead.key`).

Environment, once, on the cluster: `bash cluster/setup_env.sh`.

## Getting the code there: git, not rsync

Code travels by git; **data does not and must not**.

    # once, on copperhead
    git clone git@github.com:alexl4123/2026-PRC-Challenge.git \
        ~/03_2026_PRC_Challenge/2026-PRC-Challenge

    # thereafter
    ssh vivaldi 'cd ~/03_2026_PRC_Challenge/2026-PRC-Challenge && git pull'

`cluster/sync.sh` remains for the data only. It takes the remote host and
remote path as arguments, defaulting to `vivaldi` and
`03_2026_PRC_Challenge/2026-PRC-Challenge`.

## Getting the DATA there

The parquets cannot go through git — the repository is destined to be public
under GPLv3 and the challenge Terms & Conditions forbid publishing the dataset.
Two options:

1. **Copy the local cache** (~264 MB, one off):

       rsync -az cache/training.parquet cache/ranking.parquet \
             cache/external/ vivaldi:~/03_2026_PRC_Challenge/2026-PRC-Challenge/cache/

   Without `~/.ssh/config`, pass the key explicitly:

       rsync -az -e 'ssh -i ~/.keyfiles/copperhead.key' \
             cache/training.parquet cache/ranking.parquet cache/external/ \
             abeiser@vivaldi.kr.tuwien.ac.at:~/03_2026_PRC_Challenge/2026-PRC-Challenge/cache/

2. **Fetch from the OSN bucket on the cluster**, using the team's own S3
   credentials placed at `~/.config/prc2026/s3.env` on copperhead (mode 600).
   Cleaner, since nothing large crosses the laptop's uplink twice.

Either way the data stays inside the user's own cluster home directory.

## Usage

    ssh vivaldi 'cd ~/03_2026_PRC_Challenge/2026-PRC-Challenge && git pull'
    ssh vivaldi 'cd ~/03_2026_PRC_Challenge/2026-PRC-Challenge && bash cluster/preflight.sh'
    ssh vivaldi 'cd ~/03_2026_PRC_Challenge/2026-PRC-Challenge && sbatch --array=0-479%60 cluster/cv_array.sbatch grid'
    ssh vivaldi 'squeue -u $USER'
    ssh vivaldi 'cd ~/03_2026_PRC_Challenge/2026-PRC-Challenge && conda activate prc2026 && python cluster/collect.py grid'

The user runs these; the output is pasted back for analysis.

## What a healthy run looks like

* `sbatch` prints `Submitted batch job <id>`.
* `squeue` shows tasks in `PD` then `R`; a 240-task array with `%40` throttling
  runs ~40 at a time.
* Each task prints one line, e.g.
  `full__seed03__m07.npy: n=190,713 RMSE 419.00  (55s)`, and writes one `.npy`.
* `collect.py` refuses to score an incomplete OOF set and names the missing
  months, so a partially failed array is obvious rather than silently wrong.
* Tasks are idempotent: an existing `.npy` is skipped, so a failed array can be
  resubmitted without redoing finished work.
