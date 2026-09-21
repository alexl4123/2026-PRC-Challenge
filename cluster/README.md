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

## Prerequisites (one-off, by the user)

1. **SSH access.** `ssh -o BatchMode=yes abeiser@vivaldi.kr.tuwien.ac.at` currently
   returns `Permission denied (publickey)`. Authorise the local key:

       ssh-copy-id -i ~/.ssh/id_ed25519.pub abeiser@vivaldi.kr.tuwien.ac.at

   Then add to `~/.ssh/config` so `copperhead` is reachable in one hop:

       Host vivaldi
           HostName vivaldi.kr.tuwien.ac.at
           User abeiser
       Host copperhead
           HostName copperhead
           User abeiser
           ProxyJump vivaldi

2. **Environment on the cluster**: `bash cluster/setup_env.sh` (run on copperhead).

## Usage

    ./cluster/sync.sh                 # push code + cached parquet to the cluster
    ssh copperhead 'cd ~/prc2026 && sbatch cluster/cv_array.sbatch seeds'
    ssh copperhead 'cd ~/prc2026 && python cluster/collect.py seeds'

## Data handling

The challenge parquets must NOT be published (T&C forbid use outside the
competition until the organizers release them). They are copied to the cluster
under the user's own home directory and nowhere else. `sync.sh` never touches
`outputs/` or the credentials file.
