#!/bin/bash
# Final phase (2026-10-09): submit every cluster fit the final file needs, chained with Slurm dependencies.
# Run on the login node from the repo root, after
#   1. git pull,
#   2. final_ranking.parquet + final_submitting.parquet copied next to the repo (prc.config.DATA = repo parent),
#   3. cache_final/ranking.parquet and cache_final/v11/feat_ranking.parquet rsynced from the laptop
#      (built there with PRC_RANKING=final; this script refuses to start without them),
#   4. conda activate prc2026 (for make_final_cache.py and adsb_days.py).
#
#   bash cluster/final_phase.sh            # prints the job ids; watch with squeue -u $USER
#
# Chain:
#   stage1  adsb.lol points for the 58 Feb/Jun 2026 days (adsb_days.py indices 429-486)
#     -> stage2  events + ranking-side adsb features (--which ranking) -> cache_final/adsb_lol/features_ranking.parquet
#        -> adsb2_final (5 CPU families x 3 seeds) + adsb2_final_nn (nn, w3 x 3 seeds)
#        -> seq pool: seq_epochs_final EPOCHS 30/40/50 + seq_variant_final x 6 (all seeds 0-2)
#        -> seq_final (seq) and seq_final SEQ_FAMILY=seqnc
#   w2_final family mix (tasks 0-2) and nn_final (seeds 0-2) need no adsb features and start at once.
# Every job is idempotent (a finished seed is skipped), so a failed piece can simply be resubmitted alone.
set -euo pipefail
export PRC_RANKING=final
for f in ../final_ranking.parquet ../final_submitting.parquet cache_final/ranking.parquet cache_final/v11/feat_ranking.parquet; do
    [ -f "$f" ] || { echo "FATAL: $f missing (see the header)"; exit 2; }
done
python tools/make_final_cache.py
[ "$(python experiments/adsb/adsb_days.py | sed -n 430p)" = 2026-02-01 ] || { echo "FATAL: adsb_days index 429 is not 2026-02-01"; exit 2; }

E=--export=ALL,PRC_RANKING=final
sub() { sbatch --parsable "$@"; }
S1=$(sub --array=429-486%6 experiments/adsb/slurm_stage1_array.sbatch)
S2=$(sub --dependency=afterok:$S1 $E experiments/adsb/slurm_stage2_features.sbatch --which ranking)
W2=$(sub --array=0-2 $E cluster/w2_final.sbatch)
NN=$(sub $E cluster/nn_final.sbatch)
A1=$(sub --dependency=afterok:$S2 $E cluster/adsb2_final.sbatch)
A2=$(sub --dependency=afterok:$S2 $E cluster/adsb2_final_nn.sbatch)
EP=""
for n in 30 40 50; do EP="$EP $(sub --dependency=afterok:$S2 --export=ALL,PRC_RANKING=final,EPOCHS=$n cluster/seq_epochs_final.sbatch)"; done
VA=""
for v in ep60 ep80 ep40_d128 ep50_d128 ep40_l3 ep40_lr3; do
    VA="$VA $(sub --dependency=afterok:$S2 --export=ALL,PRC_RANKING=final,VARIANT=$v cluster/seq_variant_final.sbatch)"
done
SQ=$(sub --dependency=afterok:$S2 $E cluster/seq_final.sbatch)
SN=$(sub --dependency=afterok:$S2 --export=ALL,PRC_RANKING=final,SEQ_FAMILY=seqnc cluster/seq_final.sbatch)
cat <<EOF
submitted (PRC_RANKING=final):
  stage1 $S1 | stage2 $S2 | w2 mix $W2 | nn $NN | adsb2 $A1 | adsb2_nn $A2
  seq epochs$EP | seq variants$VA | seq $SQ | seqnc $SN
When everything has finished, pack the results for the laptop:
  tar czf final_fits.tgz cache_final/adsb_lol/features_ranking.parquet cache_final/adsb2_final cache_final/w2_final \\
      cache_final/nn_final cache_final/seq_final cache_final/seq_ep*_final
EOF
