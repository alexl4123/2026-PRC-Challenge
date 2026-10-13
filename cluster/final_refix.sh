#!/bin/bash
# Final phase, ADS-B fix (2026-10-10): re-extract the four days whose staging archives were read only up to their first
# 2 GB part (stage 1 before commit d715307: tarfile stopped silently at a damaged header) and refit everything that
# reads the ranking-side adsb.lol features. Run on the login node from the repo root after git pull and
# conda activate prc2026:
#
#   bash cluster/final_refix.sh
#   REFIX_DAYS=2026-06-06 REFIX_IDX=463 REFIX_TAG=2 bash cluster/final_refix.sh    # second round (2026-10-13):
#     06-06 is published as a stray .tar plus a .tar.aa/.ab set; the old reader stopped after the first archive.
#
# Affected days (adsb_days.py index): 2026-01-29 (394), 2026-06-07 (464), 2026-06-23 (480), 2026-06-25 (482).
# 2025-10-14 (training) is left as it is: the shipped model was trained on it.
# Old points/events of those days and the old adsb2/seq final folders are MOVED to *_pre_refix (not deleted), because
# every job skips work whose output exists. w2_final (mix) and nn_final read no ADS-B features and are kept.
set -euo pipefail
export PRC_RANKING=final
DAYS="${REFIX_DAYS:-2026-01-29 2026-06-07 2026-06-23 2026-06-25}"
IDX="${REFIX_IDX:-394,464,480,482}"
T="pre_refix${REFIX_TAG:+_$REFIX_TAG}"     # backup folder suffix; round 1 used pre_refix
for d in $DAYS; do
    i=$(python experiments/adsb/adsb_days.py | grep -n "^$d\$" | cut -d: -f1)
    case ",$IDX," in *",$(( i - 1 )),"*) ;; *) echo "FATAL: $d is not at one of the indices $IDX"; exit 2 ;; esac
done
grep -q "def iter_tar" experiments/adsb/adsb_extract_day.py || { echo "FATAL: stage-1 fix missing (git pull)"; exit 2; }
grep -q "n_dup_dropped" experiments/adsb/adsb_extract_day.py || { echo "FATAL: de-duplication missing (git pull)"; exit 2; }

[ -e "cache_final/$T" ] && { echo "FATAL: cache_final/$T exists; pass a new REFIX_TAG"; exit 2; }
P=cache/adsb_lol/points; V=cache/adsb_lol/events
mkdir -p "${P}_$T" "${V}_$T"
for d in $DAYS; do
    [ -f "$P/$d.parquet" ] && mv "$P/$d.parquet" "${P}_$T/"
    [ -f "$V/$d.parquet" ] && mv "$V/$d.parquet" "${V}_$T/"
done
mkdir -p "cache_final/$T"
for f in cache_final/adsb_lol/features_ranking.parquet cache_final/adsb2_final cache_final/seq_final cache_final/seq_ep*_final; do
    [ -e "$f" ] && mv "$f" "cache_final/$T/"
done
echo "moved aside: ${P}_$T, ${V}_$T, cache_final/$T/ ($(ls cache_final/$T | wc -l) entries)"

E=--export=ALL,PRC_RANKING=final
sub() { sbatch --parsable "$@"; }
S1=$(sub --array=$IDX experiments/adsb/slurm_stage1_array.sbatch)
S2=$(sub --dependency=afterok:$S1 $E experiments/adsb/slurm_stage2_features.sbatch --which ranking)
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
submitted (PRC_RANKING=final, ADS-B fix):
  stage1 $S1 ($DAYS) | stage2 $S2 | adsb2 $A1 | adsb2_nn $A2
  seq epochs$EP | seq variants$VA | seq $SQ | seqnc $SN
Check stage 1 when it is done (bytes_read must equal the published size, n_dup_dropped / n_skips are reported):
  grep -h '"bytes_read"' logs/prc-adsb1_${S1}_*.out
When everything has finished, pack the results for the laptop:
  tar czf final_fits_refix.tgz cache_final/adsb_lol/features_ranking.parquet cache_final/adsb2_final \\
      cache_final/seq_final cache_final/seq_ep*_final
EOF
