# Pipeline runbook: from the challenge files to the submission

How the final submission of team **understanding-accordion** is produced, which script writes which
intermediate file, and in which order things must run. Written 2026-10-02 for the reproducibility
requirement of the PRC Data Challenge 2026 (code and documentation sufficient to reproduce the results).

The challenge data is not part of this repository (Terms & Conditions). `prc/config.py` expects the
challenge parquet files in the parent directory of the repository (`DATA = REPO.parent`); every derived file
goes to `cache/` (gitignored, rebuildable) and every submission to `outputs/` (gitignored).

## 0. The final step: `assemble.py`

```
python assemble.py --arm-u U --out v27      # writes outputs/understanding-accordion_v27.parquet
python assemble.py --arm-u U --check 26     # proves equivalence with a shipped file
```

`assemble.py` reads only the model outputs listed in section 1 -- no earlier submission file -- and applies one
closed formula per row group:

| row group | rule | rows in the 2026 ranking set |
|---|---|---:|
| routes A/B/C (NM off-block `AOBT_3_flt` present), outside zone P | `rint(clip(blend_A))` | 339,551 |
| route D (no `AOBT_3_flt`), outside zone P | `rint(clip(blend_D + 0.5 * corr))` | 3,570 |
| zone P (route D and take-off > 2 h after schedule) | `rint(clip(p * off_sched + (1 - p) * pb + corr))` | 1,720 |

- `blend_X` = the 8-member blend under the production tail classifier: for each of its 5 seeds
  `clip(Wb * p_s * et + (1 - p_s) * sum_f w_f * bulk_f + w_routed * routed)`, then the mean over seeds.
- With `--arm-u U`, routes A/B/C use one weight set per tercile of the disagreement between the 27 sequence-model
  fits (fixed 2025 cut points); `--arm-u UT` adds `lambda * (1 - p) * (seq10 - seqnc10)`.
- Equivalence: `--check 24` and `--arm-u U --check 26` reproduce the shipped v24/v26 exactly on routes A/B/C and
  route D; zone P differs by at most 2 s (rounding accumulated by the old build chain, in which every submission
  was an edit of the previous one).

## 1. Components read by `assemble.py`

| file (under `cache/`) | content | written by | upstream |
|---|---|---|---|
| `final/lgb_components.npz` | production tail classifier `pt` (5 LightGBM seeds), tail anchor `et`, zone-P mask `in_P`, row ids | `experiments/exp65_submit_v8.py --stage components` | base features (§2) |
| `adsb2_final/<fam>__seed0{0,1,2}.npy` | bulk predictions of the 7 members `mix perapt lintree xgboost routed nn w3` (routed: full prediction) | `cluster/w2_final.py --adsb --extra` via `cluster/adsb2_final.sbatch` (CPU) and `cluster/adsb2_final_nn.sbatch` (GPU) | member features (§2) |
| `seq_<V>_final/seq__seed0{0,1,2}.npy`, V in `ep30 ep40 ep50 ep60 ep80 ep40_d128 ep50_d128 ep40_l3 ep40_lr3` | the 27 neighbour-context sequence-model fits (bulk) | `cluster/seq_final.py` via `cluster/seq_epochs_final.sbatch` (`EPOCHS=30/40/50`) and `cluster/seq_variant_final.sbatch` (`VARIANT=...`) | member features + movement table (`prc/models/seq_net.py`) |
| `seq_final/seq__seed0{0,1,2}.npy`, `seq_final/seqnc__seed0{0,1,2}.npy` | 10-epoch sequence model and its no-context twin (only for `--arm-u UT`) | `cluster/seq_final.sbatch` (`SEQ_FAMILY=seq` / `seqnc`) | as above |
| `w2_final/mix__seed0{0,1,2}.npy` | W2 LightGBM bulk, half of zone P's real-taxi branch | `cluster/w2_final.py` via `cluster/w2_final.sbatch` | W2 features (§2, without adsb.lol) |
| `strategy/tfm_P_lirf_ranking.npz` | TabICL tail probability for Rome's 181 zone-P rows | `experiments/strategy/s21_tfm_P_final.py` | `exp62_cand_*.parquet` |
| `tfm_w4/pool_nonrome_ranking.npz` | TabICL tail probability for the other 1,539 zone-P rows | `experiments/tfm/w4b_final.py` | `exp62_cand_*.parquet` |
| `major/m11f_candidates.npz` | route-D corrector `corr = clip(trailD_full, 0) - v9` | `experiments/major/m11f_candidate_arrays.py` | `major/m11d_pred_trailD.npy` from `experiments/major/m11d_state_routeD.py`; the v9 submission (`experiments/exp65_submit_v8.py --w-abc 0.65 --w-d 0.5`, see dev log 2026-09-23) |
| `seq/swap_seqsw_ep30__...__seqsw_ep40_lr3.json` | per-route NNLS blend weights of the 8 members | `experiments/seq/seq_swap_eval.py` (arm M) | the fold predictions in `cache/cv/` |
| `seq/arm_u.json` | tercile cut points, per-tercile weights, lambdas, gate verdicts | `experiments/seq/arm_u_eval.py` | fold predictions |

Note on the corrector: it is defined against the route-D rows of the v9 submission, so v9 is a frozen input
of the corrector (not of the assembly). Rebuilding it from raw data means rerunning exp65 (v9 settings) and m11d.

## 2. Features

| file | content | written by |
|---|---|---|
| `training.parquet`, `ranking.parquet` (in `cache/`) | the challenge files, loaded once | `prc/io.py` |
| `external/metar/`, `external/atfm_daily.csv` | METAR (Iowa Environmental Mesonet ASOS archive), EUROCONTROL daily ATFM delays | `prc/weather.py`, `prc/disruption.py` |
| `v11/feat_{training,ranking}.parquet` | NM flight-plan clocks, previous-arrival stamping, stand turnover (incl. look-ahead within the file) | `experiments/v11/v11_features.py` |
| `adsb_lol/features_{training,ranking}.parquet` | 35 adsb.lol ground-movement columns per departure | `experiments/adsb/` (stage 1: one day per task, points within 12 km / 6000 ft; stage 2: events; `adsb_build_features.py`) |
| `w2/wtc_modal_by_type.json` | wake category per aircraft type (training) | `cluster/w2_extra.py` |
| `exp62_cand_{training,ranking}.parquet` | 38 zone-P candidate features | `experiments/exp62_screen.py` |

`cluster/w2_features.py` assembles the members' 143 columns from these (`assemble_w2(adsb=True, extra=True)`).

## 3. Cluster jobs (TU Wien KR cluster, `-p gpu`, 8 x L40S)

All job files are idempotent (existing outputs are skipped), check their inputs before starting, and pin
their settings in a `_config.json` per output folder. Fold campaigns (12 leave-one-month-out folds x 3 seeds) write
`cache/cv/<campaign>/`, final fits (all of 2025 -> the ranking set) write `cache/<name>_final/`.

| purpose | fold campaign | final fits |
|---|---|---|
| 7 members (adsb2) | `adsb2_cv_array.sbatch` (180 CPU folds), `adsb2_nn_array.sbatch` (72 GPU folds) | `adsb2_final.sbatch`, `adsb2_final_nn.sbatch` |
| sequence model, 9 settings | `seq_epochs_array.sbatch` (`EPOCHS=30/40/50`), `seq_variant_array.sbatch` (`VARIANT=ep60 ep80 ep40_d128 ep50_d128 ep40_l3 ep40_lr3`) | `seq_epochs_final.sbatch`, `seq_variant_final.sbatch` |
| 10-epoch seq + twin | `seq_array.sbatch` (`SEQ_FAMILY=seq/seqnc`) | `seq_final.sbatch` |
| W2 (no adsb.lol) | `w2_array.sbatch`, `w2_nn_array.sbatch` | `w2_final.sbatch`, `w2_final_nn.sbatch` |

Approximate cost: CPU member fold 6-7 min; seq fold 4-14 min on one L40S depending on epochs; a final fit
5-15 min. The whole set above is roughly 150 GPU-hours plus 40 CPU-hours.

## 4. Validation protocol (how every change was accepted)

12-fold leave-one-month-out on 2025; blend weights are fitted on the screening months 3/6/9/12 and judged on the other
8 ("held-out") and on January + July ("scored"). A change ships only if it passes a rule written before it was run:
held-out and scored better, held-out airport x local-day block-bootstrap 95% interval below 0, January and July each
better, every seed better, and the gain not carried by the 100 most-improved rows. The scripts are under
`experiments/` (`seq/seq_swap_eval.py`, `seq/arm_u_eval.py`, `adsb/a3_eval.py`, ...); every result is in `claude_log/`.

## 5. Data sources and licences

| source | use | licence |
|---|---|---|
| PRC Data Challenge 2026 dataset | training and ranking files | challenge T&C (not redistributed) |
| adsb.lol `globe_history` (GitHub releases) | ground-movement features | ODbL 1.0 (attribution; derived databases share-alike) |
| Iowa Environmental Mesonet ASOS/METAR | weather features | public archive |
| EUROCONTROL ATFM daily delays (ansperformance.eu) | disruption features | EUROCONTROL terms (non-commercial, attribution) |
| TabICL | zone-P tail classifier | BSD-3-Clause |
| LightGBM / XGBoost / PyTorch / scikit-learn | models | MIT / Apache-2.0 / BSD-3 / BSD-3 |
