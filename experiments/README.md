# experiments/

This folder holds every experiment we ran, kept as it was run. There are two kinds:
- **numbered scripts**, `exp01`-`exp82` (2026-09-21 to 2026-09-28);
- **22 topic folders**, one per study.

Each script's header states the question, the protocol and, where one applies, the decision rule written before the
run. Many headers also record the result. The upload-by-upload outcomes are collected in
[../docs/RESULTS.md](../docs/RESULTS.md), and the validation protocol is in
[../docs/VALIDATION.md](../docs/VALIDATION.md).

**Most of this folder is not needed to reproduce the submission.** Only the scripts in the production path below
are needed; everything else is a screen, a diagnostic or a dead end.
- **Kept as run.** These files are the record of what was tried and what was rejected, so they stay in the
  repository in the form in which they were run.
- **Some cannot be run as they stand.** Some exploratory scripts read caches written in a scratch directory that is
  not published, and some still carry absolute paths of the machine they ran on (in particular `ideas_inside/`,
  `ideas_outside2/`, `review_inside/`, `review_outside2/`, `deice_0926/`, `discord_eval/`, `ground_sources/` and
  `plan_1001/live_proj.py`). The scripts on the production path resolve their paths portably; a few still show the
  development machine's interpreter path in a usage comment.

## Production path: what built v29

The order runs from the challenge files to `assemble.py`. All outputs go to the git-ignored `cache/`.
[../docs/PIPELINE.md](../docs/PIPELINE.md) lists every file and job, and [../docs/REPRODUCE.md](../docs/REPRODUCE.md)
gives the commands.

| # | Script or job | What it produces |
|---:|---|---|
| 1 | `prc/io.py`, `prc/features.py`, `prc/congestion.py`, `prc/weather.py`, `prc/disruption.py` | the challenge files loaded once; base, queue, METAR and ATFM features |
| 2 | `experiments/major/m00_base.py` | compact departure frames (`cache/major/dep_*.parquet`) read by later steps |
| 3 | `experiments/v11/v11_features.py` | flight-plan clocks, previous-arrival stamping, stand turnover, with look-ahead within the file |
| 4 | `experiments/adsb/adsb_days.py`, `adsb_download.py`, `adsb_extract_day.py` (with `adsb_events.py`, `airports.py`), run as `experiments/adsb/slurm_stage1_array.sbatch` | 429 days of adsb.lol globe_history, streamed; points within 12 km of each airport; per-day departure events |
| 5 | `experiments/adsb/adsb_build_features.py`, run as `experiments/adsb/slurm_stage2_features.sbatch` | 35 `adsb_*` columns per departure (`cache/adsb_lol/features_*.parquet`) |
| 6 | `cluster/w2_extra.py`, `cluster/w2_features.py` | wake-sequence and schedule-backlog columns; the 143 member columns |
| 7 | `experiments/exp62_screen.py` | 38 zone-P candidate features (`cache/exp62_cand_*.parquet`) |
| 8 | `experiments/exp65_submit_v8.py --stage components` | production tail classifier (5 LightGBM seeds), tail anchor, zone-P mask (`cache/final/lgb_components.npz`) |
| 9 | `cluster/w2_task.py`, `cluster/w2_final.py` via `cluster/adsb2_cv_array.sbatch` (180 CPU folds), `adsb2_nn_array.sbatch` (72 GPU folds), `adsb2_final.sbatch`, `adsb2_final_nn.sbatch` | the seven members (`mix perapt lintree xgboost routed nn w3`): 12 folds × 3 seeds, plus finals |
| 10 | `cluster/seq_task.py`, `cluster/seq_final.py` (`prc/models/seq_net.py`) via `cluster/seq_epochs_array.sbatch` and `seq_epochs_final.sbatch` (`EPOCHS=30/40/50`), and `seq_variant_array.sbatch` and `seq_variant_final.sbatch` (`VARIANT=ep60 ep80 ep40_d128 ep50_d128 ep40_l3 ep40_lr3`) | the 27-fit seq pool: 9 settings × 3 seeds, folds and finals |
| 11 | `cluster/seq_array.sbatch` and `seq_final.sbatch` (`SEQ_FAMILY=seq` / `seqnc`) | the 10-epoch seq and its no-context twin, for the U+T term (see the note below the table) |
| 12 | `cluster/w2_task.py`, `cluster/w2_final.py` via `cluster/w2_array.sbatch`, `w2_final.sbatch` | the LightGBM member without adsb.lol columns, used in zone P's real-taxi branch |
| 13 | `experiments/strategy/s21_tfm_P_final.py` | TabICL tail probability for Rome's 181 zone-P rows |
| 14 | `experiments/tfm/w4b_final.py` | TabICL tail probability for the other 1,539 zone-P rows (pooled context) |
| 15 | `experiments/major/m11_state_features.py`, then `m11d_state_routeD.py`, then `m11f_candidate_arrays.py` | the route-D state corrector, defined against the v9 file that `exp65_submit_v8.py --w-abc 0.65 --w-d 0.5` builds |
| 16 | `experiments/drivers_0926/d01_rows.py` | per-row table that defines the airport × local-day blocks of the block bootstrap |
| 17 | `experiments/tail/v24_oof.py` | the v24-structure out-of-fold predictions that the U/U+T evaluation starts from |
| 18 | `experiments/seq/seq_swap_eval.py` (arm M) | per-route NNLS weights of the eight members (`cache/seq/swap_*.json`) |
| 19 | `experiments/seq/arm_u_eval.py` | tercile cut points, per-tercile weights, λ, and the gate verdicts for U and U+T (`cache/seq/arm_u.json`) |
| 20 | `python assemble.py --arm-u UT --lobt-window --join-complement --out v29` (run from the repository root) | the submission |

- **Final phase.** The final file runs the same path with `PRC_RANKING=final`: `tools/make_final_cache.py` first,
  `cluster/final_phase.sh` for every cluster fit, `final/f01_spec_x.py` for the zone-P specialist and candidate
  features, and `final/f02_check_final.py` for the pre-upload checks ([../docs/REPRODUCE.md](../docs/REPRODUCE.md) §8).
- **How the seqnc finals were run.** For v27-v29, the three `seqnc` finals were run with
  `cluster/seq_final.py --no-context --device cpu` on a laptop CPU in fp32. The fold fits used by the evaluation ran on
  GPU with bf16 autocast. It is the same model on a slightly different numeric path.
- **Frozen inputs of the evaluation scripts.** Steps 18 and 19 also read older out-of-fold caches for the parts of
  the structure they hold fixed:
  - the v9 and v11 stand-ins (`major/m01_anatomy.py`, `strategy/s13_v11_anatomy.py`, `v11/v11_base.py`);
  - the zone-P pieces (`exp62_cv.py`);
  - the TabICL out-of-fold predictions (`tfm/w4_tail_zone.py`);
  - the route-D corrector out-of-fold predictions (`major/m11d_state_routeD.py`).

  `assemble.py` itself needs none of them.

## Which builder made which upload

From v11 to v26, each submission was built as an edit of an earlier file (builders `exp70`-`exp82`). From v27 on,
`assemble.py` computes the whole file from the components. `assemble.py --check 24/26/27/28/29` reproduces the earlier files from components alone; see the
[README](../README.md#building-the-submission).

| Version | Builder | Version | Builder |
|---|---|---|---|
| v1, v2 | `exp04_submission_v1v2.py` | v14 | `exp74_submit_w2.py --with-nonrome` (it carries `exp73`'s change) |
| v3 | `exp11_submit_v3_mixture.py` | v15 | `exp75_submit_v15.py` |
| v4 | `exp17_submit_v4.py` | v16 | `exp76_submit_v16.py` |
| v5 | `exp52_submit_v5.py` | v17 | `exp77_submit_v17.py` |
| v6 | `exp54_submit_v6.py` | v18 | `exp78_submit_v18.py` |
| v7 | `exp64_submit_v7.py` (leaderboard test) | v19 | `exp79_submit_v19.py` |
| v8, v9 | `exp65_submit_v8.py` | v20 | `exp80_submit_v20.py` |
| v10 | `exp69_submit_routeD.py` (leaderboard test) | v21-v24, v26 | `exp81_submit_v21.py` (`--epochs`, `--pool`, `--arm-u U`) |
| v11 | `exp70_submit_v11.py` | v25 | `exp82_submit_v25.py` (rule override; lost, dropped) |
| v12 | `exp71_submit_v12.py` | v27-v29, final | `assemble.py` (repository root) |
| v13 | `exp72_submit_v13.py` | | |

`exp67_submit_v10.py` was written for a family blend under the name "v10". It was renumbered, superseded by
`exp71` (v12) and never uploaded.

## Topic folders

| Folder | Date | What it holds | Role |
|---|---|---|---|
| `adsb/` | 09-25 to 09-26 | adsb.lol pipeline (download, per-day extraction, events, per-departure features, Slurm jobs); `a3_eval.py` tests the adsb.lol columns inside all seven members; feasibility measurements; the "ADS-B v2" study (re-appearance and back-calculated off-blocks), which was not adopted | production (pipeline, `a3_eval`) and screens |
| `deice_0926/` | 09-26 | de-icing / disruption indicator from adsb.lol "remote rest" aggregates; residual screen, 2026 shift, robustness; not adopted | screen |
| `discord_eval/` | 09-25 | residual-harness measurements of ideas raised by participants on the challenge Discord. Wake sequence and schedule backlog were adopted (via `cluster/w2_extra.py`); the validation discipline became gates R3/R5; the anchored network target, stand × runway encoding, dew-point spread and outlier matching were not adopted | screen |
| `drift/` | 09-28 | label-free 2025 → 2026 drift of the adsb.lol columns and the members' reliance on coverage columns; closed, no action | diagnostic |
| `drivers_0926/` | 09-26 | decomposition of the v17-structure out-of-fold error (cells, day-scale label artefacts, label-free 2026 mix); `d01_rows.py` is reused by the block bootstrap; `lb_snapshot.py` is read-only public API | diagnostic, one production helper |
| `eurocontrol/` | 10-09 | `ec01_value.py`: value of EUROCONTROL's monthly taxi-out file as a per-airport-month calibration, measured on the 2025 analogue (2025 file against our 2025 out-of-fold predictions); every variant worse on the held-out months; not used | screen |
| `final/` | 10-09 | final phase: `f01_spec_x.py` (zone-P specialist and candidate features on the final file, with an initial-mode regression check) and `f02_check_final.py` (pre-upload checks of the final file) | production |
| `ground_sources/` | 09-27 | search for open ground surveillance at the ten airports (readsb snapshots and traces, OPDI surface-event counts); nothing usable at LTFM or LFPG | screen |
| `ideas_inside/` | 09-23 | diagnostics `diag_a`-`diag_s` on ideas from inside the data (which clock BLOCK matches, stamping propensity, flight identity, target forms, a P2 specialist); the stack precursor of v11 | screen; needs scratch caches |
| `ideas_outside2/` | 09-23 | screens of outside ideas on the v6 residual: stand turnover (later in v11), runway configuration, flight-level queue position, OSM stands, revision of off-block estimates, per-airport stacking | screen; absolute paths |
| `major/` | 09-24 | "major levers" study on the v9 stand-in: error anatomy and oracles, leak checks of row order and IDs, year-ahead tests on the 2026 arrivals (provided data), the January 2026 disruption, and the route-D state corrector (`m11*`, shipped since v10) | production (`m00`, `m11`, `m11d`, `m11f`) and diagnostics |
| `plan_1001/` | 10-01 | read-only post-hoc screens of the 10-01 planning round: disagreement-tercile weights (arm U) found here, skeptic re-checks, gap anatomy, `live_proj.py` (diagnostic on our own live scores) | screen |
| `research_0925/` | 09-25 | v15 error anatomy, CatBoost screen (weight 0), calibration of large predictions, Rome day-shift label artefacts (priced, not shipped), board cut-off analysis from the public API | screen |
| `review_inside/` | 09-23 | reviewer re-checks of the `ideas_inside` results | screen; needs scratch caches |
| `review_outside2/` | 09-23 | reviewer re-checks of the `ideas_outside2` results (per-airport member, stand family rebuilt) | screen; needs scratch caches |
| `routed_0926/` | 09-26 | keep, halve, drop or refit the route-D corrector on the v17 base: `r00`-`r02` under a pre-registered rule; `r03` is the post-hoc calibration against our own v9 → v10 live delta that led to the half-strength choice in v18 | evaluation (v18) |
| `seq/` | 09-26 to 10-02 | the neighbour-context set model: subset signal tests and ablations (`s00`-`s05`); evaluation as the 8th member (`seq_eval`) and in zone P (`seq_p_branch`); settings screen; swap evaluations for epochs and the 27-fit pool (`seq_swap_eval`); arms U and U+T (`arm_u_eval`); member seeds (arm O, stopped) | production (`seq_swap_eval`, `arm_u_eval`) and evaluations |
| `strategy/` | 09-24 | clock regimes, the moderate-delay band, tabular foundation models for Rome's zone P (TabPFN and TabICL; `s21` makes the production TabICL arrays), seed scaling, season weights, cut-off projections from the public API; `s18` is a diagnostic on our own live scores | production (`s21`) and screens |
| `tail/` | 09-27 to 10-06 | v21/v24 error anatomy, Rome stamping regime, carrier stamping rate (`t06`/`t07`, the basis of the dropped v25), shared residuals; `v24_oof.py` (production helper); `v28_oof.py` (the v28-structure out-of-fold predictions) and `t08_oof_corrector.py`, a residual-corrector bet whose rule was frozen on 10-06; it failed its January ↔ July cross-fit gate and was not uploaded ([../docs/RESULTS.md](../docs/RESULTS.md) §6.5) | screens, one production helper |
| `tfm/` | 09-24 | TabICL as the zone-P classifier at every airport (`w4_tail_zone`; `w4b_final` makes the production arrays); fine-tuning (F1) and bulk retrieval (R1), both dead; new features for Rome (F2) | production (`w4b_final`) and screens |
| `v11/` | 09-24 | the v11 feature stack: features (`v11_features`, production), the base out-of-fold predictions, the residual stacker (shipped in v11, replaced from v14 by the same signals inside the members), concentration and parity checks | production (`v11_features`, `v11_base`) and evaluation |
| `w2/` | 09-24 to 09-26 | evaluations of the v11 signals inside the members (W2, v14), route D and zone P (v15), the second network W3 (v16), 10 network seeds (not adopted), and the v15/v17 error anatomy | evaluations |

## Numbered scripts

| Range | Topic |
|---|---|
| `exp01`-`exp07` | groundwork: tail-rate stability; what `AOBT_3` is relative to BLOCK; baseline ladder; v1/v2; local gate noise (absolute RMSE bootstrap sd ≈ 47 s) and the paired gate |
| `exp08`-`exp12` | the two-stage mixture, a specialist for flights without `AOBT_3`, stand occupancy as an upper bound, v3, and two fixes to the mixture |
| `exp13`-`exp19` | leave-one-month-out CV; congestion, weather and ATFM features; split feature sets for regressor and classifier; v4; which CV protocol tracks the leaderboard (forward vs LOMO, day vs month folds) |
| `exp20`-`exp23` | distributional (binned) regression: probe, CV, as a tail classifier, discretisation floor. Not adopted |
| `exp30`-`exp39` | the tail mechanism: the BLOCK = SCHED stamping artefact, a taxonomy of the 584 tail rows, BLOCK forensics, multi-anchor mixture, the Rome cell, the zone-P specialist, seed replication, propensities, seed averaging |
| `exp40`-`exp53` | alternative model families: CatBoost, XGBoost and the embedding network on screening folds; blend and decorrelation; LightGBM capacity; underfit subpopulations; v5 |
| `exp54`-`exp66` | v6; the routed model; zone P artefact vs real (`exp62`) and the three-way target (`exp63`); v7-v9 builders; the family blend (`exp66`) |
| `exp67`-`exp82` | submission builders v10-v25 (table above) |

## Glossary

- **probe:** an offline single-fold (or subsample) experiment that checks whether a model trains and deserves a full
  CV run, e.g. `exp20_distributional_probe.py`, `exp42_catboost_probe.py`, `strategy/s04_tabpfn_probe.py`. It never
  means a leaderboard probe.
- **tools/leaderboard.py:** a read-only client of the public leaderboard API. It reports our scores, rank and the
  board's movement. It uploads nothing.
- **OOF, LOMO:** out-of-fold predictions from 12-fold leave-one-month-out CV on 2025.
- **Screening months, held-out months:** blend weights are fitted on months 3, 6, 9 and 12, and judged on the other
  eight.
- **"scored" in script output:** January + July **2025** out-of-fold rows, the analogue of the scored months. It is
  never a 2026 label.
- **Label-free:** computed on 2026 without any 2026 departure label (row counts, prediction shifts, coverage).
- **Route A/B/C, route D, zone P:** with an NM off-block record (A: `0 < MVT − AOBT_3 ≤ 7200 s`, B: above, C:
  `≤ 0`); without one (D); and route D with take-off more than 2 h after schedule (P).
- **Stand-in (`v9~`, `v11~`):** a 2025 out-of-fold reconstruction of a shipped structure.
- **Finals:** fits on all of 2025 that predict the 2026 ranking file.
- **Arm:** one variant in a pre-registered comparison.
- **Gates R1-R5, X1, X2:** the decision rule. R1-R5 are defined in the [README](../README.md#validation-in-brief);
  all of them, including X1 and X2, are in the header of `seq/arm_u_eval.py`.
- **Campaign names:**
  - `W2`: the v11 signals inside the members;
  - `W3`: the second network;
  - `A3`: adsb.lol inside the members (`adsb` = arm 1, `adsb2` = arm 1 plus wake and backlog);
  - `C1`: a consolidated retrain with de-icing shares (failed);
  - `seq`/`seqnc`: the set model and its no-context twin;
  - `bet D`: the seq model's working name.
- **"LEADERBOARD TEST" and "LIVE CALIBRATION" in a header:** an upload made as a model-level test (v7, v10), or a
  script that uses our own live scores (`routed_0926/r03_calibrate.py`). Both are disclosed in the README and in
  [../docs/VALIDATION.md](../docs/VALIDATION.md).
