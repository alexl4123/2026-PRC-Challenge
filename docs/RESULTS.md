# Results: every upload, offline expectation against live score

Team **understanding-accordion**, PRC Data Challenge 2026 (departure taxi-out time, 10 European airports).

Author: Alexander Beiser (TU Wien), team understanding-accordion.

This file covers all 28 files we uploaded to the public leaderboard, from v1 (2026-09-21) to v28 (2026-10-06). For each
upload it gives the change, the script that built it, the number of rows that moved, what our 2025 validation predicted
beforehand, whether the change passed the decision rule written for it, and what the leaderboard returned. After the upload
table come the calibration of offline estimates against live deltas, a ledger of what each part of the final model was
worth live, a full account of how we used the public leaderboard, and the ideas we measured that did not work.

Related documents: the model is described in [MODEL.md](MODEL.md), the validation protocol in
[VALIDATION.md](VALIDATION.md), and the build in [PIPELINE.md](PIPELINE.md) and [REPRODUCE.md](REPRODUCE.md).

**Where the numbers come from.**

- **Live scores** come from the public leaderboard API. All 28 files are scored on the same 344,841 departures
  (`usedPairs` = 344,841). `python tools/leaderboard.py` lists them, and the values below were checked against the API on
  2026-10-06.
- **Offline estimates** were produced by the evaluation scripts named in the notes, at the time of each decision. They are
  copied from our dated internal experiment log, which is not published.
- **Rows changed** counts the departures whose integer prediction differs from the parent file. It is computed from our own
  submission files and is label-free.
- **2026 figures.** Apart from the live scores, no 2026 figure here is derived from scored labels; the one figure that
  uses a live score is the v28 bound, which applies label-free row moves to v27's live score. Every other 2026 number
  is computed from our own files and the input columns of the ranking file, and is marked *label-free*.

## At a glance

- **Best and last file:** v28 at **235.3996**, rank 21 of 232 on 2026-10-06. It is built by
  `python assemble.py --arm-u UT --lobt-window --out v28`.
- **Uploads:** 28 over 10 upload days, never more than the limit of 5 per UTC day.
- **Progress:** from v3, the first mixture model (352.2657), to v28 the score fell by 116.87 s. Five steps carry 70% of
  that:
  - v4, richer bulk features: −30.02;
  - v17, adsb.lol ground tracks: −23.53;
  - v6, zone-P tail specialist: −13.09;
  - v28, LOBT window: −7.75;
  - v14, NM-clock and stand-turnover information inside all members: −7.58.
- **Uploads that passed a written rule:** 19 changes were shipped after passing a decision rule written before their
  evaluation ran, and all 19 improved the live score.
  - Live delta divided by offline estimate: median 1.06, range 0.42–4.14.
  - Summed over the 19: −69.29 s live against −56.84 s offline (×1.22). See [§4](#4-offline-against-live-calibration).
- **Uploads outside that rule.** Three model-level leaderboard tests: the pair v1/v2, v7 and v10. One override of a
  failed rule: v25, which lost 1.07 s and was dropped. One decision taken with a live check: the v18 tie-break, which is
  still in v28. v1–v5 predate written rules. All of this is set out in [§3](#3-use-of-the-public-leaderboard).
- **v28** adds one deterministic constraint, the LOBT ±3,606 s window. The idea was adopted from public repositories, and
  in 2026 it moved 63 rows (label-free).

## Terms

| Term | Meaning |
|---|---|
| routes A/B/C | Departures with a Network Manager off-block record (`AOBT_3_flt`): 339,551 of the 344,841 ranking departures (label-free). |
| route D | Departures without that record: 5,290 ranking departures. |
| zone P | Route D **and** take-off more than 2 h after schedule: 1,720 ranking rows, 4,944 in 2025. In 2025 these rows often carry the scheduled time recorded as off-block. |
| 12m | 12-fold leave-one-month-out on 2025, all months pooled. |
| held-out | The 8 months not used to fit blend weights. Weights are fitted on months 3, 6, 9 and 12. |
| J+J | January and July 2025, the 2025 analogue of the scored months (344,419 departures). |
| CI | 95% paired bootstrap. From v18 on it is an airport × local-day block bootstrap. |
| R1–R5 | The gate set from v19 on, first in [`experiments/seq/seq_eval.py`](../experiments/seq/seq_eval.py) and in its final form, with X1/X2, in [`experiments/seq/arm_u_eval.py`](../experiments/seq/arm_u_eval.py) (details below the table). |
| Δ live | Live score minus the parent file's live score. The parent is the previous upload, except for v26, whose parent is v24. |

The gate set R1–R5:

- R1: held-out and J+J both better.
- R2: held-out block-bootstrap CI below 0.
- R3: January and July each better.
- R4: every seed better.
- R5: the held-out gain survives removing its 100 most-improved held-out rows.

Gate history:

- **Before v6:** no gate was written down.
- **v6 to v17:** the 12-month or held-out paired CI below 0. From v14 on, J+J also had to improve; from v17 on, January
  and July each had to improve separately.
- **From v19 on:** R1–R5.

## 1. Upload record

Builders are paths relative to the repository root.

- **Before v27:** each submission was built by its own builder `experiments/expNN_*`, kept for the record; from v11 to
  v26 each builder edited an earlier submission file.
- **The reproduction path:** [`assemble.py`](../assemble.py) builds every row group from stored model outputs.
  - `--check 24`, `--arm-u U --check 26` and `--arm-u UT --check 27` reproduce v24, v26 and v27.
  - Routes A/B/C and D come out exactly. Zone P matches within 2 s, the rounding that the old edit chain accumulated.

| v | date 2026 (UTC) | change | builder | rows changed | offline expectation (scope) | gate | live | Δ live |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | 09-21 | LightGBM, one regressor, ~35 features, training target trimmed to y ≤ 7,200 s | `experiments/exp04_submission_v1v2.py` | – | J+J holdout: 257.57 capped, 563.89 uncapped | none written | 543.6699 | – |
| 2 | 09-21 | **Paired with v1:** identical but untrimmed | same | 341,032 | J+J uncapped −97.5 vs v1 (capped gate then in use: +35.7) | none written; see §3 | 507.5496 | −36.12 |
| 3 | 09-21 | Mixture: P(tail)·anchor + (1 − P(tail))·bulk regressor; tail = y > 7,200 s, anchor = take-off − schedule | `experiments/exp11_submit_v3_mixture.py` | 341,181 | J+J holdout, paired vs v2: −66.1 [−131.3, −16.8] | none written | 352.2657 | −155.28 |
| 4 | 09-22 | Bulk regressor on 72 features (base + congestion + METAR + ATFM delay), tail classifier on 18 base features, 5 seeds | `experiments/exp17_submit_v4.py` | 341,422 | 12m −8.41 | none written | 322.2460 | −30.02 |
| 5 | 09-22 | LightGBM bulk at 511 leaves / 1,400 rounds / lr 0.025 (was 255 / 350 / 0.05) | `experiments/exp52_submit_v5.py` | 337,749 | bulk rows, 3 single-month folds: −1.0 to −2.1 | none written | 321.2771 | −0.97 |
| 6 | 09-23 | Zone-P tail specialist (monotone 15-leaf GBM + shrunk airport × delay table + isotonic fit, averaged) | `experiments/exp54_submit_v6.py` | 1,707 | 12m (4 seeds) −13.06 [−25.7, −4.0] | pass | 308.1903 | −13.09 |
| 7 | 09-23 | **Leaderboard test:** specialist + 42 per-flight features inside zone P | `experiments/exp64_submit_v7.py` | 1,507 (1,038 by > 1 s), all zone P | 12m −1.50 [−3.02, +0.26]; J+J −6.29 [−10.56, −3.03] | **fail**, uploaded as a test | 304.3510 | −3.84 |
| 8 | 09-23 | Entity-embedding network (3 seeds), 50/50 with LightGBM in the bulk branch | `experiments/exp65_submit_v8.py` | 337,637 | 12m −3.94 [−4.72, −3.09]; J+J −4.78 [−6.49, −3.18]; expected ≈ −3 after a seed adjustment | pass | 297.8290 | −6.52 |
| 9 | 09-23 | Network weight per route: 0.65 on A/B/C, 0.5 on D | `exp65_submit_v8.py --w-abc 0.65 --w-d 0.5` | 315,836 | 12m −0.31 [−0.40, −0.23]; J+J −0.48 [−0.72, −0.30] | pass | 297.3414 | −0.49 |
| 10 | 09-24 | **Leaderboard test:** route-D disruption corrector at full strength | `experiments/exp69_submit_routeD.py` | 5,029, all route D | 12m −0.60 [−0.94, −0.25]; J+J −0.63 | pass on 12m, but uploaded as a test (§2) | 292.3393 | −5.00 |
| 11 | 09-24 | Residual feature stack outside P: NM clock differences, stamping at the stand's previous arrival, stand turnover incl. look-ahead | `experiments/exp70_submit_v11.py` | 332,482 | 12m −4.82 [−5.35, −4.30]; J+J −5.31 [−6.54, −4.08]; 12/12 months | pass | 289.6993 | −2.64 |
| 12 | 09-24 | NNLS blend of four families on A/B/C: network, per-airport LightGBM, linear-tree LightGBM, XGBoost | `experiments/exp71_submit_v12.py` | 328,698 | 12m −1.38; J+J −1.75; 11/12 months | pass | 286.9012 | −2.80 |
| 13 | 09-24 | TabICL (in-context tabular foundation model) as the zone-P classifier at LIRF (Rome) | `experiments/exp72_submit_v13.py --variant tabicl` | 180, all Rome zone P | 12m −2.34 [−3.96, −1.03]; J+J −4.73 [−7.73, −2.20]; 10/12 | pass | 284.4106 | −2.49 |
| 14 | 09-24 | "W2": the v11 signals inside all six retrained members (stack removed); pooled TabICL for zone P at the other nine airports | `experiments/exp74_submit_w2.py --base 13 --with-nonrome` | 333,810 | held-out −1.47 [−1.81, −1.16]; J+J −2.30 [−3.19, −1.63]; + non-Rome P ≈ −0.1 / −0.2 | pass | 276.8324 | −7.58 |
| 15 | 09-25 | W2 members on route D outside P (corrector kept) and as zone P's real-taxi branch; 3-seed weights | `experiments/exp75_submit_v15.py` | 254,330 | 12m −1.14; J+J −1.02 (two parts, each passing) | pass | 275.3106 | −1.52 |
| 16 | 09-25 | W3: a second network (piecewise-linear numeric encoding, residual MLP, two heads) as a 7th member | `experiments/exp76_submit_v16.py` | 331,608 | held-out −1.10 [−1.35, −0.90]; J+J −1.21 [−1.68, −0.79]; 12/12 | pass | 274.6519 | −0.66 |
| 17 | 09-26 | adsb.lol ground-movement features (35 columns), plus wake and schedule-backlog columns, inside all 7 members | `experiments/exp77_submit_v17.py --arm adsb2` | 340,562 | held-out ≈ −19.5, J+J ≈ −16.8 (sum of two passing arms) | pass | 251.1243 | −23.53 |
| 18 | 09-26 | Route-D corrector halved outside zone P; 45 rows hit by a clipping error in the old edit chain repaired | `experiments/exp78_submit_v18.py` | 3,455, all route D outside P | 12m, all flights: −0.40 [−0.62, −0.19] | pass; **tie-break chosen with a live check** (§3) | 250.3448 | −0.78 |
| 19 | 09-26 | "seq": neighbour-context set model as the 8th member outside P | `experiments/exp79_submit_v19.py` | 330,024 | held-out −2.11 [−2.97, −1.53]; J+J −1.51 [−2.41, −0.89] | pass (R1–R5) | 248.7469 | −1.60 |
| 20 | 09-26 | seq in zone P's real-taxi branch (0.5 W2 LightGBM + 0.5 seq) | `experiments/exp80_submit_v20.py` | 1,704, all zone P | held-out −0.65 [−1.29, −0.22]; J+J −0.73 [−1.28, −0.35] | pass | 245.7300 | −3.02 |
| 21 | 09-27 | seq trained for 15 epochs (was 10) | `experiments/exp81_submit_v21.py` | 320,495 | held-out −0.65 [−0.81, −0.52]; J+J −0.62 [−0.95, −0.37] | pass | 245.4673 | −0.26 |
| 22 | 09-27 | seq at 20 epochs | `exp81_submit_v21.py --epochs 20` | 317,619 | held-out −0.26 [−0.40, −0.12]; J+J −0.31 [−0.58, −0.10] | pass | 245.1468 | −0.32 |
| 23 | 09-27 | seq at 30 epochs | `exp81_submit_v21.py --epochs 30` | 323,100 | held-out −0.60 [−1.08, −0.27]; J+J −0.40 [−0.68, −0.16] | pass | 244.6576 | −0.49 |
| 24 | 09-28 | seq = mean of 27 fits (9 settings × 3 seeds); arm defined after a grid failed, rule written before it ran | `exp81_submit_v21.py --pool <9 settings>` | 320,164 | held-out −0.61 [−0.86, −0.32]; J+J −0.65 [−0.98, −0.40] | pass | 244.2213 | −0.44 |
| 25 | 09-28 | **Rule override:** Rome zone-P TabICL with carrier stamping-rate features | `experiments/exp82_submit_v25.py` | 180, all Rome zone P | 12m −1.52 [−3.29, −0.05]; J+J −2.25 [−5.40, +0.02]; January +0.06 | **fail** (R3, R5); uploaded on the team's judgement | 245.2878 | +1.07 |
| 26 | 10-02 | "U": per-tercile NNLS weights on A/B/C, by disagreement across the 27 seq fits; built on v24 | `exp81_submit_v21.py --pool <9> --arm-u U` | 319,221 vs v24 | held-out −0.20 [−0.27, −0.15]; J+J −0.26 [−0.43, −0.15] | pass (gates frozen; design post hoc) | 243.4219 | −0.80 vs v24 |
| 27 | 10-06 | "U+T": U + λ·(1 − p)·(10-epoch seq − its no-context twin) on A/B/C | `assemble.py --arm-u UT --out v27` | 292,725 | vs v24: held-out −0.59 [−0.77, −0.43], J+J −0.60 [−0.97, −0.35]; vs U: J+J −0.34 | pass | 243.1478 | −0.27 |
| 28 | 10-06 | **63-row constraint:** LOBT ±3,606 s feasibility window | `assemble.py --arm-u UT --lobt-window --out v28` | 63 | v24 OOF (in-sample for the rule): held-out −0.55 [−0.95, −0.24], J+J −1.07 [−2.32, −0.31]; 2026 bound ≤ −5.99 if the 2025 join rule holds (label-free moves applied to v27's live score) | R1–R3 pass; R5 passes only vacuously; R4/X1/X2 not applicable; header written after the numbers | 235.3996 | −7.75 |

## 2. Notes on individual versions

**v1/v2: paired baseline (a leaderboard test).**
- *What it was:* the two files differ only in whether training rows with y > 7,200 s were dropped.
- *Why the pair went up:* our local gate at that time capped the holdout at 7,200 s, so it could not see the tail. It
  favoured trimming by 35.7 s. An uncapped holdout, which the gate did not use, pointed the other way (97.5 s in favour
  of the untrimmed model on these two files).
- *Live:* the untrimmed model was better by 36.12 s.
- *What changed as a result:*
  - The primary local metric became uncapped RMSE compared with a paired bootstrap.
  - On the uncapped Jan+Jul holdout an absolute RMSE has a bootstrap sd of 46.9 s, and pairing cuts the sd of a
    difference by 2.7× (56.2 → 20.9 s).
  - The tail was then handled by a mixture (v3) instead of by trimming.
- *Listed as:* a model-level use of the leaderboard in §3.

**v3: the mixture.** The predictor is E[y] = P(tail)·anchor + (1 − P(tail))·bulk, with the scheduled-time offset as the
tail anchor.
- In 2025, 481 of the 584 departures with y > 7,200 s have no NM record, and their median gap |y − (take-off − schedule)|
  is 4 s: the airport recorded the schedule as the off-block time.
- Live −155.28 s, 2.3× the holdout estimate.

**v4: offline under-estimated the live gain 3.6×.**
- Two other protocols gave the same 2025 effects as leave-one-month-out (exp18/exp19):
  - forward in time (train January–September);
  - day-grouped folds.
- So the protocol was not the cause. We concluded that the offline gates **order** changes but do not **size** them.

**v6: tail specialist for zone P.**
- 83% of its 2025 SSE gain came from four Rome rows.
- Live −13.09 against a cluster estimate of −13.06.

**v7: leaderboard test.**
- *Rule outcome:* the 12-month CI crossed zero, so the rule failed. J+J was clearly negative.
- *Why it went up anyway:* with the team's approval it was uploaded to measure the change directly.
- *Scope:* only zone-P rows moved (label-free), so the live delta measures the change on the 2026 zone-P rows alone.
- *Status:* the LightGBM specialist was replaced by TabICL in v13/v14. The 38 numeric candidate features built for v7
  are TabICL's inputs today.

**v10: leaderboard test.**
- *What it is:* a residual model for route-D departures. It uses the airport's trailing state, all backward-looking and
  all taken from the ranking file:
  - the last 90 min of NM taxi estimates from flights that have a record;
  - the share of flights without a record;
  - recent arrival taxi-in.
- *Why it was a test:* its target was a disruption at EHAM (Amsterdam) in early January 2026. That event is visible
  label-free in the ranking file's arrival records and in the share of departures without an NM record, and it has no
  2025 counterpart.
- *Offline:* the 12-month CI cleared zero, but it was driven by a single 2025 analogue month (LTFM, February).
- *Decision:* uploaded as a test with the team's approval. Full strength rather than half was chosen before the upload,
  from a 2025 analogue.
- *Status:* kept at full strength in zone P; halved outside P since v18.

**v11: under-delivered (×0.50).**
- In 2025, 65% of the stack's gain sat in the 0.31% of rows whose correction exceeded 300 s.
- Capping or halving those large corrections cost +0.5 to +2.0 s on 2025, so the stack was left unchanged; it was not
  tuned against the live score.
- v14 removed the stack once its signals were inside the members.

**v13: a single-airport change.**
- 180 of Rome's 181 zone-P rows changed; it swapped the classifier for that population.
- TabICL alone was chosen over a TabPFN + TabICL ensemble before the upload, for licence reasons: BSD-3 against an
  attribution clause. The offline difference to the ensemble was within its CIs (0.03 s on J+J, 0.29 s on 12m).

**v14: attribution.**
- This step, −7.58 s, comes mainly from the W2 members, not from TabICL at the nine other airports.
- Offline, W2 accounts for −2.30 of the ≈ −2.5 s J+J estimate; the non-Rome zone-P change for ≈ −0.1 to −0.2.

**v16** is the last upload without adsb.lol features (274.6519). It is the fallback if use of a flight's own surface
track were ever ruled out.

**v17: adsb.lol.**
- *Coverage:* off-block estimates exist for 19.4% of 2025 and 25.6% of 2026 departures (label-free). There are
  essentially none at LTFM and LFPG.
- *Where the gain sits:* on 2025 OOF, 95.3% of the SSE reduction from these features sits on rows where the flight's own
  surface track was observed. That locates the gain; it is not an ablation.
- *Permission:* used under the organisers' statement on the challenge Discord (recorded by us on 2026-09-25, paraphrased) that open
  trajectory data may be used to extract off-block times.

**v18: corrector strength, with a deviation from the pre-registered tie-break.**
- *Rule outcome:* outside zone P, both "drop the corrector" and "halve it" passed the pre-registered rule.
- *What the tie-break said:* pick the lower error under a 2025-pattern 2026 estimate (E1). That selects **drop**.
- *What was shipped instead:* **half**. The tie-break had been written before any calibration existed. Afterwards, a
  post-hoc calibration was run against our own v9 → v10 live delta (−5.00 s), using
  [`experiments/routed_0926/r03_calibrate.py`](../experiments/routed_0926/r03_calibrate.py).
  - E1 predicted the wrong sign for that delta.
  - A same-day estimate (E2) predicted −5.38 s.
  - Under E2, half beats drop.
- *Status:* this is the one model decision that used a live delta (§3). The halved corrector is in v28.

**v19/v20: seq.**
- The seq member reads up to 64 neighbouring movements at the same airport, month-isolated, with label-free tokens.
- Context against its no-context twin, on 2025 bulk rows: −5.68 s [−6.53, −4.82]; LTFM −17.0.
- v20's ×4.1 is not diagnosed. 2026 has about twice as many zone-P rows per month as 2025 (label-free), and v20 moved
  1,704 of them.

**v24** pools all 27 seq fits. All eight pre-registered single-setting arms had failed against 30 epochs. The pooled arm
was defined after that result, and its rule was written before the pool was evaluated.

**v25: rule override, lost.**
- *Offline:* carrier stamping rates raised Rome zone-P discrimination (Brier 0.131 → 0.101). The rule still failed:
  January got worse (R3), and the gain was carried by its top 100 rows (R5).
- *Why it went up:* the team uploaded it on its own judgement.
- *Live:* +1.07 s.
- *Afterwards:* v25 was not carried forward; v26 was built on v24. It is the second time a better Brier score on Rome's
  627 zone-P training rows failed to become a better RMSE. Rome zone P was closed after this.

**v26/v27: two variants of one design.**
- *How they were found:* U was found among about 18 post-hoc screens on 2025 OOF (2026-10-01). Its design and gates were
  then frozen in [`experiments/seq/arm_u_eval.py`](../experiments/seq/arm_u_eval.py), committed before the evaluation
  ran (9d0dd0b).
- *What the gates measured:* U and U+T each passed every gate.
- *Why both went up:* the team decided before either file was built to upload both. It also fixed a choice rule for a
  single-file final: U+T replaces U only if its live score is better by more than 0.3 s.
- *What happened to that rule:* it became moot when the organisers confirmed best-of scoring.
- *Why v28 builds on v27:* v27 also has the stronger offline evidence (held-out −0.59 against −0.20 vs v24), so the
  choice does not rest on the 0.27 s live difference.
- *Reproducibility caveat:* v27's no-context seq finals were trained on a laptop CPU in fp32, while the folds used to fit
  λ ran on GPU with bf16 autocast.

**v28: the LOBT window.**
- *Rule:*
  - **Observation (2025, in-sample for the rule):** every 2025 departure with a flight-plan match (LOBT present, 2,062,577
    rows) has its recorded off-block within ±3,606 s of LOBT. There are zero exceptions, and the edge is sharp: 215 rows
    in (3,600, 3,606], 45 in [−3,606, −3,600). No other clock gives an exact bound: within ±3,606 s are 99.914% for
    AOBT_3, 99.997% for IOBT and 99.970% for EOBT_1.
  - **Reading:** the dataset's movement-to-flight-plan join evidently required |BLOCK − LOBT| ≤ 3,606 s. This is an
    artefact of how the dataset was built, not operational knowledge.
  - **Implementation:** each prediction of a departure with an LOBT is clipped into [max(0, take-off − LOBT − 3,606), take-off − LOBT + 3,606].
- *Build safeguard:* the build refuses to run unless the 2025 data still shows this exact bound.
- *2026 checks (label-free):*
  - LOBT coverage 98.47% in 2026 against 98.92% in 2025.
  - Take-off − LOBT > 3,606 s on 2.11% against 2.22%.
  - No row would force a negative taxi time in either year.
- *Effect on 2025:* on the v24 OOF, 236 rows change and none gets worse. About 79% of the gain sits in 10 rows.
- *Effect on 2026 (label-free):* 63 v27 rows lay outside their window, 60 below and 3 above.
  - By airport: LFPG 24, EHAM 15, EGLL 5, LTFM 5, LEMD 4, EDDF 3, LIRF 2, EDDM 2, LEBL 2, LSZH 1.
  - If the rule holds in 2026, every moved row's error can only shrink, which bounds the score at ≤ 237.16 (−5.99 s).
    The bound applies these label-free moves to v27's live score.
  - 96% of that guaranteed gain comes from one LIRF July departure. Our mixture had hedged that row between two hypotheses, both
    of which lie outside its window. Without that row the bound is 242.92.
- *How the gates apply:* on the 2025 OOF, R1–R3 pass; R5 passes only vacuously, because no row can get worse, and the
  gain is concentrated (about 79% in 10 rows); R4, X1 and X2 test fitted quantities and do not apply. We do not claim
  the window passes the rule in the ordinary sense. The header in
  [`assemble.py`](../assemble.py) was written after the 2025 effect and the 2026 bound were measured, and committed before
  the build (618ba86).
- *Credit:* the idea was adopted on 2026-10-06 from the public repositories
  [EnioAguiar/prc-taxiout-2026](https://github.com/EnioAguiar/prc-taxiout-2026),
  [javidmardanov/PRC-Data-Challenge-2026](https://github.com/javidmardanov/PRC-Data-Challenge-2026) and
  [joris-decombe/taxiout](https://github.com/joris-decombe/taxiout). We took ideas only; the code is our own. Competitors'
  leaderboard statements were not used as evidence.
- *Pre-commitment, recorded before the upload:*
  - upload once;
  - no later upload changes a projected row;
  - no implied label is derived from the score.
- *Live:* −7.75 s, consistent with the bound. Per the pre-commitment, nothing further is inferred from it.

## 3. Use of the public leaderboard

The leaderboard scores every file on all 344,841 departures and keeps each team's best score; the limit is 5 uploads per
UTC day. Every file we uploaded is in the table above. This section lists every way we used the leaderboard beyond
recording our score.

**Model-level leaderboard tests: three occasions, four files.**

| Version | What the upload tested | Status in v28 |
|---|---|---|
| v1/v2 (09-21) | A paired baseline, run to learn whether the 2026 labels share the 2025 heavy tail, which our capped gate could not see. | The lesson stayed: untrimmed target and a mixture model. |
| v7 (09-23) | A zone-P classifier change whose 12-month rule failed. | Replaced by TabICL in v13/v14; the 38 features remain TabICL's inputs. |
| v10 (09-24) | A route-D corrector aimed at a 2026 event that 2025 validation cannot cover. | Kept: full strength in P, half outside. |

**One model decision used a live delta: v18.**
- Choosing between halving and dropping the corrector outside zone P, we departed from the pre-registered tie-break,
  which said drop.
- Instead we used our own v9 → v10 live delta to decide which of two label-free estimators to trust.
- The halved corrector is in v28.

**One override of a failed rule: v25.**
- It lost 1.07 s live and was not carried forward.
- The rule after it: do not override a failed rule for zone-P changes again.

**Two variants of one design: v26 and v27.**
- The team decided to upload both before either was built, and both passed frozen gates.
- v28 builds on v27, which also has the stronger offline evidence.

**Single-airport changes: v13 and v25.**
- Each changed only Rome's zone-P rows (180), by swapping the classifier for that population.
- Neither edited individually chosen rows.

**v28.**
- The idea came from public repositories, and the rule header was written after its numbers were known.
- R5 passes vacuously, and the concentration of the bound in one row is disclosed above.
- It was uploaded once, under a pre-commitment that no later upload changes a projected row and that no implied label is
  computed from its score.

**Diagnostics on our own scores (read-only).** These scripts compute aggregate quantities from our own live scores and
our own files.

| Script | What it computes |
|---|---|
| [`experiments/major/m04_live_decomp.py`](../experiments/major/m04_live_decomp.py) | Size of the 2026 bulk error implied by the v7 → v8 → v9 deltas. |
| [`experiments/major/m08_P_submission_deltas.py`](../experiments/major/m08_P_submission_deltas.py) | Where the zone-P-only uploads v5 → v7 moved predictions. |
| [`experiments/strategy/s18_nn_weight_live.py`](../experiments/strategy/s18_nn_weight_live.py) | A parabola through the v7/v8/v9 scores against network weight. Its suggested weight was not acted on. |
| [`experiments/routed_0926/r03_calibrate.py`](../experiments/routed_0926/r03_calibrate.py) | The v18 calibration above. |
| [`experiments/plan_1001/live_proj.py`](../experiments/plan_1001/live_proj.py) | For pairs of uploads v16–v25, the projection of the live change onto the prediction shift (was the shift about the right size?). Not used to tune anything. |

Only r03 fed an uploaded file (v18).

**What was never done.**
- No file was built to recover labels or to probe individual rows.
- No uploaded file differs from its parent by an edit of rows selected by id. Every change is a model or a rule applied to
  a population defined from inputs: route, zone P, airport, LOBT.
- A class of Rome rows that may carry a one-day recording shift (identified label-free) was never submitted as a
  row-subset change.
- Competitors' live results were not used as evidence.

**Data deliberately not used.**
- *What the data are:* EUROCONTROL publishes airport-level monthly taxi-out statistics, including averages and counts of
  long taxi-outs, for January and July 2026. They summarise the evaluation labels.
- *When we saw them:* they were identified and seen during scouting on 2026-09-21 and 2026-09-23. That was before the
  team decided on 2026-09-23 never to use external summaries of the scored months' labels.
- *Use:* none, in any model, check or decision. No tracked code reads them.
- *What they could have been worth, measured on 2025 only:* a perfect airport-month level correction is worth −0.14 s on
  J+J out-of-fold predictions (−0.28 s over 12 months).

## 4. Offline against live calibration

**What is compared.** The offline figure is the J+J estimate where one exists, otherwise 12m (v6, v18). Ratio = live Δ /
offline Δ.

| v | offline Δ | live Δ | ratio | | v | offline Δ | live Δ | ratio |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 6 | −13.06 | −13.09 | 1.00 | | 18 | −0.40 | −0.78 | 1.96 |
| 8 | −4.78 | −6.52 | 1.36 | | 19 | −1.51 | −1.60 | 1.06 |
| 9 | −0.48 | −0.49 | 1.02 | | 20 | −0.73 | −3.02 | 4.14 |
| 11 | −5.31 | −2.64 | 0.50 | | 21 | −0.62 | −0.26 | 0.42 |
| 12 | −1.75 | −2.80 | 1.60 | | 22 | −0.31 | −0.32 | 1.03 |
| 13 | −4.73 | −2.49 | 0.53 | | 23 | −0.40 | −0.49 | 1.22 |
| 14 | ≈ −2.5 | −7.58 | 3.03 | | 24 | −0.65 | −0.44 | 0.68 |
| 15 | −1.02 | −1.52 | 1.49 | | 26 | −0.255 | −0.80 | 3.14 |
| 16 | −1.21 | −0.66 | 0.55 | | 27 | −0.34 | −0.27 | 0.81 |
| 17 | ≈ −16.8 | −23.53 | 1.40 | | | | | |

**Outside this table:**

| Version | offline | live | ratio | Note |
|---|---:|---:|---:|---|
| v3 | −66.1 | −155.28 | 2.3 | Holdout era. |
| v4 | −8.41 | −30.02 | 3.6 | Holdout era. |
| v7 | 12m −1.50 / J+J −6.29 | −3.84 | 2.6 / 0.61 | Leaderboard test. |
| v10 | J+J −0.63 (12m −0.60) | −5.00 | 7.9 (8.3) | Leaderboard test; its event is absent from 2025. |
| v25 | J+J −2.25 | +1.07 | sign reversed | |
| v28 | bound −5.99 (label-free moves, v27 live score) | −7.75 | | The bound is a different kind of estimate. |

**What the record says.**

1. **Signs transfer; sizes do not.** All 19 rule-passing changes improved the live score. Each ratio lies between 0.42
   and 4.14, with median 1.06. Summed, they give −69.29 s live against −56.84 s offline (×1.22), or ×1.14 without v17.
2. **New information inside the members delivered more live than offline:** v14 3.0, v20 4.1, v12 1.6, v15 1.5, v17 1.4,
   v8 1.4. Residual layers and second learners delivered about half: the v11 stack 0.50, W3 (v16) 0.55, and TabICL at
   Rome (v13) 0.53 on J+J (1.06 against its 12-month estimate). We used this pattern to set expectations, not to choose
   between candidates.
3. **Small changes are near the noise level.** The seq tuning steps v21–v24 delivered ×0.76 together (−1.51 against
   −1.98). Single live deltas of 0.26–0.49 s are of the same order as the J+J CI half-widths of such changes
   (0.14–0.31 s, [VALIDATION.md §2.2](VALIDATION.md#22-paired-comparisons)). We did not read a live delta under about
   0.3 s as confirmation or refutation.
4. **2025 cannot validate a 2026-only event:** v10 delivered about 8× its offline estimate (7.9× on J+J, 8.3× on
   12m). A 627-row classifier gain can reverse: v25.
5. **Absolute levels do not transfer, only paired deltas.**
   - v5's J+J 2025 OOF RMSE was 390.94 against 321.28 live; v24's was 292.21 against 244.22.
   - The 2025 J+J RMSE is dominated by a few day-scale recording errors: removing its worst one, two and three rows gives
     364.3, 335.8 and 309.7.

## 5. Component ledger

Each row is a part of v28, with the live delta of the upload that introduced it and of the uploads that later changed it.

**Caveats:**
- These deltas are not additive shares of the final score. Later parts absorbed or replaced earlier ones: the v11 stack's
  features moved inside the members in v14, and the v8 network now has weight 0.
- An RMSE delta also depends on the level it is measured from.
- **Reference point:** v2, a single untrimmed regressor at 507.5496. Its row sums are the totals below.

| Part of v28 | live Δ at introduction and later changes | total | status in v28 |
|---|---|---:|---|
| Mixture structure, base and bulk features (congestion, METAR, ATFM), lean 5-seed tail classifier | v3 −155.28; v4 −30.02; v5 −0.97 | −186.27 | Tail probabilities and anchor read from `cache/final/lgb_components.npz`. |
| Zone-P classifier | v6 −13.09 (specialist); v7 −3.84 (per-flight features, leaderboard test); v13 −2.49 (TabICL at Rome); v14 non-Rome TabICL bundled (offline ≈ −0.1 / −0.2) | −19.42 | TabICL on all 1,720 zone-P rows: own context at Rome, pooled elsewhere. |
| NM clock, stamping and stand-turnover information | v11 −2.64 (as a residual stack); v14 −7.58 (inside the members, mainly W2); v15 −1.52 (route D and zone-P bulk) | −11.74 | Inside all members; the stack was removed. |
| Member families and blend weights | v8 −6.52 (network); v9 −0.49; v12 −2.80 (four families); v16 −0.66 (W3) | −10.47 | Non-zero weights in v28: seq, per-airport LightGBM, XGBoost, W3, routed. The v8 network, the LightGBM member and the linear-tree member have weight 0; a LightGBM (W2) is half of zone P's bulk. |
| Route-D disruption corrector | v10 −5.00 (leaderboard test); v18 −0.78 (halved outside P) | −5.78 | Full in zone P; ×0.5 on route D outside P. |
| adsb.lol ground movement, wake and backlog features | v17 −23.53 | −23.53 | Inside all members. |
| seq neighbour-context member | v19 −1.60; v20 −3.02 (zone-P bulk); v21–v24 −1.51 (epochs, 27-fit pool) | −6.12 | Weight 0.40–0.65 on A/B/C by tercile; half of zone P's bulk. |
| Disagreement-tercile weights (U) and context term (T) | v26 −0.80; v27 −0.27 | −1.07 | Routes A/B/C. |
| LOBT window | v28 −7.75 | −7.75 | 63 rows in 2026 (label-free). |
| **Sum, v2 → v28** | | **−272.15** | = 507.5496 − 235.3996 |

Not in v28:

- v1's target trim (v2, without it, was 36.12 s better);
- v25's carrier feature (+1.07, dropped);
- the residual stack of v11 (superseded in v14).

## 6. Negative results

These are ideas we built and measured that did not earn a place, with the numbers that decided them. "12m", "J+J" and
"held-out" are offline 2025 deltas in seconds; negative means better. The ideas are grouped by subject.

### 6.1 Target, loss and model form

| Idea | Measurement | Verdict |
|---|---|---|
| Trim the training target at 7,200 s (one regressor) | J+J: bulk −38 s, tail rows +6,748 s RMSE; live v1 543.67 against v2 507.55 | Rejected. The mixture handles the tail. |
| Residual target y − (take-off − AOBT_3) defined on all rows | Worse in all 6 single-month folds tested, e.g. January 397.95 vs 359.7, July 493.01 vs 424.4 | Rejected. The target is discontinuous across routes. |
| Distributional (binned) regression | Best 518.32 against the mixture's 296.22 (12m). The quantisation floor of a fine grid is only 27 s, so the fit failed, not the binning. | Stopped. |
| Three-way split: normal, delayed, recording artefact | −1.57 [−1.89, −1.26] against a matched low-capacity baseline, replicated on 8 unused months. Against the production-capacity v6: −0.17 [−0.49, +0.16]. It captures ≈ 3.4% of a 46 s oracle, the gain is on normal flights, and the delayed-band bias is unchanged (+398 s vs +380 s). | Not adopted. |
| Fitted tail regressor instead of the schedule anchor | −5.79 ± 3.81, CI [−13.6, +1.1] | Not established. |
| Set small tail probabilities to zero | Bulk RMSE unchanged (277.3) at every threshold from 1e-5 to 3e-2 | No effect. |
| Separate model for all departures without an NM record | 419.36 against 397.26 | Rejected. |

### 6.2 Recording artefacts: zone P, stamping, day shifts

| Idea | Measurement | Verdict |
|---|---|---|
| Classify "block = schedule ± 60 s" instead of y > 7,200 s | −4.47 ± 17.20, CI [−41.4, +26.7]. Inside Rome's zone P it separates tail from bulk worse (61.5 / 51.7% vs 73.98 / 52.99%). | Rejected. |
| Multi-anchor mixture (which clock the block time was copied from) | +8.04 ± 14.88; ordinary rows 233.1 → 236.0 | Rejected. |
| Out-of-fold stamping propensities per airport × pier, stand or operator | −10.42 ± 5.34 with them against −9.88 ± 5.13 without; a finer table gives −7.60 ± 6.10 | No gain. |
| Recorder-outage burstiness | Same-day concordant pairs 1,350 against 1,302 ± 44 under permutation (z = +1.1) | No signal. |
| Stand-occupancy upper bound as a tail discriminator | P(tail ∣ bound > 7,200 s) 0.031% against a 0.028% base rate | No signal. |
| 24-hour date-error override | Only 4 such rows in 2025 (y > 40,000 s with an ordinary schedule offset); base rate ≈ 0.018% | Not built. |
| Midnight-crossing flag | 2,142 rows that cross midnight with an ordinary schedule offset contain 0 tail rows | Not built. |
| Physical bounds as tail vetoes (ADS-B, wake separation, taxi distance) | The tail is a recording error, not a physical dwell | Class retired. |
| TabICL fine-tuning for Rome zone P (3 variants) | All within 0.1 s of in-context TabICL; no CI clear of 0 | Rejected. |
| Pooled all-airport TabICL context for Rome | J+J +1.09 [−0.32, +2.53]; 5/12 months | Rejected. |
| Rome zone P + stand stamping and turnover features | 12m −0.40 [−1.15, +0.11]; 9/12 | Fails its rule. |
| Rome zone P + arrival stamping regime | Brier 0.1306 → 0.1230, but 12m +0.16 and J+J +0.77 | Fails its rule. |
| Rome zone P + carrier stamping rate | Brier 0.1306 → 0.1009, 12m −1.52, but January +0.06 and carried by the top 100 rows | Fails its rule; uploaded anyway as v25, live +1.07. |
| Explicit stamp mixture for Rome A/B/C | Held-out 482.5 against 451.3 for the members | Rejected. |
| Rome route-A stamping mixture | +0.38 (12m) / +0.71 (J+J) on the stack, 3/12 months; +1.68 on the v15 base | Rejected; the members already carry it. |
| Route-D tail anchor capped by the delay | −0.039 / −0.017 s | Too small. |
| Route-D corrector re-fitted on the current base | −0.23, July +0.05 | Fails. |
| Route-D corrector re-fitted with carrier rates | 12m −0.59, but +0.11 without the top 100 rows | Fails (R5). |

### 6.3 Features and external data

| Idea | Measurement | Verdict |
|---|---|---|
| Longer congestion windows, forward windows, day load | An oracle of any airport × day × hour shared error bounds this at −4.90 on J+J (390.94 → 386.04). No observable state tracks it (Spearman ≤ 0.03). | Dropped before building. |
| Unused NM columns (ARVT_1, ARVT_3) | Spearman with the residual between −0.029 and +0.014 | Not built. |
| De-icing indicator from adsb.lol "remote rest" aggregates | −0.11 [−0.19, −0.04] (12m), but the top 100 rows carry 109% of the gain; other seeds +0.018 / −0.084 | Not adopted. |
| Retrain with de-icing shares + network NaN sentinel | Held-out +0.34 [+0.17, +0.56]; with seq in the blend −0.000. The network member got 3.54 s worse, which we attribute to the sentinel on the ADS-B columns. | Rejected. |
| ADS-B v2 estimators (back-calculated off-block, reappearance, visibility, surface congestion) | −0.03 [−0.31, +0.24] on top of the production columns; power check: the production columns are worth −16.2 | Not adopted. |
| Label-free queue signals at LTFM (neighbours' NM gap, arrival taxi-in anomaly) | LTFM SSE −0.16% / −0.09%; EGLL +0.71% | No signal. |
| Stand-to-runway distance; stand × runway encoding | −0.12 / −0.03 with CIs including 0; OpenStreetMap distance out-of-month +0.3 to +0.8 | Rejected. |
| Temperature − dewpoint spread | −0.01 [−0.07, +0.04] | No signal. |
| Runway-configuration regimes | +0.04 to +0.72 | Rejected. |
| Forward runway-queue position | ≈ −0.3, CIs touch 0 | Not built separately. |
| Callsign target encodings | 21.9% of 2026 scored-month departures carry a callsign unseen in 2025, against 1.8% within 2025 (label-free); unseen-callsign rows at Rome got worse | Dropped. |
| Season-proximity sample weights | σ = 3 months: −0.12 [−0.31, +0.07]; σ = 1.5: +0.45 | Rejected. |
| Recalibration per airport × route | Linear −0.09 [−0.42, +0.22]; isotonic +0.10 | Rejected. |
| Drop adsb.lol coverage columns against receiver drift | +3.78 (January) / +2.77 (July) bulk RMSE on 2025; 2026 predictions move little where coverage drifted most (label-free) | Columns kept. |
| Map test-set outliers onto the training distribution | +42 to +66 s (quantile map) | Rejected. |
| OpenSky state vectors | 11% of departures show an observable pushback; zero ground rows at LFPG, LIRF, LTFM; not open data for this challenge per the organisers | Not used. |
| Open ground data for LTFM and LFPG | adsb.lol: 0 ground points at LTFM in 2025; OPDI stand-exit events: 0 at LTFM in the sampled periods | None exists. |

### 6.4 Learners, capacity, seeds and blending

| Idea | Measurement | Verdict |
|---|---|---|
| Wider LightGBM | 1,023 leaves worse than 511 (month 4: 200.86 vs 200.66) | Information-limited. |
| Network hyper-parameter sweep (22 arms, 4 screening months) | Best arm −0.59 in a blend; seed spread 0.08 s; 40 epochs +4.12 solo; uncapped embeddings +0.86; Huber loss changed residual correlation by 0.000 | Closed. |
| 10 network seeds instead of 3 | +0.040 [+0.003, +0.077] (12m) | Rejected. |
| CatBoost member | Blend weight 0.000 next to six families | Dropped. |
| TabPFN as a bulk regressor | Beats LightGBM at equal n (2,000 rows) but loses to full-data LightGBM (Rome July 568 vs 544); blend weight 0 or harmful | Rejected. |
| TabICL on the bulk residual with retrieved context | 0 of 4 airport-month cells reach −0.5% | Rejected. |
| seq beyond 30 epochs; width 128, 3 layers, lower learning rate (8 pre-registered arms) | Held-out −0.22 to +0.28; per-seed checks fail | All fail. |
| seq settings screen at 10 epochs (context 128, 2-h window, width, depth) | The rule compared 2-seed configurations with a 3-seed baseline (flaw found afterwards; rule not changed). Like-for-like, only 15 epochs cleared its 1.0 s bar (−1.80). Context size was not re-tested. | Only training length was pursued. |
| seq: 6 seeds of one setting | −0.074 [−0.205, +0.042] | Fails. |
| seq: 54-fit pool instead of 27 | −0.105 [−0.238, +0.011] | Fails. |
| Member seeds 3–5 | Going from 1 to 3 seeds bought 0.069 held-out at blend level, so 3 → 6 is worth ≈ 0.02 | Stopped. |
| Per-airport blend weights | January → July cross-fit +0.375 (LIRF +3.05); route × airport +0.08 [−0.04, +0.21] | Rejected (three times). |
| Weights per route × own-ADS-B coverage | J+J +0.082 | Rejected. |
| Recalibration binned by prediction; correction binned by member disagreement | +0.267 / +0.129; +0.154 | Rejected. |
| Keep the residual stack on top of W2 | Held-out −1.50 with it against −1.89 without | Removed in v14. |
| seq as a residual model of v17 | +3.89 | The 8th-member design was used instead. |
| seq switched off at Rome (pre-registered second arm) | Passes but worse: −1.70 against −2.11 held-out | Tie-break: ungated. |
| Two-stage neighbour context (stage-1 predicted off-blocks as tokens) | Explains 0.10% of residual SSE; leaks through OOF predictions | Rejected. |

### 6.5 Pre-registered bets after v28 (2026-10-06)

| Idea | Measurement | Verdict |
|---|---|---|
| Other exact join windows, like the LOBT one | 2025 rows with \|BLOCK − X\| > 3,606 s: LOBT 0, IOBT 58, EOBT_1 614, AOBT_3 1,769, SCHED 107,499 (route D: BLOCK − SCHED spans −86,402 to +151,090 s) | Only LOBT is exact; nothing more to clip. |
| Out-of-fold residual corrector on the v28 structure ([`experiments/tail/t08_oof_corrector.py`](../experiments/tail/t08_oof_corrector.py); LightGBM on clipped residuals of routes A/B/C, rule frozen in the header before the first run; the idea was raised by a participant on the challenge Discord) | Held-out −0.591 [−0.778, −0.419], J+J −0.715 [−1.140, −0.407], January −0.331 [−1.010, +0.071], July −1.101, 11/12 months; but the January ↔ July cross-fits were +0.305 and +0.386 (X2 fails) | Not uploaded and not re-tuned, per the frozen rule. |

### 6.6 Not tried, on rule grounds

- **Published scored-month airport statistics.** Excluded by team rule (see §3).
- **Few-row, high-variance bets.** For example, re-serving a handful of possible day-shift rows at Rome, or re-tuning
  zone P by live score. These amount to probing the leaderboard.
- **Uploading several variants of one change and keeping the best live score.** v26/v27 is the one case of two variants;
  it was decided in advance with a choice rule (§2).
- **Pseudo-labels for 2026 rows from each flight's own ADS-B track.** That would mean training on the scored months.
- **A second LOBT-window variant** (point estimate, median offset, truncated mean). Excluded by the v28 pre-commitment.

## 7. How this was built

**Decisions** were taken by the team: what to test, which data sources are acceptable, which rule a change must pass,
and every upload.

**Code and analysis** were written with AI assistance (Claude, by Anthropic); the commits carry `Co-Authored-By`
trailers.

**Decision rules.**
- *v1–v5:* these predate written rules.
- *From v6 on:* every shipped change was gated by a rule written before its evaluation ran, with these exceptions:
  - v7, v10 and v25, listed in §3;
  - the v18 tie-break;
  - the v28 window, whose header was written after its 2025 effect and 2026 bound were known.
- *Commit order:* where a rule was committed on its own, commit order is consistent with it. Examples: 9d0dd0b came before
  the v26 build, and 618ba86 before the v28 build. Commit timestamps cannot prove this.
- *Where the rules are:* from v14 on, most rules sit in the headers of the evaluation scripts
  ([`experiments/w2/w2_eval.py`](../experiments/w2/w2_eval.py), [`experiments/adsb/a3_eval.py`](../experiments/adsb/a3_eval.py),
  [`experiments/seq/seq_eval.py`](../experiments/seq/seq_eval.py), [`experiments/seq/seq_swap_eval.py`](../experiments/seq/seq_swap_eval.py),
  [`experiments/seq/arm_u_eval.py`](../experiments/seq/arm_u_eval.py)). Earlier rules are recorded in the internal
  experiment log.

<!-- Later uploads (v29+) are appended to the §1 table with the same columns. -->
