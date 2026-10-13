# Validation, noise and use of the public leaderboard

Team **understanding-accordion**, PRC Data Challenge 2026 (taxi-out time of departures at ten European airports).

Author: Alexander Beiser (TU Wien), team understanding-accordion.

This document covers five things:

- how each model change was judged before it was uploaded;
- how much noise those judgements carry;
- how well the offline numbers predicted the public leaderboard;
- which guards keep labels out of the features;
- and, in full, how we used the public leaderboard.

Companion documents:

- [MODEL.md](MODEL.md) describes the model.
- [RESULTS.md](RESULTS.md) lists every upload.
- [REPRODUCE.md](REPRODUCE.md) and [PIPELINE.md](PIPELINE.md) say how to rebuild the files.

The best file on the public board is **v29**, which scores **234.8988**. It is built by

```
python assemble.py --arm-u UT --lobt-window --join-complement --out v29
```

The final-phase file is the same design rebuilt on the organisers' final ranking file (§8).

**Conventions**

- **What the offline numbers are.** Every offline number is a 2025 out-of-fold (OOF) RMSE in seconds. It is not capped and covers all departures unless stated.
- **Signs.** Δ means new minus old, so a negative Δ is an improvement.
- **Intervals.** Intervals in brackets are 95% intervals.
- **"Jan+Jul 2025".** This is the 2025 stand-in for the scored months; the code calls it `scored`. It never refers to 2026 labels.
- **2026 statements.** Each one is either one of our own public scores (marked *live*) or computed without any 2026 departure label (marked *label-free*); the mixed cases, the v28 and v29 bounds and the pricing of the 10-06 structure-search candidates, apply label-free row moves to the parent's live score. Nothing in this document is derived from 2026 labels except through our own public scores.
- **Dates.** All dates are in 2026.
- **Sources.** Numbers come from the scripts named next to them and from our internal experiment log, which is a dated record of every run and is not published.

---

## 1. The protocol

### 1.1 Folds: leave one month out on 2025

The training set has **2,085,047 departures** from January to December 2025. Every model member is fitted **12 times**, once for each held-out calendar month, at **3 seeds**. So every training row has an out-of-fold prediction from models that never saw its month (`prc/validation.py::month_cv_oof`; the cluster campaigns are in `cluster/`).

**Why calendar months.**

- **The scored set is two whole months.** January and July differ from each other and from the rest of the year. With the same model, the July 2025 error on ordinary flights was 78 s higher than January's (bulk RMSE 299.9 against 221.8). The two months also fail for opposite reasons. In January most of the error sits in a few route-D clock errors, and in July most sits at Rome.
- **Event features are built within one calendar month.** Stand turnover, sequence context and backlog are computed month by month, because the ranking file holds two months that are not contiguous (§4.2). Month folds put the same boundary into validation.

**The known weakness: training on later months.** Leave-one-month-out (LOMO) trains on months that come *after* the held-out month, which the real 2025 → 2026 task cannot do. We measured whether this changes any comparison:

| Check | Paired Δ, protocol 1 | Paired Δ, protocol 2 | Script |
|---|---|---|---|
| Forward split (train Jan–Sep, test Oct–Dec) vs LOMO | −1.76 ± 0.32 | −1.76 ± 0.32 | `experiments/exp18_forward_vs_lomo.py` |
| Day-grouped vs month-grouped 6-fold CV | −5.63 ± 0.24 | −5.71 ± 0.25 | `experiments/exp19_day_vs_month_folds.py` |

Neither check changed a comparison. We therefore use LOMO to *rank* candidates. Absolute OOF levels are optimistic, and we never used them to forecast a live score (§3).

### 1.2 Screening months, held-out months and Jan+Jul 2025

We split the months between fitting and judging:

- **Screening months 3, 6, 9 and 12 (697,218 rows).** These are used for every fitted blend quantity:
  - the per-route non-negative least-squares weights (no intercept, fitted outside zone P);
  - the disagreement-tercile cut points of arm U;
  - the λ of U+T;
  - every hyper-parameter screen.

  We chose one month per quarter. Months 1 and 7 are excluded because they stand in for the scored months. Months 4 and 10 are excluded because earlier selections had used them (LightGBM sizing on month 4, the first model-family comparison on month 10).
- **Held-out months 1, 2, 4, 5, 7, 8, 10 and 11.** The decision is made on these.
- **Jan+Jul 2025.** These two months are a subset of the held-out months. They are reported both together and separately.

**Every comparison covers all flights.** That means routes A/B/C (an NM off-block record exists), route D (no record) and zone P (route D, more than 2 h behind schedule); the definitions are in [MODEL.md](MODEL.md). A change aimed at one group is still judged on all of them.

**The metric is plain, uncapped RMSE.** On the first day we validated on a target capped at 7,200 s. The first two uploads showed that the live score follows the *uncapped* holdout (§6.2), and we switched. The capped number survives only as a diagnostic.

### 1.3 The decision rule

The rule we used for v26 and v27 is coded in `experiments/seq/arm_u_eval.py`. Design and gates were committed in 9d0dd0b, before the evaluation ran. A candidate is compared with the shipped version on all flights.

| Gate | Requirement | Why it exists |
|---|---|---|
| **R1** | Held-out (8 months) **and** Jan+Jul 2025 better (point estimates) | A change that helps in only one of them is likely fitted to one regime. |
| **R2** | Held-out airport × local-day **block-bootstrap** 95% interval entirely below 0 (1,000 draws) | The row bootstrap is too narrow (§1.4). |
| **R3** | January **and** July 2025 each better | The two months fail for different reasons (§1.1). Added on 09-25, following advice in the participants' public discussion. It later caught schedule-backlog features that gained in July and stayed flat in January: +0.03 [−0.30, +0.37]. |
| **R4** | Every seed alone better on held-out (members and sequence pool at seed *s* only, both sides refitted) | The bootstrap does not see refit noise (§2.3). |
| **R5** | The held-out gain survives removing the 100 most-improved held-out rows | 0.1% of rows carry about half the squared error (§2.1). A gain carried by a handful of rows is not a property of the model. |
| **X1** | Fitted leave-one-month-out (both sides refitted per month), better in at least 10 of 12 months | Checks that the gain does not depend on fitting weights on months 3, 6, 9 and 12. |
| **X2** | Jan → Jul and Jul → Jan single-month cross-fits both below 0 | Does a structure fitted in one season survive in the other? |
| Reported, not gated | R5 on Jan+Jul rows; Δ per airport | |
| 2026, label-free | Each tercile's share of 2026 A/B/C rows within 10 points of 33%; mean shift against the previous version within ±6 s in every airport × month cell | In 2025, the same shift statistic for earlier swaps stayed within about ±5 s. |

**What we check before every upload.**

- The built file must reproduce the previous version exactly wherever the change does not apply (`--check`).
- Its mean shift per airport × month is compared with the same statistic on the 2025 OOF.
- The largest moved rows are inspected.
- The file is validated against the submission template and read back.

**How the rule grew.** It did not exist in this form from the start. [RESULTS.md](RESULTS.md) lists each upload with the rule it was judged by.

| From | Rule in force | First used for |
|---|---|---|
| 09-21 | Paired bootstrap on the same rows; the interval must lie below 0 | v3 |
| 09-22 | 12-fold, multi-seed confirmation; screening months 3, 6, 9 and 12 reserved for selection | v4–v6 (blend weights from the screening months from v9 on) |
| 09-23 | Adopt only if the 12-month paired interval lies below 0 | v7 failed it (§6.2) |
| 09-24 | Held-out **and** Jan+Jul better, held-out interval below 0 | v14 |
| 09-25 | Also January and July each better | v17 |
| 09-26 | Block bootstrap; every seed better; not carried by 100 rows (R1–R5 in `experiments/seq/seq_eval.py`) | v19–v24 |
| 10-02 | Also X1 and X2; R5 restricted to held-out rows (§5.2 explains why this matters) | v26, v27 |

For v3–v5 the first two rows were working practice, applied when each result came in; the first rule written down before its run was the one for the v6 cluster confirmation (09-22: ship only if the paired interval is clear of zero). [RESULTS.md](RESULTS.md) therefore marks v1–v5 "none written".

**R5 has one limit.** It was written for changes to the bulk of flights. A change confined to a few hundred zone-P rows concentrates its effect by construction. The Rome zone-P changes (v13, v25) and the LOBT window (§5.4) are exactly where R5 says little.

### 1.4 The airport × local-day block bootstrap

**Why the row bootstrap is too narrow.** Errors cluster by airport and day. On a bad day at one airport (snow, de-icing, a runway closure, a recording problem) many flights miss together.

- Among the 3,653 airport-days of 2025:
  - the worst 1% carry 10.2% of the squared error of ordinary flights;
  - the worst 5% carry 23.4%;
  - the worst 10% carry 34.9%.
- The ratio of a day's MSE to the airport's mean has a Spearman correlation of 0.53 with that day's median schedule delay.

A row bootstrap treats these flights as independent, so its intervals are too narrow for changes that touch many rows.

**What `prc/validation.py::block_bootstrap` does.**

- It resamples whole airport × local-day blocks with replacement: 3,653 blocks in total, 2,446 in the held-out months, 628 in Jan+Jul 2025.
- It uses the same draws for both predictions and 1,000 draws in all.
- It reports the percentile interval of the RMSE difference, plus the plain point difference.
- The first sequence-model evaluation used the UTC day instead of the local day.

**Measured width against the row bootstrap.** Both were computed on 10-06 from the cached OOF arrays with `prc/validation.py`: 1,000 block draws and 500 row draws on the same pairs.

| Comparison | Rows changed | Window | Block interval | Row interval | Width ratio |
|---|---:|---|---|---|---:|
| v27 structure vs v24 (U+T, a blend change) | most A/B/C rows | held-out | [−0.765, −0.433] | [−0.716, −0.480] | 1.40 |
| | | Jan+Jul 2025 | [−0.969, −0.347] | [−0.854, −0.402] | 1.38 |
| v28 vs v27 structure (LOBT window) | 251 | held-out | [−0.853, −0.206] | [−0.847, −0.216] | 1.03 |
| | | Jan+Jul 2025 | [−2.069, −0.252] | [−1.906, −0.219] | 1.08 |

For bulk changes the honest interval is about 40% wider. For a change confined to a few hundred scattered rows, the two methods agree.

**What the block bootstrap still does not capture.** It sees neither refit noise (handled by R4, §2.3) nor year-to-year change (§3).

---

## 2. Noise levels

### 2.1 Absolute RMSE cannot compare two models

**The early models.** On the first day, the bootstrap sd of the uncapped Jan+Jul 2025 RMSE was **46.9 s**, and its 95% interval was 163 s wide.

- The cause was 149 of 344,419 rows with taxi-out above 2 h.
- Over the whole 2025 file, the 100 largest of 2,085,047 labels (0.005%) carry 35.8% of the total squared deviation from the label mean, i.e. of the squared error of a constant mean prediction (re-checked on 2026-10-06).
- At the time, the eight best public scores spanned 5.9 s.
- The capped RMSE had an sd of 1.62 s, but it measures a different quantity (§1.2).

**The tail is still there in late models.**

- For v21, the worst 0.1% of rows carry 50.4% of the 12-month squared error.
- For v17, fifteen rows whose recorded off-block is about a day early carry 26.7%. These are day-scale recording artefacts in the labels.
- For the v24 structure, Jan+Jul 2025 OOF is 292.21, and 224.25 without those 15 rows. Its *live* score is 244.22.

So an absolute offline RMSE says little about the live level. It depends mostly on how many day-scale artefacts a period happens to contain.

### 2.2 Paired comparisons

Every decision compares two predictions on the same rows. Rows that are hard for both models then cancel out.

- **Pairing.** On Jan+Jul 2025, resampling the same rows for both models cut the sd of the difference from 56.2 s to 20.9 s (2.7×) (`experiments/exp06_gate_and_arrivals.py`, `exp07_paired_gate.py`).
- **All 12 months.** Using the 12-month OOF instead of Jan+Jul alone cut the sd of paired differences by another 2.2–2.5×; for example, 28.33 → 11.40 s for the mixture against the untrimmed model (`experiments/exp13_month_cv.py`).
- **Late-stage changes (v21–v27).**
  - Held-out block-bootstrap half-widths are 0.06–0.40 s.
  - Jan+Jul 2025 half-widths are **0.14–0.31 s**, for changes of 0.25–0.65 s.

### 2.3 Seed noise

Seed noise is invisible to any bootstrap over rows. We measured it directly:

| Measurement | Size | Consequence |
|---|---|---|
| Absolute single-seed OOF, same configuration, different seed | 1.85 s apart (287.48 vs 289.33) | Absolute single-seed numbers are good to about 2 s. |
| Single-seed sd of the mixture, lean vs feature-heavy tail classifier | 0.91–0.97 s vs 2.81–2.88 s | More classifier features tripled the seed variance; we keep the lean classifier. |
| Embedding network, 3 seeds, screening months | spread 0.08 s | That network is limited by information, not by optimisation. |
| Sequence model: adding a third seed to a 2-seed average | −0.20 [−0.30, −0.11] on held-out | Seeds matter for this member. |
| Sequence-model settings, 3-seed sets | spread 0.3–0.8 s at blend level | Single 3-seed comparisons of settings are noisy. |
| Residual screen, control against itself at another seed | +0.043 [+0.007, +0.082] | A significant interval from the seed alone. This motivated R4. |

We also ran several **null controls**:

- **Shuffled columns.** In the residual harness, shuffling the candidate columns gave −0.04 s (12 months) and −0.11 s (Jan+Jul 2025). So |Δ| below 0.1–0.2 s is noise in that harness.
- **Permuted probabilities.** Permuting the zone-P classifier's probabilities across rows within each airport gave +38.7 s on average, against −2.52 s for the real probabilities; none of the 20 draws matched the real ones. The classifier genuinely ranks flights (`experiments/discord_eval/discipline.py`).
- **Placebo indicators** for de-icing features.

**One protocol error, recorded so it is not repeated.** A 09-26 settings screen compared a 3-seed reference ensemble with 2-seed candidates. Its per-seed condition also compared single seeds with that 3-seed ensemble, a condition the reference's own single seeds would have failed. The pre-registered result stood. Since then we compare equal seed counts (§5.3).

### 2.4 What a single live delta can tell

- The live board scores 344,841 rows, a sample of the same size as Jan+Jul 2025.
- Our Jan+Jul 2025 intervals for late changes have half-widths of 0.14–0.31 s.
- So we treat **live deltas under about 0.3 s as noise**: neither confirmation nor refutation. They are not used as validation.

---

## 3. Offline → live transfer

### 3.1 Every upload from v6 on

The offline Δ is the Jan+Jul 2025 estimate made before the upload, against the previous shipped version, unless marked. Live Δ is the change in our own public score. Ratio = live / offline.

| Ver. | Change | Offline Δ (Jan+Jul 2025) | Live Δ | Ratio |
|---|---|---:|---:|---:|
| v6 | Zone-P tail specialist | −13.06 (12 months) | −13.09 | 1.00 |
| v7 | Extra zone-P classifier features (leaderboard test, §6.2) | −6.29 (12 months −1.50, interval crosses 0) | −3.84 | 0.61 |
| v8 | Network as a second bulk learner | −4.78 | −6.52 | 1.36 |
| v9 | Per-route network weight | −0.48 | −0.49 | 1.02 |
| v10 | Route-D disruption corrector (leaderboard test, §6.2) | −0.63 | −5.00 | 7.9 |
| v11 | Residual feature stack | −5.31 | −2.64 | 0.50 |
| v12 | Multi-family blend | −1.75 | −2.80 | 1.60 |
| v13 | TabICL for Rome zone P | −4.73 | −2.49 | 0.53 |
| v14 | Stack signals inside all members | −2.30 (plus about −0.15 from the non-Rome zone P) | −7.58 | 3.1–3.3 |
| v15 | The same members on route D and in zone P's real-taxi branch | −1.02 | −1.52 | 1.49 |
| v16 | Second network architecture | −1.21 | −0.66 | 0.55 |
| v17 | adsb.lol ground-movement features inside all members | −16.8 | −23.53 | 1.40 |
| v18 | Route-D corrector halved outside zone P (§6.3) | −0.40 (12 months) | −0.78 | about 2 |
| v19 | Sequence (neighbour-context) model as the 8th member | −1.51 | −1.60 | 1.06 |
| v20 | Sequence model in zone P's real-taxi branch | −0.73 | −3.02 | 4.1 |
| v21 | Sequence model, 15 epochs | −0.62 | −0.26 | 0.42 |
| v22 | 20 epochs | −0.31 | −0.32 | 1.03 |
| v23 | 30 epochs | −0.40 | −0.49 | 1.22 |
| v24 | Pool of 27 sequence fits (arm M) | −0.65 | −0.44 | 0.68 |
| v25 | Rome zone P plus carrier stamping rate (rule failed, §6.4) | −2.25 (interval crosses 0) | **+1.07** | sign reversed |
| v26 | Arm U, disagreement-tercile weights (vs v24) | −0.255 | −0.80 | 3.1 |
| v27 | Arm U+T (vs v26) | −0.34 | −0.27 | 0.8 |
| v28 | LOBT window (§5.4) | bound ≤ −5.99 if the join rule holds (*label-free* moves applied to v27's *live* score); 2025 OOF −1.07 | −7.75 | not comparable |
| v29 | Join complement and LEBL/LEMD cap (§5.5) | bound ≤ −0.377 if the rules hold (*label-free* moves applied to v28's *live* score); 2025 OOF −0.43 | −0.50 | not comparable |

### 3.2 Reading the table

- **Direction.** From v6 to v27, 21 of 22 uploads moved the live score in the direction the 2025 OOF predicted. The exception is v25, the only upload made against its own rule.
- **Size.** Ratios range from 0.42 to 7.9, with a median of 1.06. The offline protocol orders changes correctly; it does not size them.
  - We learned this early: before v4, the OOF predicted −8.41 s and the live change was −30.02 s.
  - The offset between local holdout and live score was also unstable over the first three uploads: −20, +42 and −45 s.
- **By kind of change** (a pattern, not a law):

  | Kind of change | Versions | Ratio |
  |---|---|---|
  | New information inside the members | v14, v15, v17, v20 | 1.4–4.1 |
  | Correction layers and model diversity | v11, v16 | about 0.5 |
  | Zone-P classifier changes | v7, v13, v25 | 0.53, 0.61, and one sign reversal |
  | Events that month-wise CV cannot represent | v10 | 7.9 |

### 3.3 Year-ahead check on 2026 arrivals

The ranking file provides arrival taxi-in times for January and July 2026. These are not the scored target. We trained a taxi-in model on 2025 and tested it on those 2026 arrivals (`experiments/major/m05_arr_yearahead.py`, 09-24). This is the only year-ahead test with real labels available to us, and it concerns arrivals:

- **July: no degradation** (R² 0.648 → 0.662).
- **January: degradation only at Amsterdam** (RMSE 145 → 291 s). This pointed to the 2–7 January 2026 winter disruption, which motivated v10.
- **Identity features** (stand, operator, callsign prefix) did not hurt year-ahead.

---

## 4. Leakage guards

### 4.1 Departure labels never enter a feature

The scored target is `TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` of a departure. Neither field is available for 2026 departures, and neither may reach a feature in training.

- **Sequence model (`prc/models/seq_net.py`).**
  - `movement_table` blanks `BLOCK_TIME` on every departure row before anything else reads the table.
  - `assert_label_free` then fails loudly if any departure token carries an in-block time, a taxi-in, or in-block minus schedule.
  - `experiments/seq/s03_context_checks.py` re-runs the assertion.
- **Feature-stack and turnover features (`experiments/v11/v11_features.py`).** The departure frame passed to the feature builders has no `BLOCK_TIME` or `TAXITIME` column, and this is asserted. Labels are written separately and feed only the target and the nested target encodings.
- **Zone-P candidate features (`prc/p_features.py::load_set`, used by `experiments/exp62_screen.py`).** Departure `BLOCK_TIME` is set to missing right after loading, and this is asserted.
- **adsb.lol features (`experiments/adsb/`).**
  - The stand table is the median ADS-B pushback position for each provided `STAND_mvt`.
  - `BLOCK_TIME` and `TAXITIME` appear only in measurement scripts and as the target of the gain estimate.
  - The features observe each flight's own ground track, including in 2026. They are an independent observation of the off-block event, not a label. The information-use statement is in [MODEL.md](MODEL.md).
- **No model is fitted on any 2026 target.** 2026 rows enter only as label-free inputs, for example as context tokens. We rejected building pseudo-labels for 2026 rows from their ADS-B tracks: that would amount to training on the scored months.

**Arrival in-block times are allowed.** The ranking file provides them for 2026, so they may be used: as taxi-in, as stamping evidence, and as context.

### 4.2 Month isolation

The ranking file contains January and July 2026 back to back. Any feature that looks across rows could therefore jump five months.

- **Event features.** Stamping and stand turnover are built month by month from that month's events only. Turnover gaps are capped at 48 h, and this is asserted in training and in ranking (`v11_features.py`, CHECK 3).
- **Sequence-model contexts.** Contexts never leave an airport × calendar-month group, and 2026 contexts see only 2026 movements.
- **A bug we found and fixed.** The first July departure on each runway saw a gap of about five months to its "previous takeoff" (`prc/congestion.py::gap_prev_to`). `cluster/w2_extra.py` caps the gap and the "previous departure" link at 6 h; all members use the fix from v17 on.
- **Target encodings are nested.** The validation month is never in its own encoding, and training rows are encoded by an inner leave-one-month-out. The prior is Beta(8) toward the airport rate in the zone-P features, and a weight of 20 in the stack.
- **Foundation-model context.** In the Rome zone-P classifier evaluation, the TabICL context for a month comes from the other months' rows only.

### 4.3 Look-ahead: information from after the target's take-off

Features may use other movements that happen *after* the target's own take-off, for example:

- the next aircraft to occupy our stand;
- departures and arrivals in the following minutes.

**Where the information comes from.**

- **Which movements exist, and their times, stands and runways:** only the provided files.
- **adsb.lol observations of the neighbours:** the sequence model's tokens also carry each neighbouring departure's own adsb.lol observation (open data from the same period), including for neighbours that move after the target.
- **The target's own adsb.lol track:** its use is described in [MODEL.md](MODEL.md).

**Why we allowed it.**

- The ranking file provides these fields.
- The organisers stated on the challenge's Discord channel (recorded by us on 09-25; posted earlier in September) that features from other rows, and every field of the target row except `BLOCK_TIME` and `TAXITIME`, may be used.
- They also stated that the model's purpose is post-operations, not tactical use.

**What it is worth.**

- **Feature stack.** Look-ahead stand-turnover features were worth −2.13 [−2.37, −1.90] on top of backward-only turnover (12 months, outside zone P).
- **Sequence model.** A past-only sequence model keeps 56% of the context effect.

**One channel comes close to the label.** If the next arrival is in-block at our stand before our take-off, our off-block must precede that in-block time. These features are in the members (`sl_*`, `sn_*`, `sa_after_to`); removing same-stand neighbours from the sequence model's context changes its gain by less than 1%.

**The limit.** A real-time system could not use any of this.

**What we never did.** No feature uses any departure's `BLOCK_TIME`, and none uses an external source that summarises the scored period's labels (§6.9).

### 4.4 Leaks we could not rule out by construction, and checked

- **Row order and identifiers** (`experiments/major/m03_order_leak_check.py`, `m03b`, `m03c`). No leak:
  - `FLIGHT_ID` is NM's identifier in LOBT order and is absent on route D;
  - `MVT_ID` follows the schedule at EDDM, LEMD and LIRF and is random elsewhere;
  - neither carries off-block information.
- **Train/serve parity.**
  - Every feature-stack column has 2026 coverage within 1 percentage point of Jan/Jul 2025 (*label-free*).
  - The sequence model's token fields are filled equally in both years.
- **Drift of adsb.lol columns** (`experiments/drift/`).
  - The receiver-coverage columns drift more than the movement columns (mean Kolmogorov–Smirnov statistic 0.153 vs 0.130, *label-free*).
  - Dropping them would cost 3.78 s (January) and 2.77 s (July) of bulk RMSE on 2025 (one member, 35% subsample).
  - They do not shift 2026 predictions systematically at the airports where coverage changed most (*label-free*).

---

## 5. Post-hoc caveats

### 5.1 The same 2025 OOF judged many candidates

From 09-24 onwards, every blend evaluation fitted its weights on the same four screening months and was judged on the same eight held-out months of the same 2025 data. Dozens of candidates went through it, including:

- the pre-registered sequence arms (epochs, width, depth, pools of seeds);
- member-seed and retrain campaigns;
- zone-P and Rome variants;
- the blend restructurings U and U+T.

When one of many candidates passes a gate at 95%, the evidence is weaker than its interval suggests. The independent check is the live record in §3.

### 5.2 Arms U and U+T (v26, v27) were found after the fact

**How they were found.** On 10-01 we screened about 18 variants of the blend on the v24 OOF (scripts in `experiments/plan_1001/`).

- **U** is one weight set per tercile of the disagreement between the 27 sequence-model fits.
- **U+T** adds λ × (sequence model − its no-context twin).

On 10-02 the designs and gates were frozen in `arm_u_eval.py` (commit 9d0dd0b) and the evaluation was run; v26 was built in the next commit (19c9849). The commit order is consistent with gates before results. Three qualifications remain:

1. **Same data.** The frozen evaluation reuses the OOF and the months on which U was found. Its held-out numbers are the best of about 18 and are optimistic.
2. **One gate was narrowed with the outcome known.** In the frozen rule, R5 applies to held-out rows only; on Jan+Jul rows it is reported, not gated. The screens had already shown U at +0.008 s on that check.
3. **U+T was reshaped.**
   - The screened form also changed route D. It failed the Jan+Jul R5 check (+0.243) and was worse at Rome in the cross-fits.
   - The frozen form touches routes A/B/C only.
   - The T term itself first appeared as an "information only" arm on 09-26.

**Results.**

| Arm | Held-out vs v24 | Jan+Jul 2025 vs v24 | Live |
|---|---|---|---|
| U | −0.199 [−0.265, −0.147] | −0.255 | −0.80 |
| U+T | −0.587 [−0.765, −0.433] | −0.595 | −0.27 against v26 |

### 5.3 Other choices made after the fact

- **v24 (arm M, the pool of all 27 sequence fits).**
  - Arm M was defined after all eight pre-registered settings arms had failed, and recorded before the pool was evaluated.
  - The two commits (7a8f5d5, 58f80bc) are one minute apart, so the history cannot separate them; only the internal log records the order.
- **v21 (15 epochs).**
  - Its 2-seed screen missed the pre-registered 1.0 s bar (−0.948 s).
  - The like-for-like comparison found later was −1.80 s (§2.3). The rule was not changed afterwards.
  - The team chose to run the full campaign, which passed its own pre-registered rule. Live ratio ×0.42.
- **v20 (sequence model in zone P).** The rule header and the result landed in the same commit (04ff4df). Only the internal log records that the rule came first.
- **A gated arm that was not shipped.** The arm excluding Rome from the sequence member (09-26) was motivated by 2025 subset folds that overlap the judging months. It passed, but lost the pre-registered tie-break.

### 5.4 The LOBT window (v28)

**What it does.** v28 is v27 with one rule added (`assemble.py --lobt-window`). Every prediction for a departure that has an `LOBT_flt` is projected into the interval

`[max(0, MVT − LOBT − 3606 s), MVT − LOBT + 3606 s]` (after rounding; the lower end rounded up, the upper end down)

**Where it came from.** We adopted the idea on 10-06 from three public competitor repositories:

- EnioAguiar/prc-taxiout-2026
- javidmardanov/PRC-Data-Challenge-2026
- joris-decombe/taxiout

We took the idea only; the code is our own. The value 3606 does not appear in our records before 10-06.

**Evidence from 2025.** This is in-sample for the rule, because the window is read off these same rows.

- 2,062,577 departures have an LOBT. They are the same rows that carry `AOBT_3`.
- BLOCK − LOBT lies in [−3606, +3606] s with **no exception**. The edge is sharp:
  - 215 rows in (3600, 3606];
  - 45 rows in [−3606, −3600);
  - 0 rows in (3606, 3700].
- The median is +251 s.
- No other clock gives an exact bound. The share of rows within ±3606 s is 99.914% for AOBT_3, 99.997% for IOBT and 99.970% for EOBT_1.

Our reading: the extract's join between movements and flight plans required |BLOCK − LOBT| ≤ 3606 s. That is a property of how the dataset was built, not operational knowledge.

**Effect on the 2025 OOF.**

| Base | Rows changed | Rows worse | Held-out | Jan+Jul 2025 | Months better |
|---|---:|---:|---|---|---|
| v24 structure (the `assemble.py` header) | 236 | 0 | −0.550 [−0.946, −0.239] | −1.070 [−2.324, −0.308] | 12 of 12 |
| v27 structure (`experiments/tail/v28_oof.py`) | 251 | 0 | −0.486 [−0.853, −0.206] | −0.941 [−2.069, −0.252] | 12 of 12 |

**What the gates mean here.**

- **R5 passes vacuously** (−0.0004 s). A projection that cannot worsen any row survives "remove the 100 most-improved rows" however concentrated its gain is.
- **R4, X1 and X2 do not apply.** They test fitted quantities, and the window has none.
- **The gain is concentrated.** About 79% of the 2025 12-month gain comes from 10 rows. We do not claim the window passes the rule in the sense of §1.3.

**Consistency gate between 2025 and 2026** (*label-free*):

| Statistic | 2025 | 2026 |
|---|---:|---:|
| LOBT coverage of departures | 98.92% | 98.47% |
| Minimum MVT − LOBT | −3,382 s | −3,173 s |
| Share with MVT − LOBT > 3606 s | 2.22% | 2.11% |
| Share with \|AOBT_3 − LOBT\| ≤ 3606 s | 99.81% | 99.72% |
| Rows that would force y < 0 | 0 | 0 |

- Arrivals show the same join in both years.
- Nothing in 2026 contradicts the join.

**Effect on 2026** (*label-free*).

- **63 of 339,551 rows move**: 60 lie below their window and 3 above.
- **By airport:** LFPG 24, EHAM 15, EGLL 5, LTFM 5, LEMD 4, EDDF 3, EDDM 2, LEBL 2, LIRF 2, LSZH 1.
- **Mean shifts** per airport × month lie in [−0.02, +1.95] s.
- **Bound.** If the join rule holds in 2026, the score can be at most 237.16, i.e. −5.99 s against v27. The bound applies the label-free moves to v27's *live* score.
- **One row dominates.** 96.1% of that bound comes from one Rome July departure. The mixture had hedged it between two hypotheses, and both lay outside its window. The other 62 rows alone give a bound of 242.92.

**Order of events.**

1. The header was written on 10-06, **after** the 2025 effect and the 2026 bound had been measured.
2. Header, code and a self-test were committed in 618ba86, before the file was built and uploaded. The commit order is consistent with this sequence, but the window was not pre-registered in the sense of §1.3.
3. The self-test refuses to build unless 2025 still shows a maximum |BLOCK − LOBT| of exactly 3606 s over at least 2,000,000 rows. This guards against unit and time-zone errors.

**How it fits our own earlier rules.**

- We had retired "physical bounds" for the tail, since clock errors cannot be bounded physically. The window is not physical: it bounds the recorded field through the join.
- We had ruled that uploads staking a few high-variance rows count as probing. The window is one deterministic constraint applied to all 339,551 rows that have an LOBT, and it cannot worsen any row while the join holds.

**Live result:** **235.3996**, −7.75 s against v27. The header committed this pre-commitment (on the initial ranking file) before the upload:

- upload once;
- no later upload changes any projected row;
- nobody derives an implied label from the score;
- the score feeds no later decision.

The score is recorded, and nothing beyond it.

### 5.5 The join complement and a source cap (v29)

**What it does.** v29 is v28 with three rules for departures *without* an LOBT (`assemble.py --join-complement`, applied after the window; definitions in [MODEL.md](MODEL.md) §10.1):

- **A, one-sided.** A route-D IFR departure whose own NM record appears on its destination's arrival row (a clean link) evidently failed the join, so |BLOCK − LOBT_arr| > 3606 s. Where the lower side is impossible, the prediction is raised to the upper edge.
- **B′, two-sided, below side.** Where both sides are possible, the prediction is lowered to the lower edge. Rows stamped at the schedule and take-offs crossing local midnight relative to the schedule are excluded.
- **C, source cap.** LEBL and LEMD predictions are capped at 3606 s.

**How it was found.** A structure search on 2025 and on label-free 2026 rows for further structure of the window's kind (10-06); candidates were priced against v28's live score; 3 shipped. The rules were **not pre-registered**: they were discovered and measured on 2025, and screened on 2026 rows, before the header was written. B′'s local-midnight exclusion was added after the one 2026 row it removes had been inspected and priced under a "normal" and a "day-shift" hypothesis; in 2025 it removes 15 rows that B never moved. Header, self-tests and code were committed in df0a5ff before the build; the author decided to upload. Scripts and dropped candidates (E1, B2, DST window, twin rules T3/T4, Rome hedges): [`experiments/hunt_1006/`](../experiments/hunt_1006/README.md).

**Evidence from 2025** (in-sample for the rules; these are also the self-tests that `assemble.py` runs before building):

| Rule | 2025 check | Result |
|---|---|---|
| A | clean links with \|BLOCK − LOBT_arr\| ≤ 3606 s | 0 of 1,426 |
| A | one-sided rows below the upper edge | 0 of 70 |
| B′ | rows below the lower edge (stamping and local-midnight cases excluded) | 1,281 of 1,281 (empirical; 1,296 of 1,296 before the midnight exclusion) |
| C | LEBL / LEMD departures above 3606 s | 0 (maxima 3,604 / 3,598 s) |

On the 2025 OOF, 140 rows change and 0 get worse: held-out −0.62, Jan+Jul 2025 −0.43. As for the window, R5 says little about a change that cannot worsen a row while its rules hold, and R4, X1 and X2 do not apply.

**Effect on 2026** (*label-free*).

- **36 rows move:** 9 up by A, 21 down by B′, 6 down by C alone (C touches 10 rows, 4 of which A or B′ already moved). None of them is among the 63 window rows (asserted).
- **Consistency gates** (abort if violated): clean links are 6.8% of the 4,844 IFR departures without an LOBT (gate 5–10%); arrival NM coverage 99.40% (gate ≥ 99%); no moved row on an airport-day whose no-record share exceeds the 2025 maximum of 17.2% (worst 9.7%).
- **Bound.** If the rules hold, the score is at most 235.023, i.e. −0.377 s against v28's *live* score. About 60% of the squared moves sit on one Rome row.

**Live result:** **234.8988**, −0.50 s against v28. The window's pre-commitment was extended to these 36 rows: upload once, score logged only, no implied label, no later upload changes them on the initial ranking file.

### 5.6 Differences between what was validated and what ships

- **Tail classifier.** In cross-validation each member's mixture uses its own tail classifier. Production uses one shared tail classifier at 5 seeds.
- **Numeric path of v27's no-context sequence twin.** Its folds were trained on GPU in bf16. Its three final fits were run on a laptop CPU in fp32, because the cluster was unavailable at the time. Their mean prediction on the ranking set is 1030.2–1031.0 s, against 1030.8 s for the corresponding context model's finals (*label-free*). For the final file they were re-run by the documented GPU job (bf16), so the final file and v27-v29 differ in this numeric path (REPRODUCE §8).
- **Zone P.** In zone P, `assemble.py` reproduces earlier shipped files to within 2 s rather than exactly. The old build chain accumulated rounding.

---

## 6. Use of the public leaderboard

Our principle was that the 2025 OOF decides and the board measures. We kept to it, apart from the exceptions listed below, which we disclose in full. The scripts involved keep their original names; their headers were changed only in wording (2026-10-10 and 2026-10-13, visible in the git history). The 10-06 scripts that had stayed in a git-ignored scratch folder were added to the repository on 2026-10-13 (`experiments/hunt_1006/`, `experiments/plan_1006/`). The git history was not rewritten.

The master upload table, with every upload classified, is [RESULTS.md](RESULTS.md) §1.1.

**How live scores were used.** Our own live scores decided v1/v2 (target trimming), v7 and v10 (leaderboard tests),
the v18 tie-break and the dropping of v25; they set the v28 and v29 bounds, including the pricing of the 10-06
structure-search candidates against v28's score; and, through the offline-to-live record and early deltas (CV
protocol, `exp18`, `exp19`), they informed where we spent effort. No other modelling decision used them.

### 6.1 Facts

- **How the board works.**
  - Ranking uses each team's **best** score.
  - Every evaluation scores all 344,841 ranking departures; there is no public/private split.
  - Each team may make 5 uploads per UTC day.
- **Our uploads.** We made 29, v1–v29, between 09-21 and 10-06. The final-phase file (§8) is a separate submission
  on the final ranking file, uploaded twice (a safety net and its replacement; [RESULTS.md](RESULTS.md) §1.1).
- **What we read.** `tools/leaderboard.py` reads the public competition API and nothing else; that API returns every team's public submissions (§6.8).

### 6.2 Uploads whose keep-or-drop decision rested on the live score

These uploads were model-level leaderboard tests: whether to keep the change depended on its live score, not on our offline rule alone.

**v1 and v2 (09-21): target trimming.**

- The two files were identical except that v1 was trained on y ≤ 7,200 s and v2 on all rows. We uploaded them together as a deliberate test of whether the 2026 labels contain a heavy tail like 2025's.
- Our capped local protocol preferred v1 by 35.7 s. Live, v2 scored 507.55 and v1 543.67.
- **Consequences:**
  - the validation metric became uncapped RMSE;
  - the tail went into an explicit mixture (v3).
- Builder: `experiments/exp04_submission_v1v2.py`.

**v7 (09-23): 42 extra features for the zone-P tail classifier.**

- The 12-month rule failed: −1.50 [−3.02, +0.26]. Jan+Jul 2025 was clearly better: −6.29 [−10.56, −3.03].
- With the team's approval we uploaded it as a leaderboard test, and its builder says so in its header (`experiments/exp64_submit_v7.py`, "LEADERBOARD TEST").
- Live: −3.84. The change touched only zone-P rows, and the log at the time read its live Δ as a measurement of the change on those rows.
- **Later history:** kept in v8–v12. Replaced on all zone-P rows by TabICL in v13 (Rome) and v14 (the other airports). Each replacement passed its own offline rule, but v13 was compared against v7's classifier. **v7's classifier is not in v29;** TabICL's inputs are 38 numeric features from the same candidate set.

**v10 (09-24): the route-D disruption corrector.**

- The corrector adjusts flights without an NM record, using the airport's trailing state over the previous 90 minutes.
- It passed offline: 12 months −0.60 [−0.94, −0.25]. But its real target was the 2–7 January 2026 disruption at Amsterdam, found with the arrivals check of §3.3, which month-wise CV cannot represent. So we uploaded it explicitly as a leaderboard test (`experiments/exp69_submit_routeD.py`).
- Its strength was set before the upload from 2025 analogue days, not from the board.
- Live: −5.00.
- **The corrector remains in v29:** at full strength in zone P and at half strength on route D outside zone P (§6.3).

### 6.3 The one shipped parameter chosen with help from a live delta: the route-D half (v18)

On 09-26 a pre-registered evaluation of the corrector on the v17 base found that two options passed outside zone P:

- dropping it: −0.390 [−0.82, +0.00];
- halving it: −0.397 [−0.62, −0.19].

The pre-registered tie-break was the lower error against a label-free ADS-B estimate of the 2026 truth, called E1, and it picked *drop*. A second label-free estimate, E2, disagreed:

- **E1** = the flight's ADS-B taxi time plus the 2025 gap pattern;
- **E2** = the ADS-B taxi time plus the same day's gap on flights that have an NM record.

**How the live delta was used.** A post-hoc script tested both estimates against the measured live effect of v9 → v10, which had changed only corrector rows (`experiments/routed_0926/r03_calibrate.py`; its header says "LIVE CALIBRATION").

| | Predicted live Δ |
|---|---:|
| E2 (*label-free*) | −5.38 s |
| E1 (*label-free*) | +0.65 s |
| Measured | −5.00 s |

Under E2, halving beats dropping. The team chose to halve. v18 scored −0.78 live.

This is the only parameter in the shipped model selected with the help of a live delta, and it overrode a pre-registered tie-break. On 2025 the halved corrector is now neutral: route-D RMSE is 1191.8 with it and 1191.6 without it (measured 09-28).

### 6.4 A rule override that lost (v25)

- **The change.** v25 (09-28) changed only the 180 Rome zone-P rows: the TabICL classifier gained four carrier stamping-rate columns.
- **The rule failed.**
  - R3: January +0.063 [−0.288, +0.522].
  - R5: +1.55 s (12 months) and +2.13 s (Jan+Jul 2025) once the 100 most-improved rows are removed.
- **The upload.** After descriptive checks (`experiments/tail/t07_carrier_checks.py`), the team uploaded it as a judgement call (`experiments/exp82_submit_v25.py`).
- **The result.** Live **+1.07 s worse** than v24.
- **Afterwards.** Not carried forward: v26 was built on v24. The rule since then is that zone-P changes get no override.

### 6.5 Variants uploaded in sequence (v8/v9, v21-v24, v26/v27)

Under best-of scoring each of these series is also a selection on the scored set.
- v8 → v9 (09-23, same day): network weight 0.5, then 0.65 on routes A/B/C, fitted on the 2025 screening months.
  A day later `strategy/s18_nn_weight_live.py` fitted a live-optimal weight (0.68) through the v7-v9 scores; it was
  not acted on.
- v21, v22, v23 (09-27, same day): the seq member at 15, 20 and 30 epochs, each passing its own rule against the
  previous file; then the 27-fit pool v24 (09-28).
- v26, v27 (10-02, 10-06): two frozen arms of one evaluation, both uploaded by a decision taken before either was
  built.

For v26/v27: U and U+T were frozen together and both passed (§5.2); on 10-02 the team decided to upload both. The board keeps the better one, v27, by 0.27 s, below the noise level of §2.4. A choice rule was committed in advance for the case where only the last or a nominated file counts: U+T replaces U only if it is better by more than 0.3 s.

### 6.6 Uploads confined to a small population

An outside analyst who diffs consecutive uploads will find these nine (*label-free* counts from our own files; the same rows are marked in the master table, [RESULTS.md](RESULTS.md) §1.1):

| Version | Rows changed | Population |
|---|---:|---|
| v6 | 1,707 | Zone P |
| v7 | 1,507 (1,038 by more than 1 s) | Zone P |
| v10 | 5,029 | Route D |
| v13 | 180 | Rome zone P |
| v18 | 3,455 | Route D outside zone P |
| v20 | 1,704 | Zone P |
| v25 | 180 | Rome zone P |
| v28 | 63 | Rows with an LOBT |
| v29 | 36 | Route-D rows with a clean arrival link; LEMD rows above 3606 s |

**How they were built.** Each upload applies one model component or one rule to every row of a population defined from the provided columns. None was built from row identifiers, and none was designed to learn the label of a particular row or of a hand-picked set of rows.

**What they reveal.** The live Δ of an upload confined to a small population still carries information about that population's labels. We used it for v7, v10 and v18 as described above; v6's and v20's live Δ entered only the offline-to-live record of §3. We pre-committed not to use v28's or v29's.

### 6.7 Diagnostics on our own scores

These read-only scripts analyse the score changes between our own earlier uploads.

| Script | Date | What it does | Fed a decision? |
|---|---|---|---|
| `experiments/major/m04_live_decomp.py` | 09-24 | Decomposes v7 → v9 to estimate where our 2026 error sits | No |
| `experiments/major/m08_P_submission_deltas.py` | 09-24 | Movement of the zone-P uploads against their live changes | No |
| `experiments/strategy/s18_nn_weight_live.py` | 09-24 | Fits a parabola in the network weight through v7, v8 and v9 | No; the weight was not changed |
| `experiments/plan_1001/live_proj.py` | 10-01 | Projection of each pair of uploads; its header says "NOT used to tune anything" | No |
| `experiments/routed_0926/r03_calibrate.py` | 09-26 | The live calibration of §6.3 | Yes: the v18 half |

Apart from r03, none set a component, weight, threshold or row. They did inform our reading of transfer (§3) and our choice of where to spend effort, for example preferring new information inside the members over correction layers. Their outputs are derived from 2026 labels through our scores, so the documents in this repository do not reproduce them.

### 6.8 Other teams' scores

**Two early design arguments used competitors' scores:**

1. **09-21.** We argued from the then-leading score and our own 2025 tail that the 2026 labels could not contain a 2025-like tail, and validated on a capped target. Our own v1/v2 refuted this within hours (§6.2).
2. **09-23.** We used the then-leading score to bound how much of our error must sit in the bulk, and briefly shifted effort there. The bound was overstated, because "bulk" had been defined from the true labels, and we corrected it the same day.

Since then, other teams' scores have been used only for rank tracking and planning.

**Leaderboard tooling.** `tools/leaderboard.py` pulls every team's public submission history from the competition
API (`--wait` polls until our file is scored). `experiments/strategy/s08_threshold_drift.py` (cut-off drift),
`s15_probability.py` (our probability of reaching the top-25% cut), `s16_outside_view.py` (how teams at our level
improved), `research_0925/lb_analysis.py`, `drivers_0926/lb_snapshot.py`, `plan_1006/ff/ff_rankat.py` and
`plan_1006/pattern.py` (other teams' upload timing) analyse that public data. They fed no model decision.

### 6.9 Information about the scored months from outside the board

**EUROCONTROL's monthly taxi-out file: our exposure.** EUROCONTROL publishes each airport's monthly taxi-out figures
(`taxi_out_additional_time_<year>.csv`, also shown on the ansperformance.eu dashboard). The 2026 file already covered
January-July 2026, so it summarises the labels we are scored on. Dates are 2026, times UTC.
- **09-21, first working session.** The session downloaded the file, checked its 2025 rows against the training data
  (agreement within ±0.03 min) and noted each airport's January and July 2026 change in mean taxi-out as a possible
  bias correction. This was never implemented.
- **09-23, about 13:40.** An internal scouting report quoted Rome's January and July 2026 counts of taxi-outs over
  120 min next to the summed tail probabilities of our v8 and proposed four uses. The team rejected all four that day
  and set a rule: no summary of the scored months' labels as a feature, calibration target, count or upload gate.
- **v13 (09-24) and the v25 override (09-28)** rested on 2025 evaluations: v13 on the leave-one-month-out test of
  TabICL on Rome's 627 zone-P rows of 2025; v25 on `experiments/tail/t06_carrier_stamp.py` (2025; rule failed) and
  `t07_carrier_checks.py` (descriptive checks on 2025 and label-free 2026). Neither our dev logs nor these scripts
  refer to the published Rome counts. That does not rule out an indirect influence of the 09-23 report on where we
  spent effort. One coincidence a reader may notice: v13's expected numbers of Rome zone-P tails (the summed probabilities, about 42 for January and 72 for July 2026) are close to the published Rome counts of taxi-outs over 120 min quoted on 09-23 (44 and 75). Those counts were not used by v13 or by any later build; a calibrated classifier is expected to land near the true count.
- **09-26 to 09-28.** Research passes flagged the file again; it was not opened. We recorded an organiser answer on
  the challenge Discord (2026-09-28) allowing the published monthly averages; the team kept its rule (re-confirmed
  10-06).
- **10-09.** After a second organiser confirmation (Discord, 2026-10-09), the author decided the file could be used if
  measured and gated first. The 2025 and 2026 files were downloaded (09:40). `experiments/eurocontrol/ec01_value.py`
  measured the 2025 file on our 2025 out-of-fold predictions only, as an additive per-airport-month shift: every variant
  was worse on the held-out months (raw +4.17 s, 1 of 12 months better; half +1.18 s, 4/12; corrected for each
  airport's published-minus-recorded gap +2.94 s, 0/12; gap-corrected half +0.87 s, 2/12). A label-free comparison of
  v29's January/July airport-month means with the 2026 file's January/July figures was also looked at; it showed no
  evidence of value and fed nothing. The file was not used.
- **Never viewed:** the February and June 2026 figures (the downloaded 2026 files contain them). The final file and
  its check script `experiments/final/f02_check_final.py` use none of the file. No tracked code reads the 2026 file.

We cannot rule out that having seen the 2026 figures on 09-21 and 09-23 influenced our judgement indirectly, so we state the exposure rather than claim we never saw them. None of their 2026 figures appear in this repository.

**For scale (2025 only).** A perfect level shift per airport and month, applied to our 2025 OOF, improves Jan+Jul 2025 by 0.14 s (0.28 s over 12 months). The value of the tail counts was never measured.

**Competitor READMEs.** On 10-06, while evaluating the LOBT window, we read the READMEs of public competitor repositories. Two contain statements about the live effect of their own uploads on the scored set; one of them concerns a row that the window moves.

- We read them before deciding to upload v28.
- We did not use them as evidence. The v28 decision rests on the 2025 data and the label-free checks in §5.4.
- We do not repeat those statements.

Otherwise, participants' public statements about their own scores, on Discord and in READMEs, were read only as sources of ideas.

### 6.10 What we did not do

- **No upload was built to recover labels.** That rules out constant-value files, per-airport or per-row perturbations, binary searches, and selecting rows by identifier.
- **No implied label for any row was ever computed from a score**, including v28's and v29's.
- **No variant was uploaded without passing its own rule**, but variants of one change were uploaded in sequence and the board keeps the best (§6.5).
- **No strength or threshold was tuned against the board**, apart from the route-D half of §6.3.
- **No uploads of window variants.** We did not upload alternatives to the LOBT window (AOBT_3 point, median offset, truncated mean).
- **The pre-commitments of §5.4 and §5.5 stand:** no later upload changes a row moved by v28 or v29 on the initial ranking file. On the final ranking file (§8) these rows are rebuilt mechanically by the same rules.

---

## 7. How this was built

**How this was built.** Team understanding-accordion is one person, Alexander Beiser (TU Wien), working with a
multi-agent Claude Code workflow (Anthropic's Claude): a main session wrote code and ran analyses, and sub-agents ran
searches, reviews and documentation drafts. Code, analysis and documentation were produced this way; the author
directed the work and took the decisions: which data to use or decline, what to test, which rule a change must pass,
and what to upload. Every upload was approved by the author, normally per version; v16 and v19 were approved in advance
on condition that they passed their rule, and v9 went up under a general permission given on 09-23 to upload a model
that was certainly better offline. From 2026-10-06 a standing approval, stated in advance, allowed a
pre-registered experiment (called "bet" in the code) to be uploaded if it passed its pre-registered rule; none did.
Uploads were executed by the agent through `prc/s3.py`; the author ran every `git push` and every cluster job. Every
commit carries a `Co-Authored-By` trailer. Every result was recorded in an internal, date-stamped dev log; code
comments refer to it and to internal reports, which are not published and are available to the organisers on request.

The rules in §1.3 were written before the evaluations they judged, except where §5 and §6 say otherwise. This document and [RESULTS.md](RESULTS.md) report the numbers of the internal dev log.

---

## 8. Final phase

**Task.** On 10-09 the organisers announced one blind final submission on `final_ranking.parquet` (January, February, June and July 2026; 670,790 departures), with the January and July times re-dithered, deadline 2026-10-14 10:00 UTC. The design must not change.

**What we did.** The v29 design was re-run on the final file (`PRC_RANKING=final`): we re-ran the frozen fit-and-predict jobs on the unchanged 2025 training data with the same seeds, with no model, weight or rule change, and nothing selected or tuned on the final file. Commands and the table of every difference from the v29 build are in [REPRODUCE.md](REPRODUCE.md) §8.

**Checks** (*label-free*; `experiments/final/f02_check_final.py` and the regression checks of REPRODUCE §8):

- **Regression in initial mode.** Rerun on the initial ranking file, the steps we checked reproduce the shipped artefacts: training features byte-identical, the zone-P specialist identical (max |Δp| 0), Rome's TabICL probabilities identical (max 0.0), the base frames identical. The cluster fits were not re-run in initial mode.
- **Re-dither.** January/July movement times (and arrival in-block times) differ between the two files by a triangular ±12 s distribution, consistent with two independent ±6 s integer dithers. `SCHED`, `LOBT`, `IOBT`, `EOBT` and `AOBT` are unchanged, so the 3606 s window still applies.
- **Replay against v29** on the 344,841 shared January/July departures: RMS difference 16.11 s (routes A/B/C 10.24 s, route D outside zone P 42.2 s, zone P 166.4 s); 0.25% of rows differ by more than 60 s. 82% of the zone-P difference is at Rome, where the TabICL probabilities moved (median 0, 90th percentile 0.027). The difference is not fully decomposed; REPRODUCE §8 lists the known contributors.
- **Pre-committed rows.** The 63 window rows and 36 join-complement rows are rebuilt by the same rules on the re-dithered inputs. 57 of 63 and 26 of 36 differ from v29 by a median of 4–5 s (at most 23 s), because the window edge moves with the re-dithered take-off time. Nothing was edited by hand.
- **The file.** Valid, 670,790 rows; 174 predictions above 7,200 s. The join complement moved 52 rows in the final file.
- **New months.** Mean prediction February 988 s and June 989 s, against 2025 mean labels of 1,002 s and 973 s for the same months.
- **adsb.lol short reads, fixed before upload.** On 2026-10-10 we found that stage 1 had read only the first 2 GB part of five adsb.lol "staging" archives (2025-10-14, 2026-01-29, 06-07, 06-23, 06-25): Python's stream tar reader stops silently at a damaged header, and each of these archives has one (35-76 KB) just past the part boundary. Stage 1 now uses its own resynchronising reader (`iter_tar`), fails loudly on any unread remainder, and drops exact duplicate rows (these archives repeat their first part). The four final-file days were re-extracted and every ADS-B-dependent job re-run: 11,350 of 670,790 predictions changed, 11,328 of them on those days (6.6 s RMS over the file). The training day 2025-10-14 (0.3% of training departures) was left as the shipped model saw it. After the fix 06-25 is still thin in the source (31% of departures matched).

**Status.** Two uploads (master table, [RESULTS.md](RESULTS.md) §1.1): a safety net of the pre-fix build on 2026-10-10 08:35 UTC and its replacement on 2026-10-12 16:50 UTC, read back byte-identical. The scorer accepted all 670,790 departures and returns no score.
