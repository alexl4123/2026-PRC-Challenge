# PRC Data Challenge 2026: taxi-out time prediction

Team **understanding-accordion** (authors: see [Authors](#authors))

Repository: <https://github.com/alexl4123/2026-PRC-Challenge> · Challenge: <https://prc-data-challenge-2026.netlify.app/>

The task is to predict the taxi-out time in seconds of every departure in January and July 2026 at ten European
airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM. That is 344,841 departures, scored by RMSE.
The training data are the labelled 2025 movements, which include 2,085,047 departures. In the final phase, announced
on 2026-10-09, a second ranking file adds February and June 2026 (670,790 departures; see [Final phase](#final-phase)).

**In brief.** The model estimates the conditional mean explicitly: a tail probability for recording artefacts, per-route
blends of feature-based members and a neighbour-context set model, and a TabICL classifier for the rows most exposed to
those artefacts (zone P). As a taxi-out model (v27) it scores **243.15 s** on the public leaderboard. Three rules that
undo artefacts of how the dataset was joined (LOBT window, join complement, LEBL/LEMD cap; v28-v29) are post-processing
worth another 8.25 s, to **234.8988 s** for v29: rank 21 of 232 on 2026-10-06 (23 on 2026-10-09).
`python assemble.py --arm-u UT --lobt-window --join-complement --out v29` rebuilds v29 from the stored model outputs.
The final-phase file is the same design re-run on the final ranking file ([Final phase](#final-phase)).

| Document | Contents |
|---|---|
| [docs/MODEL.md](docs/MODEL.md) | model design and the domain reasoning behind each component |
| [docs/RESULTS.md](docs/RESULTS.md) | every upload v1-v29: change, builder, rows changed, offline expectation, rule outcome, live score |
| [docs/VALIDATION.md](docs/VALIDATION.md) | protocol, noise analysis, offline-to-live transfer, full account of leaderboard use |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | environment, data downloads, rebuild levels L0-L2 and their checks |
| [docs/PIPELINE.md](docs/PIPELINE.md) | runbook: which script or cluster job writes each component |
| [experiments/README.md](experiments/README.md) | index of the experiment folders, and the production path |

## The problem

- **The label is an identity.** On every training departure, `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt −
  BLOCK_TIME_UTC_mvt` exactly. The ranking file gives the take-off time and blanks the off-block, so the task is to
  estimate *when the recorded off-block happened*.
- **Network Manager sees the same event.** `AOBT_3_flt` is present for 98.9% of the 2025 departures and for 98.5% of
  the 2026 departures (label-free). It is the strongest single input.
- **A few rows dominate the metric.**
  - 584 training departures have a taxi-out above 2 h. 149 of them fall in January and July 2025, which together
    have 344,419 departures.
  - 481 of the 584 have no NM record. For those, the recorded off-block sits at the scheduled time (median
    |y − (MVT − SCHED)| = 4 s). This is a recording artefact, not a long taxi.
  - They concentrate in **zone P**: no NM record and take-off more than 2 h after schedule. In 2025, zone P has
    4,944 rows, and 450 of them are above 2 h.
  - In the 2025 out-of-fold predictions of the v17 model, 15 rows whose off-block was recorded about a day early carry
    26.7% of the squared error.
- **Ground surveillance is uneven.** The open adsb.lol archive yields an off-block estimate for 19.4% of the 2025
  departures and for 25.6% of the 2026 departures (label-free). At LFPG and LTFM it yields almost none.

## Approach

RMSE rewards the conditional mean, so we estimate the mean explicitly and serve each departure by its route. In the
ranking file (label-free counts):
- **routes A/B/C** have an NM off-block record (339,551 rows);
- **route D** has none (5,290 rows);
- **zone P** is the part of route D that takes off more than 2 h after schedule (1,720 rows).

Outside zone P, a mixture `p·e_T + (1 − p)·m` combines three parts:
- a LightGBM tail probability `p`;
- a tail anchor `e_T`;
- a bulk estimate `m`, blended from eight members, of which five carry non-zero weight in v29 (`mix`, `nn` and
  `lintree` have weight 0; [docs/MODEL.md](docs/MODEL.md) §5.2).

Seven of the members are feature-based models on 143 columns. The columns cover flight-plan clocks, queue and
stand-turnover state from the movement file, METAR weather, daily ATFM delay, and 35 adsb.lol ground-track columns.
The eighth member is a set model over the 64 neighbouring movements at the same airport.

Inside zone P, TabICL gives the probability that the recorded off-block is the scheduled time, and the prediction
mixes take-off minus schedule with a real-taxi estimate. Where the NM record is missing, a corrector adds the
airport's current state. Finally, every prediction for a departure with a flight-plan LOBT (routes A/B/C) is clipped
into the ±3606 s window around that LOBT; the window is a property of how the dataset was joined. Three further
label-free rules (v29) bound 36 route-D rows whose own flight-plan join evidently failed, and cap LEBL/LEMD at 3606 s.

### Components of the shipped model (v29)

1. **Routes.** `MVT − AOBT_3` is Network Manager's view of the quantity we predict. Rows with it and rows without it
   get separate blend weights (`assemble.py`).
2. **Tail classifier and anchor.** The classifier averages 5 LightGBM seeds. The anchor is `MVT − SCHED` when that
   exceeds 2 h, and the training-tail mean otherwise (`experiments/exp65_submit_v8.py --stage components`).
3. **Seven members, 3 seeds each:** `mix` (LightGBM), `perapt` (LightGBM per airport), `lintree` (LightGBM with
   linear leaves), `xgboost`, `routed` (models the residual on `MVT − AOBT_3`), `nn` (entity-embedding network) and
   `w3` (second network). Features come from `cluster/w2_features.py` and `cluster/w2_extra.py`.
   - Adding the adsb.lol columns inside the members improved the 2025 held-out RMSE by −17.74 s [−19.64, −15.59].
   - The live step v16 → v17, which also added wake-sequence and backlog columns, was −23.53 s.
   - 95.3% of the adsb.lol squared-error reduction sits on rows where the flight's own track was seen. That locates
     the gain; it is not an ablation.
4. **Neighbour-context set model, `seq`** (`prc/models/seq_net.py`).
   - **Input.** Up to 64 departures and arrivals within ±90 min, at the same airport and in the same month, as
     label-free tokens: clocks, runway, stand, wake category, arrival in-block, adsb.lol fields.
   - **Model.** A transformer reads the tokens, conditioned on the target's 143 columns. Departures' `BLOCK_TIME` is
     blanked, and this is asserted.
   - **Effect.** Against its no-context twin on 2025 out-of-fold bulk rows: −5.68 s [−6.53, −4.82] overall, −17.0 s
     at LTFM and −6.3 s at LFPG, the two airports without open ground data.
   - **Shipped form.** A pool of 27 fits (9 settings × 3 seeds).
5. **Blend** (`experiments/seq/seq_swap_eval.py`, `experiments/seq/arm_u_eval.py`).
   - Per-route NNLS weights are fitted on the 2025 months 3, 6, 9 and 12.
   - On A/B/C there is one weight set per tercile of the disagreement among the 27 seq fits. The cut points are fixed
     from 2025: 13.9 s and 25.4 s.
   - The term `λ·(1 − p)·(seq10 − seqnc10)` adds the context effect of a 10-epoch seq.
6. **Zone P:** `p·(MVT − SCHED) + (1 − p)·(0.5·LightGBM without adsb.lol + 0.5·seq) + corrector`.
   - `p` comes from TabICL.
   - At Rome the context is Rome's 627 labelled 2025 zone-P rows; at the other nine airports, all 4,944
     (`experiments/strategy/s21_tfm_P_final.py`, `experiments/tfm/w4b_final.py`).
7. **Route-D corrector** (`experiments/major/m11d_state_routeD.py`).
   - A LightGBM residual model on the airport's current state: the median NM taxi proxy of the last 90 min, arrival
     taxi-in over the last 60 min, and the route-D share over the last 3 h.
   - It was built for the early-January 2026 EHAM disruption, when 20-40% of departures lacked an NM record
     (label-free).
   - It runs at full strength in zone P and at half strength elsewhere on route D.
8. **LOBT window** and **join complement**, below.

### The LOBT window (v28)

- **The rule.** In the 2025 file, every departure with a flight-plan LOBT has `BLOCK − LOBT` within ±3606 s.
  - This covers 2,062,577 rows, the same rows that carry `AOBT_3`, with zero exceptions.
  - The edge is sharp: 215 rows lie in (3600, 3606] and 45 in [−3606, −3600).
  - It is an artefact of the dataset's movement-to-flight join, not operational knowledge.
- **What v28 does.** v28 is v27 with each prediction clipped into `[max(0, MVT − LOBT − 3606), MVT − LOBT + 3606]`.
  - In 2026, LOBT exists exactly on the 339,551 A/B/C rows.
  - 63 rows moved: 60 up and 3 down.
  - If 2026 follows the same join rule, no row can get worse. The pre-upload bound was ≤ 237.16 s, and about 96% of
    the bound came from one Rome row. The 2026 row counts in this list are label-free; the bound applies
    those label-free moves to v27's live score.
- **Live result.** 243.1478 → 235.3996, −7.75 s.
- **2025 evidence** (in-sample for the rule).
  - 236 out-of-fold rows change, none for the worse.
  - Held-out months: −0.550 s [−0.946, −0.239]. January + July: −1.070 s [−2.324, −0.308].
  - About 79% of the gain sits in 10 rows, so the "not carried by 100 rows" gate passes only vacuously.
- **Order of events.**
  - The `assemble.py` header was written after these numbers were measured, and committed before the upload
    (618ba86).
  - A self-test refuses to build unless 2025 shows the exact window.
  - The idea comes from public competitor repositories ([Credits](#credits)).

### The join complement (v29)

- **The rules** (`assemble.py --join-complement`, applied after the window). They extend the same join reading to
  departures *without* an LOBT.
  - **A, one-sided.** A route-D IFR departure whose own NM record appears on its destination's arrival row (a "clean
    link") evidently failed the join, so `|BLOCK − LOBT_arr| > 3606 s`. Where the lower side is impossible, the
    prediction is raised to the upper edge. 2025: 1,426 clean links, 0 inside ±3606 s; 70 one-sided rows, 0 below the
    upper edge.
  - **B′, two-sided, below side.** Where both sides are possible, the prediction is lowered to the lower edge;
    stamped-at-schedule rows and take-offs crossing local midnight are excluded. 2025: 1,281 of 1,281 such rows lie
    below the lower edge (1,296 of 1,296 before the midnight exclusion). This is empirical, not a forced identity.
  - **C, source cap.** LEBL and LEMD record no departure taxi above 3,606 s (2025 maximum 3,604 / 3,598 s).
- **What v29 does** *(label-free)*. 36 rows moved: 9 up (A), 21 down (B′), 6 down by C alone (C touches 10 rows, 4 of
  which A or B′ already moved). None of them is among the 63 window rows. If the rules hold, the bound was ≤ 235.023 s.
- **Live result.** 235.3996 → 234.8988, −0.50 s.
- **2025 out-of-fold** (in-sample for the rules): 140 rows change, none for the worse; held-out −0.62 s, January +
  July −0.43 s.
- **Order of events.** The rules came from a structure search on 2025 and on label-free 2026 rows; candidates were
  priced against v28's live score; 3 shipped. They were not pre-registered. The midnight exclusion of B′ was added
  after one 2026 row it removes had been inspected and priced under two hypotheses. Header, self-tests and code were
  committed before the build (df0a5ff); the author decided to upload. The window's pre-commitment extends to these 36
  rows. The search scripts, the dropped candidates and this history are in
  [experiments/hunt_1006/](experiments/hunt_1006/README.md).

### Information the model uses

- **Post-operational setting.** The organisers said "the model is for post-ops, not for tactical use" (challenge
  Discord, recorded by us on 2026-09-25).
- **Other movements.** Features may use any provided field of any movement in the ranking file, including movements
  after the flight's take-off (look-ahead within the file). The exceptions are a departure's `BLOCK_TIME` and
  `TAXITIME`, which are blank there. The sequence model's tokens also carry other departures' adsb.lol fields,
  including those of flights that move after the target.
- **The flight's own adsb.lol track after pushback.** The organisers, in the same exchange: "if you can find open
  trajectory data with good ground coverage and extract off-block times, we are ok".
- **Daily ATFM delay** is an ex-post operational input like METAR: the file contains the actual delays of the scored
  months, as the METAR archive contains their actual weather. It is arrival ATFM delay, not a taxi-time statistic.
- **No 2026 departure labels.** Neither the labels nor any published summary of them is a model input. No airport
  or developer API with per-flight off-block times (e.g. Schiphol, Heathrow, Fraport, Munich) and no commercial
  flight-tracking service was queried; such sources were flagged in research notes and not downloaded. OPDI flight
  events, including 10 days inside January 2026, were read only to count surface events near the airports (2026-09-22
  and 09-27). The only label-dependent 2026
  information we used is our own leaderboard scores; see [Use of the public leaderboard](#use-of-the-public-leaderboard).

## Validation in brief

Details are in [docs/VALIDATION.md](docs/VALIDATION.md).

- **Folds.** 12-fold leave-one-month-out CV on 2025. Blend weights are fitted on months 3, 6, 9 and 12. Changes are
  judged on the other eight ("held-out") months and on January + July 2025.
- **Gates.** The decision rule grew from a paired-bootstrap test (09-21) to the following form, used from 2026-09-26.
  Each evaluating script carries its rule in its header:
  - R1: held-out and January + July both better;
  - R2: the airport × local-day block-bootstrap 95% CI below 0;
  - R3: January and July each better;
  - R4: every seed better;
  - R5: the gain survives dropping the 100 most-improved held-out rows.

  At least 17 commit subjects record a rule written before its run. Commit order is consistent with that; it is not
  proof. The exceptions are under [Use of the public leaderboard](#use-of-the-public-leaderboard).
- **Noise.** The absolute local RMSE has a bootstrap sd of about 47 s, because a few hundred long rows dominate it.
  Pairing the two models on the same resampled rows cuts the sd of a difference by 2.7×, and pooling all 12 months
  cuts it by another 2.2-2.5×. Late changes (v21-v27) had January + July CI half-widths of 0.14-0.31 s, so a single
  live delta under about 0.3 s is read as noise.
- **Transfer.** v6 was predicted at −13.06 s and scored −13.09 s live. All 19 uploads that passed a rule written
  before their run improved the live score; their live/offline ratios range from 0.42 to 4.14 (median 1.06).
  Information placed inside the members over-delivered (v14 ×3, v17 ×1.4, v20 ×4); correction layers and model
  diversity delivered about half (v11, v16).
- **2025 out-of-fold RMSE.** The v28 structure scores 234.37 s on the held-out months and 290.68 s on January + July.
  The v24 structure scored 235.44 s and 292.21 s. The v29 rules take another 0.62 s and 0.43 s off.
  - The window and join-complement parts of these figures are in-sample.
  - These levels are not comparable with the live score, so decisions use paired differences.

## Results

The table shows live public-leaderboard RMSE in seconds, on all 344,841 rows. The board ranks each team by its best
submission. Δ is measured from the previous row, except for v25 and v26-v27, which are measured from v24. All
uploads are listed in [docs/RESULTS.md](docs/RESULTS.md).

| Versions | Date | Change | Live | Δ |
|---|---|---|---:|---:|
| v1-v2 | 09-21 | one LightGBM regressor; v1 trimmed the >2 h tail from training (543.67), v2 kept it | 507.55 | |
| v3 | 09-21 | two-stage mixture with an explicit tail probability | 352.27 | −155.28 |
| v4 | 09-22 | queue, METAR and ATFM-delay features | 322.25 | −30.02 |
| v5-v6 | 09-22/23 | LightGBM capacity; a zone-P tail specialist | 308.19 | −14.06 |
| v7-v9 | 09-23 | zone-P specialist features; embedding network in the bulk | 297.34 | −10.85 |
| v10 | 09-24 | route-D state corrector (a leaderboard test) | 292.34 | −5.00 |
| v11-v14 | 09-24 | flight-plan clock and stand-turnover signals; six model families; TabICL in zone P | 276.83 | −15.51 |
| v15-v16 | 09-25 | those signals on route D and in zone P; a second network | 274.65 | −2.18 |
| v17 | 09-26 | adsb.lol columns inside all seven members | 251.12 | −23.53 |
| v18-v20 | 09-26 | corrector halved; seq as the eighth member and in zone P | 245.73 | −5.39 |
| v21-v24 | 09-27/28 | longer seq training; a pool of 27 seq fits | 244.22 | −1.51 |
| v25 | 09-28 | Rome zone-P rule override (lost; dropped) | 245.29 | +1.07 |
| v26-v27 | 10-02/06 | disagreement-tercile weights (U), then the context term (U+T) | 243.15 | −1.07 |
| v28 | 10-06 | LOBT window | 235.40 | −7.75 |
| **v29** | 10-06 | **join complement and LEBL/LEMD cap (36 rows)** | **234.90** | **−0.50** |

## Building the submission

The challenge files (`training_2025-*.parquet`, `ranking.parquet`, `submitting.parquet`) are not in this
repository. By default, `prc/config.py` looks for them in the repository's parent directory. Derived data go to
`cache/` and submissions to `outputs/`. Both directories are git-ignored, because they hold data derived from the
challenge files and from adsb.lol.

The environment is Python 3.12 with `pip install -r requirements.txt`. These are the pinned versions of the laptop
environment that built v24-v29 and the final-phase file; the cluster environments were not pinned
([docs/REPRODUCE.md](docs/REPRODUCE.md) §5). TabPFN is optional (a diagnostic only). `pyarrow >= 20` is required.

Level L0 assembles the submission from the stored model outputs in a few seconds, without training:

```
python assemble.py --arm-u UT --lobt-window --join-complement --out v29    # writes outputs/understanding-accordion_v29.parquet
python assemble.py --arm-u UT --lobt-window --join-complement --check 29   # compares with a shipped file in outputs/
python assemble.py --arm-u UT --lobt-window --check 28
python assemble.py --arm-u UT --check 27
python assemble.py --arm-u U --check 26
python assemble.py --check 24
```

- **Checks.**
  - On 2026-10-06, `--check 27` and `--check 28` found 0 differing rows; on 2026-10-09, `--check 29` likewise.
  - `--check 24` and `--check 26` match exactly on routes A/B/C and route D. In zone P they differ by at most 2 s,
    the rounding accumulated by the old build chain, in which each file was an edit of the previous one.
- **Inputs.** `assemble.py` never reads an earlier submission. Its header gives the formula for each row group, the
  LOBT rule and the join-complement rules.
- **Output.** `--out` refuses to overwrite an existing file; if `outputs/` already holds the shipped v29, write the
  rebuild under another name (e.g. `--out v129`) or use `--check 29`.
- **L1 and L2.** [docs/REPRODUCE.md](docs/REPRODUCE.md) describes the deeper rebuilds: L1 from the features and L2
  from the raw data.
  - L2 streams 1.331 TB of adsb.lol archives (429 days).
  - With their fold campaigns, the model fits take an estimated 150 GPU-hours on a Slurm node with 8 NVIDIA L40S
    GPUs, plus about 40 CPU-hours.

## Final phase

On 2026-10-09 the organisers announced one blind final submission, `understanding-accordion_final.parquet`, on
`final_ranking.parquet`: January, February, June and July 2026, 670,790 departures, with the January and July times
re-dithered. The design must not change. The deadline for the file and for this repository is 2026-10-14 10:00 UTC.

- **What we submit.** The v29 design, re-run on the final file. No model, weight, cut point, λ or rule was changed.
  `PRC_RANKING=final` switches the pipeline to the final file, with its own `cache_final/` and `outputs_final/`;
  [docs/REPRODUCE.md](docs/REPRODUCE.md) §8 has the commands, the checks and a table of every difference from the
  v29 build (data fix, plumbing, numeric path of one member, transductive medians, two uploads).
- **New inputs.** adsb.lol days for February and June 2026 (58 days). We re-ran the frozen fit-and-predict jobs on
  the unchanged 2025 training data with the same seeds and predicted the final file.
- **Checks** *(label-free)*. In initial mode the rebuilt steps we checked reproduce the shipped
  artefacts: training features, the zone-P specialist, Rome's TabICL probabilities and the base frames. On the
  344,841 January/July departures the final file shares with v29, the predictions differ by 16.11 s RMS (against
  v29's live RMSE of 234.90 s). This difference is not fully decomposed; REPRODUCE §8 lists the known contributors.
  The re-dither moves movement times by up to ±12 s and leaves the flight-plan clocks unchanged, so the 3606 s window
  still holds; the 99 rows moved by v28/v29 are rebuilt by the same rules, not copied.
- **ADS-B fix.** Four adsb.lol days of the final file had been read only in part (a silent stop of Python's tar reader
  at a damaged archive header); they were re-extracted with a fixed reader and the ADS-B-dependent jobs re-run before
  upload ([docs/VALIDATION.md](docs/VALIDATION.md) §8).
- **Status.** Uploaded 2026-10-12 16:50 UTC, replacing a safety-net upload of the pre-fix build made on 2026-10-10;
  read back byte-identical. The scorer accepted all 670,790 departures and returns no score.

## Repository layout

```
assemble.py          canonical builder: stored model outputs -> submission (built v27-v29 and the final file)
requirements.txt     pinned laptop environment
docs/                MODEL, RESULTS, VALIDATION, REPRODUCE, PIPELINE
prc/                 package: config, io (memory-aware loading), features, congestion (queue state),
                     weather (IEM METAR), disruption (ATFM), p_features (zone P), validation (paired and
                     block bootstrap), submit (file checks), s3 (upload), models/ (tail specialist, routed,
                     embedding network, w3, seq set model, ...)
cluster/             Slurm jobs (*.sbatch) and their Python tasks: fold fits and final fits of every member
experiments/         exp01-exp82 and 25 topic folders, including the archived 10-06 searches and 10-13 checks; see experiments/README.md
tools/leaderboard.py read-only client of the public leaderboard API (all teams' public submissions)
tools/make_final_cache.py  mirrors the training-side cache for the final phase (PRC_RANKING=final)
LICENSE              GNU GPL v3
```

## Data sources and licences

| Source | Use | Licence or terms | Status |
|---|---|---|---|
| PRC Data Challenge 2026 dataset | training and ranking files | challenge Terms & Conditions | used; **not redistributed** |
| adsb.lol globe_history [2025](https://github.com/adsblol/globe_history_2025), [2026](https://github.com/adsblol/globe_history_2026) | ground-movement features | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). A CC0 file of unstated scope also ships; we treat the data as ODbL | used; notice below |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) ASOS/METAR | weather at take-off | "in the public domain and may be used freely by anyone for any lawful purpose" ([disclaimer](https://mesonet.agron.iastate.edu/disclaimer.php)) | used |
| EUROCONTROL Airport Arrival ATFM Delay, daily by cause ([ansperformance.eu](https://ansperformance.eu/)) | disruption features: an operational input like weather, not a taxi-out statistic | may be copied with EUROCONTROL named as the source, not for commercial use ([terms](https://ansperformance.eu/about/disclaimer/)). Organisers: "can be considered open data for the challenge" (Discord, recorded 2026-09-25) | used; not redistributed |
| TabICL 2.2.0, code and checkpoint | zone-P tail probability | BSD-3-Clause | used |
| LightGBM, XGBoost, PyTorch, scikit-learn | models | MIT, Apache-2.0, BSD-3-Clause, BSD-3-Clause | used |
| pandas, NumPy, SciPy, pyarrow | data handling | BSD-3-Clause for the first three, Apache-2.0 for pyarrow | used |
| TabPFN v2; CatBoost | evaluated | Prior Labs License 1.1; Apache-2.0 | **not** in any shipped file |
| OpenSky state vectors / Trino | none | excluded by the organisers | **not used**; one public sample day was read on 2026-09-22 to measure ground coverage |
| OPDI flight events | coverage check (surface-event counts) | | **not used** in the model |
| EUROCONTROL monthly airport taxi-out figures (2025; January-July 2026) | 2025 file: measured as a calibration on 2025 (2026-10-09) | | **not used** in any submission; see below |

**adsb.lol notice (ODbL §4.3):**

> Contains information from adsb.lol globe_history (https://github.com/adsblol/globe_history_2025,
> https://github.com/adsblol/globe_history_2026), which is made available here under the Open Database License (ODbL)
> (https://opendatacommons.org/licenses/odbl/1-0/).

No derived database is published, and no adsb.lol-derived data is in this repository or in its history. The
extraction code in `experiments/adsb/` is the published method (ODbL §4.6 b). Its stage-1 archive reader changed on
2026-10-10; the training day 2025-10-14 was extracted with the old reader ([docs/REPRODUCE.md](docs/REPRODUCE.md) §3.5).

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
  spent effort.
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

For scale, on 2025 only: even a perfect per-airport-month level correction of our out-of-fold predictions would
have gained 0.14 s on January + July.

## Use of the public leaderboard

Every upload is scored on the full evaluation set, and the organisers monitor for attempts to learn from that.
[docs/VALIDATION.md](docs/VALIDATION.md) §6 gives the full account; in summary:

- **Every upload, classified.** [docs/RESULTS.md](docs/RESULTS.md) §1.1 is the master upload table: v1-v29 and the
  two uploads of the final file, each with its rows changed and the population they belong to. Of the 29 public
  uploads, 19 passed a rule written before their run, 4 files were leaderboard tests (v1/v2, v7, v10), 1 overrode a
  failed rule (v25), 2 applied post-hoc dataset-artefact rules (v28, v29) and 3 predate written rules (v3-v5); 9 were
  confined to a small population and 8 belong to three series of variants.
- **Model-level leaderboard tests.** These uploads settled questions that offline validation could not:
  - v1/v2 (09-21) tested trimming the training tail.
  - v7 (09-23) changed 1,507 zone-P rows (1,038 by more than 1 s; label-free counts); its 12-month offline
    interval crossed zero, while the January + July one did not.
  - v10 (09-24) added the route-D corrector on 5,029 rows (label-free count).

  The builders of v7 and v10 say so in their headers.
- **One live-informed choice still in the shipped model.**
  - For v18, the pre-registered rule passed both options for the corrector outside zone P: dropping it or halving it.
    Its tie-break estimator favoured "drop".
  - That estimator failed a check against our own v9 → v10 live delta. The alternative estimator favoured "half",
    and the team chose "half" (`experiments/routed_0926/r03_calibrate.py`).
  - The choice remains in v29 and in the final file as the `0.5` in `assemble.py`.
- **One rule override, which lost.** v25 (09-28) changed Rome's zone-P probabilities as a deliberate judgement call,
  after the change had failed its rule: January was worse and the gain was concentrated. It scored 1.07 s worse and
  was not carried forward.
- **Variants uploaded in sequence.** Under best-of scoring each of these series is also a selection on the scored set.
  - v8 → v9 (09-23, same day): network weight 0.5, then 0.65 on routes A/B/C, fitted on the 2025 screening months.
    A day later `strategy/s18_nn_weight_live.py` fitted a live-optimal weight (0.68) through the v7-v9 scores; it was
    not acted on.
  - v21, v22, v23 (09-27, same day): the seq member at 15, 20 and 30 epochs, each passing its own rule against the
    previous file; then the 27-fit pool v24 (09-28).
  - v26, v27 (10-02, 10-06): two frozen arms of one evaluation, both uploaded by a decision taken before either was
    built.
- **Uploads confined to a small population** *(label-free counts)*. These are disclosed because they can look like
  probing: v6 (1,707 zone-P rows), v7 (1,507 zone-P rows), v10 (5,029 route-D rows), v13 (180 of Rome's 181 zone-P
  rows), v18 (3,455 route-D rows outside zone P), v20 (1,704 zone-P rows), v25 (the same 180 Rome rows), v28 (63 rows
  with an LOBT) and v29 (36 rows). Each applies one model component or rule to every row of a population defined
  from input columns. We committed before v28 that no later upload would change a row the window moved and that no
  implied label is derived from its score; v29's 36 rows are under the same pre-commitment. Both apply to the
  initial ranking file; in the final file these rows are rebuilt by the same rules ([Final phase](#final-phase)).
- **Diagnostics on our own scores.** Some scripts do algebra on the squared-error difference between two of our
  uploads, to study offline-to-live transfer: `major/m04_live_decomp.py`, `major/m08_P_submission_deltas.py`,
  `strategy/s18_nn_weight_live.py`, `plan_1001/live_proj.py` and `routed_0926/r03_calibrate.py`.
- **How live scores were used.** Our own live scores decided v1/v2 (target trimming), v7 and v10 (leaderboard tests),
  the v18 tie-break and the dropping of v25; they set the v28 and v29 bounds, including the pricing of the 10-06
  structure-search candidates against v28's score; and, through the offline-to-live record and early deltas (CV
  protocol, `exp18`, `exp19`), they informed where we spent effort. No other modelling decision used them.
- **Leaderboard tooling.** `tools/leaderboard.py` pulls every team's public submission history from the competition
  API (`--wait` polls until our file is scored). `experiments/strategy/s08_threshold_drift.py` (cut-off drift),
  `s15_probability.py` (our probability of reaching the top-25% cut), `s16_outside_view.py` (how teams at our level
  improved), `research_0925/lb_analysis.py`, `drivers_0926/lb_snapshot.py`, `plan_1006/ff/ff_rankat.py` and
  `plan_1006/pattern.py` (other teams' upload timing) analyse that public data. They fed no model decision.
- **No file was ever built to recover labels.** Competitors' statements about their own live results are not used as
  evidence.

## How this was built

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

- **Rules.** From v6 on, most changes were gated by a rule written before their run. The exceptions are listed in the
  previous section. In addition, v1-v5 predate written rules, the v28 header was written after its 2025 numbers
  were measured, and the v29 rules came from a structure search on 2025 and on label-free 2026 rows, with candidates
  priced against v28's live score (3 shipped).

## Credits

We adopted the ideas below from other participants. All code in this repository is our own. Discord messages are
dated by when we recorded them; some were posted earlier in September.

- **LOBT feasibility window** (adopted 2026-10-06), from three public repositories:
  [EnioAguiar/prc-taxiout-2026](https://github.com/EnioAguiar/prc-taxiout-2026),
  [javidmardanov/PRC-Data-Challenge-2026](https://github.com/javidmardanov/PRC-Data-Challenge-2026) and
  [joris-decombe/taxiout](https://github.com/joris-decombe/taxiout). We did not find it ourselves. Our decision to
  use it rests on our own 2025 measurements, not on their leaderboard statements.
- **Overdue-queue and schedule-backlog features:**
  [kagervig/PRC-Challenge-2026](https://github.com/kagervig/PRC-Challenge-2026).
- **Wake-sequence feature, January ↔ July cross-fits, and the drift check of new inputs:** GREKI, on the challenge
  Discord (2026-09-25 and 2026-09-28).
- **Judging January and July separately, and rejecting gains carried by fewer than 100 rows:** Piyush Patil, on the
  challenge Discord (2026-09-25). These became our gates R3 and R5.
- **adsb.lol as a ground-movement source:** raised by participants on the challenge Discord (September 2026).
- **Ideas from the challenge Discord that we tested but did not ship** (measured; see [docs/RESULTS.md](docs/RESULTS.md)):
  - the carrier habit of copying the planned time into the off-block when there is no NM match, and a corrector on
    out-of-fold predictions: Vudueprajacu (2026-09-28). The carrier rate improved Rome's zone-P classifier but not the
    RMSE (the one upload that overrode our rule on it, v25, lost); the corrector failed our January <-> July cross-fit gate.
  - making the submission's outliers resemble the training outliers: lbragado (September 2026); not adopted.
  - an anchored target for the network member and a 2025-vs-2026 distribution check of new columns: Piyush Patil
    (September 2026); measured, the parity check became part of our drift audit.
- **Data and organisation:** the organisers of the PRC Data Challenge (EUROCONTROL's Performance Review Commission,
  with the OpenSky Network), the adsb.lol contributors, and the Iowa Environmental Mesonet.

## Authors

Team understanding-accordion: Alexander Beiser (TU Wien).

## Licence

The code and documentation are released under the GNU General Public License v3.0 ([LICENSE](LICENSE)), as the
challenge's prize-eligibility conditions require. Neither the challenge data nor any third-party data is included.
