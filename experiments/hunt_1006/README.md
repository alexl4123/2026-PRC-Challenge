# experiments/hunt_1006/: the structure search that produced v29 (2026-10-06)

Archived as run. Not on the production path; `assemble.py --join-complement` is the shipped implementation of what
survived here.

## Provenance

- **Written on 2026-10-06**, after v28 (live 235.3996) and before v29, in a git-ignored scratch folder (`cache/hunt/`).
  Copied here on 2026-10-13 so that the record is complete.
- **Copied:** all 111 `*.py` files, unchanged. **Not copied:** their outputs (`*.out`, `*.log`) and intermediate files
  (`*.parquet`, `*.npz`, `*.csv`, including the dry-run candidate submissions): they hold data derived from the
  challenge files. Nothing was excluded for any other reason.
- **Paths are as run.** The scripts use absolute paths of the development machine (`/home/thinklex/...`) and read
  caches under `cache/hunt/` that are not published. They will not run unchanged.
- **Who ran it.** A multi-agent Claude Code session (four searchers, two skeptics, one synthesiser) launched by the
  author with the instruction to learn from the LOBT-window gain (dataset-construction structure) and look for
  further improvements. No uploads, git or cluster commands were made by it. Its internal report is not published;
  the numbers below come from these scripts and that report.
- **A few scripts name individual 2026 departures** (callsign or row id) in comments or literals, because they
  inspect or price those rows (`sk_02_rome.py`, `sk_03_cfg.py`, `sk_04_pkg.py`, `syn_01_verify.py`, `h23_bcheck.py`,
  `hedge_05_lirf_format.py`, `hedge_07_plink.py`). These are input fields of the ranking file, not labels.

## What the search was

- **On 2025 (labels):** look for constraints that the dataset's construction places on the recorded off-block (join
  conditions and their complements, source caps, timestamp noise, row order, identifiers), require 0 exceptions in all
  12 months, and measure each rule on the v28-structure 2025 out-of-fold predictions.
- **On label-free 2026 rows:** for each rule, find which ranking rows it would move and by how much, and check that
  2026 was built the same way (link shares, NM coverage, caps on 2026 arrivals). Several scripts also screened where
  v28 hedges between hypotheses on 2026 rows (`h16_hedges26.py`, `ids_10_hedge2026.py`, `hedge_10_2026.py`,
  `hedge_13_routeD26.py`, `hedge_17_dist.py`), i.e. which 2026 rows carry the most expected squared error under the
  model's own probabilities.
- **Priced against v28's live score.** 22 of the 111 scripts use 235.3996 (v28's public score): a rule that holds
  bounds the score by `sqrt(235.3996² − Σ move² / N)`, and an estimate [E] multiplies that by assumed probabilities
  of row classes. No 2026 label and no published aggregate was used; the live score is the only label-dependent
  input.
- **Not pre-registered.** Rules were found and measured on 2025, then screened and priced on 2026 rows, before the
  `assemble.py` header (df0a5ff) was written.

## Script families

| prefix | files | subject |
|---|---:|---|
| `h00`-`h34` | 35 | bounds of BLOCK against every clock; day-scale rows; duplicate rows; the link from a route-D departure to its own NM arrival record (`h07`, `h10`, `h15`, `h22`); the LEBL/LEMD cap (`h20`, `h21`); dry-run candidate files (`h25_candidate.py`, including `+E1`); per-rule 2025 OOF (`h30`); 2026 hedge screen (`h16`); raw row order (`h17`) |
| `hedge_01`-`hedge_24` | 24 | Rome zone-P rows: stamp / day-shift / normal classes, label-free fingerprints, date conventions, MVT_ID order at LIRF; 2026 screens of v28's hedge rows (`hedge_10`, `hedge_13`, `hedge_17`); Rome decision candidates (`hedge_16`) |
| `ids_01`-`ids_31` | 31 | construction forensics: MVT_ID allocation and raw row order (`ids_04`-`ids_07`, `ids_11`-`ids_15`), FLIGHT_ID, timestamp noise, local-date artefacts, twins (`ids_08`, `ids_09`, `ids_18`, `ids_28`), the destination-NM exclusion "DST window" (`ids_19`-`ids_23`, `ids_30`, `ids_31`), 2026 hedge screen (`ids_10`) |
| `stand_*` | 15 | physical stand and airframe sequence bounds; FLIGHT_ID and MVT_ID order |
| `sk_01`-`sk_04`, `skep_rules_overlap.py` | 5 | skeptic re-checks; `sk_04_pkg.py` builds the 36-row package and prices one Rome row under two hypotheses |
| `syn_01_verify.py` | 1 | synthesiser's independent re-check of the package file |

The row-order and identifier scripts (`ids_04`, `ids_11`-`ids_15`, `h17`, `stand_s07`, `stand_s09`, `stand_s12`,
`hedge_22`, `hedge_23`) found no leak: MVT_ID is batched per airport, month and phase and sorted by schedule, FLIGHT_ID
is NM's LOBT order, and raw file order is uncorrelated with the out-of-fold residual (|ρ| ≤ 0.019).

## Candidates and outcomes

2025 figures are in-sample for the rules. 2026 figures are label-free row counts; "bound" applies those moves to
v28's live score.

| candidate | rule | 2025 | 2026 | outcome |
|---|---|---|---|---|
| **A** | join complement, one-sided: a route-D IFR departure whose own NM record sits on its destination's arrival row failed the join; raise to the upper edge where the lower side is impossible | 0 of 1,426 clean links inside ±3,606 s; 70 of 70 one-sided rows above | 9 rows up | **shipped (v29)** |
| **B** | two-sided, below side, stamping excluded | 1,296 of 1,296 below the lower edge (empirical) | 22 rows down | replaced by B′ |
| **B′** | B without take-offs that cross local midnight relative to the schedule | 1,281 of 1,281 | 21 rows down | **shipped (v29)**; see the note below |
| **C** | LEBL/LEMD source cap at 3,606 s | 0 of 391,947 above | 6 rows (10 touched, 4 already moved by A/B′) | **shipped (v29)** |
| E1 | LIRF one-sided rows take the stamped value | 21 of 23 stamped; rule chosen after seeing the rows | 12 rows; package + E1 bound 234.775 if E1 holds | dropped: statistical, not exact |
| B2 | two-sided rows where stamping is still possible | 58 of 58 below, but stamping undecided | 6 rows | dropped |
| DST window | destination-NM exclusion with another link key | 0 of 830 after filters chosen on 23 exceptions; OOF 88 rows, 0 worse | 28 rows, 24 identical to A/B moves | dropped; corroborates A/B |
| twin T3/T4 | return-to-stand pairs: the NM row is the aborted first off-block; move by a learned offset | structure 99 of 99; OOF 87/99 rows changed, 20/23 worse | 14 LFPG rows | dropped: move size not exact |
| ROME-X, ROME-T, ROME-DS | Rome zone-P date-crossing, delay > 8 h, day-shift hedges | in-sample thresholds | 22 rows (98.7% of the squared moves on one row) / decided by one row / 3 rows | killed: row-concentrated, would act as probes |
| A1, FID, INB, stand and airframe bounds | other link and physical rules | exact only where vacuous, or harmful | 0 rows or harmful | killed |

**How B′ came about.** B's 2026 moves included exactly one row whose take-off crosses local midnight relative to
its schedule (a Rome departure). A skeptic inspected that row's label-free fields and priced it under a "normal" and a
"day-shift" hypothesis against v28's live score (`sk_04_pkg.py`: −0.11 s against +3.12 s). The local-midnight
exclusion was then written as a rule over input columns, which drops that row and no other 2026 row; in 2025 it
removes 15 rows, none of which B had moved. The rule is not an edit by row id, but it was motivated by one inspected
2026 row. The dry-run file of the 36-row package was built here, outside `assemble.py`, before the header was
written; the shipped build reproduces it row for row.

## What this means for the documents

The v29 rules came from a structure search on 2025 and on label-free 2026 rows, with candidates priced against v28's
live score; three were shipped (A, B′, C). See [../../docs/RESULTS.md](../../docs/RESULTS.md) §2 (v29) and
[../../docs/VALIDATION.md](../../docs/VALIDATION.md) §5.5.
