# experiments/final_1013/: last checks before the deadline (2026-10-13)

Archived as run. Nothing here changed the final file. All work is on 2025 labels or label-free on the final ranking
file; no live score exists for the final file.

## Provenance

- Run on 2026-10-13 by Claude Code agents in a session scratch folder and copied here the same day: the `*.py`
  scripts, and for `lever_search/` also the pre-registration (`prereg.md`) and the printed results (`results.txt`,
  `anatomy.txt`). Intermediate arrays and tables (`*.npz`, `*.npy`, `*.parquet`, `*.csv`) were not copied.
- **Paths are as run**: absolute paths of the development machine and of the session scratch folder
  (`/tmp/claude-1000/.../scratchpad/...`). The scripts will not run unchanged.

## `lever_search/`: assembly-level candidates on the 2025 out-of-fold predictions

- **Question.** Is there a cheap assembly-level change to the v29 structure that passes a written rule on 2025?
- **Baseline.** The v29 structure on the 2025 out-of-fold predictions (`base.py`): v27 blend, LOBT window, join
  complement. Every candidate leaves the window and join-complement rows unchanged and re-applies both rules.
- **Protocol.** Nested leave-one-month-out (every parameter fitted on the other 11 months, hyper-parameters by an
  inner leave-one-month-out); date-block bootstrap; rule R1-R4 of `prereg.md` (12-month CI below 0, January + July
  better, January and July each better, at least 8 of 12 months better).
- **Pre-registration.** `prereg.md` was written at 09:35 UTC before any candidate result (K1-K6). Three addenda
  (K7-K9, then K10, K11 and K3b, then K3c) were written after earlier results and are marked post hoc in the file.

| candidate | change | result |
|---|---|---|
| K1, K2 | additive shift per airport × route (× hour band) | no change selected in any month: fail |
| K3 | shift by distance to the active LOBT-window edge | pass, −0.008 s (negligible) |
| K4 | soft clip around the NM off-block | no change selected: fail |
| K5 | zone-P tail-probability recalibration | +0.75 s: fail |
| K6 | per-airport prediction floor | fail (R3, R4) |
| K7, K9 | route-D weights refitted on 11 months, with and without a schedule-anchor column | +0.031 / +0.026 s: fail |
| K8 | U+T weights refitted on 11 months | +0.096 s: fail |
| K10, K11 | zone-P and route-D corrector multipliers | fail (K10: R1; K11: R1, R3) |
| K3b *(post hoc)* | additive shift by bucket of take-off − LOBT above 3,606 s | pass on `prereg.md`'s rule, −0.083 s [−0.140, −0.031]; **fails our own R5** (+0.082 s on the held-out months without their 100 most-improved rows; carried mostly by about 400 rows above 6,000 s) |
| K3c *(post hoc)* | isotonic version of K3b | pass, −0.093 s; January + July CI crosses 0 |

**Outcome: none shipped.** K3 is negligible; K3b and K3c are post hoc, sit next to the window rows that the v28
pre-commitment protects, and K3b fails R5 of [../../docs/VALIDATION.md](../../docs/VALIDATION.md) §1.3.
`recipe_k3b.py` and `final_apply.py` show how K3b would be applied (label-free on the final file: it would move
11,574 rows); it was not applied.

## `audit_inputs/` and `audit_outputs/`: label-free audits of the final file

- **Inputs** (`audit_inputs/`): raw columns, cached feature files, METAR / ATFM / congestion coverage, adsb.lol days
  and match rates, timestamp re-dither, initial against final ranking rows, unseen categories; February/June 2026
  against the same months of 2025.
- **Outputs** (`audit_outputs/`): the final predictions by airport × day and month, against v29 on the shared
  January/July rows and against the 2025 analogue months.
- **Findings.**
  - One more adsb.lol extraction failure: **2026-06-06**, whose release holds a stray `.tar` next to a split
    `.tar.aa`/`.tar.ab` set; the old reader stopped after the first archive. Its status is in
    [../../docs/REPRODUCE.md](../../docs/REPRODUCE.md) §8.
  - Otherwise source gaps or real changes, not pipeline errors: 2026-06-11 is thin in the adsb.lol source; LIRF loses
    ground coverage in June/July; EDDF's ground share drops in February; EDDM has new stands 101-108 from 06-01;
    LFPG shows a level step in take-off − AOBT_3 from 06-08 that the predictions do not follow. Without labels we
    cannot tell whether the last one is an error.
