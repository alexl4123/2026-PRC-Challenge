# Lever search pre-registration — written 2026-10-13 09:35:08 UTC, BEFORE any candidate result was computed

Context read before writing: assemble.py, docs/RESULTS.md §6 (negative results), v28_oof.py, arm_u_eval.py, v24_oof.py.
Nothing about residuals of the v28/v29 OOF has been looked at yet (only array shapes / column names).

## Baseline B (v29 structure, 2025 OOF)
B = jc( window( rint(clip(f27,0)) ) ) where f27 = cache/w2/v28s_oof.npz['f27'], window = LOBT +-3606 projection (as
assemble.lobt_window), jc = assemble.py join-complement rules A, B', C applied to the training file via
assemble.clean_links('training') (DEP order == dep_training order, checked). "Projected rows" = rows where B differs from
rint(clip(f27)). Every candidate leaves projected rows unchanged (v28/v29 pre-commitment), and re-applies window + jc +
rint to its output.

## Evaluation (identical for all candidates)
Nested leave-one-month-out on 2025: for each evaluation month m, every parameter is fitted on the other 11 months only;
any hyper-parameter (shrinkage lambda, grid choice) is chosen by an inner leave-one-month-out over those 11 months
(inner criterion: pooled SSE over the 11 inner held-out months). The month-m prediction never sees month-m labels.
Metric: RMSE over ALL departures (all routes), candidate G vs baseline B.
Blocks for the paired bootstrap: UTC calendar date (365 blocks), 1000 resamples, seed 0 (prc.validation.block_bootstrap).
Secondary (reported, not gated): airport x local-day blocks; RMSE after removing the 100 most-improved rows.

## Decision rule (ship only if ALL hold)
R1 12-month nested held-out RMSE diff, 95% date-block CI upper bound < 0.
R2 Jan+Jul subset point diff < 0.
R3 January diff < 0 AND July diff < 0.
R4 >= 8 of 12 months with diff < 0.
R5 parameters only from nested/inner LOMO (by construction above).
Plus disclosed: any candidate flagged as a near-repeat of a RESULTS.md negative result or as touching a pre-commitment
must be reported as such; the team decides.

## Candidates (all pre-registered here; any later one gets its own timestamped entry)
K1 additive shift per airport x route group (A/B/C outside P, D outside P; zone P untouched):
   delta_c = sum(resid_c)/(n_c + lambda), resid = y - B on the 11 fitting months, lambda in {0, 1e2, 1e3, 1e4, 1e5, inf}
   chosen by inner LOMO. NEAR-REPEAT of RESULTS 6.3 "Recalibration per airport x route (linear)" on an older base.
K2 as K1 but cells = airport x route group x local-hour band (hour_loc // 4, 6 bands). Same lambda grid.
K3 LOBT-window margin calibration (A/B/C rows with LOBT, not projected): additive shift by bin of the distance to the
   nearer active bound: d_lo = B - lo (only rows with lo = off_lobt-3606 > 0), d_hi = hi - B; bins (s) [0,60),[60,180),
   [180,400),[400,800),[800,1600) for each side separately (10 cells); other rows untouched; delta = shrunk mean
   residual, lambda grid as K1, inner LOMO. FLAG: close in spirit to the "truncated-mean LOBT-window variant" excluded
   by the v28 pre-commitment (it does not move projected rows, but moves their neighbours).
K4 soft clip of A/B/C rows relative to the NM off-block: G = clip(B, off_aobt3 - c_lo, off_aobt3 + c_hi), c_lo, c_hi in
   {600, 900, 1200, 1800, 2400, 3600, inf} (independent), chosen by inner LOMO.
K5 zone-P tail-probability recalibration: p' = sigmoid(a*logit(p) + b), separately Rome / non-Rome; (a, b) on a grid
   a in {0.5,0.6,...,2.0}, b in {-2,-1.75,...,2} minimising SSE of the zone-P prediction on the 11 fitting months; zone-P
   prediction rebuilt as P_rest + clip(p'*so + (1-p')*pbP, 0) (v24_oof.py formula), other rows untouched.
K6 per-airport prediction floor: G = max(B, Q_a(alpha)), Q_a = alpha-quantile of y at airport a on the fitting months,
   alpha in {0, 0.0005, 0.001, 0.005, 0.01} chosen by inner LOMO.
Exploratory (no decision): error anatomy of B by airport x route x hour x off_lobt bucket; bias vs variance share.

## Addendum 1 — written 2026-10-13 09:39:26 UTC, AFTER K1-K6 results and the exploratory anatomy (post hoc; disclosed)
Results so far: K1, K2, K4 select "no change" in every outer month; K3 PASS but -0.008 s; K5 FAIL (+0.75); K6 FAIL.
Anatomy: bias components are tiny (airport x route bias^2 = 0.26% of SSE); top-100 rows = 36% of SSE; route D outside P =
22.6% of SSE, 19% in 84 rows with |e| > 3600 (many with y ~= off_sched, block stamped at schedule, off_sched in 1-2 h).
Same evaluation and decision rule as above. New candidates (motivated by the anatomy => post hoc):
K7 route D outside P: NNLS weights refit (8 members as v24 + a 9th column x9 = clip(off_sched, 0, 7200), the schedule
   anchor), fitted on the D-outside-P rows of the 11 non-evaluation months (y on members, no intercept, as v24_oof.py),
   then + 0.5 * corr exactly as now. No hyper-parameter.
K8 routes A/B/C: the U+T structure (per-tercile NNLS weights over the 8 members + lambda_bin * dl, as arm_u_eval.py)
   refit on the 11 non-evaluation months instead of the screening months 3/6/9/12. No hyper-parameter. (Note: the
   baseline is in-sample on months 3/6/9/12 for its weights, which handicaps K8 there; 8 held-out months also reported.)
K9 route D outside P: as K7 but WITHOUT x9 (pure weight refit on 11 months) — isolates the anchor column's effect.

## Addendum 2 — written 2026-10-13 09:41:33 UTC, AFTER K7-K9 results (all FAIL; K8 +0.096, K9 +0.026, K7 +0.031)
Same evaluation and decision rule. Post hoc, disclosed.
K10 zone-P corrector multiplier: zone-P prediction = kappa * corr + clip(p*so + (1-p)*pbP, 0) (production: kappa = 1),
    kappa in {0, 0.25, 0.5, 0.75, 1, 1.25} chosen on the 11 non-evaluation months' zone-P SSE (nested). (Checked first:
    the OOF zone-P non-branch part P_rest equals corr exactly.)
K11 route-D-outside-P corrector multiplier: f = clip(blendD + kappa_D * corr, 0), kappa_D in {0, 0.25, 0.5, 0.75, 1}
    chosen on the 11 non-evaluation months (production 0.5; FLAG: 0.5 vs alternatives involved the v18 live tie-break).
K3b additive shift for A/B/C non-projected rows with off_lobt > 3606, cells = off_lobt buckets (3606,4200], (4200,5000],
    (5000,6000], (6000,8000], (8000,inf); shrunk mean residual, lambda grid as K1 by inner LOMO. Motivated by the anatomy
    (bucket off_lobt > 5000: mean residual +207 s, n = 1,911). FLAG: window-adjacent (see K3).

## Addendum 3 — written 2026-10-13 09:42:45 UTC, AFTER K10/K11 (FAIL) and K3b (PASS, -0.083; diagnostics: gain
carried mostly by the 49 rows with off_lobt > 8000 and the 355 rows in (6000, 8000]; w/o top-100 rows +0.067)
K3c (one more, then stop): monotone (isotonic, non-decreasing in off_lobt) correction of the residual for A/B/C
    non-projected rows with off_lobt > 3606: delta(off_lobt) = isotonic regression of (y - B) on off_lobt fitted on the 11
    non-evaluation months (sklearn IsotonicRegression, increasing=True, out_of_bounds='clip'); G = B + delta. No
    hyper-parameter. Same rule. If both K3b and K3c pass, K3b (pre-registered earlier) is the primary; K3c is reported
    as a robustness variant, not a choice between them by result.

## Outcomes — written 2026-10-13 09:44:30 UTC (full numbers in results.txt)
13 candidates: K1 K2 K4 FAIL (no change selected); K5 K6 K7 K8 K9 K10 K11 FAIL; K3 PASS (-0.008, negligible);
K3b PASS (-0.083 [-0.140,-0.031]); K3c PASS (-0.093 [-0.162,-0.023]; robustness variant, J+J CI crosses 0).
K3b diagnostics: team R5 (held-out w/o top-100 held-out rows) +0.082 => would FAIL the team's v19+ gate; 12m w/o top-30
rows +0.016; cross-fits Jan->Jul -0.240, Jul->Jan -0.058; drop-one-airport -0.061..-0.091.
Final-file (label-free): rebuild == shipped final (0 rows differ); K3b moves 11,574 rows, 36 of them by +1094 s.
