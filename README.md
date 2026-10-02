# PRC Data Challenge 2026 — taxi-out time prediction

Team **understanding-accordion**.

Repository: <https://github.com/alexl4123/2026-PRC-Challenge>

Predicting taxi-out time (seconds) for departures at ten major European
airports — EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM — scored
by RMSE over 344,841 departures in January and July 2026.

<https://prc-data-challenge-2026.netlify.app/>

## The problem in one paragraph

`TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`, exactly, for all
2,085,047 training departures. In the ranking set the takeoff time is given and
the off-block time is blanked, so the task is really to predict *when the
aircraft pushed back*, working backwards from a known takeoff. Network Manager's
`AOBT_3_flt` measures the same physical event from a different system and is
left in the file at 98.5% coverage, which makes it the strongest single feature.

## Layout

    prc/            importable package
      config.py     paths, airports, constants
      io.py         memory-aware loading and caching
      features.py   base features; every one computable on the ranking set
      congestion.py queue and throughput features from the movement stream
      weather.py    METAR from the Iowa Mesonet ASOS archive
      disruption.py daily EUROCONTROL ATFM delay by cause
      validation.py metrics, month-wise CV, paired bootstrap
      submit.py     building and validating a submission file
      s3.py         upload to the team bucket
    experiments/    numbered, reproducible; each states what it tests
    cluster/        Slurm array jobs

Data lives **outside** this repository, one directory up. That is deliberate:
the challenge dataset may not be published, and this repo is destined to be
public under GPLv3.

## Running

    python3 -m venv ~/prc2026-venv
    ~/prc2026-venv/bin/pip install "pyarrow>=20" pandas numpy scikit-learn lightgbm

`pyarrow >= 20` is required — earlier versions fail on the challenge parquets
with `Repetition level histogram size mismatch`.

    ~/prc2026-venv/bin/python experiments/exp13_month_cv.py

Long local runs should be wrapped so a runaway fails small rather than taking
the desktop with it:

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \
        systemd-inhibit --what=idle:sleep:handle-lid-switch \
        ~/prc2026-venv/bin/python experiments/expNN_....py

## Building the submission

The submission is assembled from stored model outputs by one script:

    python assemble.py --arm-u U --check 26     # equivalence with the shipped v26
    python assemble.py --arm-u U --out v27      # writes outputs/understanding-accordion_v27.parquet

[docs/PIPELINE.md](docs/PIPELINE.md) is the runbook: which script and cluster job writes each
component, the feature pipeline, the validation protocol, and every data source with its licence.

## Method notes

The metric is dominated by a tiny tail: about 150 of 344,841 rows exceed two
hours, and they drive the score. Two consequences shape everything here.

**Validation.** An absolute local RMSE has a bootstrap sd of ~47s because those
rows dominate it, while the top leaderboard teams are separated by ~6s. Models
are therefore compared with a *paired* bootstrap over leave-one-month-out
predictions, which cuts the sd by 2.5x.

**Modelling.** No single L2 model wins both halves: restricting the training
target to the bulk improves bulk RMSE by 38s and costs 6,748s on the tail. The
model is a mixture that estimates the conditional mean explicitly,

    E[y|x] = P(T|x)·E[y|T,x] + (1 − P(T|x))·E[y|¬T,x]

which is worth ~155s against a single regressor on the official leaderboard.

## Licence

GPLv3, per the challenge's prize-eligibility conditions.
