# Reproducing the submission

Team **understanding-accordion**, PRC Data Challenge 2026 (taxi-out time at ten European airports).
Author: Alexander Beiser (TU Wien), team understanding-accordion.

This document explains how to rebuild our final submission at three depths. The final submission is v28: public-leaderboard
RMSE 235.3996, uploaded 2026-10-06. Its predecessors are rebuilt the same way. The runbook [PIPELINE.md](PIPELINE.md)
lists which script writes each intermediate file. The model is described in [MODEL.md](MODEL.md) and the validation
protocol in [VALIDATION.md](VALIDATION.md).

**Conventions.**
- "2025" is the labelled training year: 12 monthly files, 2,085,047 departures.
- "2026" is the ranking file: every departure of January and July 2026, 344,841 rows. We never had its per-flight
  labels; on published monthly summaries of them, see §3.8.
- Every 2026 number below is computed from our own predictions or from the unlabelled ranking file, and is marked
  *(label-free)*. The one exception is the public-leaderboard score of our own uploads, which is marked as such.
- Measurements without a date come from our internal experiment log, which is not published. Measurements taken while
  writing this document are dated 2026-10-06.
- Paths are relative to the repository root. `<cluster>` stands for the Slurm cluster we used.

## Summary

| level | starts from | rebuilds | compute | what it demonstrates |
|---|---|---|---|---|
| **L0** | 75 stored model-output files (159 MB) and the challenge files | the shipped v24, v26, v27 and v28 | about 1 s and under 1 GB per build, on a laptop | each shipped file follows from one closed formula over the stored model outputs (v24 and v26 up to 2 s of rounding on 750 rows), with no hand edits and no earlier submission file read |
| **L1** | the cached feature tables | the 75 model outputs, then L0 | final fits only: about 7-8 GPU-hours on L40S, plus 18 CPU fits and three laptop CPU steps. With the fold campaigns that re-derive the blend weights: roughly 150 GPU-hours and 40 CPU-hours. All of these figures are estimates (§2.5). | the model outputs come from the code at the stated settings, up to run-to-run variation (not measured) |
| **L2** | the raw challenge files and public data | the cached feature tables, then L1 | a 1.331 TB stream of adsb.lol archives (429 days), plus METAR and ATFM downloads | the features use only the challenge files and the open sources listed in §3 |

L0 is checked: all four checks passed on 2026-10-06, and the v26 check also passed from a fresh clone on 2026-10-02.
L1 and L2 are the procedures that produced the stored files, written down step by step. We have not re-run them end to
end.

---

## 0. Setup

### 0.1 Environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- [`requirements.txt`](../requirements.txt) pins the versions of the laptop environment that built v24-v28: CPython 3.12.3
  on Ubuntu 24.04.
- **pyarrow ≥ 20 is required.** Older versions fail on the challenge parquet files with
  `Repetition level histogram size mismatch`.
- L0 imports only numpy, pandas and pyarrow.
- **TabICL checkpoint** (needed at L1 for the zone-P classifier):
  - Hugging Face repository `jingang/TabICL`, snapshot `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`.
  - File `tabicl-classifier-v2-20260212.ckpt`, 110.4 MB,
    SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
  - `tabicl` downloads it into the standard Hugging Face cache on first use. We ran with `HF_HUB_OFFLINE=1` after that.
- We wrapped every long local run in a memory cap, so that a runaway job fails small:

  ```bash
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 nice -n 5 .venv/bin/python <script>
  ```

### 0.2 Where the data must be

The challenge data is not in this repository: the Terms & Conditions forbid redistributing it. `prc/config.py` looks for
it in the **parent directory of the clone** (`DATA = REPO.parent`). Every derived file goes to `cache/` and every
submission to `outputs/`. Both are gitignored.

```
<parent>/
  training_2025-01-01_2025-02-01.parquet  ...  training_2025-12-01_2026-01-01.parquet   (12 files)
  ranking.parquet
  submitting.parquet
  <clone>/                  this repository, under any directory name
    cache/                  derived files (created on first use)
    outputs/                submission files
```

The first call to `prc.io.load()` builds `cache/training.parquet` and `cache/ranking.parquet`. It streams one month at a
time and asserts 4,167,797 and 689,534 movement rows; arrivals and departures both count.

There is no environment variable for the data location yet. To use another layout, change `DATA` in `prc/config.py`.

---

## 1. L0: rebuild the shipped files from the stored model outputs

### 1.1 Inputs

`assemble.py` reads 75 files, 159 MB in total, all under `cache/`. They are not in the repository: every one of them is
derived from the challenge data. [PIPELINE.md §1](PIPELINE.md) gives the writer of each file.

| files | count | content |
|---|---:|---|
| `final/lgb_components.npz` | 1 | the production tail classifier (5 LightGBM seeds), the tail anchor, the zone-P mask and the row ids |
| `adsb2_final/<family>__seed0{0,1,2}.npy` + `ranking_ids.npy` | 22 | the 7 member families `mix perapt lintree xgboost routed nn w3`, 3 seeds each |
| `seq_<V>_final/seq__seed0{0,1,2}.npy` + `ranking_ids.npy`, for 9 settings V | 36 | the 27-fit pool of the neighbour-context sequence model. These folders also hold seeds 3-5, which assemble.py does not read |
| `seq_final/{seq,seqnc}__seed0{0,1,2}.npy` + `ranking_ids.npy` | 7 | the 10-epoch sequence model and its no-context twin; only `--arm-u UT` reads them |
| `w2_final/mix__seed0{0,1,2}.npy` + `ranking_ids.npy` | 4 | the LightGBM bulk model without adsb.lol, used in zone P |
| `strategy/tfm_P_lirf_ranking.npz`, `tfm_w4/pool_nonrome_ranking.npz` | 2 | TabICL tail probabilities for the 1,720 zone-P rows: 181 at Rome, 1,539 elsewhere *(label-free counts)* |
| `major/m11f_candidates.npz` | 1 | the route-D corrector. **It embeds the route-D values of the v9 submission** (§4) |
| `seq/swap_seqsw_ep30__…__seqsw_ep40_lr3.json`, `seq/arm_u.json` | 2 | the blend weights, the tercile cut points, the λ values of the U+T arm, and the gate verdicts |

`assemble.py` also reads the ranking file's ids and five of its columns. With `--lobt-window` it also reads two columns of
the 2025 training departures, for a self-test. `--check N` compares against `outputs/understanding-accordion_vN.parquet`,
and `--out` uses `submitting.parquet` as the template.

### 1.2 Commands and expected output

```bash
python assemble.py --check 24
python assemble.py --arm-u U --check 26
python assemble.py --arm-u UT --check 27
python assemble.py --arm-u UT --lobt-window --check 28
```

Output of the last command, run on 2026-10-06 *(label-free: it compares our predictions with our predictions)*:

```
LOBT self-test: 2,062,577 training departures, max |BLOCK - LOBT| = 3606 s, 0 outside -> OK
A/B/C outside P  339,551 rows: rows differing     0, max |diff|    0 s -> OK
D outside P        3,570 rows: rows differing     0, max |diff|    0 s -> OK
zone P             1,720 rows: rows differing     0, max |diff|    0 s -> OK
v28: EQUIVALENT (P differs by old-chain rounding only)
```

The script prints the last line whenever the tolerances hold, even when nothing differs at all. All four checks report
EQUIVALENT. Measured with the cache already built: 1.0-1.2 s wall time, peak memory 0.41 GB (`--check 24`) to 0.85 GB
(`--check 28`).

| check | routes A/B/C (339,551 rows) | route D outside zone P (3,570) | zone P (1,720) |
|---|---|---|---|
| `--check 24` | 0 rows differ | 0 rows differ | 750 rows differ, max 2 s |
| `--arm-u U --check 26` | 0 | 0 | 750, max 2 s |
| `--arm-u UT --check 27` | 0 | 0 | 0 |
| `--arm-u UT --lobt-window --check 28` | 0 | 0 | 0 |

- **v27 and v28 match exactly.** `assemble.py` built them.
- **v24 and v26 were built by the earlier delta chain** (builders `experiments/exp70`-`exp82`), in which every submission
  was an edit of the previous file. On 750 zone-P rows the chain accumulated rounding of at most 2 s. Every other row
  matches exactly.
- **What `assemble.py` reads.** It reads no earlier submission file (commit `10c3cb7`). It applies one formula per row
  group, given in its header and in [PIPELINE.md §0](PIPELINE.md).
- **What `--lobt-window` checks first.** On the 2025 training file it asserts that the off-block time lies within ±3606 s
  of the flight plan's LOBT, with the maximum exactly 3606 s. This guards against unit and time-zone bugs. It also asserts
  that no route-D or zone-P row carries an LOBT.
- **v28 differs from v27 in 63 rows only** *(label-free)*. Each of them is moved exactly to the edge of its window.

**Writing a file.**

```bash
python assemble.py --arm-u UT --lobt-window --out v128   # -> outputs/understanding-accordion_v128.parquet
```

`--out` behaves as follows:
- It refuses to overwrite an existing file.
- It writes to the schema of `submitting.parquet`.
- It validates the result as the organisers' validator would: 344,841 rows, the id set equal to the ranking departures, no
  duplicates or nulls, on-disk types equal to the template's.
- It reads the file back and compares it with the in-memory build.

On 2026-10-06 we wrote rebuilds with the same writer code under pyarrow 25.0.1 to a scratch location:
- The v27 and v28 rebuilds have the **same SHA-256 as the shipped files** below.
- A v24 rebuild does not, because of the zone-P rounding described above.

**Clean-clone test (2026-10-02).** We made a fresh `git clone` at commit `1c20e86` and symlinked `cache/` and `outputs/`
into it. `python assemble.py --arm-u U --check 26` reported EQUIVALENT, the same result as in the working tree.

### 1.3 Manifest of the shipped files

These are the SHA-256 hashes of the files we uploaded for scoring, computed on 2026-10-06. The organisers hold the
uploaded copies, so they can compare them.

| file | built by | L0 command | public leaderboard RMSE | SHA-256 |
|---|---|---|---:|---|
| `understanding-accordion_v24.parquet` | `experiments/exp81_submit_v21.py --pool <9 settings> --prev 23 --version 24` (2026-09-28) | `--check 24` | 244.2213 | `2cd656bc2fcba539a7d22038d3676cb9d68bd57be890ae6f20ae97f0cc409494` |
| `understanding-accordion_v26.parquet` | `exp81 --pool <9> --arm-u U --prev 24 --version 26` (2026-10-02) | `--arm-u U --check 26` | 243.4219 | `888ab1d8ff168186debbb8991bfaebcb3ab338851154a8dee14362c019068111` |
| `understanding-accordion_v27.parquet` | `assemble.py --arm-u UT --out v27` (2026-10-02) | `--arm-u UT --check 27` | 243.1478 | `8a5e97bdc3b9eaa602caec841e5a8fef36bb1963c9d9ff638b14465f20142d5e` |
| `understanding-accordion_v28.parquet` | `assemble.py --arm-u UT --lobt-window --out v28` (2026-10-06) | `--arm-u UT --lobt-window --check 28` | **235.3996** | `0d1929c78ddc395187458f1824e3228e32922594e1033c4235e956b52c7cd4a2` |

- **Rows changed between consecutive files** *(label-free)*: v24 → v26, 319,221 rows; v26 → v27, 292,725 rows; v27 → v28,
  63 rows.
- **v25 is not in this lineage.** It was a judgement call against our own validation rule and scored 1.07 s worse than v24
  on the public leaderboard. We dropped it, and `assemble.py` has no option that builds it. [VALIDATION.md](VALIDATION.md)
  gives the full record.

---

## 2. L1: refit the members and the sequence pool from the cached features

### 2.1 Inputs (the cached feature tables)

| file under `cache/` | content | written at L2 by |
|---|---|---|
| `training.parquet`, `ranking.parquet` | the challenge files, loaded once | `prc/io.py` (§0.2) |
| `external/metar/<ICAO>.parquet` (10 files), `external/atfm_daily.csv` | weather and ATFM delay | §3.2, §3.3 |
| `v11/feat_{training,ranking}.parquet` | flight-plan clocks, previous-arrival stamping, stand turnover | `experiments/v11/v11_features.py` (§3.4) |
| `adsb_lol/features_{training,ranking}.parquet` | 35 adsb.lol ground-movement columns per departure | §3.5 |
| `exp62_cand_{training,ranking}.parquet` | 38 zone-P candidate features | `experiments/exp62_screen.py` (§3.4) |
| `major/`, `exp62_oof.npy`, `exp62_spec_P.npy`, `blend_oof_mix_nn3.npy` | inputs of the route-D corrector | §3.4, §3.6 |

`cluster/w2_features.py` assembles the members' 143 columns from these tables. The wake-category map
`cache/w2/wtc_modal_by_type.json` is written from the training frame on the first member job.

### 2.2 Cluster preparation

**The cluster.** `<cluster>` runs Slurm. Two partitions were used:
- `gpu`: one node with 8 × NVIDIA L40S, 64 CPUs and 2 TB RAM.
- `any`: CPU nodes, with a 1 h time limit.

**Getting code and data there.**
- Code travels by `git clone` / `git pull`.
- The data never goes through git: copy the cached files above into `cache/` of the clone on the cluster (for example
  with `rsync`).

**Setting up.** Run these once, on the cluster:

```bash
bash cluster/setup_env.sh        # conda env prc2026: CPU members
bash cluster/setup_env_gpu.sh    # conda env prc2026gpu: torch from the cu124 index (L40S = sm_89)
bash cluster/preflight.sh        # pyarrow >= 20, required cache files, all 10 METAR stations, row counts
```

Then, in the GPU environment, install the pinned `tabicl==2.2.0 transformers==5.17.0`, but only if TabICL is to run on the
cluster. We ran it on the laptop.

**How the jobs behave.**
- Every job file checks its inputs before starting.
- Every job is idempotent: an existing output is skipped, so a failed array can simply be resubmitted.
- Every job pins its settings in a `_config.json` in its output folder and stops on a mismatch.
- Fold campaigns write `cache/cv/<campaign>/`. Final fits (all of 2025 → the ranking file) write `cache/<name>_final/`.

### 2.3 Final fits: the outputs `assemble.py` reads

| outputs | job | tasks | resources requested per task |
|---|---|---:|---|
| `adsb2_final/{mix,perapt,lintree,xgboost,routed}__seed0{0,1,2}` | `sbatch cluster/adsb2_final.sbatch` | 15 | 8 CPUs, 24 GB |
| `adsb2_final/{nn,w3}__seed0{0,1,2}` | `sbatch cluster/adsb2_final_nn.sbatch` | 6 | 1 GPU, 4 CPUs, 48 GB |
| `seq_ep{30,40,50}_final/` | `sbatch --export=ALL,EPOCHS=30 cluster/seq_epochs_final.sbatch`, then the same with `EPOCHS=40` and `EPOCHS=50` | 9 | 1 GPU, 8 CPUs, 96 GB |
| `seq_{ep60,ep80,ep40_d128,ep50_d128,ep40_l3,ep40_lr3}_final/` | `sbatch --export=ALL,VARIANT=<V> cluster/seq_variant_final.sbatch`, once per variant | 18 | 1 GPU, 8 CPUs, 96 GB |
| `seq_final/seq__seed0{0,1,2}` | `sbatch cluster/seq_final.sbatch` | 3 | 1 GPU, 8 CPUs, 96 GB |
| `seq_final/seqnc__seed0{0,1,2}` | `sbatch --export=ALL,SEQ_FAMILY=seqnc cluster/seq_final.sbatch`. **The shipped files were not made this way** (§4) | 3 | 1 GPU, 8 CPUs, 96 GB |
| `w2_final/mix__seed0{0,1,2}` | `sbatch --array=0-2 cluster/w2_final.sbatch` (tasks 0-2 are the `mix` family) | 3 | 8 CPUs, 24 GB |

**The nine sequence-pool settings.** The shared base: K = 64 neighbours, a ±5,400 s context window, batch 4,096, width 64,
2 layers, learning rate 2e-3. The settings vary epochs and one capacity knob:
- `ep30`, `ep40`, `ep50`, `ep60`, `ep80`: epochs only.
- `_d128`: width 128.
- `_l3`: 3 layers.
- `_lr3`: learning rate 3e-3.

The pinned values are in each folder's `_config.json`.

Three further steps run on a CPU; we ran them on the laptop:

1. **Production tail classifier and tail anchor**:
   `python experiments/exp65_submit_v8.py --stage components` writes `cache/final/lgb_components.npz`.
   - It refits the five LightGBM seeds of the v6 configuration on base, congestion, METAR and ATFM features.
   - Its header records about 30 min with 4 threads and about 5.5 GB.
2. **Zone-P tail probabilities (TabICL, CPU, 3 random states each, `HF_HUB_OFFLINE=1`)**:
   - Rome: `python experiments/strategy/s21_tfm_P_final.py` fits on the 627 Rome zone-P rows of 2025.
   - The other nine airports: `python experiments/tfm/w4b_final.py --threads 3` fits on all 4,944 zone-P rows of 2025, with
     an airport one-hot encoding.
   - `s21` imports TabPFN unconditionally (§5).
3. **Route-D corrector**: `experiments/major/m11d_state_routeD.py`, then `experiments/major/m11f_candidate_arrays.py`. This
   needs the v9 submission (§4).

### 2.4 Fold campaigns: only to re-derive the blend weights and cut points

There are 12 leave-one-month-out folds × 3 seeds = 36 fits per model.

| model | fold campaign |
|---|---|
| 7 members with adsb.lol | `adsb2_cv_array.sbatch`: 180 CPU fits (5 families); `adsb2_nn_array.sbatch`: 72 GPU fits (`nn`, `w3`) |
| sequence pool, 9 settings | `seq_epochs_array.sbatch` with `EPOCHS=30/40/50`; `seq_variant_array.sbatch` with the six `VARIANT` values. 324 GPU fits |
| 10-epoch sequence model and twin | `seq_array.sbatch` with `SEQ_FAMILY=seq` and `SEQ_FAMILY=seqnc` |
| members without adsb.lol | `w2_array.sbatch`, `w2_nn_array.sbatch` |

Then, on any machine:

```bash
python experiments/seq/seq_swap_eval.py --old seqsw_ep30 \
    --new seqsw_ep30+seqsw_ep40+seqsw_ep50+seqsw_ep60+seqsw_ep80+seqsw_ep40_d128+seqsw_ep50_d128+seqsw_ep40_l3+seqsw_ep40_lr3
python experiments/seq/arm_u_eval.py          # writes cache/seq/arm_u.json
```

These write the two JSON files that `assemble.py` reads. Each script's header holds its decision rule; the rules are
explained in [VALIDATION.md](VALIDATION.md). `assemble.py` refuses weights whose recorded verdict is not `PASS`.

### 2.5 Cost

**Measured unit costs:**
- a CPU member fold: 6-7 min;
- a network (`nn`) fold: 1 min 25 s on one L40S;
- a 10-epoch sequence-model fold: 243 s on one L40S (14.5 s per epoch plus setup), peak RSS 7.99 GB;
- a no-context twin final on the laptop CPU: 6.9-8.4 min, peak 8.6-8.7 GB.

**Estimates:**
- **Final fits only.** About 7-8 GPU-hours. We derived this from the per-epoch time above: the 27 pool fits add up to 469
  epoch-equivalents per seed, counting `_d128` and `_l3` at +30%. Add 18 CPU fits and the three laptop steps.
- **Everything in §2.3 and §2.4.** Roughly 150 GPU-hours and 40 CPU-hours, as estimated in [PIPELINE.md §3](PIPELINE.md).
  The original campaigns ran 4-8 tasks at a time on the 8-GPU node.

### 2.6 How close an L1 rebuild will come

We have not measured this, and we will not use the leaderboard to measure it.
- Every fit takes an explicit `--seed`.
- The GPU fits use bf16 autocast, which is not bit-reproducible in general. A refit is therefore a new draw of the same
  model.

The closest records we have:
- Pooling 54 instead of 27 sequence fits changed the 2025 held-out RMSE by −0.105 s [−0.238, +0.011].
- Using three member seeds instead of one changed it by −0.069 s.

We therefore expect a refit to move the 2025 out-of-fold RMSE by about a tenth of a second, not by seconds. This is our
estimate, not a measurement.

---

## 3. L2: from the raw challenge files

Run the subsections in order. Everything except the adsb.lol extraction (§3.5) runs on the laptop. The adsb.lol
extraction needs outbound HTTPS from the compute nodes.

### 3.1 Challenge cache

```bash
python -c "from prc.io import load; load('ranking', columns=['MVT_ID_mvt'])"   # builds cache/{training,ranking}.parquet
```

The first `load()` builds both cached files and checks their row counts (§0.2).

### 3.2 METAR

`prc/weather.py` (`download()`) fetches the ten airports' METARs from the Iowa Environmental Mesonet ASOS archive. It
needs no key and covers 2025-01 to 2026-08, one observation every 30 minutes, into `cache/external/metar/`. The service
throttles with HTTP 503 when called too quickly. The function retries with back-off and caches what it gets, so do not
delete that cache to fix a downstream bug. `exp65 --stage components` calls `download()` itself if the cache is missing.

### 3.3 ATFM daily delays

**No script in the repository writes `cache/external/atfm_daily.csv`.** We extracted it interactively on 2026-09-21. This
is the source and the procedure; we checked the procedure on 2026-10-06.

- **Source.** EUROCONTROL, *Airport arrival ATFM delay*:
  <https://www.eurocontrol.int/performance/data/download/xls/Airport_Arrival_ATFM_Delay.xlsx>.
  - We downloaded it on 2026-09-21 (HTTP 200, 97,452,493 bytes, SHA-256
    `1bbdb726130e97f99c55f7e61fa211a3ad70c61c1e40042e0319cb99efdecca2`).
  - Its META sheet gives release date 2026-09-15, version V.1.0.
  - EUROCONTROL updates the file at the same URL, so a fresh download will differ.
- **Extraction.** Take sheet `DATA`, the ten airports, and dates 2025-01-01 to 2026-08-31. Write one row per airport-day,
  with the arrivals `FLT_ARR_1` and one column per delay-cause code (`DLY_APT_ARR_<code>_1`, empty cells set to 0). The
  result has 6,080 rows.
- **Verification.** On 2026-10-06 the code below, run on the stored workbook, rebuilt our CSV **byte for byte**: SHA-256
  `ca14ab70eb7e68c72315c19b6e6c86b593283ca2ddc85a6a630b0e93800fd16c`. It took 98 s and 0.22 GB.

```python
# Usage: python atfm_extract.py atfm.xlsx cache/external/atfm_daily.csv
import sys, hashlib, datetime as dt
import openpyxl, pandas as pd

APTS = {'EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM'}
LO, HI = dt.datetime(2025, 1, 1), dt.datetime(2026, 8, 31)
src, dst = sys.argv[1], sys.argv[2]
rows = openpyxl.load_workbook(src, read_only=True)['DATA'].iter_rows(values_only=True)
ix = {h: i for i, h in enumerate(next(rows)) if h}
causes = sorted(h[12:-2] for h in ix if h.startswith('DLY_APT_ARR_') and h != 'DLY_APT_ARR_1')
keep = [[r[ix['FLT_DATE']].date().isoformat(), r[ix['APT_ICAO']], r[ix['FLT_ARR_1']]]
        + [r[ix[f'DLY_APT_ARR_{c}_1']] or 0 for c in causes]
        for r in rows if r[ix['APT_ICAO']] in APTS and r[ix['FLT_DATE']] and LO <= r[ix['FLT_DATE']] <= HI]
df = pd.DataFrame(keep, columns=['date', 'apt', 'flt_arr'] + [f'dly_{c}' for c in causes])
df.sort_values(['date', 'apt']).to_csv(dst, index=False, lineterminator='\r\n')
print(len(df), 'rows, sha256', hashlib.sha256(open(dst, 'rb').read()).hexdigest())
```

This is arrival ATFM delay by day and airport. It is not a taxi-time statistic and contains no challenge labels.

### 3.4 Feature tables built on the laptop

```bash
python experiments/exp65_submit_v8.py --stage components   # cache/final/lgb_components.npz (needs METAR + ATFM)
python experiments/v11/v11_features.py                     # cache/v11/feat_{training,ranking}.parquet
python experiments/exp53_v5_oof.py                         # cache/exp53_oof.npy
python experiments/exp62_screen.py                         # cache/exp62_cand_{training,ranking}.parquet, exp62_pool_training.parquet
python experiments/exp62_cv.py                             # cache/exp62_oof.npy, cache/exp62_spec_P.npy
```

The order matters:
- `v11_features.py` checks the ranking row order against `lgb_components.npz`, so that file must exist first.
- `exp62_cv.py` prints a sanity comparison against `exp53_oof.npy`, so that file must exist first.

`v11_features.py` has two safeguards:
- It asserts that the departure frame it builds from has no off-block or taxi-time column.
- It builds every event-based feature month by month, so no ranking feature crosses between January and July.

### 3.5 adsb.lol ground movements

- **Source.** `github.com/adsblol/globe_history_2025` and `globe_history_2026`: one release per day, a tar split into
  parts. Licence ODbL 1.0.
- **Days.** `experiments/adsb/adsb_days.py` lists the 429 days used:
  - 2024-12-31 to 2025-12-31 (the eve of 1 January is needed for early-morning departures);
  - 2026-01-01 to 2026-01-31;
  - 2026-06-30 to 2026-07-31.
- **Releases.** `adsb_download.py` picks the release that the archive's own `PREFERRED_RELEASES.txt` names for each day.
  The listed days add up to **1.331 TB**. That volume is streamed, never stored.

```bash
sbatch experiments/adsb/slurm_stage1_array.sbatch                                     # 429 tasks, at most 8 at a time
sbatch --dependency=afterany:<stage-1 job id> experiments/adsb/slurm_stage2_features.sbatch
```

- **Stage 1** (`adsb_extract_day.py`) streams one day's tar without unpacking it to disk. It keeps the trace points within
  12 km of one of the ten airports and below 6,000 ft above the field (or flagged on the ground), and writes
  `cache/adsb_lol/points/<date>.parquet`.
  - On the laptop, one July 2025 day of 3.5 GB took 149-176 s to download at 17-24 MB/s and 50-70 s to parse with 6
    workers, with a peak of 0.86 GB.
  - On the cluster, 60 days were done after about 20 minutes, and all 429 completed the same day.
  - The compute nodes need outbound HTTPS to github.com. For a laptop-only run there is `run_days_local.sh` (§5).
- **Stage 2** (`adsb_events.py`, `adsb_build_features.py`) detects take-offs, matches them to departures and writes
  `cache/adsb_lol/features_{training,ranking}.parquet` (35 `adsb_*` columns, keyed by `MVT_ID_mvt`). Events for 417 of
  the 429 days were built in the first 16 minutes.
  - No feature uses a label. The stand table is the median adsb.lol push-back position per `STAND_mvt`.
  - Altitudes are corrected for the per-airport-hour pressure offset seen on the runway.
- **Not used.** The "ADS-B v2" scripts (`adsb_events_v2.py`, `adsb_build_features_v2.py`) were evaluated and not
  adopted. The shipped members do not need them.

### 3.6 Route-D corrector chain

```bash
python experiments/major/m00_base.py               # cache/major/dep_{training,ranking}.parquet
python experiments/major/m01_anatomy.py            # cache/major/v9s_oof.npy (2025 out-of-fold stand-in for v9)
python experiments/major/m11_state_features.py     # cache/major/m11_state_{training,ranking}.parquet
python experiments/major/m11d_state_routeD.py      # cache/major/m11d_pred_trailD.npy
python experiments/major/m11f_candidate_arrays.py  # cache/major/m11f_candidates.npz (reads outputs/..._v9.parquet)
```

`m01_anatomy.py` reads `cache/blend_oof_mix_nn3.npy`. No tracked script writes that file; we saved it interactively on
2026-09-23. Its two rows come from the early model-family study (`cluster/study_task.py`):
- row 0: the out-of-fold prediction of the one-seed LightGBM mixture (family `mix`, seed 2);
- row 1: the mean of the three network seeds' out-of-fold mixtures.

`m11f_candidate_arrays.py` reads the shipped v9 submission file. v9 came out of a chain of builders, each of which checks
identity with the shipped file of the step before:
1. `experiments/exp54_submit_v6.py` built v6.
2. `experiments/exp64_submit_v7.py` built v7.
3. `experiments/exp64_export_spec.py` exported the v7 specialist (`cache/final/spec_x_ranking.npz`).
4. `cluster/nn_final.sbatch` fitted three GPU network seeds (`cache/nn_final/`).
5. `python experiments/exp65_submit_v8.py --stage assemble --specialist x --w-abc 0.65 --w-d 0.5 --version 9` built v9.

The scripts are all in the repository. We have not re-run this chain since 2026-09-23.

### 3.7 Then L1 and L0

With the tables of §2.1 in place, run §2 and then §1.

### 3.8 Inputs that are deliberately not part of the pipeline

- **EUROCONTROL's published monthly airport taxi-out figures for January and July 2026.** They summarise the evaluation
  labels. We identified them and saw them during scouting on 2026-09-21 and 2026-09-23, and deliberately did not use
  them. No code in this repository reads them.
- **OpenSky Trino and state vectors.** Not used.
- **OPDI.** Evaluated with event counts only; not used in the model.
- **Our own leaderboard client.** `tools/leaderboard.py` reads the public leaderboard API. Nothing in the pipeline uses it.

---

## 4. Frozen inputs, stated plainly

These are the places where the shipped model depends on something a clean L1/L2 run would not recreate by itself.

1. **The route-D corrector depends on v9.** In `cache/major/m11f_candidates.npz`, `corr = clip(trailD_full, 0) − v9`, where
   `v9` holds the route-D values of the v9 submission (uploaded 2026-09-23).
   - The corrector was trained on 2025 against `v9s_oof.npy`, a reconstruction of v9's out-of-fold predictions that
     depends on the interactively saved `blend_oof_mix_nn3.npy` (§3.6).
   - v9 is therefore a frozen input of the corrector, though not of the assembly formula.
   - Reproducing the corrector from raw data means rebuilding v9's components first (§3.6).
2. **The corrector and its strength on route D were partly chosen with the public leaderboard.**
   - The corrector entered the model at full strength through v10 (2026-09-24), a model-level leaderboard test: its
     target was a disruption that month-wise cross-validation cannot see.
   - The factor 0.5 on route D outside zone P dates from v18 (2026-09-26). Both "drop it" and "halve it" passed our 2025
     rule. The rule's pre-registered tie-break pointed to dropping. Halving was chosen after a label-free 2026 estimate
     had been calibrated against the live v9 → v10 score change.
   - Both choices are still in the shipped model: full strength in zone P, half strength on route D outside it.
     [VALIDATION.md](VALIDATION.md) has the full account.
3. **The no-context twin finals of v27 and v28 were CPU runs, not the documented GPU job.** The cluster was unreachable,
   so on 2026-10-02 we trained `cache/seq_final/seqnc__seed0{0,1,2}.npy` on the laptop.
   - Command: `python cluster/seq_final.py --seed N --no-context --device cpu --threads 6`, in fp32.
   - The documented route is `sbatch --export=ALL,SEQ_FAMILY=seqnc cluster/seq_final.sbatch`, and the out-of-fold twin
     predictions from which the U+T weights were fitted came from GPU runs with bf16 autocast. It is the same code and
     model on a different numerical path.
   - Label-free check: on routes A/B/C the U+T term has an sd of 13.4 s on 2025 out-of-fold predictions and 16.2 s on
     2026 *(label-free)*, +21%. That is the same ratio as the raw difference between the two networks.
4. **The production tail classifier is the v6 configuration.** `lgb_components.npz` (exp54's LightGBM parameters, five
   seeds) has not been refitted since 2026-09-23. Every later change was layered on top of it.
5. **External snapshots:**
   - the ATFM workbook release of 2026-09-15 (hash in §3.3);
   - the TabICL checkpoint `tabicl-classifier-v2-20260212` (hash in §0.1);
   - the adsb.lol releases named by `PREFERRED_RELEASES.txt` when we fetched them (2026-09-25);
   - the METAR archive as fetched on 2026-09-21.
6. **The LOBT window constant.** 3606 s was read off the 2025 training file. Over 2,062,577 departures with a
   flight-plan LOBT, the off-block time lies within ±3606 s of LOBT with zero exceptions.
   - We take it to be a property of how the dataset's movement-to-flight join was built, not operational knowledge.
   - The idea was adopted on 2026-10-06 from the public repositories
     [EnioAguiar/prc-taxiout-2026](https://github.com/EnioAguiar/prc-taxiout-2026),
     [javidmardanov/PRC-Data-Challenge-2026](https://github.com/javidmardanov/PRC-Data-Challenge-2026) and
     [joris-decombe/taxiout](https://github.com/joris-decombe/taxiout). We took only the idea; the code in `assemble.py`
     is our own.
   - The decision to ship it rests on the 2025 data alone.
   - The rule and the code were committed (`618ba86`) before v28 was built, but after the rule's 2025 effect had been
     measured. The commit order is consistent with that account.

---

## 5. Known gaps

- **No tests.** `tests/` is empty. The checks that exist are inside the scripts: `assemble.py --check`, the `--lobt-window`
  self-test, the row-count assertions in `prc/io.py`, input and config checks in every cluster job, and validation of
  every written submission in `prc/submit.py`.
- **`cache/external/atfm_daily.csv` has no writer in the repository.** §3.3 gives the source, the procedure and code that
  reproduces it byte for byte from the stored workbook.
- **`cache/blend_oof_mix_nn3.npy` has no writer in the repository.** It was saved interactively. §3.6 says what it holds.
- **`experiments/strategy/s21_tfm_P_final.py` imports TabPFN unconditionally**, although `assemble.py` uses only its TabICL
  output. To run it as written, install `tabpfn==9.0.0` (commented out in `requirements.txt`).
- **The data location is fixed** to the parent directory of the clone (§0.2). There is no environment variable for it.
- **The cluster environments were not pinned.** `cluster/setup_env*.sh` install without versions, and the resulting
  versions were not recorded. Only the laptop environment is pinned.
- **The no-context twin finals took a different route** (CPU fp32 instead of GPU bf16, §4 item 3).
- **Absolute paths remain off the reproduction path.** The scripts named in this document derive their paths from their
  own location; a few of them (`prc/io.py`, `prc/config.py`, `exp62_screen.py`, `exp62_cv.py`, `exp64_submit_v7.py`,
  `exp64_export_spec.py`) still show the development machine's interpreter path in a usage comment, which the code does
  not read. 38 exploratory scripts under `experiments/` still use absolute paths of the development machine in their
  code: in `ideas_outside2/`, `ideas_inside/`, `review_*/`, `deice_0926/`, `discord_eval/`, `ground_sources/` and
  `plan_1001/live_proj.py`. So does the interpreter path in the laptop driver `experiments/adsb/run_days_local.sh`.
  None of them is needed for L0-L2.
- **L1 and L2 have not been re-run end to end.** The stored components were produced by the runs described here, over
  2026-09-21 to 2026-10-02.

---

## 6. Hardware used

| machine | specification | used for |
|---|---|---|
| laptop | AMD Ryzen 7 PRO 7840U (8 cores, 16 threads), 29 GiB RAM visible to the OS, no CUDA GPU; Ubuntu 24.04, CPython 3.12.3 | feature building (§3.4, §3.6), the production tail classifier, TabICL, the route-D corrector, every blend and gate evaluation, the no-context twin finals of v27/v28, all assembly; adsb.lol prototyping. Every run was memory-capped (`systemd-run`, 6-12 GB) |
| `<cluster>`, partition `gpu` | one node with 8 × NVIDIA L40S, 64 CPUs, 2 TB RAM | sequence-model and network folds and finals; the CPU members' folds and finals (8 CPUs per task) |
| `<cluster>`, partition `any` | CPU nodes, 1 h job limit | adsb.lol stage 1 (4 CPUs, 6 GB per day) and stage 2 (8 CPUs, 16 GB) |

---

## 7. How this was built

**Who decided what.** Every decision was directed by the team: which ideas to test, which data to use or decline, and
which file to upload. Code and analysis were written with AI assistance; the commits that came out of it carry
`Co-Authored-By` trailers.

**How changes were accepted.** Changes were accepted on 2025 leave-one-month-out evidence, in most cases against a
decision rule written down before the deciding run. Commit subjects mark these rules as pre-registered. The exceptions:
- v1-v5 predate written rules; v1/v2 were a paired model-level leaderboard test.
- v7 and v10 were model-level leaderboard tests.
- v18's corrector factor was settled with a live check (§4 item 2).
- v25 overrode our own rule, scored 1.07 s worse, and was dropped.
- The v28 window rule was written down after its 2025 effect had been measured.

[VALIDATION.md](VALIDATION.md) gives the full account, including how the public leaderboard was and was not used.
