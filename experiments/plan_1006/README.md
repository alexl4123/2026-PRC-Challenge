# experiments/plan_1006/: endgame planning round that led to v28 (2026-10-06)

Archived as run. Not on the production path; `assemble.py --lobt-window` is the shipped implementation of the window.

## Provenance

- **Written on 2026-10-06**, after v27 (live 243.1478) and before v28, by a multi-agent Claude Code planning session
  (analysts for field forensics, score levers, documentation, rules and public repositories, plus skeptics) launched
  by the author. Kept in a git-ignored scratch folder (`cache/plan_1006/`) and copied here on 2026-10-13.
- **Copied:** all 22 `*.py` files, unchanged. **Not copied:** outputs and data files, including `lb_rows.json`, a
  dump of the public leaderboard API (every team's public scores and upload times) that `pattern.py` and
  `ff/ff_rankat.py` read. Nothing was excluded for any other reason.
- **Paths are as run** (absolute paths of the development machine); the scripts will not run unchanged.

## Contents

| script | what it does | data |
|---|---|---|
| `lobt.py`, `lobt1.py`-`lobt4.py`, `lobtwin.py`, `lobt_oof.py`, `lobt_oof2.py`, `lobt2026.py` | the LOBT ±3,606 s window: exactness on 2025, value on the v24 out-of-fold predictions, rows v27 would move in 2026 and the bound on v27's live score | 2025 labels; 2026 label-free; v27's live score |
| `skep/lobt_check.py`, `skep/oof_proj.py` | skeptic re-checks of the window claim and of its 2025 out-of-fold effect | as above |
| `sl/v27proj.py`, `sl/proj_eval.py` | the window applied to v27 (2026, label-free) and evaluated on 2025 | as above |
| `sl/lirfrow.py` | inspection of the 2026 departures that take off more than 40,000 s after their LOBT, i.e. the Rome July row that carries 96% of the window's bound (all clocks of the row, label-free), with their July 2025 analogues | 2026 input columns of one row; 2025 labels |
| `sl/arvt.py`, `sl/nonm.py`, `sl/dst_excl.py` | other clocks as windows; departures without an NM record; first form of the destination-record exclusion (continued in `../hunt_1006/`) | 2025 labels; 2026 label-free |
| `ff/ff_lobt_month.py`, `ff/ff_lobt2026.py` | window statistics per month, 2025 against 2026 (label-free) | |
| `ff/ff_rankat.py` | board ranks at given cut times | public leaderboard dump |
| `pattern.py` | upload-timing profile of the top 40 teams (number of uploads, new bests, bursts of three uploads within 10 minutes, uploads just after midnight UTC) | public leaderboard dump |
| `agg_value.py` | value of a perfect airport-month level correction on our 2025 out-of-fold predictions (−0.14 s on January + July 2025) | 2025 only |

`pattern.py` profiled the public upload timing of other teams. Its result was read as circumstantial only, no team was
named or reported, and it fed no model or upload decision. `ff_rankat.py` and `pattern.py` are the only scripts here
that read other teams' data, and that data is the public leaderboard.

The window idea came from public competitor repositories (credited in the [README](../../README.md#credits)); the
decision to ship it rests on the 2025 measurements in these scripts.
