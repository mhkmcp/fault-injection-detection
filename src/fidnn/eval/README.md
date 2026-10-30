# fidnn.eval

Fault splits, the leakage guard, metrics and the results build. SPEC §7, §9. The `/report-results`
skill covers the reporting rules.

| File | Role |
|---|---|
| `splits.py` | `fault_dev` / `fault_test` by injection **instance**; `fault_gen` tagged configs |
| `leakage.py` | The guard that fails a results build instead of reporting a leaked number |
| `metrics.py` | TPR at calibrated thresholds, precision/recall/F1, AUROC + AUPR, per track |
| `stats.py` | Bootstrap CIs, paired-bootstrap differences, the ≥ 5-seed rule |
| `localisation.py` | H4: Spearman ρ, bucket confusion, propagation depth, pre-add share |
| `run.py` / `report.py` | `fidnn eval`, `fidnn report` → `docs/M5_results.md` |
| `ablations.py` | §9.6 items 1–3 and 5–7: K sweep, tap position, block drops, training budget, kernel, transfer |
| `granularity.py` | `fidnn granularity`: single-layer (D1) vs per-layer late fusion (D3) vs tap subsets vs joint (D2), the H2 test, → `docs/M5_granularity.md` |

## What the guard refuses to build

`leakage.assert_clean` raises before any table is written when: an injection instance is in both
`fault_dev` and `fault_test`; a `fault_gen` configuration (stratum `sign`, budget 4) appears in
`fault_dev`; `fault_test` is missing any bucket, stratum or budget of the grid; a fault record id
turns up in a clean split; or the Kaggle/canonical pixel-hash check found collisions.

## Reporting rules this package enforces

- **Tracks S and H never merge.** Every metric row carries its track.
- **No aggregate-only table.** `metrics.breakdowns` produces the mandatory per-category, per-stratum,
  per-bucket, per-budget, per-mode, per-precision and per-attacker rows.
- **`fault_gen` is its own row**, never folded into the headline.
- **A CI that includes 0 is "no detected difference"**, never a win — `stats.paired_difference`
  returns that verdict string, and the report prints it verbatim.
- **Fewer than 5 seeds is not reportable** (C4): the report prints a warning banner instead of
  quietly showing a single-run number.
- **Detection is read with overhead** (C5): the results page links the overhead record rather than
  standing alone.

## Ablations that must not cheat

The §9.6(1) K sweep selects taps by **clean-validation FPR**, never by fault TPR — picking taps on
fault data would be a C2 violation, so `greedy_tap_selection` takes no fault argument at all. Block
ablations zero a block rather than removing its columns, so widths stay comparable across rows.
Transfer (§9.6(7)) truncates both models to their shared tap width and reports that truncation with
the number, since M2 and M3 expose different tap counts.

CIFAR-10-C (§9.6(8)) is loaded by `fidnn.data.cifar10c` as a **clean-input shift** and is never
combined with faults; the per-type (15) and per-family (4) breakdowns are mandatory, not optional.

## Granularity comparison (`granularity.py`)

One fixed design list, all fitted in one process so they compare like for like:

| Granularity | SPEC | Designs |
|---|---|---|
| `single` | §6.2 D1 | one per tap: that tap's SVDD from a shared `D1`, judged on its own |
| `per_layer` | §6.2 D3 | `D3` max / mean / fpr_weighted over the same `D1` |
| `subset` | §9.6(1)(2) | fused SVDD (`D2` on a tap subset): position thirds, backbone, leave-one-out, greedy K = 2–4 by clean_cal FPR |
| `joint` | §6.2 D2 | fused SVDD over all taps |

- **No design is chosen with faults (C2).** Subsets come from structure or from
  `greedy_tap_selection`, which takes no fault argument. `--exhaustive` adds every other subset
  with `oracle=True`: a map of the space for the UI and appendix, kept out of every table and
  verdict.
- **Paired differences resample injections, not probes.** The probes of one injection share a
  corrupted model, so `cluster_paired` bootstraps injection instances; resampling probes would
  give intervals several times too narrow (tested). The clean side is held fixed.
- **AUROC is bootstrapped through per-sample contributions** (`auroc_contrib`: share of clean
  scores below each fault score), whose mean is exactly the AUROC, so 2,000 resamples cost a
  weighted mean each.
- **H2 is tested as SPEC §1 words it** — fused TPR@1 % against the best single layer — per track.
  The best single layer is picked on the evaluation data, which favours it; the verdict is
  therefore conservative for H2 and is never a design choice.
- **Latency** is per image, normalise + SVDD + combine, batch 1 and 32, after warm-up. A
  single-layer design scores only its own tap. Feature-extraction cost per K comes from the M-6
  record and is reported next to it (C5).
- The `D1` row of `fidnn eval` scores the **max of raw per-tap scores**, which sit on different
  scales per tap. That is neither §6.2 D1 (each tap alone) nor D3 (CDF-mapped); this module reports
  both properly. Keep it in mind when reading `*_headline.parquet`.
- Without fault features (e.g. FP32 before an FP32 sweep) only the clean side is written: FPR at
  each α and scoring cost.

## Decisions

- **Spearman is rank-based**, so H4 correlates the injected bucket index directly with the
  first-alarm tap index; no scaling between the two axes is needed or applied.
- **Output-only baselines** are scored from the per-probe scores recorded during the sweep and the
  matching clean scores from extraction, so both sides see the same probes and the same threshold
  rule.
