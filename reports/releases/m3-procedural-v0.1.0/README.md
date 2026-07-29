# Fusion Fault Bench — m3-procedural-v0.1.0

This is the aggregate-only evidence release for the frozen CPU M3
procedural estimator-output benchmark. It publishes every matrix entry,
all 429 aggregate rows, all 10 crossover rows, complete validation
summaries, both run records per experiment, and three deterministic
figures. No result, method, severity, direction, or outcome was selected
after inspection.

## What is included

| # | Experiment | Fault axis | Aggregate rows | Crossovers | Validation |
|---:|---|---|---:|---:|---|
| 0 | `procedural-lidar-y-bias` | `y` (m) | 66 | 2 | PASS |
| 1 | `procedural-camera-noise-correctly-reported` | `xy` (std-scale) | 30 | 1 | PASS |
| 2 | `procedural-camera-noise-underreported` | `xy` (std-scale) | 30 | 1 | PASS |
| 3 | `procedural-camera-calibration-x` | `x` (m) | 66 | 2 | PASS |
| 4 | `procedural-camera-calibration-yaw` | `yaw` (rad) | 66 | 2 | PASS |
| 5 | `procedural-camera-timestamp-offset` | `time` (s) | 66 | 2 | PASS |
| 6 | `procedural-camera-dropout` | `availability` (probability) | 72 | 0 | PASS |
| 7 | `procedural-common-mode-x-fov-edge` | `x` (m) | 33 | 0 | PASS |

The only omitted scientific member is `sequence-metrics.ndjson`:
71,700 generated sequence rows remain local. Each experiment retains
the omitted member's exact source byte length, SHA-256, and an
independently manifest-derived record count in `release-summary.json`.
The public validator authenticates that commitment but cannot
recompute aggregates or inspect omitted rows without regenerating the
source artifacts.

## Crossover outcomes

These are signed, per-physical-axis stress-test results. An observed
crossover is not a physical sensor tolerance.

| Experiment | Direction | Status | Point estimate | 95% interval / censoring | Unit |
|---|---|---|---:|---|---|
| `procedural-lidar-y-bias` | negative | observed | 1.3968849130460446 | [1.389999281805564, 1.4037114600964853] | m |
| `procedural-lidar-y-bias` | positive | observed | 1.3942566175140465 | [1.3874881459452584, 1.4008480010938364] | m |
| `procedural-camera-noise-correctly-reported` | increase | not-observed | undefined | [4.0, +∞) | std-scale |
| `procedural-camera-noise-underreported` | increase | observed | 1.4475323333484358 | [1.4258107359522219, 1.468957584519013] | std-scale |
| `procedural-camera-calibration-x` | negative | observed | 1.383889433246956 | [1.345955729458017, 1.4209496094435923] | m |
| `procedural-camera-calibration-x` | positive | observed | 1.4237279263474245 | [1.3806381804922199, 1.4627830366308743] | m |
| `procedural-camera-calibration-yaw` | negative | observed | 0.03521127235349965 | [0.03409377684383041, 0.036400627447693006] | rad |
| `procedural-camera-calibration-yaw` | positive | observed | 0.03408296916867806 | [0.032988505139442084, 0.03507853034101522] | rad |
| `procedural-camera-timestamp-offset` | negative | observed | 0.3535791578648241 | [0.3417235221585074, 0.3662034173717133] | s |
| `procedural-camera-timestamp-offset` | positive | observed | 0.36708164425734324 | [0.3527324507856683, 0.3800755226303754] | s |

Observed, not-observed, undetermined, negative, and contrary outcomes
are retained. The fused-delta figure shows every one of the 54 unique
signed aggregate rows, their pointwise intervals, and all 10
direction-specific PAVA fits in native units.

![M3 signed fusion delta curves](figures/fusion-delta-curves.svg)

## Availability and common-mode controls

The dropout figure contains all 72 method-by-probability aggregate
rows. It displays coverage and undefined-output rate before
conditional matched-center loss. Undefined conditional loss is marked
and never imputed as zero. Dropout has no crossover estimand.

![M3 dropout controls](figures/dropout-controls.svg)

The common-mode figure contains all 33 camera-only, LiDAR-only, and
fixed-fusion absolute-loss rows. It shows those result-derived curves
alongside the independently validated camera-LiDAR
disagreement-invariance blind spot. Common mode has no
healthy-reference crossover.

![M3 common-mode control](figures/common-mode-control.svg)

## Provenance and repeat evidence

- Scientific source revision: `e8595fe428bcb9dfb269069e4b02972aff10f4ee`
- Lockfile SHA-256: `ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`
- Artifact-set SHA-256: `a870f05a372f727a2dbca432079299c58b3bea04a0053d1fc923ebf232f95cef`
- CPU: `Apple M3 Pro`
- Public CI: [GitHub Actions run 30456056647](https://github.com/6ickomod3/fusion-fault-bench/actions/runs/30456056647) on `e8595fe428bcb9dfb269069e4b02972aff10f4ee`; smoke only, not release evidence
- Independent adversarial results review: `pass`; [included report](evidence/results-review.md), tracked source `docs/reviews/m3-results-review.md`; artifact set `a870f05a372f727a2dbca432079299c58b3bea04a0053d1fc923ebf232f95cef`; included byte-for-byte
- Primary complete-matrix measurement: 1218.376157041639 s wall, 369000448 bytes peak RSS
- Repeat complete-matrix measurement: 1258.861101000104 s wall, 389431296 bytes peak RSS
- Indexed scientific comparisons: 48
- Indexed scientific mismatches: 0

Both measurements are reported; neither was selected as preferred.
Wall time and peak memory are self-reported by the tracked `wait4`
driver and cannot be independently recomputed from this package.
Distinct paths, inodes, logical commands, volatile run records, and
completion markers are consistency evidence, not cryptographic proof
that two executions occurred. The Git-bound official identity and source
revision are the local provenance boundary. The CI and review entries
are tracked attestations linked to public or human-readable references;
the offline validator does not query GitHub or authenticate the reviewer.

## Claim boundary

M3 measures matched-center estimator-output behavior under declared
procedural proxy faults. It does not evaluate a detector, raw sensor
noise, association, planning, safety, production readiness,
fleet-scale behavior, or real-world sensor tolerance. M3 uses no
nuScenes data. The CI smoke matrix is explicitly not release evidence.

See `claim-evidence.md` for exact selectors and `verification.md` for
the strict validation command.
