# Claim-evidence ledger - m3-procedural-v0.1.0

Every quantitative claim below is conditional on the frozen procedural
population and proxy-fault contract. Shared provenance for every row:

- scientific revision `e8595fe428bcb9dfb269069e4b02972aff10f4ee`;
- named hardware `Apple M3 Pro`;
- artifact set `a870f05a372f727a2dbca432079299c58b3bea04a0053d1fc923ebf232f95cef`; and
- pointwise paired complete-sequence bootstrap inference.

| Claim family | Exact manifest / record selector | Figure | Validation evidence |
|---|---|---|---|
| Signed fusion benefit/harm and crossover for `procedural-lidar-y-bias` | manifest `e5a4aa3ddf9832cd8cc88eb0be87151bfb44efbf99363f826f3b360c81960056`; `records/procedural-lidar-y-bias/aggregate-metrics.ndjson` where `metric_name=fused-minus-healthy`, all severities and directions; all rows in `records/procedural-lidar-y-bias/crossovers.ndjson` | `figures/fusion-delta-curves.svg`, panel 1 | `records/procedural-lidar-y-bias/procedural-validation.json`, all gates |
| Signed fusion benefit/harm and crossover for `procedural-camera-noise-correctly-reported` | manifest `4359f4f5cc172017b4cbf2eb5d7470692a25846501066d999ae416b56caa1add`; `records/procedural-camera-noise-correctly-reported/aggregate-metrics.ndjson` where `metric_name=fused-minus-healthy`, all severities and directions; all rows in `records/procedural-camera-noise-correctly-reported/crossovers.ndjson` | `figures/fusion-delta-curves.svg`, panel 2 | `records/procedural-camera-noise-correctly-reported/procedural-validation.json`, all gates |
| Signed fusion benefit/harm and crossover for `procedural-camera-noise-underreported` | manifest `900b4893ca33eb8ce84d10cc14a3e350e6b07de4783641c4212b4ee0337c549d`; `records/procedural-camera-noise-underreported/aggregate-metrics.ndjson` where `metric_name=fused-minus-healthy`, all severities and directions; all rows in `records/procedural-camera-noise-underreported/crossovers.ndjson` | `figures/fusion-delta-curves.svg`, panel 3 | `records/procedural-camera-noise-underreported/procedural-validation.json`, all gates |
| Signed fusion benefit/harm and crossover for `procedural-camera-calibration-x` | manifest `463637c5dc2b8a8135e40dab4b23e1a3fd61b9475364a523b999e61d1787cdce`; `records/procedural-camera-calibration-x/aggregate-metrics.ndjson` where `metric_name=fused-minus-healthy`, all severities and directions; all rows in `records/procedural-camera-calibration-x/crossovers.ndjson` | `figures/fusion-delta-curves.svg`, panel 4 | `records/procedural-camera-calibration-x/procedural-validation.json`, all gates |
| Signed fusion benefit/harm and crossover for `procedural-camera-calibration-yaw` | manifest `b6ec17bd745483af6863857db93376e5d560ccc9d4a6cfffc13e027e2e4289fa`; `records/procedural-camera-calibration-yaw/aggregate-metrics.ndjson` where `metric_name=fused-minus-healthy`, all severities and directions; all rows in `records/procedural-camera-calibration-yaw/crossovers.ndjson` | `figures/fusion-delta-curves.svg`, panel 5 | `records/procedural-camera-calibration-yaw/procedural-validation.json`, all gates |
| Signed fusion benefit/harm and crossover for `procedural-camera-timestamp-offset` | manifest `292d47f2711223382cca48e229a2cb7a1bd6ebfe392bb36d0730505b5e9f3d57`; `records/procedural-camera-timestamp-offset/aggregate-metrics.ndjson` where `metric_name=fused-minus-healthy`, all severities and directions; all rows in `records/procedural-camera-timestamp-offset/crossovers.ndjson` | `figures/fusion-delta-curves.svg`, panel 6 | `records/procedural-camera-timestamp-offset/procedural-validation.json`, all gates |
| Dropout availability, undefined rate, and conditional localization loss | manifest `79ae8c67ff9994b7d6b764e8ef8b7c2185c3cb4871b6489d64bb4385f786022a`; `records/procedural-camera-dropout/aggregate-metrics.ndjson`, all methods, probabilities, and all source metrics; figure presentation order is coverage, undefined-output-rate, conditional-matched-center-mse | `figures/dropout-controls.svg`, panels 1-3 | `records/procedural-camera-dropout/procedural-validation.json`, dropout and all other gates |
| Common-mode absolute loss and disagreement blind spot | manifest `1b78059d62b016ca8a25cc23d22a73576ef1e61742c08f35394e9ad273c06d3c`; `records/procedural-common-mode-x-fov-edge/aggregate-metrics.ndjson`, all methods, severities, and directions | `figures/common-mode-control.svg` | `records/procedural-common-mode-x-fov-edge/procedural-validation.json`, `common_mode_validation` and all other gates |
| Deterministic scientific repeat | every indexed member pair in `evidence/repeat-verification.json`; no run selection | no derived result figure | `evidence/matrix-validation.json` and `evidence/repeat-verification.json` |

All aggregate and crossover selectors mean every matching source row
in frozen order. No statement may omit an observed, not-observed,
undetermined, negative, or contrary outcome. Physical severity units
are never pooled.

The strongest supported public wording is: “A deterministic CPU
estimator-output benchmark measured matched-center fusion behavior
under declared procedural geometry and proxy metadata faults.”

Unsupported claims include real sensor-noise transfer, detector
robustness, physical fault tolerance, planning or safety benefit,
production readiness, and fleet generalization.
