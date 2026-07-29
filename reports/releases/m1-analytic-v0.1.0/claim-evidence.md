# M1 Claim–Evidence Ledger

This ledger maps every public M1 quantitative statement to a committed
machine-readable record and a presentation artifact. All selectors are applied
to files beneath this release directory.

## Shared provenance

The source revision for every claim is
[`524c8f70ece3eca2e61796165b23ffe51baadfbc`](https://github.com/6ickomod3/fusion-fault-bench/commit/524c8f70ece3eca2e61796165b23ffe51baadfbc).
Every `run.json` records `source_dirty=false`, package version `0.1.0`, and
lockfile SHA-256
`ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`.
The common named environment is Apple M3 Pro, arm64, 11 logical CPUs,
19,327,352,832 memory bytes, Darwin 24.5.0, and Python 3.12.13.

The experiment identity keys used below are:

| Key | Experiment | Record directory | Manifest SHA-256 | Artifact SHA-256 | Run ID |
|---|---|---|---|---|---|
| B | Signed camera x-bias | `records/analytic-camera-x-bias-a603d090f77a/` | `a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e` | `3717c2b3fdce9e9f2bc43463434fde28a4d24dd9ca1d72e451b3d9d5273c2959` | `run:cc67cf35ac9b74116cbb2f39c934bbfddb041036c58194722090979549460687` |
| C | Correctly reported camera noise | `records/analytic-camera-noise-correctly-reported-3ea7ffc2949c/` | `3ea7ffc2949cf99f20d20ec18844f0b8dc3b3ebb81e13e926f7440b7c5084176` | `51abb5043ddffd633c0fa81ea5d69dc6c1246d092185f22261f183915f911467` | `run:4d601cfe04c83839a25088a061ab9a6d4b3c29ffcec71ea4f6ce64c3343f4340` |
| U | Underreported camera noise | `records/analytic-camera-noise-underreported-9d26e1b33f1f/` | `9d26e1b33f1fd2e35b0de90703a960d2eba6bb26bd2219bce6f0bb82480f4ac4` | `8a3c2179e49cdc2ae994d9c791185a546788252710a0e80fca73cf39305165e7` | `run:a58e65ece1915d49c485f36ee478f97fce35c18763f7fa4cec2ffd44bd90b234` |

For each key, `manifest.json` commits experimental intent,
`source-payload-index.json` commits the five scientific payload-member hashes,
`source-success.json` commits the artifact and run-record digests, and
`run.json` supplies the source, environment, lock, command, and run identity.

## Result claims

| Claim | Exact public quantity | Machine-record selector | Figure | Required interpretation |
|---|---|---|---|---|
| Q1: design | \(N=200\), \(B=2000\), 95% pointwise paired-sequence bootstrap for B, C, and U | Each `manifest.json`: `$.source.sequence_count`, `$.evaluation.bootstrap.replicates`, `$.evaluation.bootstrap.confidence_level`, `$.evaluation.bootstrap.unit`, and `$.evaluation.bootstrap.resampling` | Both figures state N and B | This is a synthetic one-object analytic experiment, not a dataset sample. |
| Q2: negative x-bias | Population grid root `3.8282790927021715` m; continuous root `3.869066367512064` m; finite-sample root `3.2126457655205014` m; interval `[1.3103527307270806, 4.971038275465616]` m; \(q=1.0\); `observed`; tested maximum `8.0` m | B `analytic-validation.json`: `$.crossover_references[?(@.direction=="negative")]`; B `crossovers.ndjson`: record with `direction=="negative"` | `figures/bias-fused-minus-healthy.svg`, negative series | A controlled additive output-bias magnitude, not a physical calibration tolerance. |
| Q3: positive x-bias | Population grid root `3.8282790927021715` m; continuous root `3.869066367512064` m; finite-sample root `2.9641840935526584` m; interval `[1.128775473537445, 4.58019364069497]` m; \(q=1.0\); `observed`; tested maximum `8.0` m | B `analytic-validation.json`: `$.crossover_references[?(@.direction=="positive")]`; B `crossovers.ndjson`: record with `direction=="positive"` | `figures/bias-fused-minus-healthy.svg`, positive series | Population symmetry does not require identical finite-sample branch estimates. |
| Q4: correctly reported noise control | No population grid root through `4.0`; no finite continuous population root; finite-sample point root `3.539641384241362` std scale; no two-sided interval; \(q=0.6325=1265/2000\); `undetermined`; tested maximum `4.0` | C `analytic-validation.json`: `$.crossover_references[0]`; C `crossovers.ndjson`: `direction=="increase"` | `figures/noise-reporting-fused-minus-healthy.svg`, correctly reported series | The point-curve root is not an observed crossover. Mixed bootstrap support requires the published `undetermined` status. |
| Q5: correctly reported endpoint | At `severity.direction=="increase"` and `severity.magnitude==4.0`, \(D_H=0.0002931315153226384\) m² with pointwise 95% interval `[-0.0013976621161069004, 0.002087427419543242]` | C `aggregate-metrics.ndjson`: record with `metric_name=="fused-minus-healthy"`, `method_id=="fixed-fusion"`, `severity.direction=="increase"`, and `severity.magnitude==4.0` | `figures/noise-reporting-fused-minus-healthy.svg`, rightmost blue raw point and interval | The interval spans zero; this finite raw endpoint does not replace the independent population reference. |
| Q6: underreported noise | Population grid root `1.4630684126547195` std scale; continuous root `1.4657551414886727` std scale; finite-sample root `1.2916426005640154`; interval `[1.044728776068505, 1.5825214913938184]`; \(q=1.0\); `observed`; tested maximum `4.0` | U `analytic-validation.json`: `$.crossover_references[0]`; U `crossovers.ndjson`: `direction=="increase"` | `figures/noise-reporting-fused-minus-healthy.svg`, underreported series | Standard-deviation scale is a configured model coordinate, not meters or a real sensor-fault prior. |
| Q7: analytic-oracle agreement | All frozen six-standard-error Monte Carlo checks pass; maximum absolute standardized errors are `1.0144140839301548` (B), `0.9162214715685516` (C), and `1.0842355904362782` (U) | Each `analytic-validation.json`: `$.all_monte_carlo_checks_passed`, `$.monte_carlo_standard_error_multiplier`, and `max($.population_points[*].absolute_standardized_error)` | Not a plotted claim | The closed-form module checks the declared Gaussian estimator-output model only. |

The result rows in `README.md` contain no meter/std-scale conversion. Q2 and
Q3 use meters; Q4 and Q6 use dimensionless standard-deviation scale. Every
plotted raw point and pointwise interval comes from the corresponding
`aggregate-metrics.ndjson` record satisfying
`metric_name=="fused-minus-healthy"` and
`method_id=="fixed-fusion"`. PAVA curves are recomputed from those ordered raw
points by the preregistered equal-severity-weight rule.

## Integrity, environment, and omission claims

| Claim | Exact quantity | Machine-record selector or audit source | Caveat |
|---|---|---|---|
| Q8: named hardware and locked software | Apple M3 Pro; arm64; 11 logical CPUs; 19,327,352,832 memory bytes; Darwin 24.5.0; Python 3.12.13; lock SHA-256 `ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`; source `524c8f70ece3eca2e61796165b23ffe51baadfbc` | `release-index.json`: `$.environment`, `$.lockfile_sha256`, `$.source_revision`; each B/C/U `run.json`: `$.environment`, `$.lockfile_sha256`, `$.git_revision` | Runtime and memory fields identify the evidence machine; M1 does not claim cross-architecture byte identity. |
| Q9: primary/repeat determinism | Six strict replacement-bundle validations and 18/18 byte-identical comparisons: six stable files for each of three primary/repeat pairs | `release-index.json`: `$.verification.replacement_bundle_validations`, `$.verification.stable_file_comparisons`, and `$.verification.stable_files_per_experiment`; artifact and primary/repeat run digests in `$.experiments[*]` | The five payload members plus their `payload-index.json` envelope form the six-file stable allowlist. `run.json` is volatile across reruns; the archival release separately freezes the original run digests. |
| Q10: provenance-amendment equivalence | Three withheld comparison bundles were strictly loaded; 15/15 normalized scientific-file comparisons and 23,148 normalized records matched the excluded `1649dec5de387dd8b408a14678fa0acad0818735` execution after removing only provenance-bound `run_id` values | `release-index.json`: `$.verification.withheld_comparison_bundle_validations`, `$.verification.withheld_source_revision`, `$.verification.withheld_normalized_file_comparisons`, `$.verification.withheld_normalized_record_count`, and `$.verification.run_id_exclusion_only` | The old artifacts are deliberately not committed and cannot support a public result; this audit cannot be rerun from public files alone. |
| Q11: omitted bias sequence rows | 13,000 rows; 8,033,267 bytes; SHA-256 `82af401303cb7783db46ff6c769cba285bc08486930e2847dfae40349e623000` | `release-index.json`: B `$.omitted_sequence_metrics`; B `source-payload-index.json`: `$.files[?(@.path=="sequence-metrics.ndjson")]` | Rows are deterministic synthetic intermediates and can be regenerated from the frozen manifest and source. |
| Q12: omitted correct-noise sequence rows | 5,000 rows; 3,195,026 bytes; SHA-256 `33b888922c7af9341198a47dc576a903468055df0ca7643fa4919d78f55b3887` | `release-index.json`: C `$.omitted_sequence_metrics`; C `source-payload-index.json`: `$.files[?(@.path=="sequence-metrics.ndjson")]` | Same omission policy. |
| Q13: omitted underreported-noise sequence rows | 5,000 rows; 3,169,944 bytes; SHA-256 `7427ae25dddc60c55d10a10c548de8288422efe5d60db69b0693f1e250615478` | `release-index.json`: U `$.omitted_sequence_metrics`; U `source-payload-index.json`: `$.files[?(@.path=="sequence-metrics.ndjson")]` | Same omission policy. |
| Q14: figure completeness | Bias figure: 14 raw points and two PAVA curves. Noise figure: 10 raw points and two PAVA curves. Each has one zero line. | `release-index.json`: `$.figures[*].raw_point_count` and `$.figures[*].pava_curve_count`; validator exactly regenerates both SVG byte strings from the curated aggregate rows and checks their indexed hashes and lengths | Raw non-monotonic points remain visible; the fitted line does not replace them. |

## Claim boundary

The ledger supports only behavior of the declared analytic estimator-output
models and the integrity of their CPU execution. It does not support claims
about raw sensors, detectors, naturally occurring faults, physical
calibration/timing thresholds, nuScenes transfer, collision outcomes,
operational fallbacks, or fleets. See
[verification.md](verification.md) for the complete amendment and
determinism audit.
