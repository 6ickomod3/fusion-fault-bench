# M2 Claim–Evidence Ledger

This ledger maps every public M2 quantitative statement to the curated
machine-readable record beneath this release. M2 validates implementation
conventions; it reports no fault crossover or perception-performance result.

## Shared provenance

The scientific source revision is
[`cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4`](https://github.com/6ickomod3/fusion-fault-bench/commit/cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4).
The primary `records/m2-geometry/run.json` records `source_dirty=false`,
package version `0.1.0`, lockfile SHA-256
`ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`,
and Apple M3 Pro / arm64 / 11 logical CPUs / 19,327,352,832 memory bytes /
Darwin 24.5.0 / Python 3.12.13.

The manifest digest is
`7bfb5427c1ea5450a795bedef327457e5959860316bcd23f930e6eaa5917a068`,
the scientific artifact digest is
`09159042ca063b50762bf4150fb275b8a1760e4317ab74da8b0d24c133f42c90`,
and the run ID is
`run:697f42275ec0c2bffd91718bf6806c4e8900318e597e7f5da727643200a88ff6`.

## Synthetic geometry and covariance claims

All selectors below are relative to
`records/m2-geometry/geometry-validation.json`.
Normalized gate ratios derived from Q1–Q5, the Q7 headline counts, the Q9
reference-check count, and the Q11 authentication boundary are also presented
in `figures/geometry-validation-summary.svg`. The release validator regenerates
that figure byte-for-byte from the curated manifest and validation record.

| Claim | Exact public quantity | Machine-record selector | Interpretation |
|---|---|---|---|
| Q1: SE(3) properties | Rotation `1.7763568394002505e-15`; translation `2.8421709430404007e-13` m; point round trip `9.947598300641403e-14` m | `$.synthetic_geometry_validation.rotation_max_abs_error`, `.translation_max_abs_error_m`, `.point_round_trip_max_abs_error_m` | Maxima from the frozen 256-transform PCG64DXSM property workload, not dataset residuals. |
| Q2: quaternion sign | `0.0`, gate `1e-12` | `$.synthetic_geometry_validation.quaternion_sign_max_abs_error`; tolerance in `manifest.json` | Confirms equivalent `q` and `-q` rotation matrices for the synthetic workload. |
| Q3: independent projection fixture | Projection `1.4779288903810084e-12` px; depth `1.4210854715202004e-14` m; box corner `4.440892098500626e-16` m | `$.synthetic_geometry_validation.projection_max_abs_error_px`, `.depth_max_abs_error_m`, `.box_corner_max_abs_error_m` | These are repository-owned synthetic-reference disagreements, not local nuScenes residuals. |
| Q4: finite-difference covariance check | Maximum Jacobian error `2.1762947000070199e-10` against `1e-7` | `$.covariance_validation.finite_difference_max_abs_error`; manifest finite-difference tolerance | Validates the declared bearing/depth-to-BEV Jacobian only. |
| Q5: nonlinear covariance sampling | 200,000 samples; maximum gate ratio `0.019455935342375528`; `xx`, `xy`, `yy` all pass | `$.covariance_validation.monte_carlo_sample_count`, `.covariance_entry_max_gate_ratio`, `.covariance_entries[*]` | Sampling uses the manifest's **actual** covariance; reported covariance remains a separate estimator input. |
| Q6: component result | Synthetic geometry, covariance, and top-level `all_checks_passed` are true | `$.synthetic_geometry_validation.all_checks_passed`, `$.covariance_validation.all_checks_passed`, `$.all_checks_passed` | The strict loader recomputes all public numeric gates and conjunctions. |

The independent fixture is
`tests/fixtures/m2_geometry_reference_v1.json`, committed by SHA-256
`0676993f48e5a40034dfe497df7165b33f2d2f96dad234afd62af8e461beb252`.
Its conventions were reviewed against official nuScenes-devkit revision
`d9de17a73bdc06ce97a02f77ae7edb9b0406e851`.

## Local-data claims and attestation boundary

| Claim | Exact public quantity | Machine-record selector | What is and is not independently checkable |
|---|---|---|---|
| Q7: official-mini headline profile | 10 scenes, 404 samples, 18,538 sample annotations; profile pass true | `$.dataset_validation.expected_headline_counts`, `.headline_profile_passed_attested` | The counts and boolean are public; source metadata rows are intentionally absent. |
| Q8: structural integrity | Structural-integrity attestation true | `$.dataset_validation.structural_integrity_passed_attested` | Attests the adapter's fixed 12-table, link, chain, channel, calibration, pose, and box checks. A public clone cannot recompute it without independently obtaining data. |
| Q9: referenced key-frame blobs | 808 checks; attestation true | `$.dataset_validation.keyframe_blob_check_count`, `.keyframe_blob_validation_passed_attested` | Checks bounded referenced-file existence only; M2 does not read point-cloud or image contents. |
| Q10: local projection | Scalar/production cross-check and diagnostic-generation attestations true | `$.dataset_validation.local_projection_crosscheck_passed_attested`, `.diagnostic_svg_generated_attested` | No token, selected count, pixel/depth value, residual, or SVG is released. |
| Q11: dataset authentication boundary | `summary-does-not-authenticate-dataset-bytes` | `$.dataset_validation.dataset_authentication` | The artifact cannot identify or authenticate the exact archive or table bytes used locally. |

The five local-data booleans and the reported-covariance role-separation
boolean are explicitly named `_attested`. The strict public loader validates
their schema and recomputes their conjunctions, but does not claim to
re-execute the omitted local checks. Synthetic numeric gates are independently
recomputed from public values and frozen tolerances.

The figure's normalized numeric bars use each observed synthetic error divided
by its preregistered tolerance; the three Monte Carlo rows use the recorded
entry-wise gate ratio directly. A value at or below one passes. Its local-data
panel displays only the allowlisted headline counts, key-frame-reference check
count, and attestation status. It is a release summary, not the omitted local
projection diagnostic.

## Integrity and repeatability claims

| Claim | Exact quantity | Evidence | Caveat |
|---|---|---|---|
| Q12: primary/repeat determinism | 3/3 byte-identical stable-file comparisons; same artifact SHA-256; ignored diagnostic SVG byte-identical | `release-index.json`: `$.verification` plus primary/repeat identities | `run.json` and `_SUCCESS` vary because timestamps and run-record digests are volatile. Cross-architecture byte identity is not claimed. |
| Q13: exact primary run | Run SHA-256 `462e15bb9da0b8caa43b6040b7f147fa64d84c7cdf51264acd02a1bd2eadc50f` | `release-index.json`; `records/m2-geometry/source-success.json` | This is the exact archived primary run record. |
| Q14: exact repeat run | Run SHA-256 `5931411b25a169533a08849593af4416dd4a4d930054b93067f53199f0d27449` | `release-index.json` | Repeat volatile bytes are not tracked; only their digest and comparison attestation are retained. |
| Q15: implementation CI | GitHub Actions run `30437837817` passed for source revision `cd9ce423...` | `release-index.json`: `$.verification.public_ci` | CI used synthetic tests and did not access nuScenes. |

## Terms and claim boundary

The aggregate local-data record is derived from **nuScenes v1.0-mini,
Motional** and is governed by **CC BY-NC-SA 4.0 plus Motional Dataset Terms**,
with attribution to “nuScenes: A multimodal dataset for autonomous driving,
Caesar et al., 2020.” Motional does not sponsor, approve, or endorse Fusion
Fault Bench.

The ledger supports convention, implementation, artifact-integrity, and
local-profile attestation claims only. It supports no real sensor-noise
transfer, detector robustness, physical fault threshold, temporal-fault,
crossover, fallback, safety, production, or fleet claim.
