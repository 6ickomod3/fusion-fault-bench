# M4 Claim-Evidence Ledger

Every numeric M4 statement in the public overview, figure, README, and results
page is an exact projection of the strict
[`m4-health-v0.1.0`](m4-health-v0.1.0/) release. Aggregate selectors are the
tuple `(condition_id, method, metric_name, window)` in
[`aggregate-metrics.ndjson`](m4-health-v0.1.0/aggregate-metrics.ndjson).

Scientific revision:
`a829a9f3af541c1b92b89d051b7c8b7003dc5a15`

Intent:
`c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357`

Named release CPU: Apple M3 Pro

## Identity, selection, and completeness

| Public statement | Exact source |
|---|---|
| Release artifact digest `f41dc1…325be` | [`_SUCCESS`](m4-health-v0.1.0/_SUCCESS) and [`release-index.json`](m4-health-v0.1.0/release-index.json) |
| Candidate `27`, self `0.999`, cross `0.995` | [`primary-fit-fit-summary.json`](m4-health-v0.1.0/primary-fit-fit-summary.json) and fit claim rows |
| 47 conditions and 11,515 aggregates | [`release-summary.json`](m4-health-v0.1.0/release-summary.json) |
| 8,900 sequence-condition pairs and 264,600 / 133,500 / 35,600 raw row counts | [`evaluation-validation.json`](m4-health-v0.1.0/evaluation-validation.json) |
| 433,700 omitted rows and 245,920,746 bytes | [`release-summary.json`](m4-health-v0.1.0/release-summary.json) and [`source-member-commitments.ndjson`](m4-health-v0.1.0/source-member-commitments.ndjson) |

## Primary policy-gain claims

All rows below use `combined-health-gate`, `policy-gain-vs-fixed`, and the
`event` window unless marked otherwise. Positive values mean lower
matched-center MSE than fixed fusion on the row's defined common support.

| Condition | Estimate and pointwise 95% interval, m² | Exact condition selector |
|---|---:|---|
| LiDAR output \(y\)-bias, \(+3\) m | `+5.4589961961` `[5.3908193270, 5.5093657489]` | `test-lidar-output-y-bias.value-03` |
| LiDAR timestamp offset, \(+0.6\) s | `+2.9466293127` `[2.8685921634, 3.0237926323]` | `test-lidar-timestamp-offset.value-03` |
| Camera noise, `3×`, covariance underreported | `+0.0861431264` `[0.0830889704, 0.0893188743]` | `test-camera-noise-underreported.value-01` |
| Camera timestamp offset, \(+0.6\) s | `+0.0234110854` `[0.0217760060, 0.0249890751]` | `test-camera-timestamp-offset.value-03` |
| Camera calibration \(x\), \(+3\) m | `+0.0015240031` `[0.0004904883, 0.0027795646]` | `test-camera-calibration-x.value-03` |
| Camera output \(y\)-bias, \(+3\) m | `+0.0014603752` `[0.0004905327, 0.0025904551]` | `test-camera-output-y-bias.value-03` |
| Held-out camera yaw, \(+0.06\) rad | `+0.0006738515` `[0.0001739007, 0.0012663493]` | `test-camera-calibration-yaw.value-03` |
| LiDAR noise, `3×`, covariance underreported | `−0.5785755347` `[−0.6080402101, −0.5532472348]` | `test-lidar-noise-underreported.value-01` |
| Shared common-mode \(x\)-bias, \(+4\) m | `−0.1429770381` `[−0.2011339883, −0.0865535057]` | `test-common-mode-x-edge.value-03` |
| Main clean | `−0.0000105295` `[−0.0000315884, 0]` | `test-main-clean.value-00`, window `score` |
| Edge clean | `−0.0041989398` `[−0.0100137990, −0.0000204478]` | `test-edge-clean.value-00`, window `score` |
| Clean bounded acceleration | `−0.0000105295` `[−0.0000315884, 0]` | `test-clean-bounded-acceleration.value-00`, window `score` |
| Correctly reported camera/LiDAR `3×` noise | exactly `0` | respective `test-*-noise-correctly-reported.value-01` rows |
| Cold-start camera calibration / LiDAR bias | exactly `0` | respective `test-cold-start-*.value-00` rows |

The fixed primary selector set is serialized in
[`quantitative-claims.ndjson`](m4-health-v0.1.0/quantitative-claims.ndjson);
the ledger additionally names supporting control rows used to interpret those
primary outcomes.

## Attribution, action, and control evidence

| Public statement | Aggregate selector(s) |
|---|---|
| LiDAR \(+3\) m bias attribution `0.995` | `test-lidar-output-y-bias.value-03`, `combined-health-gate`, `attribution-fraction`, `event` |
| Camera underreported-noise frame-oracle-recoverable-loss fraction `0.858215` | same condition, `frame-oracle-recoverable-loss-fraction`, `event` |
| Held-out yaw detection `1.0`, attribution `0.03`, LiDAR action occupancy `0.0272917` | `test-camera-calibration-yaw.value-03`, corresponding event metrics |
| LiDAR underreported-noise detection and attribution `1.0` | `test-lidar-noise-underreported.value-01`, corresponding event metrics |
| Common-mode detection `0.77`; first-latch labels `0.23` ambiguous and `0.54` LiDAR-fault | `test-common-mode-x-edge.value-03`, detection and first-latch-label event metrics; no target makes correct/wrong attribution undefined |
| Main / bounded-acceleration / edge false-alert starts `0.025 / 0.03 / 0.17` | respective conditions, `combined-health-gate`, `false-alert-episode-starts`, `score` |
| Cold-start detection `1.0`, attribution `0`, latency `3.14 / 3.11` frames | respective cold-start conditions and event metrics |
| LiDAR `+3 m` bias recovery fraction `1.0`, mean latency `4.07` frames, recovery-window gain `−0.620387` `[−0.642015,−0.597787] m²` | `test-lidar-output-y-bias.value-03`, `combined-health-gate`, corresponding `recovery-fraction`, `recovery-latency`, and `policy-gain-vs-fixed` recovery rows |
| Full-dropout fixed coverage `0`, health coverage `1` | respective `value-02` conditions, method coverage rows, `event` |
| Full-dropout health loss `0.1804525251 / 2.0051304651 m²` | respective `value-02` conditions, `combined-health-gate`, `matched-center-mse`, `event` |
| Full-dropout fixed conditional loss and policy gain undefined | corresponding fixed MSE and combined policy-gain rows with status `undefined` |

## Repeat and resource evidence

[`repeat-verification.json`](m4-health-v0.1.0/repeat-verification.json)
records zero mismatches across seven fit and nine evaluation scientific-member
comparisons, distinct volatile run records, independent paths/inodes, and one
runtime environment.

| Run | Wall-time / RSS claim rows | Raw evidence |
|---|---|---|
| Primary fit | `279.62 s`, `121585664 bytes` | [`primary-fit-time-l.txt`](m4-health-v0.1.0/primary-fit-time-l.txt) |
| Repeat fit | `286.17 s`, `123486208 bytes` | [`repeat-fit-time-l.txt`](m4-health-v0.1.0/repeat-fit-time-l.txt) |
| Primary evaluation | `592.01 s`, `169869312 bytes` | [`primary-evaluation-time-l.txt`](m4-health-v0.1.0/primary-evaluation-time-l.txt) |
| Repeat evaluation | `559.82 s`, `168116224 bytes` | [`repeat-evaluation-time-l.txt`](m4-health-v0.1.0/repeat-evaluation-time-l.txt) |

The resource claim rows are also retained in
[`quantitative-claims.ndjson`](m4-health-v0.1.0/quantitative-claims.ndjson).
The measurement scope is operator-recorded and self-reported, not an
independent attestation.

## Figure mapping

[`m4-health-policy-outcomes.svg`](../../docs/figures/m4-health-policy-outcomes.svg)
uses only rows named above. It preserves sign and physical units, marks
full-dropout loss comparison as undefined, and states the estimator-output,
pointwise-interval, and non-safety boundaries in the figure itself.

## Unsupported claims

The evidence does not support claims about physical sensor tolerances, natural
fault prevalence, raw camera or point-cloud degradation, detector AP,
association, nuScenes persistence, planning, collision reduction, operational
fallback selection, production readiness, vehicle safety, or fleet
generalization.
