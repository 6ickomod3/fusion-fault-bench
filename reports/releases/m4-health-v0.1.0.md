# M4 Observable Health-Aware Fallback Release

Release evidence:
[`m4-health-v0.1.0/`](m4-health-v0.1.0/)

Scientific source revision:
`a829a9f3af541c1b92b89d051b7c8b7003dc5a15`

Release artifact:
`f41dc1a10c0e5ac940e2760baee159a911f95446345dc70c3fb1a5afa8d325be`

## Question

Given only pre-update residual consistency and direct availability/timestamp
telemetry, can one frozen rule detect a transient sensor event, attribute it,
route around it, and reduce matched-center loss without unacceptable clean
regression?

M4 fits eight empirical-CDF calibration channels on 200 clean training
sequences, selects one pair from 36 threshold candidates on 200 validation
sequences, freezes that fit, and evaluates 47 conditions on 200 held-out main
sequences or 100 held-out edge sequences. Inference resamples complete paired
sequences with 2,000 bootstrap replicates.

The selected candidate was index `27`, with self threshold `0.999` and cross
threshold `0.995`. The test stage evaluated 8,900 sequence-condition pairs and
retained 11,515 aggregate records.

## Main result

Observable fallback was conditionally useful, not uniformly robust. Positive
policy gain means lower matched-center MSE than fixed fusion on common support:

| Test condition | Combined-gate event gain \(L_F-L_P\), m² | Pointwise 95% interval | Interpretation |
|---|---:|---:|---|
| LiDAR output \(y\)-bias, \(+3\) m | +5.458996 | [5.390819, 5.509366] | Large benefit; 99.5% correct attribution |
| LiDAR timestamp offset, \(+0.6\) s | +2.946629 | [2.868592, 3.023793] | Large benefit from direct telemetry |
| Camera noise, \(3\times\), covariance underreported | +0.086143 | [0.083089, 0.089319] | Recovered 85.8% of the frame-oracle-recoverable loss |
| Camera timestamp offset, \(+0.6\) s | +0.023411 | [0.021776, 0.024989] | Benefit from direct telemetry |
| Held-out camera yaw, \(+0.06\) rad | +0.000674 | [0.000174, 0.001266] | Small benefit despite only 3% first-event attribution |
| LiDAR noise, \(3\times\), covariance underreported | **−0.578576** | [−0.608040, −0.553247] | 100% attribution, but dropping LiDAR was harmful |
| Shared common-mode \(x\)-bias, \(+4\) m | **−0.142977** | [−0.201134, −0.086554] | No uniquely healthy target; routing worsened loss |

The LiDAR-noise counterexample is central: event detection and attribution
were both 100%, yet camera-only fallback increased loss. Sensor health and
downstream action utility are different estimands.

![M4 observable health outcomes](../../docs/figures/m4-health-policy-outcomes.svg)

## Controls and boundary cases

- Correctly reported `3×` camera and LiDAR noise had exactly `0.0 m²`
  event-window policy gain. Reported covariance normalized the evidence as
  intended.
- Main clean and bounded-acceleration score-window gain was
  `−1.0529e-5 m²`; the corresponding false-alert episode rates were `0.025`
  and `0.03` per sequence.
- The held-out edge clean condition was worse:
  `−0.00419894 m²` \([−0.0100138,−0.00002045]\), with `0.17`
  false-alert episode starts per sequence.
- Cold-start camera calibration and LiDAR bias were detected after about
  `3.1` frames, but first-event attribution was `0%`. The ambiguous state kept
  fixed fusion, so policy gain was exactly zero.
- After the LiDAR `+3 m` bias ended, all latched episodes recovered in a mean
  `4.07` frames. Recovery-window policy gain was nevertheless
  `−0.620387 m²` \([−0.642015,−0.597787]\): the frozen three-frame clearing
  rule incurred a measurable post-event cost.
- Under full camera or LiDAR dropout, fixed-fusion event coverage was `0%`
  and its conditional loss was undefined. The health gate restored `100%`
  coverage with respective conditional losses `0.180453 m²` and
  `2.005130 m²`. This is a coverage result; no missing output was assigned zero
  loss and no unsupported fixed-versus-policy contrast was reported.

## Determinism and resources

Two complete fits matched on all seven indexed scientific members. Two complete
evaluations against the same official fit matched on all nine indexed
scientific members. Volatile run records remained distinct.

| Run | Wall time | Peak RSS |
|---|---:|---:|
| Primary fit | 279.62 s | 121,585,664 bytes |
| Repeat fit | 286.17 s | 123,486,208 bytes |
| Primary evaluation | 592.01 s | 169,869,312 bytes |
| Repeat evaluation | 559.82 s | 168,116,224 bytes |

These Apple M3 Pro measurements are raw `/usr/bin/time -l` logs plus strict
sidecars. They are operator-recorded, self-reported evidence, not independent
or cryptographic execution proof.

## Public evidence boundary

The strict release is `7,108,875` bytes (`6.8 MiB`) and retains:

- both complete fit artifacts;
- the complete 11,515-row aggregate matrix;
- primary and repeat evaluation provenance envelopes;
- four raw resource logs and their sidecars;
- all 29 predeclared quantitative claim projections; and
- exact digest, byte-length, and record-count commitments for the omitted
  sequence files.

The omitted local sequence evidence contains 264,600 loss, 133,500 contrast,
and 35,600 event records (`433,700` total; `245,920,746` bytes for one
evaluation). Third parties can regenerate the public presentation from the
retained release evidence. Scientific outcome tables and the figure project
aggregate rows; fit and resource tables use the retained fit summaries and raw
measurement records. Third parties cannot independently recompute the
bootstrap intervals without rerunning the benchmark.

See the [claim-evidence ledger](m4-health-v0.1.0-claim-evidence.md),
[verification record](m4-health-v0.1.0-verification.md), frozen
[pre-registration](../../docs/m4-health-plan.md), and
[technical walkthrough](../../docs/m4-technical-walkthrough.md). The
[independent adversarial results review](../../docs/reviews/m4-results-review.md)
records the release decision and counterexamples.

## Claim boundary

This is a CPU-only procedural estimator-output benchmark with known object
identity. It does not evaluate a learned detector, raw sensor degradation,
natural fault rates, nuScenes latent-scene persistence, planning, collisions,
or a production fallback. The reported meter/radian/second/probability axes
remain separate controlled coordinates; results are not physical tolerances or
safety thresholds.
