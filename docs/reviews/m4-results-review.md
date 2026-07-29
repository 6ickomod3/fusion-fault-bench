# M4 adversarial results review — m4-health-v0.1.0

Verdict: **PASS after public-document revision**

Release artifact:
`f41dc1a10c0e5ac940e2760baee159a911f95446345dc70c3fb1a5afa8d325be`

Scientific revision:
`a829a9f3af541c1b92b89d051b7c8b7003dc5a15`

Frozen intent:
`c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357`

## Review scope

An independent adversarial agent reviewed the retained strict release against
the frozen M4 plan and benchmark contract. The review covered hypothesis
outcomes, signed-estimand semantics, bootstrap support, held-out yaw, clean,
edge, maneuver, common-mode, cold-start, and dropout controls, repeat and
resource evidence, privacy, omissions, provenance, and public claim wording.
The reviewer did not implement M4, fit the official policy, run the official
evaluation, or build the curated release.

The reviewer independently:

- strict-validated the curated release and checked its digest and scientific
  revision;
- inspected all 11,515 retained aggregate rows, including 11,061 `ok` and 454
  intentionally `undefined` rows;
- checked that the 29 fixed quantitative claim rows are exact projections of
  retained aggregate, fit-summary, or resource evidence;
- audited interval direction, defined-bootstrap support, frame-oracle
  dominance, held-out/control coverage, repeat comparisons, resource caps,
  omission commitments, and privacy-scan results; and
- adversarially reread the public overview, figure, results, limitations,
  reproducibility, benchmark-card, and claim-evidence language.

The public package omits sequence rows, so this review could not independently
recompute the paired bootstrap intervals from the curated tree. The review
instead verified the retained aggregates, strict validation records, exact
commitments, and honest disclosure of that boundary.

## Initial public-document findings and resolution

The machine release passed on the first review. The initial public promotion
review returned P1, not P0, for claim wording and status consistency. The
release documents were revised to:

1. mark M4 consistently as released while preserving the frozen
   preregistration as a pre-execution document;
2. publish the post-event recovery-window latch penalty alongside event-window
   benefits;
3. distinguish the unbounded
   `frame-oracle-recoverable-loss-fraction` from event recovery metrics;
4. describe common-mode first-latch labels as targetless outcomes, not as
   correct attribution, misattribution, or fault mislocalization;
5. label `0.23` as the common-mode first-latch ambiguous-label fraction rather
   than the distinct `0.77` ambiguous event-outcome fraction;
6. report the curated tree as 7,108,875 bytes, or 6.780 MiB, rather than
   6.9 MiB; and
7. state that public evidence regenerates from retained release evidence,
   because fit-selection and resource tables do not come from aggregate rows
   alone.

After those corrections, no P0 or P1 result or claim blocker remained.

## Hypothesis outcomes

M4 does not support a universal health-gating benefit. H1 is only partially
supported. At the positive high-severity endpoints, combined-gate
event-window policy gain \(L_F-L_P\) was:

| Condition | Gain, m² | Pointwise 95% interval |
|---|---:|---:|
| LiDAR output \(y\)-bias, \(+3\) m | `+5.4589961961` | `[5.3908193270, 5.5093657489]` |
| LiDAR timestamp offset, \(+0.6\) s | `+2.9466293127` | `[2.8685921634, 3.0237926323]` |
| Camera noise, `3×`, covariance underreported | `+0.0861431264` | `[0.0830889704, 0.0893188743]` |
| Camera timestamp offset, \(+0.6\) s | `+0.0234110854` | `[0.0217760060, 0.0249890751]` |
| Camera calibration \(x\), \(+3\) m | `+0.0015240031` | `[0.0004904883, 0.0027795646]` |
| Held-out camera yaw, \(+0.06\) rad | `+0.0006738515` | `[0.0001739007, 0.0012663493]` |

The decisive counterexample is `3×` underreported LiDAR noise. Detection and
attribution were both `1.0`, but policy gain was
`−0.5785755347 m²`, interval
`[−0.6080402101, −0.5532472348]`. Correctly identifying a degraded modality
did not imply that discarding its still-useful information was the better
action. This negative result prevents a claim that H1 holds across all
preregistered high-severity axes.

The correctly reported camera- and LiDAR-noise controls had exactly
`0 m²` event-window policy gain, supporting H2's covariance-awareness
control. Direct availability and timestamp telemetry drove the high-severity
timestamp and dropout detection results, while self/cross evidence alone did
not reproduce that attribution behavior. The held-out yaw family therefore
supports only a narrow procedural held-out-family result: its extreme positive
gain was small, detection was `1.0`, first-event attribution was only `0.03`,
and the first event was ambiguous in `0.97` of sequences.

## Recovery, clean, and edge controls

Event-window benefit did not eliminate the cost of the frozen three-healthy-
frame release rule. After a `+3 m` LiDAR output \(y\)-bias, recovery-window
gain was `−0.6203874 m²`,
`[−0.642015, −0.597787]`, despite the large positive event-window result.
The corresponding recovery latency was about `4.07` frames. The public result
must therefore present detection, event utility, and recovery cost as separate
estimands rather than summarize the condition as an unqualified recovery.

Main-clean score-window gain was `−0.0000105295 m²`,
`[−0.0000315884, 0]`, with `0.025` false-alert episode starts per sequence.
The bounded-acceleration control had the same score-window gain and `0.03`
false-alert starts per sequence. The held-out edge-clean control was a clearer
support-shift regression: gain `−0.0041989398 m²`,
`[−0.0100137990, −0.0000204478]`, with `0.17` false-alert starts per
sequence. These outcomes satisfy the preregistered negative-control reporting
requirement; they do not establish operationally acceptable false-alert
rates.

Cold-start camera-calibration and LiDAR-bias events were both detected in
every sequence after about `3.14` and `3.11` frames, respectively, but
first-event attribution was `0`. The ambiguous latch retained fixed fusion,
so policy gain was exactly zero. This is evidence of delayed or prevented
attribution, not successful recovery.

## Common-mode and dropout semantics

The shared common-mode \(+4\) m bias has no uniquely healthy modality and
therefore no valid correct-attribution target. Combined-gate detection was
`0.77`; first-latch labels were `0.23` ambiguous and `0.54` LiDAR-fault, while
the targetless event-outcome ambiguous fraction was `0.77`. Routing worsened
loss: policy gain was `−0.1429770381 m²`,
`[−0.2011339883, −0.0865535057]`. The result supports the preregistered
common-mode blind-spot claim only with targetless semantics.

Dropout is coverage-first. At full camera or LiDAR dropout, fixed-fusion event
coverage was zero, so its conditional MSE and the fixed-versus-policy contrast
were undefined. The combined gate restored coverage to one, with conditional
loss `0.1804525251 m²` for camera dropout and `2.0051304651 m²` for LiDAR
dropout. These are coverage and conditional-loss results, not localization
wins over a zero-imputed fixed baseline. On defined common support at partial
dropout, combined-gate policy gain was negative, including
`−0.007695972 m²` for camera dropout probability `0.5` and
`−0.973615920 m²` for LiDAR dropout probability `0.5`.

## Statistics and support

Inference aggregates object/frame losses into sequence losses and resamples
complete paired sequences with 2,000 pointwise bootstrap replicates. All
retained `ok` estimates had valid ordered intervals and sufficient defined
replicate support; 11,035 used all 2,000 replicates, and the lowest remaining
defined count was 1,962. All 645 `gap-vs-frame-oracle` rows were non-negative.

The fit was frozen before test application. Candidate `27`, with self
threshold `0.999` and cross threshold `0.995`, was one of five feasible
validation candidates and satisfied the preregistered clean constraints.
Intervals condition on that selected fit and are pointwise, not simultaneous.
No physical unit axes are combined.

`frame-oracle-recoverable-loss-fraction` is an unclipped ratio, not a bounded
percentage. It may be negative when the policy is worse than fixed fusion; for
example, the underreported-LiDAR-noise row is approximately `−4.03258`.
Public wording must preserve that definition.

## Repeat, resources, privacy, and omissions

The two fits matched on all seven indexed scientific members, and the two
evaluations matched on all nine, for 16 comparisons and zero mismatches.
Distinct paths, inodes, and volatile records support independence checks, but
they are not cryptographic proof of two executions.

| Run | Wall time | Peak RSS |
|---|---:|---:|
| Primary fit | `279.62 s` | `121,585,664 B` |
| Repeat fit | `286.17 s` | `123,486,208 B` |
| Primary evaluation | `592.01 s` | `169,869,312 B` |
| Repeat evaluation | `559.82 s` | `168,116,224 B` |

All four records are below the preregistered `1,800 s` and
`1,073,741,824 B` caps. They are raw Darwin `/usr/bin/time -l` records with
strict sidecars, and remain operator-recorded, self-reported evidence rather
than independent execution attestation.

The strict privacy checks found no local dataset path, dataset payload,
private interview path, credential pattern, or prohibited absolute user path
in the curated release. This is a release-scope scan, not a general guarantee
about untracked local files.

The release retains every aggregate row but omits 264,600 sequence-loss,
133,500 contrast, and 35,600 event rows: 433,700 rows and 245,920,746 bytes
for one evaluation. Exact digests, byte lengths, and record counts bind those
omissions. They authenticate the omitted content identity but do not let a
third party independently recompute bootstrap inference without rerunning the
benchmark.

## Claim boundary

The supported conclusion is narrow: one frozen, causal, observable health rule
was conditionally useful and conditionally harmful in a deterministic CPU
estimator-output benchmark with known object identity. The evidence supports
matched-center benchmark loss, coverage, detection, attribution, action, and
latency statements only for the declared procedural conditions and named
revision.

It does not establish raw-sensor behavior, detector performance, association,
natural fault prevalence, physical calibration or timing tolerances, nuScenes
latent-scene persistence, planning or collision benefit, operational fallback
selection, production readiness, vehicle safety, or fleet generalization.
Resource evidence is self-reported, intervals are pointwise and conditional on
the frozen fit, and no public-CI claim is made by this review.

The reviewer identity and process are not cryptographically authenticated by
the offline validator. This tracked report's exact release-digest and
scientific-revision binding is the review boundary.
