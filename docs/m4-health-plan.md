# M4 Observable Health-Aware Fallback Pre-registration

Status: **pre-registered; not executed**.

This document freezes the M4 question, population, transient-event semantics,
observable-feature boundary, policy, threshold selection, estimands, inference,
acceptance gates, controls, resource limits, and claim boundary before M4 code
or results are produced. The machine-readable intent is
[`m4-health-v1.json`](../examples/health/m4-health-v1.json), canonical SHA-256
`c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357`.
The normative [v0.1 benchmark contract](benchmark-contract-v0.1.md) remains
controlling.

M4 is a new versioned event benchmark. It does not reinterpret M3's
sequence-persistent fault manifests or result rows. M4 remains CPU-only,
dataset-independent, known-ID, and estimator-output-level.

## 1. Question and hypotheses

M3 established when fixed fusion became harmful under persistent faults. M4
asks a complementary online question:

> Given only observable, pre-update consistency and direct telemetry, can a
> frozen rule detect and attribute a transient sensor event, route around it,
> and reduce matched-center loss without unacceptable clean regression?

The predeclared hypotheses are:

1. At high test severities, the combined gate has positive sequence-level gain
   over fixed fusion for additive bias, underreported noise, camera calibration
   translation/yaw, and timestamp events.
2. Correctly reported noise remains normalized by its reported covariance;
   health routing is not expected to improve fixed fusion and any unnecessary
   fallback must be published.
3. Current dropout and nonzero timestamp metadata are directly observable.
   The direct-telemetry ablation should separate this evidence from NIS-based
   diagnosis.
4. Camera calibration yaw, excluded from train and validation, measures
   held-out-fault-family behavior.
5. A shared common-mode bias can increase absolute loss without uniquely
   attributable inconsistency.
6. A clean bounded-acceleration maneuver can expose false alerts from a
   constant-velocity predictor.
7. Cold-start events can delay or prevent numeric attribution because two
   strictly prior observations are unavailable.

Every hypothesis may be false. M4 completion requires valid evidence, not a
minimum gain, detection rate, attribution rate, or favorable comparison.

## 2. Frozen population and data separation

M4 reuses the content-addressed procedural profiles:

| Population | Profile | Sequences | Use |
|---|---|---:|---|
| Train | `constant-velocity-front-roi-v1` train | 200 | clean ECDF fit only |
| Validation | same profile, validation split | 200 | threshold selection only |
| Test | same profile, test split | 200 | frozen-fit evaluation only |
| Edge test | `constant-velocity-fov-edge-v1` | 100 | held-out support/common mode |

The profile digests are respectively
`4771a6e69d75b9af41f99ab794c0af1b51e6103e43474c8e0f07df3e6f3ca68c`
and
`ca1544f69023847af7bdad9f1306ae3885f2e5d067d6afc026038f87ae36448d`.
All use data seed `1729`.

Train receives no fault event. Validation selects one global threshold pair.
Test loads the resulting fit artifact in apply-only mode: no ECDF refit,
threshold override, replay normalization, or policy reselection is permitted.
Layout/range/motion families and sequence identifiers remain disjoint.

All variants of one base sequence share one statistical cluster. Fault family,
target, direction, severity, event phase, method, and policy never create new
inference units.

## 3. Event semantics

Intervals are zero-based and half-open.

### 3.1 Standard event

| Phase | Frames |
|---|---|
| Predictor initialization | `[0,2)` |
| Clean prefix | `[0,12)` |
| Numeric score window | `[2,48)` |
| Fault active | `[12,36)` |
| Recovery | `[36,48)` |

### 3.2 Cold-start diagnostic

The fault is active on `[0,24)` and recovery is `[24,48)`. Numeric prediction
becomes available only after each object/modality history contains two prior
available observations.

Outside the active interval, observation and metadata configuration is exactly
identity. A validator must prove clean-before and clean-after observations
equal the paired clean condition. Faults do not accumulate across frames.
Generation always uses true state, calibration, and time; only estimator-
consumed reconstruction/alignment metadata is corrupted.

## 4. Exact validation and test matrix

Generic output bias, noise reporting, timestamps, and dropout are mirrored
between camera and LiDAR. Camera extrinsic faults remain camera-specific.

### 4.1 Validation-only values

| Family | Target | Axis / unit | Exact values | Selection role |
|---|---|---|---|---|
| Additive output bias | camera | \(y\), m | \(-2,-.5,.5,2\) | numeric utility |
| Additive output bias | LiDAR | \(y\), m | \(-2,-.5,.5,2\) | numeric utility |
| Underreported noise | camera | \(xy\), std-scale | \(1.5,4\) | numeric utility |
| Underreported noise | LiDAR | \(xy\), std-scale | \(1.5,4\) | numeric utility |
| Correctly reported noise | camera | \(xy\), std-scale | \(1.5,4\) | negative-control utility |
| Correctly reported noise | LiDAR | \(xy\), std-scale | \(1.5,4\) | negative-control utility |
| Timestamp offset | camera | time, s | \(-.4,-.1,.1,.4\) | direct validation only |
| Timestamp offset | LiDAR | time, s | \(-.4,-.1,.1,.4\) | direct validation only |
| Dropout | camera | probability | \(.25,.75\) | direct validation only |
| Dropout | LiDAR | probability | \(.25,.75\) | direct validation only |
| Calibration translation | camera | \(x\), m | \(-2,-.5,.5,2\) | numeric utility |

The literal `validation-main-clean` identity condition on the main validation
population is the sole source for clean-regression, clean-coverage, and
false-alert feasibility gates.

### 4.2 Frozen-fit test values

| Family | Target | Axis / unit | Exact values | Test role |
|---|---|---|---|---|
| Additive output bias | camera | \(y\), m | \(-3,-.75,.75,3\) | unseen severity |
| Additive output bias | LiDAR | \(y\), m | \(-3,-.75,.75,3\) | unseen severity |
| Underreported noise | camera | \(xy\), std-scale | \(1.25,3\) | unseen severity |
| Underreported noise | LiDAR | \(xy\), std-scale | \(1.25,3\) | unseen severity |
| Correctly reported noise | camera | \(xy\), std-scale | \(1.25,3\) | negative control |
| Correctly reported noise | LiDAR | \(xy\), std-scale | \(1.25,3\) | negative control |
| Timestamp offset | camera | time, s | \(-.6,-.15,.15,.6\) | unseen/direct |
| Timestamp offset | LiDAR | time, s | \(-.6,-.15,.15,.6\) | unseen/direct |
| Dropout | camera | probability | \(.1,.5,1\) | unseen/direct |
| Dropout | LiDAR | probability | \(.1,.5,1\) | unseen/direct |
| Calibration translation | camera | \(x\), m | \(-3,-.75,.75,3\) | unseen severity |
| Calibration yaw | camera | yaw, rad | \(-.06,-.015,.015,.06\) | held-out family |

### 4.3 Test-only controls

- Main clean identity.
- Edge-profile clean identity, described only as a support/layout shift.
- Clean bounded acceleration on the main test population: for transitions
  \(k=0,\ldots,46\) from frame \(k\) to \(k+1\), use \(dt=.1\) s,
  \(a_x=0\), and
  \(a_y=\mathrm{side}\,8\ \mathrm{m/s^2}\) for \(18\leq k<24\),
  \(a_y=-\mathrm{side}\,8\ \mathrm{m/s^2}\) for \(24\leq k<30\), and zero
  otherwise. `side` is \(-1\) for object indices 0–2 and \(+1\) for indices
  3–5. Update
  \(p_{k+1}=p_k+v_kdt+\tfrac12a_kdt^2\) and
  \(v_{k+1}=v_k+a_kdt\) from the unchanged main-test frame-0 state.
  Recompute eligibility once from this true maneuver trajectory before
  generating observations and reuse it for every method. This is a
  predictor-mismatch stress control, not a realism claim.
- Edge-profile shared \(x\)-bias at \(-4,-1,1,4\) m.
- Cold-start camera calibration \(x=+3\) m.
- Cold-start LiDAR \(y\)-bias \(+3\) m.

Every fault family retains an identity control. Identity observations across
compatible manifests must be exactly equal after narrow provenance
normalization.

## 5. Persistent-frame observable input

The health scorer consumes timestamped Cartesian measurements and reported
covariances in one persistent scene BEV frame. Procedural M4 uses its stationary
world frame. This interface is intentionally frame-agnostic so M5 can transform
moving-ego replay estimates into a scene-world frame without changing or
refitting the scorer.

The typed policy input may contain only:

- opaque object ID for history lookup;
- current aligned estimate;
- current reported covariance;
- current modality availability;
- reference timestamp; and
- reported sensor timestamp.

It cannot contain truth, latent velocity, fault family/target, severity,
direction, event phase/boundary, seed, split, sequence ID, manifest metadata,
or absolute frame index as a feature.

## 6. Pre-update predictor and NIS

Each modality/object has an independent history containing its two most recent
available observations strictly before frame \(k\), their own reported
covariances, and reference times \(t_a<t_b<t_k\). Define

\[
h=\frac{t_k-t_b}{t_b-t_a},
\qquad
\widehat z_k=(1+h)z_b-hz_a,
\]

\[
P_k^-=(1+h)^2R_b+h^2R_a.
\]

Reference-frame timestamps determine \(h\). Corrupted reported timestamps are
used only by the direct timestamp residual. Process noise is zero because the
primary latent model is exactly constant velocity.

Current self NIS is

\[
\mathrm{NIS}_{m,k}^{self}
=(z_{m,k}-\widehat z_{m,k})^\top
(P_{m,k}^-+R_{m,k})^{-1}
(z_{m,k}-\widehat z_{m,k}).
\]

Directional cross NIS is

\[
\mathrm{NIS}_{C\leftarrow L,k}
=(z_{C,k}-\widehat z_{L,k})^\top
(P_{L,k}^-+R_{C,k})^{-1}
(z_{C,k}-\widehat z_{L,k}),
\]

\[
\mathrm{NIS}_{L\leftarrow C,k}
=(z_{L,k}-\widehat z_{C,k})^\top
(P_{C,k}^-+R_{L,k})^{-1}
(z_{L,k}-\widehat z_{C,k}).
\]

Predictions are formed and committed before current measurements update either
history. Missing measurements do not update. Executed actions never affect
monitoring histories.

At least two mature current objects are required for numeric evidence. Mature
object values become a frame arithmetic mean and maximum. Eight separate clean
arrays are fitted: self/cross × camera/LiDAR × mean/maximum.

A required numeric channel with fewer than two mature current objects has
status `insufficient-support`; this is not healthy evidence. Direct evidence
remains eligible. Without direct evidence, any required undefined numeric
channel emits raw `ambiguous` with `insufficient-support` status, but the frame
holds the latched state, activation candidate/count, and recovery count: it
neither advances nor resets any counter.

## 7. Clean ECDF scores

For each of the eight statistics, clean train frames `[2,48)` produce exactly
9,200 finite frame values. Sort and retain the exact float64 arrays. Define

\[
\operatorname{rank}(x)
=\frac{\#\{v_i<x\}}{n}.
\]

Strict-less-than tie handling is fixed. Each self or directional cross channel
score is the maximum of its mean-statistic rank and maximum-statistic rank.
Alarms use strict `score > threshold`; threshold `1.0` therefore disables
numeric alarms, including for degenerate tied distributions.

Direct features remain distinct from ECDF scores:

- current availability; and
- \(|t^{reported}-t^{reference}|>10^{-12}\) s.

Trailing four-frame missing fraction and maturity fractions are released as
diagnostics but do not drive the v1 raw decision.

These ECDF values are anomaly ranks, not probabilities. Brier score, ECE, and
probability-calibration claims are not applicable.

## 8. Raw label, latched state, and executed action

M4 records three different outputs at every frame:

1. raw observable label;
2. latched health state; and
3. executed estimator action.

Conflating them would incorrectly treat immediate missing-input routing as
fault detection.

### 8.1 Raw decisions

- **Direct-only:** exactly one currently missing or timestamp-suspicious sensor
  receives that sensor label; both suspicious is ambiguous; otherwise healthy.
- **Self-NIS:** exactly one self alarm receives that sensor label; two alarms is
  ambiguous; otherwise healthy.
- **Cross-NIS:** any directional cross alarm is ambiguous. Cross-only never
  invents target attribution.
- **Combined:** direct evidence has priority; absent direct evidence, exactly
  one self alarm receives that sensor label and two are ambiguous; absent self
  alarms, any cross alarm is ambiguous; otherwise healthy.

### 8.2 State transition table

| Current state | Raw evidence | Counter effect | State/action timing |
|---|---|---|---|
| healthy | same nonhealthy eligible label twice | activation reaches 2 | latch on the second current frame |
| healthy | eligible healthy | clear candidate and set activation count to 0 | remain healthy |
| healthy | eligible contrary nonhealthy label | replace candidate and set activation count to 1 | remain healthy |
| nonhealthy label | healthy three times | recovery reaches 3 | restore healthy on the third current frame |
| nonhealthy label | any nonhealthy label | reset recovery | retain original latch; never switch directly |
| any | insufficient-support without direct evidence | hold all activation and recovery state | retain current state/action |

### 8.3 Action table

| Availability / state | Nonabstaining action | Abstaining action |
|---|---|---|
| neither available | undefined | undefined |
| camera only available | camera immediately | camera immediately |
| LiDAR only available | LiDAR immediately | LiDAR immediately |
| both, healthy | fixed fusion | fixed fusion |
| both, camera-fault | LiDAR | LiDAR |
| both, LiDAR-fault | camera | camera |
| both, ambiguous | fixed fusion | undefined |

Immediate availability routing changes the executed action but not the
detection definition. Detection uses the first nonhealthy latched state.

## 9. Methods and hindsight boundaries

The exact M4 method set is:

1. camera only;
2. LiDAR only;
3. fixed fusion;
4. self-NIS gate;
5. cross-NIS gate;
6. direct-telemetry gate;
7. combined health gate;
8. combined health gate with ambiguous abstention;
9. fault-target-drop diagnostic; and
10. frame-action performance oracle.

Target-drop knows the injected target and is not deployable. It executes fixed
fusion outside the active event and the non-target modality inside a
single-target event. If that selected modality is unavailable, its output is
undefined; it never falls back to the injected target. It equals fixed fusion
on identity conditions and is not defined for common-mode events.

The M3 complete-sequence oracle is not an M4 ceiling because M4 can change
actions by frame. M4 therefore versions a new oracle at matching granularity:
for each frame, select camera, LiDAR, or fixed fusion with the lowest
current-frame mean object loss among currently defined actions, then aggregate
the selected object losses over the ordinary eligible-object-frame sequence
denominator. Exact loss ties resolve in declared order: camera, then LiDAR,
then fixed fusion. It uses hindsight and is not deployable. On identical
support it must have loss no greater than every nonabstaining policy. Common
mode has no healthy modality, target-drop policy, or performance oracle.

## 10. Threshold selection

One global pair is selected:

\[
(\theta_{self},\theta_{cross})
\in\{.95,.975,.99,.995,.999,1\}^2.
\]

There are exactly 36 candidates, iterated with self threshold ascending in the
outer loop and cross threshold ascending in the inner loop. Every ablation
uses the selected pair; no ablation gets private tuning.

The combined nonabstaining candidate is feasible only when:

- mean clean regression is at most `0.002 m²`;
- its upper pointwise 95% paired-bootstrap bound is at most `0.005 m²`;
- mean false-alert episode starts are at most `0.05` per clean sequence; and
- clean coverage equals fixed-fusion coverage exactly.

Among feasible candidates, minimize score-window policy-minus-frame-oracle
regret. Weight camera and LiDAR targets equally, then families within target
equally, then declared conditions within family equally. The utility includes
mirrored bias, underreported noise, correctly reported noise, and camera
calibration translation. Direct timestamp/dropout families validate direct
rules but cannot affect numeric threshold selection.

Values within `1e-12 m²` are tied. Break ties by:

1. fewer clean false-alert episode starts;
2. lower clean regression;
3. larger self threshold; then
4. larger cross threshold.

Retain all 36 candidate rows. If no candidate is feasible, M4 fails; gates may
not be relaxed after inspection. The content-addressed fit artifact freezes
ECDF arrays, thresholds, feature/state/action semantics, validation rows, and
source identity before test execution.

## 11. Estimands and support

For method \(m\), sequence \(j\), and window \(W\),

\[
L^W_{m,j}
=\frac{1}{n_{m,j,W}}
\sum_{i\in W,\;m\ defined}
\|\widehat p_{m,ji}-p_{ji}\|_2^2.
\]

For the standard schedule, score, event, and recovery windows are `[2,48)`,
`[12,36)`, and `[36,48)`. For the cold-start schedule they are `[0,48)`,
`[0,24)`, and `[24,48)`. A cold-start row may not inherit standard windows.

The primary policy gain on equal support is

\[
G_P^W=\frac1N\sum_j(L^W_{F,j}-L^W_{P,j}),
\]

so positive values favor the policy. Publish score `[2,48)`, event `[12,36)`,
and recovery `[36,48)` windows. Also publish:

- clean regression \(L_P-L_F\);
- target-drop gap \(L_P-L_T\);
- frame-oracle gap \(L_P-L_O\); and
- frame-oracle recoverable-loss fraction
  \[
  \rho=\frac{\bar L_F-\bar L_P}{\bar L_F-\bar L_O}.
  \]

Report \(\rho\) only on identical support when the aggregate denominator
exceeds `1e-12 m²`; recompute it within every bootstrap replicate and do not
clip it. Abstaining policies publish coverage and undefined-output rate before
conditional loss and never receive an unequal-support recovery fraction. For
dropout, fixed-policy loss contrasts are computed only on paired common support
where both outputs are defined; coverage and undefined rate are reported
first. No all-support policy gain, oracle gap, or recovery fraction is
published on method-specific unequal support. Missing output is never assigned
zero loss.

Attribution and action utility are separate estimands. Correctly identifying
degradation does not imply dropping the modality improves localization.

## 12. Event metrics and censoring

Event outcomes are correct, ambiguous, wrong-sensor, or missed and sum to the
complete sequence denominator. The event outcome is the label of the first
healthy-to-nonhealthy latched transition during the active interval; no such
transition is `missed`. Later latch episodes do not rewrite the outcome.
Report:

- detection fraction before conditional detection latency;
- correct, ambiguous, wrong, and missed fractions;
- attribution fraction before conditional attribution latency;
- state and executed-action occupancy;
- early-clear fraction;
- latched state at the final active frame;
- recovery denominator fraction before recovery fraction/latency;
- coverage and undefined-output rate; and
- false-alert episode starts per clean sequence.

Detection latency is first nonhealthy latch minus event start. Undetected
events remain misses and never become zero latency. For dropout, first report
the fraction of sequences with at least one realized missing frame during the
active event. Latency relative to the first realized missing frame is
conditioned only on that subset. A sequence with no realized miss remains a
`missed` regime event regardless of any latch; such a latch is additionally
counted in the false-alert diagnostic.

Early clear is a nonhealthy-to-healthy latched transition during the active
event after detection. Recovery is evaluated only among sequences latched
nonhealthy at the final active frame; latency is first healthy latch in
recovery minus recovery start. A policy that cleared during the fault is
reported as early-clear, not zero-latency recovery.

Multiple latch episodes remain visible in episode counts, state/action
occupancy, early-clear, and final-active-frame state. Attribution latency uses
the first correctly targeted latch during the event, even when an earlier
episode was ambiguous or wrong; its defined fraction is published first.
Common-mode controls have no healthy target: report first-latch labels,
ambiguous and missed fractions, and state/action occupancy without defining
correct or wrong-sensor attribution.

## 13. Inference

- NumPy PCG64DXSM.
- Validation seed `2718`; test seed `314159`.
- 2,000 paired bootstrap replicates.
- One bootstrap index matrix per split.
- Complete base sequences are resampled; every fault/severity/policy variant
  follows the sampled sequence index.
- Equal-target/equal-family metrics are recomputed inside each replicate.
- Pointwise 95% percentile intervals use NumPy linear quantiles.
- Conditional ratios require more than 97.5% valid bootstrap denominators.
- Test intervals condition on the frozen selected fit and do not include
  training/model-selection uncertainty.
- Intervals are pointwise, not simultaneous matrix-wide coverage.

M4 has no crossover or PAVA estimand.

## 14. Acceptance gates and negative controls

Release requires:

- new event, fit, evaluation, artifact, and release contracts without changing
  frozen M3 method or manifest unions;
- exact event masks, identity outside events, paired base draws, and no fault
  accumulation;
- zero-noise constant-velocity prediction and fusion oracles;
- analytic \(\chi^2_2\) NIS checks under matched reported covariance;
- independent scalar predictor, covariance, NIS, feature, state, action, and
  oracle references;
- metadata-label mutations that cannot change observable features;
- future-observation mutations that cannot change any earlier output;
- current-observation mutations that can change residuals but not the
  precommitted prediction;
- independent sensor histories;
- exact ECDF endpoint/tie and threshold-1.0 tests;
- full state/action transition-table tests;
- deterministic 36-candidate selection and independently recomputed fit;
- hard separation of train, validation, test, held-out yaw, edge, common-mode,
  bounded-acceleration, and cold-start controls;
- clustered bootstrap and censored latency reconstruction;
- coverage-first dropout/abstention checks;
- frame-oracle dominance on identical support;
- two clean executions with byte-identical indexed scientific members;
- independent adversarial implementation/results review;
- full Ruff, Pyright, pytest, build, wheel-smoke, privacy, and CI gates.

The required negative controls are correctly reported noise, clean main,
clean edge support, clean bounded acceleration, common-mode bias, cold start,
partial/full dropout, zero-noise identity, feature-metadata leakage, current-
update leakage, and future-prefix causality.

## 15. Artifact and resource contract

The no-overwrite fit artifact (`ffb.health-fit-payload/v1`) contains frozen
intent/profile identities, eight exact sorted clean arrays, all 36 validation
candidate rows, selected thresholds/policy semantics, validation, an index,
run record, and success marker.

The no-overwrite evaluation artifact (`ffb.health-eval-payload/v1`) binds the
fit digest and contains sequence, aggregate, event, validation, index, run, and
success records. Raw frame features and sequence rows remain ignored/local.
The curated release publishes aggregates, figures, methods, fit/validation
summaries, named-CPU resources, review, and exact omitted-row commitments.

Frozen caps:

- fewer than 50,000,000 candidate-frame evaluations;
- fewer than 100,000,000 bootstrap cells;
- peak RSS below 1 GiB;
- each full run below 1,800 seconds on the named release CPU;
- curated release below 50 MiB;
- one CPU process; no GPU; and
- feature computation once followed by immutable reuse.

## 16. Explicit non-goals and claim boundary

M4 does not add a learned classifier, probability calibration, Torch, CUDA,
detector training, raw-sensor generation, nuScenes replay, association, set
loss, robust fusion, random onsets, compound faults, fault priors, planning,
collision, production fallback, or physical tolerance.

Valid wording is limited to measured matched-center loss, coverage, observable
event attribution, latency, clean regression, and oracle gaps under this
frozen procedural contract. “Deployable feature” means only that prohibited
labels/hindsight are excluded; it does not establish deployment suitability,
production readiness, safety benefit, or vehicle validation.
