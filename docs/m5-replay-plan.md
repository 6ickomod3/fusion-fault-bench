# M5 nuScenes-mini Latent Replay Pre-registration

Status: **frozen before replay descriptors or fault outcomes after adversarial
plan review PASS**.

This document freezes the M5 question, source population, asynchronous geometry,
motion proxy, pre-fault support, persistent-fault matrix, apply-only health
transfer, inference, acceptance gates, artifact boundary, and explicit non-goals
before replay execution. The machine-readable intent is
[`m5-nuscenes-mini-replay-v1.json`](../examples/replay/m5-nuscenes-mini-replay-v1.json).
Its frozen canonical SHA-256 is
`d429e36e2ce17ec8628c9bad4b5051fd54e0d88bcdeb966d112972e4c3dc2836`.
See the [M5 adversarial plan review](reviews/m5-plan-review.md).

The normative [v0.1 benchmark contract](benchmark-contract-v0.1.md), the
[M2 geometry plan](m2-geometry-plan.md), the released
[M3 persistent-fault plan](m3-temporal-plan.md), and the released
[M4 observable-health plan](m4-health-plan.md) remain controlling. M5 does not
rewrite any frozen M3 manifest union or M4 fit. It introduces a content-addressed
replay profile that binds local nuScenes metadata to those already released
scientific definitions.

## 1. Question, panels, and hypotheses

M3 measured persistent-fault behavior on seeded constant-velocity scenes. M4
fit and evaluated an observable health rule on scene-disjoint procedural data.
M5 asks:

> Do the released fixed-fusion mechanisms and the frozen observable-health rule
> persist when latent geometry, object motion, ego motion, calibration, image
> support, timing, track lifecycle, variable object count, and the recorded
> LiDAR-support proxy are replaced by ten recorded nuScenes-mini scenes?

M5 has exactly two panels:

1. **M5-A:** the complete M3 persistent-fault matrix replayed without favorable
   family or direction selection;
2. **M5-B:** the released M4 fit and candidate applied without refitting,
   normalization, threshold changes, policy selection, or latch tuning.

The preregistered M5-A hypotheses are:

1. Identity fused-minus-healthy deltas are negative for every ordinary
   single-target experiment.
2. Both maximum signed endpoints are positive for LiDAR output bias, camera
   calibration translation, camera calibration yaw, and camera timestamp
   offset.
3. The maximum camera underreported-noise endpoint is positive.
4. Camera correctly reported noise remains negative through \(4\times\)
   standard deviation.
5. Dropout masks are nested; at full camera dropout fixed-fusion coverage is
   zero and its conditional loss is undefined, while target-drop retains the
   healthy-modality coverage.
6. Common-mode bias increases absolute loss at both signed \(4\) m endpoints
   while leaving camera-LiDAR disagreement unchanged at every severity; it has
   no healthy target and no crossover.

Every M5-B hypothesis below selects method `combined-health-gate`, metric
`policy-gain-vs-fixed`, window `event`, and unit `m²`.
The preregistered M5-B hypotheses are:

1. The seven released positive M4 transport checks retain positive
   event-window point gain:
   LiDAR \(+3\) m output bias, LiDAR \(+0.6\) s timestamp offset, camera
   \(3\times\) underreported noise, camera \(+0.6\) s timestamp offset, camera
   \(+3\) m calibration translation, camera \(+3\) m output bias, and camera
   \(+0.06\) rad calibration yaw.
2. The released LiDAR \(3\times\) underreported-noise harmful-fallback
   counterexample retains negative event-window policy gain.
3. Exactly `replay-camera-noise-correctly-reported:3` and
   `replay-lidar-noise-correctly-reported:3` do not produce positive
   event-window health-policy gain.
4. Exactly `replay-common-mode-x:+4` remains without a uniquely faulty target;
   policy gain is reported without a healthy-target interpretation.

Every hypothesis may fail. A valid negative or partial-persistence result is
releasable. Condition- or window-level undefined results are also releasable
when the base source-validity gates passed. A scene with no base eligible
object-frame or an invalid health schedule instead blocks scientific release
and emits only a sanitized validation failure. No favorable outcome is an
implementation acceptance gate.

## 2. Why M5 does not add a v0.2 method

The roadmap permits a selected association, miss/false-positive, set-loss,
robust-fusion, or learned-health extension only after matched-center replay
remains valid. Developing a method on the same ten scenes used to establish
that validity would mix domain transport, method selection, and confirmatory
evaluation.

M5 and M6 therefore have a frozen **no-go** for v0.2. A later, separately
preregistered extension is allowed only if:

1. every M5 implementation, support, reproducibility, privacy, and review gate
   passes;
2. at least four of output bias, underreported noise, calibration translation,
   calibration yaw, and timestamp offset are robustly persistent at every
   preregistered maximum;
3. correctly reported noise retains the expected beneficial sign at
   \(4\times\);
4. one exact failure has the same direction in procedural and replay evidence;
5. untouched procedural and recorded evaluation data are available; and
6. exactly one extension is frozen before those new evaluation data are used.

The preferred later candidate is a fixed CPU robust-fusion baseline. The ten
mini scenes may motivate it but may not confirm it. M6 proceeds without v0.2
regardless of the M5 outcome; this gate authorizes only a separately
preregistered future milestone with fresh evaluation data. The roadmap's
conditional extension is permission, not a required M5 deliverable.

## 3. Exact source population and order

M5 uses all ten scenes from the already validated official
`v1.0-mini` profile, in this exact UTF-8 order:

1. `scene-0061`
2. `scene-0103`
3. `scene-0553`
4. `scene-0655`
5. `scene-0757`
6. `scene-0796`
7. `scene-0916`
8. `scene-1077`
9. `scene-1094`
10. `scene-1100`

The sequence identifier is `nuscenes:{scene_name}`. Each scene follows its
declared `first_sample_token -> next` chain. Frame index is the zero-based chain
position. Filesystem and dictionary enumeration never determine scientific
order.

There is no replay train/validation/test split. M3 parameters and M4 fit were
frozen before these scenes were used for outcomes. All ten scenes are one
finite exploratory population. A scene cannot be dropped, replaced, retried,
or moved to another panel because of support or results.

Every sample must have exactly one key-frame `LIDAR_TOP` record and one key-frame
`CAM_FRONT` record. The LiDAR key-frame timestamp is the reference time and its
ego frame \(E_k\) is the per-frame scoring frame. Relative times subtract the
first integer microsecond timestamp before conversion to seconds.

The recorded sample annotation center is assigned to the LiDAR reference time
and treated as benchmark truth. This preserves M2's snapshot approximation. It
does not assert that a moving annotation is physically synchronized to both
sensors.

The dataset root is resolved only from `NUSCENES_ROOT`. It is absent from
intent, digest, stdout, errors, generated public members, and tracked files.

## 4. Opaque known identities and ordering

M5 retains every sample-annotation category; it introduces no detection-class
filter. nuScenes instance identities do not cross scene boundaries.

Within a scene, private instance tokens are sorted as UTF-8 bytes and assigned
opaque public ordinals `track:{ordinal:04d}`. The token-to-ordinal map remains
local. Tokens, filenames, paths, poses, calibrations, annotation coordinates,
and per-frame timestamps never enter release artifacts, exception text, reprs,
or public commitments.

At most one annotation for an instance may occur in one sample. Frames are
ordered by chain position, objects by opaque ID, and eligible rows by
`(frame_index, object_id)`. Shuffling source-table insertion order must not
change any row, draw, commitment, or aggregate.

## 5. Pre-fault common support

Eligibility is calculated once from the recorded annotation snapshot and
nominal transforms, before noise or any fault. The exact mask is reused across
both panels, all families, targets, signs, severities, methods, and policies.

An object-frame is eligible only when:

1. its center in \(E_k\) has \(x>0\),
   \(5\le x\le60\) m, and \(|y|\le40\) m;
2. the same recorded center, transformed through the nominal
   camera-time ego pose and true `CAM_FRONT` calibration, has depth greater
   than \(0.1\) m and lies strictly inside the calibrated image:
   \(0<u<W,\ 0<v<H\);
3. recorded `num_lidar_pts > 0`, the frozen M2 LiDAR-support proxy; and
4. both proxy estimators are nominally available.

The center snapshot is not motion-interpolated for eligibility. This exactly
preserves M2's common-support rule. Box-corner `ANY` visibility, annotated
visibility level, category, speed, track length, measurement draw, and fault
response do not change support. Box dimensions and visibility remain
descriptors.

Reported calibration and timestamp metadata are unauthorized for eligibility.
Faulted transforms use a separate typed role. Every scene must have at least
one base eligible object-frame. Failure blocks scientific release and emits
only a sanitized validation record; a nine-scene substitute is forbidden.

Frames with zero eligible objects remain in the temporal schedule and dropout
draw vector. For health scoring they produce insufficient numeric support.
With healthy frame-level availability and no timestamp evidence, the
numeric-dependent self, cross, and combined methods hold their latch state;
the direct-only method remains update-eligible under its exact rule in
Section 12.

## 6. Frames and asynchronous geometry

All transforms use right-handed frames, column vectors, and
\(T_{a\leftarrow b}p_b=p_a\).

For sample \(k\):

- \(t_r\) is the `LIDAR_TOP` timestamp;
- \(E_k\) is the ego frame at \(t_r\);
- \(t_c\) is the `CAM_FRONT` timestamp;
- \(E_c\) is the ego frame at \(t_c\);
- \(C\) is the calibrated camera frame; and
- \(G\) is the log-qualified global frame.

Recorded transforms are

\[
T_{G\leftarrow E_k},\qquad
T_{G\leftarrow E_c},\qquad
T_{E_c\leftarrow C}.
\]

The nominal camera-to-scoring chain is

\[
T_{E_k\leftarrow C}
=T_{E_k\leftarrow G}
 T_{G\leftarrow E_c}
 T_{E_c\leftarrow C}.
\]

Primary camera support and reconstruction use the full recorded \(SE(3)\)
chain above. M5-B retains each modality's full reconstructed 3D center before
any BEV truncation, maps that point through the full current ego pose into
global coordinates, and only then projects global XY into a yaw-anchored
scene frame \(S\).

Let \(P_{xy}\) select the first two coordinates,
\(o_0=P_{xy}t_{G\leftarrow E_0}\), and \(\psi_0\) be the yaw extracted from
the scalar-first first LiDAR ego-pose quaternion. With
\(\widehat p^3_{m,E_k}\) the retained full 3D reconstructed center for modality
\(m\),

\[
\widehat p_{m,S}
=R_2(-\psi_0)
\left[
P_{xy}\left(
R_{G\leftarrow E_k}\widehat p^3_{m,E_k}
+t_{G\leftarrow E_k}
\right)-o_0
\right].
\]

Let \(e_{m,k}\in\mathbb R^2\) be the modality's base Cartesian error and
\(B_{m,k}=\partial\widehat p^3_{m,E_k}/\partial e_{m,k}\in\mathbb R^{3\times2}\)
be the exact rigid-chain propagation Jacobian. For LiDAR, \(B_{L,k}\) is the
zero-\(z\) lift. For camera, it includes the exact true-generation and
reported-reconstruction chain, including a reported calibration perturbation
when present. Define the exact \(2\times2\) health Jacobian

\[
J_{m,k}=R_2(-\psi_0)P_{xy}R_{G\leftarrow E_k}B_{m,k},
\qquad
C_{m,S}=J_{m,k}C^{reported,base}_{m,k}J_{m,k}^{\top}.
\]

The covariance must remain finite and positive definite. Roll, pitch, and
object \(z\) therefore influence the full 3D point and Jacobian before global
XY projection; no upper-left block of a full \(SE(3)\) rotation is treated as
an \(SO(2)\) rotation. Planar changes of global origin and yaw must leave loss,
NIS, labels, latch states, actions, and aggregates invariant.

## 7. Annotation motion and natural asynchrony

M5 uses a declared annotation-motion proxy, not a physical-velocity label:

- if both adjacent annotations exist, use the previous-to-next centered
  secant;
- at a track endpoint, use the current-to-neighbor one-sided secant;
- allow at most \(3.0\) s for the centered span and \(1.5\) s for a one-sided
  span, matching the pinned devkit velocity limits; and
- when velocity is otherwise undefined, use a declared zero-order hold and
  report its fraction.

Differences use sample timestamps as in the pinned official devkit. Velocity is
generator/oracle metadata only. It cannot enter observable health features.

The recorded center \(p_G(t_r)\) is transported to camera acquisition time:

\[
p_G(t_c)=p_G(t_r)+v_G(t_c-t_r).
\]

Camera Gaussian error is drawn in \(E_k\), lifted with zero \(z\)-error,
rotated to \(G\), and included before true camera backprojection. The physical
proxy is generated using \(p_G(t_c)\), the true camera-time ego pose, and the
true extrinsic. Nominal reconstruction returns it to \(E_k\), then known proxy
motion aligns it back to \(t_r\). Identity therefore recovers exactly the
noisy reference-time center even when camera and LiDAR key frames are
asynchronous.

Natural camera-LiDAR capture offset is a descriptor, not a fault. Health input
at identity receives the aligned estimator-state timestamp \(t_r\), never the
raw camera acquisition timestamp.

## 8. Calibration and timestamp fault causality

A camera calibration fault changes only reported reconstruction metadata in the
camera-time ego frame:

\[
\widetilde T_{E_c\leftarrow C}
=\Delta T_{E_c}T^{\mathrm{true}}_{E_c\leftarrow C}.
\]

The physical camera proxy, true pose, stochastic draw, and eligibility must be
identical across calibration severities. A mutation that uses the corrupted
extrinsic for both generation and reconstruction is a release blocker.

For an alignment-timestamp fault:

\[
t_{\mathrm{reported}}=t_{\mathrm{true}}+\delta
\]

and the alignment error is

\[
\widehat p_G(t_r;\delta)-\widehat p_G(t_r;0)=-v_G\delta.
\]

The physical proxy and recorded ego pose stay fixed. This tests
estimator-state alignment metadata, not ego-pose clock lookup, pose
interpolation, or raw-sensor latency. After alignment, M4 receives reported
state time \(t_r+\delta\), so direct telemetry observes the injected offset but
does not flag healthy natural asynchrony.

## 9. Estimator-output and covariance model

M5 retains M3's Cartesian estimator-output model:

- camera actual and identity-reported standard deviation:
  \((1.0,1.0)\) m;
- LiDAR actual and identity-reported standard deviation:
  \((0.3,0.3)\) m;
- zero-mean, diagonal, temporally IID Gaussian errors;
- independent base camera and LiDAR draws; and
- actual error distributions separate from reported covariance.

This is not pixel noise, a depth estimator, a point-cloud return model, a
detector, or raw-sensor simulation. Calibrated camera geometry determines
support and metadata-fault reconstruction only.

For M5-A, the BEV estimator value is
\(\widehat p_{m,E_k}=P_{xy}\widehat p^3_{m,E_k}\). With the Section 6
reconstruction Jacobian \(B_{m,k}\), define

\[
A_{m,k}=P_{xy}B_{m,k},\qquad
C_{m,E_k}=A_{m,k}C^{reported,base}_{m,k}A_{m,k}^{\top}.
\]

This is the exact full-chain replay generalization of the frozen M3 v0.1
reported-covariance behavior and reduces to its planar equations on the M3
source. The resulting covariance must be finite and positive definite. All
M5-A values and covariances are expressed in the common current \(E_k\) BEV
frame before fixed information fusion.

M5-B makes a separate health-only copy and pushes its reported covariance into
\(S\) with the exact modality- and severity-specific
reconstruction/projection Jacobian \(JCJ^\top\) from Section 6; this reduces to
an ordinary covariance rotation only for a planar orthogonal frame change.
The \(S\) copy is used only as input to the observable health monitor. It is
never fused and never supplies a localization value or loss.
Additive and common-mode biases are expressed in the current \(E_k\) axes.
Calibration perturbations are expressed in \(E_c\). Dropout uses one
target-modality uniform draw per frame, shared by all objects, and nested
across probability.

## 10. Paired randomness

M5 uses:

- NumPy PCG64DXSM;
- the frozen name-and-sequence SHA-256 stream derivation;
- data master seed `1729`;
- bootstrap seed `1618033`;
- exact sequence IDs `nuscenes:{scene_name}`;
- latent, camera, LiDAR, and fault streams; and
- one camera/LiDAR draw row per eligible object-frame in frame/object order.

The latent stream is declared for compatibility with the frozen M3 stream
contract but is not consumed because the replay source supplies recorded
latent state. Its non-consumption is itself part of the intent.

The same base draws are reused across both panels, all conditions, directions,
severities, methods, and policies. Dropout draws one vector of length
`frame_count` and reuses it across probabilities. There are no retries.

### 10.1 Replay experiment identity

M5 never claims that a replay experiment is one of the frozen M3 v1alpha1
manifests with a field silently changed. After this complete replay intent is
frozen, each replay experiment identity is the SHA-256 of canonical JSON with
the following exact envelope:

```json
{
  "schema": "ffb.replay-experiment-identity/v1",
  "replay_intent_sha256": "<canonical M5 intent digest>",
  "panel_id": "<m5-a or m5-b panel ID>",
  "source_sha256": "<M3 source-manifest digest or M4 fit-artifact digest>",
  "experiment_id": "<replay experiment or condition ID>"
}
```

The canonical serializer, not textual JSON key order, controls the digest. For
M5-A, `source_sha256` is the digest of the exact source manifest named by the
frozen M3 matrix. For M5-B, it is the released M4 fit-artifact digest. Every
local sequence row, local result row, curated aggregate, figure record, index
member, and success marker binds this replay identity. Completeness validation
reconstructs the exact intent-defined panel and execution order and rejects a
missing, extra, duplicate, or mismatched identity. The released M3 and M4
contracts remain unchanged.

## 11. M5-A: complete M3 persistent matrix

The source matrix is
[`m3-procedural-v1.json`](../examples/matrices/m3-procedural-v1.json), canonical
SHA-256
`7162a01bb766a38e8f7196cba9eb2c6c546f5866bc640f220252bb6e7aab6f9b`.
M5 may change only the experiment prefix, source population, and bootstrap
seed. Fault semantics, severity grids, observation parameters, methods,
evaluation, dropout rules, common-mode rules, PAVA, crossover, and result
selection are unchanged.

| Experiment | Target and axis | Exact grid |
|---|---|---|
| Additive bias | LiDAR \(y\) | \(0,.25,.5,1,2,4\) m, paired signs |
| Correctly reported noise | camera \(xy\) | \(1,1.25,1.5,2,4\times\) |
| Underreported noise | camera \(xy\) | \(1,1.25,1.5,2,4\times\) |
| Calibration translation | camera \(x\) | \(0,.25,.5,1,2,4\) m, paired signs |
| Calibration yaw | camera yaw | \(0,.005,.01,.02,.04,.08\) rad, paired signs |
| Timestamp offset | camera time | \(0,.05,.1,.2,.4,.8\) s, paired signs |
| Dropout control | camera availability | \(0,.1,.25,.5,.75,1\) |
| Common-mode control | both \(x\) | \(0,.25,.5,1,2,4\) m, paired signs |

The fixed M3 methods remain camera-only, LiDAR-only, fixed fusion,
fault-target drop, and the sequence performance oracle where defined. Dropout
has coverage, undefined-output rate, and conditional loss but no crossover.
Common mode has absolute loss, no healthy reference, no target-drop policy,
and no crossover.

## 12. M5-B: apply-only M4 health transfer

The health panel binds:

- M4 intent SHA-256
  `c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357`;
- fit artifact SHA-256
  `abd1540f292fe51a7a23a47b679fe8e1522d8c5e20a03125a880eb9242a608ee`;
- fit run SHA-256
  `0311aec90df031bd0d1720d5fa15aae91e2e5c3dfea923dacc2eefe518134fcd`;
- selected candidate `27`;
- self threshold `0.999`; and
- cross threshold `0.995`.

The eight clean ECDF arrays, thresholds, nonempty-frame feature equations,
raw-decision priority, two-frame activation, three-frame recovery, action
mapping, abstention semantics, state/action tie order, and method/oracle loss
formulas are unchanged.

M5-B uses new, versioned replay-only contracts:

- `ffb.replay-health-schedule/v1`;
- `ffb.replay-health-frame-input/v1`;
- `ffb.replay-health-evidence/v1`;
- `ffb.replay-health-sequence-event/v1`; and
- `ffb.replay-health-result/v1`.

These contracts add only variable-length windows, zero-object frames, dynamic
event/recovery counts, and observation-step plus elapsed-time latency. They do
not mutate or broaden any released M4 schema or class. Regression tests must
prove that all released M4 nonempty behavior remains byte-for-byte and
value-for-value unchanged.

No replay outcome may refit an ECDF, normalize a channel, select a threshold,
change a policy, tune a latch, choose a severity, or remove a condition.

For a scene with \(K\) samples, \(K\ge16\):

\[
a=\lfloor K/4\rfloor,\qquad b=\lfloor3K/4\rfloor.
\]

The schedule is:

| Phase | Frames |
|---|---|
| Predictor initialization | \([0,2)\) |
| Clean prefix | \([0,a)\) |
| Score window | \([2,K)\) |
| Fault active | \([a,b)\) |
| Recovery | \([b,K)\) |

Endpoints depend only on \(K\), not eligibility or observations.
They must additionally satisfy

\[
a\ge4,\qquad b-a\ge2,\qquad K-b\ge3.
\]

This retains at least two numeric clean-prefix steps and permits the complete
three-step M4 recovery recurrence.

The complete M4 standard-event test matrix is repeated:

| Family | Target | Exact values |
|---|---|---|
| Output \(y\)-bias | camera, LiDAR | \(-3,-.75,.75,3\) m |
| Underreported noise | camera, LiDAR | \(1.25,3\times\) |
| Correctly reported noise | camera, LiDAR | \(1.25,3\times\) |
| Timestamp offset | camera, LiDAR | \(-.6,-.15,.15,.6\) s |
| Dropout | camera, LiDAR | \(.1,.5,1\) |
| Calibration translation | camera \(x\) | \(-3,-.75,.75,3\) m |
| Calibration yaw | camera | \(-.06,-.015,.015,.06\) rad |
| Common-mode bias | both \(x\) | \(-4,-1,1,4\) m |
| Clean identity | neither | \(0\) |

M5 does not invent replay analogues for M4's procedural edge population,
bounded-acceleration trajectory, or cold-start diagnostic.

Health inputs are limited to opaque ID, aligned estimate, reported covariance,
availability, reference-state time, and reported-state time. Truth, actual
covariance, annotation velocity, category, visibility, point count, scene/log
identity, fault metadata, severity, seed, and event phase are prohibited.

Only the monitoring copy enters those health inputs. Every executed
camera-only, LiDAR-only, fixed-fusion, target-drop, and frame-oracle value uses
the Section 9 \(E_k\) BEV representation and the frozen M3/M4 action formula.
The post-transition health action selects among those \(E_k\) outputs.
Matched-center loss is always against recorded truth projected into the same
current \(E_k\) BEV frame; it is never computed from the scene-frame monitoring
copy.

M5 performs no object-count normalization beyond M4's frozen per-frame mean and
maximum NIS channels. Track entry/exit, variable eligible-object count, and
`num_lidar_pts > 0` support therefore shift the clean score distribution and
are intentionally part of the apply-only domain-transport test.

For a zero-object frame, the replay frame contains `objects=[]`. Each numeric
channel records `current_object_count=0`, `mature_object_count=0`,
`mature_fraction=0.0`, status `insufficient-support`, and null mean, maximum,
and score. Per-object measurement histories do not change. The strictly
increasing reference time advances, and current frame-level camera/LiDAR
availability is appended to each four-step missingness history. With no object
measurement, timestamp evidence is absent; direct telemetry uses only
frame-level availability.

The direct-only decision is always update-eligible on an empty frame. If
exactly one modality is unavailable it labels that modality, if both are
unavailable it is ambiguous, and otherwise it is healthy. For the combined
rule, a nonhealthy direct label has priority and is update-eligible. With
healthy direct evidence, the undefined numeric channels produce ambiguous
`insufficient-support`. Self- and cross-only decisions likewise produce
ambiguous `insufficient-support`. An insufficient-support decision holds every
latch counter and state. An update-eligible direct decision uses the unchanged
M4 recurrence. The executed action is selected after that current transition
from the post-transition latch and current availability.

Every zero-object frame counts as one observation step in state/action
occupancy, event detection, attribution, recovery, and censoring. It produces
no object row and therefore adds nothing to a localization-loss numerator or
denominator. The complete scene remains in event and detection denominators.

Primary latency remains an integer count of observation steps. Active-frame
count is \(b-a\), recovery-frame count is \(K-b\), and observed detection and
attribution latencies lie in \([0,b-a-1]\), with censor bound \(b-a\).
Observed recovery latency lies in \([0,K-b-1]\), with censor bound \(K-b\).
Dropout first-missing step lies in \([0,b-a-1]\). Detection minus first
missing is their signed step difference, bounded by
\([-(b-a-1),b-a-1]\) when both are defined. Active-window latch-episode
counts lie in \([0,b-a]\). Clean-identity false-alert episodes are counted
over \([2,K)\) and lie in \([0,K-2]\); unrealized-dropout false alerts are
counted over \([a,b)\) and lie in \([0,b-a]\); other fault rows record zero
false alerts. Each active state-occupancy and action-occupancy partition sums
exactly to \(b-a\). A separately named
`elapsed-reference-time-latency` diagnostic uses recorded reference timestamps.
For detection frame \(d\), first correct-attribution frame \(c\), recovery
frame \(r\), and first realized missing frame \(m\), the exact seconds are
\(t_d-t_a\), \(t_c-t_a\), \(t_r-t_b\), \(t_m-t_a\), and dropout response
\(t_d-t_m\), respectively. The last quantity may be negative, matching the
signed step response. Each seconds value is defined exactly when its
corresponding step event and required conditioning event are observed. Its
defined fraction is published before the conditional arithmetic mean over
observed scene values. Each complete-scene bootstrap replicate reconstructs
that mean as the sum of observed values divided by their observed count. The
point and interval are defined only when the point count is nonzero and
strictly more than 97.5% of replicates are defined. Censored elapsed time is
never imputed, and no nominal frame-rate conversion is allowed. The same
conditional aggregation applies to observation-step latency.

The replay event classification otherwise preserves M4 semantics: detection is
the first healthy-to-nonhealthy latch transition in \([a,b)\); identity and an
unrealized dropout are forced `missed`, with their relevant latch episodes
counted as false alerts; a targetless first latch is `ambiguous`; and
single-target outcomes are correct, ambiguous, wrong-sensor, or missed. Later
correct attribution uses the first correct-target latch in the active window.
Early clear is a later nonhealthy-to-healthy transition before \(b\). Recovery
is eligible only when the final active state is nonhealthy and uses the first
nonhealthy-to-healthy transition in \([b,K)\). An unobserved latency is null and
remains in the denominator of its separately published defined fraction.

## 13. Estimands

For scene \(j\), condition \(s\), and method \(m\), with frozen eligible set
\(E_j\):

\[
L_{m,j}(s)
=\frac{1}{|E_j|}\sum_{i\in E_j}
\|\widehat p_{m,ji}(s)-p_{ji}\|_2^2.
\]

For a single-target persistent fault:

\[
d_j(s)=L_{F,j}(s)-L_{H,j}(s),\qquad
D_{\mathrm{mini}}(s)=\frac{1}{10}\sum_{j=1}^{10}d_j(s).
\]

Scenes, not object-frames, receive equal inferential weight. Variable frame and
object counts do not make a scene more representative.

For health policy \(P\) and declared window \(W\):

\[
G_P^W
=\frac{1}{10}\sum_{j=1}^{10}
\left(L_{F,j}^W-L_{P,j}^W\right).
\]

Positive gain means lower matched-center MSE than fixed fusion on the same
defined support. Dropout and abstention report coverage first and conditional
loss second; no missing output is assigned zero loss.

Support is frozen separately for every named health window. If any scene has
zero eligible rows in a window, that all-ten equal-scene window loss or
contrast is undefined. If any scene has zero paired common-support rows, the
equal-scene policy gain is undefined. The scene remains in event and detection
denominators, and no reduced-scene mean is permitted. Pooled availability
coverage and pooled availability conditional loss are the explicit exceptions;
they follow the count reconstruction below.

Availability's v0.1 primary aggregate remains the pooled valid/eligible count
ratio reconstructed inside each scene bootstrap. A zero-eligible scene
contributes the count pair `(valid=0, eligible=0)` and remains in the bootstrap
input; a zero-valid but positive-eligible scene likewise remains. The pooled
point is defined only when total eligible count across all ten scenes is
positive. A bootstrap replicate with zero pooled eligible count is undefined
and excluded from its conditional interval; the pooled ratio is reported only
when strictly more than 97.5% of replicates are defined. Conditional loss
is total loss sum divided by total valid count, is defined at the point only
when the all-ten total valid count is positive, excludes a bootstrap replicate
with zero total valid count as undefined, and retains the same strict
greater-than-97.5% defined-bootstrap rule.

A scene-equal coverage diagnostic is also required so a large scene cannot hide
heterogeneity. If any scene has zero eligible count in the named window, that
all-ten-scene diagnostic is undefined rather than averaged over a reduced
scene set.

## 14. Small-\(n\) inference and persistence labels

One shared \(2000\times10\) PCG64DXSM integer matrix resamples complete scenes
for every method, severity, panel, window, and contrast. Pointwise 95%
percentile intervals use NumPy linear quantiles.

These intervals describe sensitivity to composition of the finite ten-scene
mini split. They are not fleet-population confidence intervals. Frames,
tracks, annotations, conditions, signs, and severity variants never become
independent samples.

Every primary contrast additionally publishes:

- positive, zero, and negative scene counts;
- all ten leave-one-scene-out aggregate estimates;
- the number of distinct private log groups; and
- all leave-one-log-group-out aggregate estimates, carrying every scene in
  the omitted log group together.

Log tokens remain private. Public groups use opaque ordinals.
Local ignored sequence rows retain the v0.1-required
`nuscenes:{scene_name}` identifier. Opaque labels apply only to public
leave-one-cluster sensitivity rows; they do not silently rewrite the sequence
contract.

There are no asymptotic frame-level tests, matrix-wide significance statements,
or inferential comparisons of M3 versus M5 roots or M4 versus M5 gains.
Cross-milestone differences are descriptive.

For a preregistered expected direction:

- **robustly persistent:** the point sign matches, its pointwise interval lies
  wholly on the expected side, at least eight of ten individual scene signs
  match, every leave-one-scene-out and leave-one-log-group-out estimate is
  defined with the expected sign, and at least two log groups exist;
- **directionally consistent:** the point sign matches but at least one
  robustness condition fails;
- **non-persistent:** the point is exactly zero or has the opposite sign; and
- **undefined:** required support is absent.

Expected-positive rows require point \(>0\) and, for robust persistence,
interval lower bound \(>0\). Expected-negative rows require point \(<0\) and
interval upper bound \(<0\). Exact zero never passes directional persistence.
Any undefined leave-out estimate, or fewer than two log groups, prevents the
robust label. H5-B3 is a nonpositive control rather than a persistence row: it
is supported only when point \(\le0\) and interval upper bound \(\le0\).

A global “M3 mechanism persisted” headline requires every H5-A1 through H5-A4
check to be robustly persistent. Otherwise the public wording is partial or
non-persistence and the complete matrix remains visible.

## 15. Frozen descriptive comparison

Descriptions are generated only after this intent is frozen and before faults
are executed. Each quantity is summarized within scene first, then by
minimum/median/maximum across ten scenes. Pooled object-frame histograms are
secondary and explicitly descriptive.

The release reports:

- sample count and reference \(\Delta t\);
- raw camera-minus-LiDAR acquisition offset;
- support-waterfall counts;
- eligible object-frame and unique-track count;
- eligible track length;
- ego range and bearing;
- box width, length, and height;
- finite-difference speed and acceleration;
- visibility level;
- LiDAR point count;
- zero-order-hold velocity fraction;
- category composition; and
- distinct log-group count.

The fixed support-waterfall order is all annotations, ROI pass, camera-center
pass, positive LiDAR-point support, final eligible. These are cumulative
filters in that order.

Numeric quantiles use NumPy's linear rule at
\(0,.25,.5,.75,1\). Eligible track length is the number of final eligible
frames for an opaque track. Range is \(\operatorname{hypot}(x_e,y_e)\), bearing
is \(\operatorname{atan2}(y_e,x_e)\), and speed is the global-XY norm of the
declared velocity proxy. Acceleration is the global-XY norm of a centered or
one-sided secant of that velocity proxy, using the same \(3.0/1.5\) s gap
limits and zero-order hold. Recorded dimensions retain upstream
width/length/height order. The nuScenes visibility level describes visibility
across the upstream six-camera annotation, not `CAM_FRONT` alone; an empty
token is `unknown`. Category and visibility summaries are counts and fractions
in frozen UTF-8 label order, followed by minimum/median/maximum of per-scene
fractions. Numeric across-scene summaries are minimum/median/maximum of each
named within-scene quantile. Log-group ordinals follow UTF-8 ordering of
private log tokens.

The comparator is the released 200-sequence M3 main test profile. Shared
fields are frame count/period, support count, track length, range, bearing, and
speed. Box dimensions, visibility, point counts, category, and sensor
asynchrony are reported as not modeled in M3. No significance test compares
the populations.

## 16. Analytic, property, integration, and local-data gates

Required synthetic or independent checks include:

1. zero-noise identity recovers truth in \(E_k\) and \(S\) within
   \(10^{-12}\) m;
2. natural camera-LiDAR asynchrony cancels exactly at identity;
3. timestamp displacement matches \(-v\delta\);
4. a stationary track has zero timestamp displacement;
5. calibration translation/yaw match an independent scalar oracle while the
   physical proxy is invariant;
6. using one corrupted calibration for generation and reconstruction fails;
7. a stationary global object under ego translation, yaw, roll, and pitch
   remains stationary in \(S\) and catches early XY truncation or use of the
   wrong sensor pose;
8. centered, one-sided, and zero-order-hold velocity cases have analytic
   answers;
9. strict image, depth, ROI, and `num_lidar_pts == 0` boundaries match the
   frozen decisions;
10. reported metadata mutations cannot alter eligibility;
11. shuffled table insertion order produces identical rows;
12. a stationary replay fixture is metric-equivalent to M3;
13. planar scene-origin translation and yaw changes preserve loss, covariance,
    NIS, labels, and actions;
14. healthy-modality rows and eligibility commitments are invariant over
    severity;
15. dropout masks are nested and full dropout preserves missing-output
    semantics;
16. common-mode bias leaves cross-modal disagreement unchanged;
17. empty health frames produce insufficient numeric support and hold the
    self/cross and combined-with-healthy-direct latches, while direct-only
    availability evidence uses its normal recurrence and action timing;
18. extending health inputs to empty frames does not change any released M4
    nonempty-input behavior;
19. substituting scene-frame monitoring values for executed \(E_k\)
    localization outputs fails;
20. current measurements do not enter their own predictions and future
    mutations cannot change earlier outputs;
21. prohibited metadata mutations cannot change health features; and
22. independent sequence aggregation, bootstrap, PAVA, crossover, event,
    policy, and oracle implementations match production.

Local execution must additionally prove that all ten declared scenes validate,
every scene has base support, every health schedule has at least 16 frames and
satisfies the exact phase constraints, and no raw payload bytes are read.

## 17. Artifacts, repeatability, and resources

Raw sequence rows are local ignored NDJSON. Frame, track, annotation,
coordinate, token, pose, calibration, and timestamp rows are never released.
Curated evidence contains only intent, aggregate descriptors, aggregate panel
results, crossover records, cluster-sensitivity aggregates, figures,
validation, run provenance, exact source-member commitments, index, and
success marker.

Commitments store role, byte length, record count, and SHA-256, not a dataset
path or private identifier. Two clean local executions must have byte-identical
indexed scientific members.

The caps are:

- one CPU process;
- no GPU, Torch, CUDA, or raw payload reads;
- peak RSS below \(1\) GiB;
- each full run below 1,800 seconds on the named release CPU;
- curated release below 50 MiB; and
- no-overwrite artifact publication.

Every public number must bind to intent, software revision, aggregate record,
figure, and named hardware run.

## 18. Dataset, privacy, and license boundary

The adapter may read validated metadata tables and inspect already required
key-frame filesystem metadata. It may not read image or point-cloud contents.
No dataset file is copied into the repository or generated-output tree.

Public aggregate replay evidence is marked
`CC BY-NC-SA 4.0 plus Motional Dataset Terms`, with attribution and
non-endorsement. The repository's Apache-2.0 license does not relicense it.
The adapter summary does not authenticate dataset bytes.

Tracked-file, staged-file, candidate-artifact, stdout/stderr, exception, and
secret scans must reject:

- archives, maps, images, point clouds, and metadata tables;
- scene/sample/annotation/instance/calibration/log tokens;
- filenames and dataset paths;
- per-frame timestamps, poses, calibrations, or coordinates;
- credentials and private interview material; and
- ignored raw outputs.

Scene names are intentionally present only because the frozen v0.1 contract
uses them as public sequence IDs. They do not authenticate the underlying
dataset.

## 19. Release acceptance

M5 can be released only when:

1. this intent is frozen before replay descriptors or outcomes;
2. independent adversarial plan review has no blocker;
3. all ten scenes run in fixed order without retry, selection, or exclusion;
4. transform, timing, support, causality, and leakage oracles pass;
5. M3 and M4 matrices are complete;
6. complete-scene bootstrap and log-group sensitivity are present;
7. two executions have identical scientific evidence;
8. full unit, property, analytic-oracle, integration, local-data, artifact,
   release, and privacy tests pass;
9. independent adversarial implementation review has no blocker;
10. independent adversarial results and claims review has no blocker;
11. methodology, limitations, reproducibility, results, evidence ledger, and
    learning material agree;
12. Ruff format/check, Pyright, pytest, package build, and wheel smoke pass;
13. tracked and staged files contain no dataset/private payload;
14. the release commit and tag are pushed and resolve exactly on the remote;
    and
15. GitHub CI passes at that release commit.

An unsupported hypothesis stays visible. Condition/window undefined support
cannot be converted into a reduced-population positive claim. Base-source
invalidity blocks the scientific release rather than becoming an outcome row.

## 20. Claim boundary and non-goals

M5 may describe:

- matched-center estimator-output loss under the declared proxy;
- persistence or non-persistence across ten recorded mini latent scenes;
- apply-only transport of the frozen observable-health rule;
- scene-composition and log-group sensitivity; and
- measured CPU runtime and memory.

M5 cannot establish real sensor-noise transfer, raw-sensor robustness, detector
quality, a physical fault tolerance, fault prevalence, an independent random
population result, fleet generalization, planning/collision benefit, production
readiness, or safety.

M5 does not add Torch, CUDA, GPU work, detector inference/training, images,
point clouds, photorealistic rendering, association, false positives, misses,
set loss, learned health, a new robust-fusion method, planning, collision, or
closed-loop evaluation.

## 21. Pinned upstream reference

Schema, transform, annotation interpolation, and velocity convention checks are
pinned to official nuScenes-devkit revision
[`d9de17a73bdc06ce97a02f77ae7edb9b0406e851`](https://github.com/nutonomy/nuscenes-devkit/tree/d9de17a73bdc06ce97a02f77ae7edb9b0406e851),
the same revision used by M2. The devkit is a convention oracle, not a runtime
dependency.
