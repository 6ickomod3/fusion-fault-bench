# M2 Geometry and nuScenes Grounding

Status: **pre-registered; adversarial plan, artifact, privacy, and licensing
reviews passed before implementation or geometry execution**. A readiness-only
check had already confirmed the expected directory layout and the three
documented official-mini headline counts; no projection, covariance,
referential-integrity, or experimental result was evaluated before this plan.

Release note: M2 subsequently passed its frozen gates and was promoted as
[`m2-geometry-v0.1.0`](../reports/releases/m2-geometry-v0.1.0/README.md).
The scientific execution revision is
`cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4`. This note records disposition
without changing the preregistered questions, thresholds, or claim boundary
below.

This document freezes the M2 validation questions, numerical gates, local-data
boundary, and explicit non-goals before geometry code or nuScenes-mini
diagnostics are executed. M2 is an implementation-validation milestone. It
does not estimate a camera-LiDAR fault crossover and releases no claim about
raw-sensor robustness.

## 1. Validation questions

M2 asks whether Fusion Fault Bench can:

1. represent and compose named rigid transforms without ambiguity;
2. reproduce the documented nuScenes sensor-to-ego, ego-to-global, and
   scalar-first quaternion conventions;
3. project global annotation geometry into the front-camera image using the
   same transform direction and pinhole convention as the official devkit;
4. propagate a declared camera bearing/depth covariance into ego-frame BEV;
5. identify the same pre-fault object support for camera-only, LiDAR-only, and
   fused methods; and
6. load and validate user-provided nuScenes-mini metadata without copying
   dataset content or local paths into a tracked artifact.

These are pass/fail implementation questions. Counts, residual tolerances, and
runtime from a passing local validation may be reported as software-validation
evidence only.

## 2. Coordinate contract

All calculations use float64, right-handed frames, column vectors, and

\[
T_{a\leftarrow b}p_b=p_a.
\]

Composition is legal only when the intermediate frame names match exactly:

\[
T_{a\leftarrow c}
=T_{a\leftarrow b}T_{b\leftarrow c}.
\]

An M2 transform stores both its target and source frame. Applying, composing,
or comparing incompatible frames must fail rather than silently relabel a
matrix.

The supported named frame kinds are:

- `global`, qualified by the scene's log/map namespace;
- `ego`, qualified by log namespace and the relevant sample-data timestamp;
- `camera`, qualified by channel, calibration instance, and sample-data
  timestamp; and
- `lidar`, qualified by channel, calibration instance, and sample-data
  timestamp.

Runtime qualifiers may contain local dataset tokens, but validation summaries,
candidate public logs, tracked fixtures, stdout, stderr, and failure artifacts
must not serialize those qualifiers. Cross-log transforms fail even if two logs
refer to the same named map; an explicit bridge would be required and is
outside M2.

nuScenes key-frame sensors are not assumed synchronous. For every sample M2
uses two distinct ego frames:

- \(e_r\): the canonical BEV scoring frame at the `LIDAR_TOP` key-frame
  sample-data timestamp; and
- \(e_c\): the ego frame at the `CAM_FRONT` key-frame sample-data timestamp.

The sample timestamp is not substituted for either sensor timestamp. The
recorded sample-level global annotation center is treated as benchmark truth
and expressed in \(e_r\) for forward/lateral support and later localization
loss. Camera image support is evaluated by transforming that same recorded
snapshot through \(e_c\), as the official key-frame devkit path does. This is
not a claim that a moving object's annotation is physically synchronized to
both sensor timestamps. Any camera estimate used by later fusion must
ultimately be reconstructed into \(e_r\); a camera-frame or \(e_c\)-frame value
may never be fused directly with an \(e_r\)-frame LiDAR value.

For a nuScenes camera sample-data record:

\[
T_{g\leftarrow e}
=\operatorname{Pose}(
q_\text{ego}^{wxyz},t_\text{ego}),
\qquad
T_{e\leftarrow c}
=\operatorname{Pose}(
q_\text{calib}^{wxyz},t_\text{calib}).
\]

Therefore

\[
T_{c\leftarrow g}
=T_{c\leftarrow e}T_{e\leftarrow g}
=T_{e\leftarrow c}^{-1}T_{g\leftarrow e}^{-1}.
\]

The calibrated-sensor transform is never interpreted as ego-to-sensor, and the
ego pose is never interpreted as global-to-ego. Recorded calibration and
localization metadata are treated as the nominal benchmark reference, not as
exact physical ground truth.

For a later camera estimate reconstructed into the scoring frame, the nominal
cross-time chain is

\[
T_{e_r\leftarrow c}
=T_{e_r\leftarrow g}
T_{g\leftarrow e_c}
T_{e_c\leftarrow c}.
\]

M2 validates this chain but does not interpolate annotations, compensate
object motion, or score temporal loss. The devkit-compatible local diagnostic
uses the camera sample-data ego pose. Substituting the LiDAR ego pose is a
required negative control.

## 3. Rigid-transform and quaternion behavior

The production geometry layer will provide:

- immutable named rigid transforms;
- scalar-first quaternion-to-rotation conversion;
- transform inverse, checked composition, and column-vector application;
- homogeneous-matrix export for diagnostics only; and
- camera projection, box-corner, bearing/depth, covariance, and ROI helpers.

Quaternion inputs must be finite, have four elements in explicitly typed
`[w, x, y, z]` order, and have norm within \(10^{-6}\) of one. Accepted inputs
are normalized before matrix construction. A zero, non-finite, or grossly
non-unit value is rejected. The API cannot infer that a unit-length array was
actually supplied in scalar-last order; it must never guess or reorder one.

Rotation matrices must be finite and right-handed. Construction rejects a
matrix when either

\[
\lVert R^\top R-I\rVert_\infty>10^{-10}
\]

or

\[
|\det R-1|>10^{-10}.
\]

The implementation may remove only round-off-scale asymmetry when constructing
derived covariance matrices; it may not project an invalid rotation onto
\(\mathrm{SO}(3)\).

## 4. Projection and independent oracle

The pinhole camera convention is:

\[
\begin{bmatrix}\tilde u\\\tilde v\\\tilde w\end{bmatrix}
=Kp_c,\qquad
(u,v)=(\tilde u/\tilde w,\tilde v/\tilde w),
\]

where the nuScenes camera depth is \(p_{c,z}\). Mathematical projection is
valid only for finite points with \(p_{c,z}>0\). Center ROI and devkit box
visibility apply their stricter thresholds separately.

Camera intrinsics must be finite \(3\times3\), have positive focal lengths and
the standard pinhole third row \([0,0,1]\) within \(10^{-12}\).

Calibrated image eligibility uses strict bounds

\[
0<u<W,\qquad 0<v<H,
\]

and a minimum center depth of \(0.1\) m. Center support and diagnostic box
visibility are different:

- the benchmark ROI scores the projected object center; and
- the local diagnostic draws the eight official-order box corners and applies
  the official devkit `ANY` rule: at least one corner lies in strict image
  bounds with depth greater than \(1\) m, and every corner has depth greater
  than \(0.1\) m.

Two independent checks are frozen:

1. A committed synthetic fixture contains poses, points, intrinsics, and
   expected pixel/depth values calculated without importing production
   geometry code. Production projection must agree within \(10^{-9}\) pixels
   and \(10^{-12}\) m depth.
2. A separate scalar reference path, which must not import the production
   adapter, projection, or transform modules, independently loads and indexes
   raw metadata for the selected local diagnostic. Its valid-point decisions
   must match exactly; finite pixels must agree within \(10^{-9}\) px and
   finite depths within \(10^{-10}\) m.

The local diagnostic deterministically selects the lexicographically first
scene name, follows its declared first-sample link, and uses that sample's
`CAM_FRONT` key frame. It processes every annotation belonging to the sample
in UTF-8 token order; it cannot select a favorable object after projection.
The run fails as vacuous unless at least one annotation and at least one finite
positive-depth center are checked. Selection never depends on filesystem
enumeration. The selector is fixed before results, but the selected scene,
sample, tokens, coordinates, counts, projected pixels, residuals, depths, and
category labels remain local-only.

For diagnostic boxes, annotation size is interpreted exactly as
`[width, length, height]`, center and orientation begin in the global frame,
and both are transformed into the camera frame. The eight-corner order is
frozen by the independent fixture.

## 5. Bearing/depth covariance

M2 defines a horizontal camera observation

\[
y=(\beta,d),
\]

where \(\beta\) is horizontal bearing in radians and \(d>0\) is optical-axis
depth in meters. Conditional on a declared camera vertical coordinate \(h_c\),

\[
p_c(\beta,d)=
\begin{bmatrix}
d\tan\beta\\
h_c\\
d
\end{bmatrix}.
\]

For \(y=(\beta,d)\), the camera-frame Jacobian is

\[
J_c=
\begin{bmatrix}
d\sec^2\beta & \tan\beta\\
0 & 0\\
0 & 1
\end{bmatrix}.
\]

Given the full cross-time nominal rotation \(R_{e_r\leftarrow c}\), the
ego-BEV Jacobian and propagated covariance are

\[
J_{xy}=R_{e_r\leftarrow c}[0{:}2,:]J_c,\qquad
\Sigma_{xy}=J_{xy}\Sigma_{\beta d}J_{xy}^{\top}.
\]

M2 fixes \(h_c=0.5\) m. Translation is constant and therefore contributes
nothing to this covariance. Actual sampling covariance and reported estimator
covariance are separate typed inputs. The frozen manifest uses, in
\((\beta,d)\) order,

\[
\Sigma^\text{actual}_{\beta d}=
\begin{bmatrix}
10^{-8} & 5\times10^{-6}\\
5\times10^{-6} & 0.04
\end{bmatrix},
\qquad
\Sigma^\text{reported}_{\beta d}=
\begin{bmatrix}
2.25\times10^{-8} & -3.75\times10^{-6}\\
-3.75\times10^{-6} & 0.0625
\end{bmatrix}.
\]

Monte Carlo draws use only the actual covariance. Both covariances are
propagated analytically and retain their role labels.

This is a first-order covariance model. It does not claim that a learned camera
detector emits Gaussian bearing/depth errors.

The full map into \(e_r\)-BEV must match a central finite-difference Jacobian
using steps \(10^{-6}\) rad and \(10^{-4}\) m within \(10^{-7}\) absolute
error. The Monte Carlo gate uses:

- NumPy `PCG64DXSM`;
- seed `13464654573299691533`;
- 200,000 independent Gaussian bearing/depth draws;
- mean \((0.2\ \mathrm{rad}, 25\ \mathrm{m})\);
- standard deviations \((10^{-4}\ \mathrm{rad}, 0.2\ \mathrm{m})\);
- correlation \(0.25\); and
- a non-axis-aligned synthetic full cross-time
  \(R_{e_r\leftarrow c}\) from the independent fixture.

The generator makes one float64 `standard_normal(size=(200000, 2))` call.
Each row is mapped with NumPy's lower Cholesky factor of
\(\Sigma^\text{actual}_{\beta d}\), then the fixed mean is added. A
non-positive sampled depth fails the gate; samples are never clipped,
conditioned, or redrawn. Empirical covariance uses `ddof=1`. Only the
three unique `xx`, `xy`, and `yy` entries are checked.

For each covariance entry \(i,j\), the empirical nonlinear covariance must
satisfy

\[
|\widehat\Sigma_{ij}-\Sigma_{ij}|
\le
6\sqrt{
\frac{\Sigma_{ij}^2+\Sigma_{ii}\Sigma_{jj}}{n-1}
}
+10^{-8}\ \mathrm{m}^2.
\]

The final \(10^{-8}\ \mathrm{m}^2\) allowance is fixed before execution for
second-order nonlinear remainder and float64 round-off. M2 reports the gate,
not a tuned post-hoc tolerance.

The propagated \(2\times2\) covariance remains full and symmetric. M2 does not
silently diagonalize it or connect it to M1's diagonal manifest model. Any
fusion integration is a later versioned change.

## 6. Common front-camera/LiDAR support

The M2 pre-implementation erratum in Benchmark Contract v0.1 resolves its
original `radial` label in favor of the already-frozen `x_min_m`,
`x_max_m`, and `abs_y_max_m` fields. These are inclusive ego-frame forward and
lateral bounds:

\[
x_{\min}\le x_e\le x_{\max},\qquad |y_e|\le y_{\max},
\]

with the separate strict requirement \(x_e>0\).

For a procedural source without calibrated image bounds, the symmetric
half-FOV boundary is inclusive:

\[
|\operatorname{atan2}(y_e,x_e)|\le\theta_{\mathrm{half}}.
\]

The current manifest requires a half-FOV field even in calibrated-image mode;
it is inactive in that mode. A typed camera-support union is deferred to a
later schema revision rather than silently changing v1alpha1.

An object center is pre-fault eligible only when all of the following are true:

1. its recorded center in \(e_r\) satisfies the ego-forward and lateral
   limits;
2. the nominal LiDAR-support proxy marks the object available;
3. using the nominal recorded camera pose, either:
   - the center is in calibrated image bounds and beyond minimum depth, or
   - for a procedural source without intrinsics, its ego-BEV bearing is within
     the declared symmetric camera half-FOV; and
4. both modality estimators mark it eligible before any fault is inserted.

Faulted reported calibration may change the reconstructed estimate but may not
change this denominator. ROI code accepts a typed eligibility transform
separately from a typed reported reconstruction transform so that a caller
cannot accidentally use corrupted metadata for eligibility.

For later nuScenes replay, recorded `num_lidar_pts > 0` is a LiDAR-support
proxy. It is not detector availability or proof that an estimator would emit a
valid center. M2 validates and exposes this field but does not yet publish
replay losses.

## 7. Minimal nuScenes-mini adapter

The adapter uses the Python standard library plus the existing NumPy
dependency. The official devkit is an external validation reference, not a
base runtime dependency.

The adapter's local Python API accepts an explicit `Path`, but the M2 public
and release CLI accepts only the fixed environment name `NUSCENES_ROOT`
through `--dataset-root-env NUSCENES_ROOT`; it rejects literal or relative
dataset-root arguments. The environment value is held only in local runtime
state. It must not appear in a manifest, validation record, digest input,
generated public figure, success message, or tracked test fixture.

M2 reads only the metadata required for grounding:

- `scene`;
- `sample`;
- `sample_data`;
- `attribute`;
- `sensor`;
- `calibrated_sensor`;
- `ego_pose`;
- `sample_annotation`;
- `instance`;
- `category`;
- `log`;
- `visibility`; and
- referenced `CAM_FRONT`/`LIDAR_TOP` key-frame blob existence.

The adapter must reject:

- a missing or non-list table;
- duplicate or empty primary tokens;
- missing foreign keys;
- broken `prev`/`next` reciprocity;
- a scene sample chain whose endpoints or length disagree with the scene row;
- an instance annotation chain whose endpoints or length disagree with the
  instance row;
- non-increasing timestamps in sample, annotation, or sample-data chains;
- a sample-data chain that changes scene, channel, or calibration identity;
- a key sample without exactly one `CAM_FRONT` and one `LIDAR_TOP` record;
- modality/channel mismatches;
- an ego-pose timestamp that differs from its sample-data timestamp;
- invalid timestamps, image dimensions, intrinsics, translations,
  quaternions, annotation dimensions, or point counts; and
- a referenced key-frame camera/LiDAR blob that is absent, empty, a symlink, or
  escapes the dataset root.

An empty annotation visibility token is allowed exactly as documented by the
upstream schema. Positive image dimensions and \(3\times3\) pinhole intrinsics
are required only for camera calibration/sample-data records. Blob validation
checks exactly one `CAM_FRONT` and one `LIDAR_TOP` key-frame file for each of
the 404 samples—808 references total. It does not read file contents or inspect
sweeps, radar, maps, or other camera channels.

The official mini profile additionally expects 10 scenes, 404 samples, and
18,538 sample annotations. A mismatch fails the profile check but is not
presented as evidence that arbitrary nuScenes versions are corrupt.

## 8. Public and local artifacts

The tracked pre-execution manifest is
[`examples/validation/m2-geometry-v1.json`](../examples/validation/m2-geometry-v1.json)
with `schema="ffb.geometry-validation-manifest/v1"`. It contains no local
scene name, token, timestamp, filename, or path. It freezes the dataset
profile, frame roles, exact checks, diagnostic selector, fixture identity,
property-test draw contract, tolerances, covariance Monte Carlo contract,
artifact layout, dataset-terms notice, and exact output field allowlists.
It contains no local dataset filename/path, absolute path, scene name, token,
timestamp, or credential; repository-relative source and generated-output
paths are intentionally part of reproducibility.

The independent synthetic fixture is
[`tests/fixtures/m2_geometry_reference_v1.json`](../tests/fixtures/m2_geometry_reference_v1.json).
Its exact file SHA-256 is
`0676993f48e5a40034dfe497df7165b33f2d2f96dad234afd62af8e461beb252`.
After the final adversarial preregistration amendments, the canonical manifest
SHA-256 is
`7bfb5427c1ea5450a795bedef327457e5959860316bcd23f930e6eaa5917a068`.

The clean-source runner consumes that tracked manifest and resolves the local
dataset root only from the named runtime environment variable. The root and
its value are never passed to the artifact writer. The complete artifact tree
has exactly these five regular, non-symlink files:

```text
manifest.json
geometry-validation.json
payload-index.json
run.json
_SUCCESS
```

Each file is canonical single-object JSON plus one LF, is capped at 1 MiB, and
the complete tree is capped at 5 MiB. No subdirectory or extra entry is
allowed.

The identity graph is acyclic:

```text
manifest.json
  -> manifest_sha256
  -> run_id
  -> geometry-validation.json
  -> payload-index.json
  -> artifact_sha256
  -> run.json
  -> run_sha256
  -> _SUCCESS
```

- `manifest_sha256` is the existing key-sorted canonical-intent digest.
- `run_id` reuses M1's domain-separated, length-framed derivation over the
  manifest digest, Git revision, lockfile digest, package version, and the new
  literal artifact contract
  `ffb.geometry-validation-payload/v1`. The generic helper must accept that
  contract as an explicit argument; released M1 behavior remains unchanged.
- `payload-index.json` has
  `schema="ffb.geometry-payload-index/v1"`, the literal artifact contract,
  `run_id`, `manifest_sha256`, and exactly two ordered raw-file entries:
  `manifest.json`, then `geometry-validation.json`. Each entry contains path,
  byte length, and raw SHA-256.
- `artifact_sha256` reuses M1's
  `fusion-fault-bench/artifact/v1` domain and length-framed exact canonical
  index-file bytes.
- `run.json` reuses `ffb.run/v1alpha1` without adding duration or memory
  fields. Its fixed logical command is
  `ffb geometry validate examples/validation/m2-geometry-v1.json
  --dataset-root-env NUSCENES_ROOT --output-dir
  reports/generated/m2-geometry`. The environment variable's value and any
  literal/relative dataset argument are prohibited.
- `run_sha256` reuses M1's
  `fusion-fault-bench/run-record/v1` domain and length-framed finalized
  canonical run-file bytes.
- `_SUCCESS` reuses `ffb.success/v1alpha1`, contains only
  `artifact_sha256` and `run_sha256`, and is written last.

A single immutable M2 layout drives scan, caps, indexed order, write order,
cleanup, and strict load. Descriptor-safe staging and no-replace publication
may be factored from M1, but M1 wrappers and its released five-member analytic
payload remain unchanged.

`geometry-validation.json` is a code-owned strict model; the manifest does not
create arbitrary output fields. The exact top-level and nested field names are
duplicated in the manifest as a cross-check and include only:

- fixed schema/run/manifest identities and a fixed dataset-terms notice;
- the official three expected headline counts; explicitly named attestations
  for headline match, structural integrity, key-frame blob validation, local
  projection cross-check, and diagnostic generation; the literal 808
  key-frame check count; and a fixed no-dataset-authentication notice;
- synthetic fixture identity plus transform, round-trip, projection, depth,
  quaternion-sign, and box-corner maximum errors;
- synthetic finite-difference and three ordered covariance-entry Monte Carlo
  gate records plus their maxima; and
- component pass flags plus their recomputed conjunction.

The exact result typing and derivation rules are:

- top-level `schema` is the literal `ffb.geometry-validation/v1`; `run_id`
  uses the existing bounded identifier grammar; digests are 64 lowercase hex
  characters and must match the bundle;
- `dataset_terms` must equal the manifest's five fixed strings exactly;
- `dataset_validation.profile`, its three `expected_headline_counts`, the
  `keyframe_blob_check_count=808`, and
  `dataset_authentication="summary-does-not-authenticate-dataset-bytes"` are
  fixed literals;
- the five dataset fields ending in `_attested` are strict booleans written by
  the local runner. They deliberately attest checks whose source rows and
  residuals are absent from the public artifact. The loader cannot recompute
  those checks from sanitized evidence and must not claim that it can;
- `dataset_validation.all_checks_passed` is recomputed as the conjunction of
  those five attestations and the exact 808 count;
- every synthetic error is a finite float greater than or equal to zero;
  `synthetic_geometry_validation.all_checks_passed` is recomputed by comparing
  each error with its named manifest tolerance, including the exact fixture
  identity and digest;
- `covariance_entries` contains exactly `xx`, `xy`, and `yy` in that order.
  Each record has finite `absolute_error_m2 >= 0`,
  `allowed_error_m2 > 0`, `gate_ratio >= 0`, and boolean `passed`.
  `gate_ratio = absolute_error_m2 / allowed_error_m2` and
  `passed = gate_ratio <= 1`, reconciled at absolute tolerance \(10^{-12}\);
- the three covariance maximum fields are the ordinary maxima of their three
  corresponding entry values; `monte_carlo_sample_count` is exactly 200,000;
  and `actual_sampling_gate_passed` is the conjunction of the three entry
  flags;
- `reported_role_separation_passed_attested` is a strict boolean attestation
  because the public artifact intentionally omits the full propagated
  matrices;
- `covariance_validation.all_checks_passed` is recomputed as finite-difference
  error at or below \(10^{-7}\), exact sample count, actual sampling gate pass,
  and reported-role attestation; and
- top-level `all_checks_passed` is the conjunction of the three component
  `all_checks_passed` values. Public release requires it to be true.

The strict bundle validator independently recomputes every identity, numeric
derivative above, synthetic tolerance decision, and conjunction that is
recomputable from sanitized evidence. It validates but does not mislabel the
five local-data booleans or role-separation boolean as independently
recomputed. It rejects noncanonical JSON, extra/missing fields or files,
invalid bounds, wrong order/literals, non-finite values, manifest/result
mismatch, and unrecognized strings.

Numeric transform/projection disagreement in public evidence comes only from
the repository-owned synthetic fixture. The local nuScenes projection exposes
one boolean. It never exposes a residual, selected-sample count, pixel, depth,
token, or SVG. The sanitized summary authenticates only its own bytes and
declared aggregate checks; it does not authenticate the exact dataset archive
or metadata-table bytes and must not be used as a dataset cache key.

All source nuScenes files and the dataset root remain outside this repository;
they may not be copied even to an ignored directory. Only the generated local
diagnostic SVG and private validation detail may be written under ignored
`reports/generated/`, outside the candidate artifact directory. Every
candidate public stdout, stderr, exception, and failure log is built from
fixed sanitized messages; generic filesystem exceptions or adapter objects
are never serialized.

The code and repository-owned synthetic evidence remain under Apache-2.0.
Any tracked local nuScenes-derived aggregate summary is separately marked
`CC BY-NC-SA 4.0 plus Motional Dataset Terms`, includes the frozen attribution
and non-endorsement notice, and is limited to non-commercial use under the
user-accepted upstream terms. It is not relicensed by this repository's
Apache-2.0 license. Commercial users must obtain appropriate permission.

## 9. Negative controls and failure tests

Required negative controls are:

- a unit scalar-last array is never silently reordered, while explicit
  `wxyz` interpretation and \(q/-q\) equivalence match the fixture;
- reversing either nuScenes pose direction changes the synthetic projection
  and fails the oracle;
- substituting the LiDAR-time ego pose for the camera-time pose fails the
  independent projection fixture;
- composing transforms with mismatched intermediate or log-qualified frames
  raises an error;
- projecting a zero/negative-depth point returns invalid without division;
- a reflection matrix is rejected;
- invalid, asymmetric, indefinite, role-free, or wrong-unit covariance inputs
  are rejected;
- center-in-image and devkit-box-`ANY` counterexamples demonstrate that the two
  support rules are not interchangeable;
- exact open image/depth and closed longitudinal/lateral/FOV boundary cases
  pass their declared decisions;
- corrupting a foreign key, chain link, channel modality, or blob path is
  detected in a tiny synthetic metadata fixture;
- calibration corruption applied to both generation and reconstruction is
  prohibited by API separation tests;
- reported calibration cannot change pre-fault ROI membership; and
- local absolute paths and dataset tokens are absent from every candidate
  tracked artifact, stdout, stderr, and failure record.

## 10. Acceptance gates

M2 may be released only when:

1. transform composition, inverse, and round-trip property tests pass for 256
   PCG64DXSM-generated rigid transforms using the frozen manifest seed, draw
   order, distributions, and per-quantity tolerances;
2. scalar-first quaternion and nuScenes transform-direction fixtures pass;
3. the independent projection fixture and local scalar cross-check satisfy the
   frozen tolerances;
4. the finite-difference and 200,000-draw covariance gates pass;
5. synthetic referential-integrity mutation tests fail closed;
6. the local official mini profile passes all structural, link, calibration,
   and referenced-blob checks;
7. the deterministic local projection diagnostic is generated and visually
   inspected, while remaining untracked;
8. base package import remains free of optional devkit, plotting, image, Torch,
   and CUDA dependencies;
9. the five-file geometry-validation artifact round-trips through its strict
   loader and rejects identity, contradiction, path-leak, extra-file,
   noncanonical, race, and partial-publication mutations;
10. Ruff format/check, strict Pyright, the full test suite, package build, and
    wheel smoke test pass;
11. an independent adversarial implementation review finds no release blocker;
12. the tracked-file and candidate-output audit finds no dataset payload,
    derived per-frame payload, absolute local path, private interview material,
    credential, or generated raw output;
13. dataset-derived aggregate evidence carries the declared separate terms,
    attribution, and non-endorsement notice; and
14. public methodology, limitations, reproducibility, release evidence, and
    private interview-learning material are updated together.

## 11. Explicit non-goals

M2 does not:

- read point-cloud contents or infer visibility from raw returns;
- run a detector, neural network, Torch, CUDA, or GPU workload;
- train or tune an estimator;
- estimate velocity, interpolate annotations, or build temporal sequences;
- inject a calibration or timing sweep;
- publish per-scene or per-frame nuScenes results;
- claim real sensor-noise transfer, raw-sensor robustness, fault prevalence,
  safety, or fleet generalization; or
- change the M1 analytic result.

Temporal generation and fault sweeps begin in M3. nuScenes matched-center
temporal replay begins in M5.

## 12. Pinned external references

The independent convention review is pinned to official nuScenes-devkit
revision
[`d9de17a73bdc06ce97a02f77ae7edb9b0406e851`](https://github.com/nutonomy/nuscenes-devkit/tree/d9de17a73bdc06ce97a02f77ae7edb9b0406e851):

- [metadata schema](https://github.com/nutonomy/nuscenes-devkit/blob/d9de17a73bdc06ce97a02f77ae7edb9b0406e851/docs/schema_nuscenes.md);
- [official transform and sample-data path](https://github.com/nutonomy/nuscenes-devkit/blob/d9de17a73bdc06ce97a02f77ae7edb9b0406e851/python-sdk/nuscenes/nuscenes.py);
- [projection, visibility, and transform helpers](https://github.com/nutonomy/nuscenes-devkit/blob/d9de17a73bdc06ce97a02f77ae7edb9b0406e851/python-sdk/nuscenes/utils/geometry_utils.py);
  and
- [official box-corner order](https://github.com/nutonomy/nuscenes-devkit/blob/d9de17a73bdc06ce97a02f77ae7edb9b0406e851/python-sdk/nuscenes/utils/data_classes.py).
