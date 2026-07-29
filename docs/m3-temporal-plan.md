# M3 Temporal Procedural Benchmark Pre-registration

Status: **pre-registered; not executed**.

This document freezes the M3 population, generator, fault operators, experiment
matrix, estimands, inference, validation gates, and claim boundary before any
M3 result is generated. The normative v0.1 definitions in
[Benchmark Contract v0.1](benchmark-contract-v0.1.md) remain controlling.

M3 is CPU-only and dataset-independent. It evaluates matched camera/LiDAR
estimator outputs produced from known procedural BEV centers. It does not
evaluate a detector, raw images, point clouds, association, planning, or a
learned health policy.

## 1. Question and predeclared hypotheses

The primary question is:

> Under the frozen procedural population and one declared proxy fault, when
> does fixed information fusion have higher complete-sequence matched-center
> loss than the modality designated healthy?

The predeclared hypotheses are:

1. Persistent output bias, calibration translation/yaw, timestamp offset, and
   underreported noise can make fixed fusion harmful at a tested severity.
2. Correctly reported increased independent Gaussian noise remains beneficial
   in the population at every finite tested scale because information fusion
   can downweight it.
3. Dropout reduces target and fixed-fusion coverage; it has no localization
   crossover estimand.
4. A shared common-mode bias can increase absolute localization loss while
   leaving camera-LiDAR disagreement unchanged.

Observed, not-observed, undetermined, or contrary outcomes are all releasable.
No minimum improvement or number of observed crossovers is an acceptance gate.
The complete frozen matrix will be published; favorable roots may not be
selected after inspection.

## 2. Immutable profiles and populations

The exact JSON inputs are content-addressed:

| Profile | Purpose | SHA-256 |
|---|---|---|
| `constant-velocity-front-roi-v1` | Main train, validation, and test population | `4771a6e69d75b9af41f99ab794c0af1b51e6103e43474c8e0f07df3e6f3ca68c` |
| `constant-velocity-fov-edge-v1` | Separate edge-support/common-mode control | `ca1544f69023847af7bdad9f1306ae3885f2e5d067d6afc026038f87ae36448d` |
| `constant-velocity-ci-smoke-v1` | Small committed CI-only profile | `7f2479c064e0f8104789dfc3ce704a78aabdd46c1be7a31fdd2e75dbe3b407ed` |

The runner must resolve the tracked profile by `profile_id`, recompute its
canonical digest, and fail closed on any mismatch. Scientific CLI overrides are
forbidden.

### 2.1 Main constant-velocity profile

Each split contains exactly 200 complete sequences. Each sequence has six known
objects, `object:00` through `object:05`, and 48 frames:

\[
t_k=0.1k\ \mathrm{s},\quad k=0,\ldots,47,\qquad
p_{o,k}=p_{o,0}+v_o t_k.
\]

One sequence-scoped latent-stream call produces
`Generator(PCG64DXSM(seed)).random((6, 4), dtype=float64)`. For row
\(U_o=(U_0,U_1,U_2,U_3)\), the split mappings are:

- Train, `near-two-lane-flow`:
  \(x_0=10+18U_0\);
  \(y_0\) is \(-3.5\) for even objects and \(+3.5\) for odd objects,
  plus \(0.25(2U_1-1)\);
  \(v_x=-1+2U_2\);
  \(v_y=0.1(2U_3-1)\).
- Validation, `mid-lateral-crossing`:
  \(x_0=30+10U_0\);
  side is \(-1\) for even objects and \(+1\) for odd objects;
  \(y_0=\mathrm{side}(5+3U_1)\);
  \(v_x=-1+2U_2\);
  \(v_y=-\mathrm{side}(1.5+1.5U_3)\).
- Test, `far-fast-approach`:
  \(x_0=44+12U_0\);
  lateral centers are \([-7,-4,-1,1,4,7]\) m in object order plus
  \(0.25(2U_1-1)\);
  \(v_x=-(3+2U_2)\);
  \(v_y=0.2(2U_3-1)\).

M3 releases only the 200 main-profile test sequences. Train and validation are
committed now to establish disjoint layout/range/motion families for M4; M3
must not tune on them.

### 2.2 Edge-support control

The edge profile contains 100 test sequences, four objects, and the same
timeline. For each object:

\[
r=20+20U_0,\quad
a=\mathrm{side}\,[0.7-(0.005+0.015U_1)],
\]

\[
p_0=r[\cos a,\sin a]^\top,\quad
q=-0.5+U_2,\quad
v=q[\cos a,\sin a]^\top.
\]

Side is \(+1\) for even objects and \(-1\) for odd objects. This profile tests
support bookkeeping near the declared field-of-view edge. Constant,
translation-invariant Cartesian noise does not make these frames intrinsically
harder, so M3 will not claim range- or image-edge-dependent measurement
realism.

### 2.3 Eligibility and ordering

Eligibility is computed once from true, noiseless states before any fault or
random observation:

- ego-forward \(x\in[5,60]\) m;
- \(|y|\le40\) m;
- \(|\operatorname{atan2}(y,x)|\le0.7\) rad;
- nominal support from both estimators.

The ordered eligible object-frame identifiers and their hash are frozen per
sequence and reused across every method, severity, direction, and dropout
probability. Noisy or corrupted estimates leaving the ROI remain scored.
Dropout changes valid count, never eligible count. There are no retries or
result-dependent exclusions.

Canonical order is sequence index, frame index, UTF-8 object ID, then
coordinate \(x,y\). Result rows are ordered by sequence; condition identity
then increasing magnitude, with negative before positive for signed axes; the
manifest method order; and, for availability rows, coverage, conditional loss,
then undefined-output rate.

## 3. Sensor and reconstruction model

The base estimator-output errors are temporally IID, independent between
modalities, isotropic, constant-range, zero-mean Cartesian Gaussians:

- camera actual and reported standard deviation: \([1.0,1.0]\) m;
- LiDAR actual and reported standard deviation: \([0.3,0.3]\) m.

Isotropic camera uncertainty is deliberate: a yaw perturbation preserves its
Cartesian covariance, so M3 does not silently introduce an undeclared full
covariance or uncertainty-misreporting effect.

The stationary rig uses a synthetic true front-camera extrinsic
\(T_{e\leftarrow c}^{\mathrm{true}}\) with translation
\([1.5,0,1.5]\) m and scalar-first quaternion
\([0.5,-0.5,0.5,-0.5]\). It maps optical camera axes
\((x\ \mathrm{right},y\ \mathrm{down},z\ \mathrm{forward})\) to ego axes
\((x\ \mathrm{forward},y\ \mathrm{left},z\ \mathrm{up})\).

For a camera error \(\epsilon_C=[\epsilon_x,\epsilon_y,0]^\top\),

\[
q_C=(T_{e\leftarrow c}^{\mathrm{true}})^{-1}(p_e+\epsilon_C),
\qquad
z_C=T_{e\leftarrow c}^{\mathrm{reported}}q_C.
\]

Generation always uses the true transform. Only reconstruction receives
reported metadata. LiDAR uses \(z_L=p_e+\epsilon_L\). Calibration faults use
left ego-frame composition:

\[
T_{e\leftarrow c}^{\mathrm{reported}}
=\Delta T_eT_{e\leftarrow c}^{\mathrm{true}}.
\]

Consequently, pure ego-frame calibration translation is exactly equivalent to
the same additive camera-output bias. That equivalence is an implementation
oracle, not a second independent scientific finding.

Timestamp offset uses a versioned oracle constant-velocity alignment proxy:

\[
t_\mathrm{reported}=t_\mathrm{true}+\delta,\qquad
z_\mathrm{aligned}(t_k)
=z_\mathrm{raw}+v(t_k-t_\mathrm{reported})
=z_\mathrm{raw}-v\delta.
\]

It does not re-pair frames, clip endpoints, change noise draws, or generate the
observation at \(t_k+\delta\). The exact latent velocity isolates timestamp
metadata error from velocity-estimation error; it is not a deployable health
feature.

## 4. RNG and counterfactual pairing

All manifests use:

- NumPy PCG64DXSM;
- SHA-256 name-and-sequence stream derivation;
- data master seed `1729`;
- bootstrap seed `2718`;
- named `latent`, `camera`, `lidar`, and `fault` streams.

Each sequence receives one camera standard-normal array and one independent
LiDAR array in eligible frame/object order. The arrays are reused across all
methods, severities, and signed directions. Dropout receives one uniform draw
per target-modality frame, shared by all eligible objects in that frame. The
same uniforms are reused at every probability, so masks are nested.

## 5. Frozen experiment matrix

The ordered matrix is committed at
`examples/matrices/m3-procedural-v1.json`, with canonical SHA-256
`7162a01bb766a38e8f7196cba9eb2c6c546f5866bc640f220252bb6e7aab6f9b`.
All primary manifests use the main test profile; the common-mode control alone
uses the edge profile.

| Experiment | Target / axis | Severity grid |
|---|---|---|
| LiDAR additive bias | LiDAR \(y\) | \(0,.25,.5,1,2,4\) m, paired ± |
| Correctly reported noise | camera \(xy\) | \(1,1.25,1.5,2,4\) std-scale |
| Underreported noise | camera \(xy\) | \(1,1.25,1.5,2,4\) std-scale |
| Calibration translation | camera \(x\) | \(0,.25,.5,1,2,4\) m, paired ± |
| Calibration yaw | camera yaw | \(0,.005,.01,.02,.04,.08\) rad, paired ± |
| Timestamp offset | camera time | \(0,.05,.1,.2,.4,.8\) s, paired ± |
| Dropout control | camera availability | \(0,.1,.25,.5,.75,1\) probability |
| Common-mode control | both sensors, \(x\) | \(0,.25,.5,1,2,4\) m, paired ± |

Meters, radians, seconds, probabilities, and standard-deviation scales are
separate physical axes and are never pooled into one severity.

## 6. Methods, estimands, and inference

Crossover manifests use camera only, LiDAR only, fixed information fusion,
fault-target-drop, and the complete-sequence performance oracle.
Fault-target-drop uses fixed fusion at identity and the healthy modality at
every nonidentity severity. The performance oracle selects exactly one of
camera, LiDAR, or fixed fusion from complete-sequence losses; per-frame or
per-object hindsight is forbidden.

For complete sequence \(j\),

\[
L_{m,j}(s)=\frac{1}{n_j}\sum_{i=1}^{n_j}
\|\widehat p_{m,ji}(s)-p_{ji}\|_2^2,
\qquad
d_j(s)=L_{F,j}(s)-L_{H,j}(s).
\]

The primary population estimand is the equal-sequence mean
\(D_H(s)=N^{-1}\sum_j d_j(s)\). Inference remains signed. Harmful-fusion gap is
formed only after inference. Each manifest uses 2,000 paired complete-sequence
bootstrap replicates, pointwise 95% percentile intervals, the frozen
equal-severity nondecreasing PAVA rule, linear first-zero interpolation, and
\(10^{-12}\,\mathrm{m}^2\) zero tolerance.

The correctly reported-noise population reference has no finite crossover;
the underreported case may cross. Raw sampled points remain visible even when
PAVA is fitted. Pointwise intervals across the matrix are not simultaneous
family-wise inference, and the release will state that multiplicity boundary.

Dropout is an availability control. It reports coverage, undefined-output
rate, and matched-center MSE conditional on valid output. Fixed fusion is
undefined whenever the target input is absent. At nonidentity probability,
target-drop uses the healthy modality on all frames. No zero localization
penalty is imputed, and there is no crossover or performance oracle.

The common-mode control uses camera, LiDAR, and fixed fusion absolute losses
only. It has no healthy reference, target-drop, performance oracle, or
crossover.

## 7. Exact statistical validation

The main test profile has exactly
\(n=200\times 48\times 6=57{,}600\) eligible object-frames at identity.
Moment checks pool only those ordered main-test identity draws; the edge profile
is checked separately with \(n=100\times48\times4=19{,}200\). No train or
validation draw enters an M3 release gate.

For one zero-mean configured error coordinate with standard deviation
\(\sigma\), let \(\bar e\) be the sample mean and \(s^2\) the unbiased sample
variance computed with `ddof=1`. The frozen checks are:

\[
|\bar e|\le \frac{6\sigma}{\sqrt n},
\qquad
|s^2-\sigma^2|\le
6\sigma^2\sqrt{\frac{2}{n-1}}.
\]

For each of the four camera-LiDAR coordinate pairs, sample cross-covariance is
computed with `ddof=1` after subtracting the two sample means and must satisfy

\[
|\widehat{\operatorname{cov}}(e_{C,a},e_{L,b})|
\le \frac{6\sigma_C\sigma_L}{\sqrt{n-1}},
\quad a,b\in\{x,y\}.
\]

The within-camera and within-LiDAR \(x,y\) sample covariances use the analogous
\(6\sigma_x\sigma_y/\sqrt{n-1}\) bound. Reported covariance is not estimated
from these samples.

Expected-loss checks use an independent affine-Gaussian representation
conditional on every frozen latent object-frame. Stack its four independent
base normal coordinates as \(u\sim\mathcal N(0,I_4)\) and express one method
error as \(A u+b\). For one squared loss,

\[
\mathbb E[\|Au+b\|^2]
=\operatorname{tr}(A^\top A)+b^\top b,
\]

\[
\operatorname{Var}[\|Au+b\|^2]
=2\operatorname{tr}[(A^\top A)^2]
+4b^\top AA^\top b.
\]

For a signed contrast between methods 1 and 2, use
\(M=A_1^\top A_1-A_2^\top A_2\),
\(c=A_1^\top b_1-A_2^\top b_2\), and
\(d=b_1^\top b_1-b_2^\top b_2\):

\[
\mathbb E[u^\top Mu+2c^\top u+d]=\operatorname{tr}(M)+d,
\qquad
\operatorname{Var}=2\operatorname{tr}(M^2)+4c^\top c.
\]

Object-frame errors are independent conditional on each persistent fault.
Their expectations and variances are combined into each declared
object-frame-mean sequence loss, then into the equal-sequence population mean:
if sequence \(j\) has \(n_j\) rows and per-row variances \(v_{ji}\),

\[
\operatorname{SE}(\bar L)
=\sqrt{\frac{1}{N^2}\sum_j
\frac{1}{n_j^2}\sum_i v_{ji}}.
\]

Every released raw non-availability affine method-loss aggregate (camera,
LiDAR, fixed fusion, and target-drop) and signed contrast, at every condition,
must differ from this independently computed expectation by no more than six
analytic standard errors, using inclusive `<=` comparison. The same formulas
are used for common-mode absolute losses. The complete-sequence performance
oracle is a non-affine minimum and is excluded from the analytic-moment gate;
its exact sample-row selection semantics are independently recomputed instead.
PAVA values and roots are never inputs to this gate. Availability rows instead
use the exact-mask checks below and independent bundle ratio recomputation.

## 8. Independent validation and release gates

Implementation may not run the release matrix until an adversarial plan review
passes. M3 evidence may not be promoted until all applicable gates below pass.

1. A zero-error lower-level fixture recovers every eligible true center within
   \(10^{-12}\) m for camera, LiDAR, and fusion.
2. Profile loading is fail-closed: schema, profile ID, canonical digest, split
   count, ROI, isotropic-yaw requirement, and manifest linkage are checked.
3. The independent latent oracle verifies sequence IDs, the exact
   \(p_k=p_0+v t_k\) mapping, family supports, split disjointness, object/frame
   order, eligibility hashes, and absence of retries.
4. Independent scalar formulas match calibration translation, yaw displacement
   \((R_\theta-I)p\), and timing displacement \(-v\delta\) within
   \(10^{-12}\) m. Static tracks are timing-invariant.
5. A mutation using corrupted calibration in both generation and reconstruction
   must fail the independent reconstruction oracle.
6. Calibration translation and equal-target/axis additive bias agree in a
   lower-level metamorphic fixture within \(10^{-12}\) m per center and
   \(10^{-12}\,\mathrm{m}^2\) per sequence loss.
7. Empirical per-coordinate means, `ddof=1` variances, within-sensor
   cross-covariances, and all four camera-LiDAR cross-covariances pass the exact
   formulas in section 7.
8. Correctly reported variance scales exactly as \(k^2\); underreported
   covariance remains exactly nominal and is never estimated from samples.
9. The independent affine-Gaussian oracle in section 7 verifies every
   applicable raw non-availability affine method loss and signed contrast at
   the fixed six-SE bound. The complete-sequence performance oracle is verified
   by exact candidate-loss selection rather than an affine moment formula.
   Expected curves—not PAVA output—must have the declared response:
   quadratic/sign-symmetric bias and timing terms, bounded monotone yaw
   displacement, increasing underreported-noise contrast, and
   correctly-reported contrast approaching zero from below. On the frozen yaw
   grid, each noiseless center displacement is exactly
   \(2r\sin(|\theta|/2)\), its squared loss is
   \(2r^2(1-\cos\theta)\), and the conditional expected signed contrast must be
   nondecreasing under exact floating-point comparison up to the independent
   \(10^{-12}\,\mathrm{m}^2\) oracle-reconciliation tolerance.
10. The healthy unimodal output, loss, count, and status are invariant across
    every severity and direction. Identity equality is required only among
    manifests sharing profile, split, sequence count, observations, seeds, and
    comparable methods. The dropout \(p=0\) conditional losses must equal the
    corresponding main-profile localization losses for camera, LiDAR, fixed
    fusion, and target-drop. The different edge-profile control is excluded
    from cross-profile identity equality. Identity comparison removes only
    `run_id`, `manifest_sha256`, `fault_family`, `fault_axis`, the severity
    coordinate, and the manifest experiment identifier; sequence ID, method,
    status, value, unit, eligible count, and valid count must remain identical.
    For dropout identity only, `conditional-matched-center-mse` is mapped to
    `matched-center-mse` before comparison.
11. Eligible denominators and ordered eligibility hashes never change after a
    fault. Complete-sequence aggregation and sequence-level oracle selection
    have mutation tests against object-frame weighting and per-frame hindsight.
12. An independent scalar reference reimplements the v0.1 SHA-256 framed
    fault-stream seed derivation without calling the production seed helper,
    constructs exactly one PCG64DXSM `float64` uniform vector of length 48 per
    sequence, and compares every `uniform < probability` frame mask at every
    registered probability byte-for-byte. It also verifies that every eligible
    object in a frame shares the same mask, masks are nested, and frame order is
    increasing. At \(p=0\), all coverage is one and target-drop equals fusion.
    At \(p=1\), target and fusion coverage are zero and conditional loss is
    undefined, while healthy and target-drop coverage remain one.
13. Common-mode injection preserves camera-LiDAR disagreement exactly and
    shifts each noiseless method by the declared common bias.
14. The generic bundle validator independently recomputes row completeness,
    method identities, aggregates, bootstraps, PAVA roots, crossing fractions,
    statuses, censoring, and availability ratios.
15. Unit, property, analytic-oracle, integration, resource-cap, privacy, and
    isolated-wheel smoke tests pass. Two clean executions have byte-identical
    indexed scientific payloads.
16. An independent adversarial implementation/results review finds no
    unresolved release blocker.

The PAVA fit cannot satisfy the raw monotonic-response gate because it is
monotonic by construction.

## 9. Matrix and validation-record contracts

The strict `ffb.experiment-matrix/v1` loader accepts only a top-level object
with the matrix ID, ordered execution entries, ordered profile entries, release
split, result-selection rule, and scientific-override rule. For
`m3-procedural-v1` it requires exactly the frozen eight manifests in their
committed order and exactly the three profiles in their committed order. Paths
must be normalized repository-relative POSIX paths under `examples/manifests`
or `examples/profiles`; absolute paths, `..`, symlinks, duplicates, extra
entries, and missing entries are rejected. Every referenced canonical digest
and the matrix digest are recomputed before execution and loading. The only
other accepted M3 matrix ID is `m3-ci-smoke-v1`, which requires exactly its one
committed smoke manifest and one committed smoke profile; it is never accepted
as release evidence.

Each indexed `procedural-validation.json` must contain these typed fields rather
than an unauditable top-level assertion:

- schema, run ID, manifest digest, profile ID/digest, split, sequence count,
  frame count, object count, and total eligible count;
- individual booleans for profile schema, ID, digest, split count, ROI,
  isotropic-yaw compatibility, split-family support, and canonical ordering;
- the SHA-256 of the ordered per-sequence eligibility commitments plus minimum
  and maximum eligible counts and an eligibility-invariance boolean;
- maximum absolute identity, translation, translation/bias-equivalence, yaw,
  timing, and static-timing oracle discrepancies with their units and fixed
  tolerances, plus the fault-cancellation mutation result;
- one typed moment-check row per declared mean, variance, within-sensor
  covariance, and camera-LiDAR covariance, containing sample count, `ddof`,
  expectation, observed value, six-SE bound, absolute discrepancy, and pass;
- one typed expected-loss row per applicable affine manifest condition, method,
  and metric, containing the expected value, empirical value, analytic SE,
  absolute standardized error, and pass;
- for dropout, the independently derived uniform-vector commitment, exact-mask
  comparison count, frame-sharing/nesting/endpoint booleans, and maximum mask
  discrepancy; otherwise an explicit not-applicable dropout section;
- identity comparison scope/count/max discrepancy, common-mode disagreement
  discrepancy when applicable, implied row/bootstrap-cell resource counts, each
  resource-cap boolean, and a recomputed conjunction over all applicable
  single-run gates.

The strict loader independently rebuilds those values from the profile,
manifest, and sequence rows wherever possible and rejects contradictory pass
flags. Independently recomputed expected values, standard errors, discrepancies,
and bounds reconcile to the record at absolute tolerance \(10^{-12}\) in their
declared unit with zero relative tolerance; categorical values and counts are
exact.

Deterministic repeat comparison cannot be embedded in one indexed artifact
without a circular dependency. The curated release therefore additionally
contains a typed `repeat-verification.json` with both artifact digests, the
ordered indexed-member SHA-256 pairs, exact comparison count, mismatch count,
and recomputed all-equal conjunction. It also contains measured named-CPU wall
time and peak memory. The release validator rebuilds this record from both
strictly loaded artifacts. Curated aggregate records, crossover records,
validation summaries, figures, and repeat evidence are tracked; raw sequence
NDJSON remains ignored and is committed to by hash/count only.

## 10. Artifact and execution contract

M1's released analytic artifact loader remains unchanged. M3 uses a distinct
`ffb.procedural-payload/v1` artifact with:

```text
manifest.json
procedural-profile.json
sequence-metrics.ndjson
aggregate-metrics.ndjson
crossovers.ndjson
procedural-validation.json
payload-index.json
run.json
_SUCCESS
```

The new contracts are `ffb.procedural-profile/v1`,
`ffb.procedural-validation/v1`, and `ffb.payload-index/v1alpha2`.
`crossovers.ndjson` may be canonical empty bytes only for availability and
common-mode manifests. The loader has an exact allowlist, canonical-byte and
digest checks, symlink/path hardening, row caps, and independent scientific
recomputation.

The indexed member order is exactly:

1. `manifest.json`;
2. `procedural-profile.json`;
3. `sequence-metrics.ndjson`;
4. `aggregate-metrics.ndjson`;
5. `crossovers.ndjson`;
6. `procedural-validation.json`.

`PayloadIndexV1Alpha2` contains schema, literal artifact contract
`ffb.procedural-payload/v1`, run ID, manifest digest, profile digest, and
exactly six file entries in that order. Each entry contains its literal path,
byte length, and SHA-256. All indexed members except `crossovers.ndjson` have
length in `[1, 512 MiB]`. Crossover manifests require a nonempty crossover
member; availability and common-mode manifests require it to be exactly zero
bytes. The whole artifact allowlist is the six indexed members followed by
`payload-index.json`, `run.json`, and `_SUCCESS`, with no symlinks or other
entries.

The procedural artifact digest is:

```text
SHA256(
  b"fusion-fault-bench/procedural-artifact/v1\0"
  || uint64_big_endian(len(canonical_payload_index_bytes))
  || canonical_payload_index_bytes
)
```

The deterministic run ID retains the v0.1 domain and four-byte framing:

```text
"run:" || SHA256(
  b"fusion-fault-bench/run-id/v1\0"
  || frame4(manifest_sha256)
  || frame4(git_revision)
  || frame4(lockfile_sha256)
  || frame4(package_version)
  || frame4("ffb.procedural-payload/v1")
)
```

where `frame4(x)=uint32_big_endian(len(UTF8(x))) || UTF8(x)`. The profile
digest is already committed inside the manifest and is not duplicated in the
run-ID preimage. The finalized canonical run-record digest is:

```text
SHA256(
  b"fusion-fault-bench/run-record/v1\0"
  || uint64_big_endian(len(canonical_run_bytes))
  || canonical_run_bytes
)
```

`_SUCCESS` retains `ffb.success/v1alpha1` and contains exactly the procedural
artifact digest plus finalized run-record digest. It is written last after an
atomic, no-overwrite publication.

The inherited execution caps are 10,000 sequences, 20,000 bootstrap
replicates, 20,000,000 bootstrap cells, 2,000,000 sequence rows, 512 MiB per
scientific member, and 1 GiB per artifact. Release execution requires a clean
tracked checkout and locked repository-local environment.

CI runs only the separate content-addressed matrix `m3-ci-smoke-v1`, canonical
SHA-256
`fd52418c58867d7ddd6a09a0907f797c030b3c424f6b49eca7fd09334d48d186`.
That matrix references committed manifest
`procedural-ci-smoke-v1alpha1` (SHA-256
`cc1c26f8ebce3bf17143cef89238719dde14568091838fcabf6ecaeef0e702fd`)
and the smoke profile. It uses four sequences, eight frames, three objects,
camera \(x\)-calibration severities \([0,0.5]\) m with paired directions, the
same observations/seeds/methods, and 200 bootstraps. CI never truncates or
overrides a release manifest, and smoke evidence is not promoted as an M3
result.

## 11. Public claim and privacy boundary

The strongest permitted M3 description is:

> A deterministic CPU estimator-output benchmark measured matched-center fusion
> behavior under declared procedural geometry and proxy metadata faults.

Any quantitative statement must trace to the frozen manifest, scientific source
revision, aggregate record, figure, named hardware run, and verification
command. Crossover values are conditional stress-test results, not physical
sensor tolerances. The release must publish negative and undetermined outcomes,
raw points, uncertainty, common-mode blind spots, and coverage before
dropout-conditional loss.

M3 uses no nuScenes data. No dataset file, local path, raw generated output,
credential, or private interview material may be tracked. M3 does not justify
claims about detector robustness, association, real sensor-noise transfer,
fleet behavior, safety, production readiness, or nuScenes persistence.

## 12. Amendment policy

Any change to a profile mapping, population, split, sensor model, fault
operator, severity grid, seed, method, estimand, interval, crossover rule, or
acceptance threshold after execution starts requires a new profile or manifest
identifier and a documented pre-result amendment. Result-driven tuning is
forbidden.
