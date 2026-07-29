# Benchmark Contract v0.1

Status: **frozen v0.1 contract; M0 foundation validated**.

This document fixes the quantities, fault semantics, aggregation, and inference
rules that could otherwise be chosen after results are visible. Any change that
affects the estimand requires a new manifest schema and a documented benchmark
revision.

M2 pre-implementation erratum: section 3 originally said `radial` limits while
the frozen manifest schema, examples, and field names specify
`x_min_m`/`x_max_m`. The term is corrected below to **ego-forward** limits.
This resolves a contradictory label before geometry execution; it changes no
schema, manifest value, M1 analytic result, or estimand.

## 1. Question and estimand

Fusion Fault Bench asks:

> Under a declared estimator-output model and controlled proxy fault, when does
> fixed camera-LiDAR fusion have higher matched-center localization loss than
> the modality designated healthy?

This is a conditional benchmark result. A crossover is not a universal physical
sensor threshold and does not estimate a real fleet fault distribution.

For a single-sensor crossover experiment at severity \(s\), method \(m\), and
complete sequence \(j\), let \(L_{m,j}(s)\) be the mean squared BEV center error
across the same pre-fault eligible matched object-frames in that sequence. The
term “squared center error” means squared Euclidean error, not a coordinate
mean. For \(n_j\) eligible object-frames,

\[
L_{m,j}(s)=\frac{1}{n_j}\sum_{i=1}^{n_j}
\left\|\widehat p_{m,ji}(s)-p_{ji}\right\|_2^2.
\]

Thus both \(x\) and \(y\) squared errors are summed for each center, then object
frames are averaged. The primary paired sequence contrast is

\[
d_j(s)=L_{F,j}(s)-L_{H,j}(s),
\]

where \(F\) is fixed fusion. The healthy modality \(H\) is derived from the
single-sensor target: a camera fault implies healthy LiDAR and a LiDAR fault
implies a healthy camera. It is not a selectable manifest field. The population
estimand is

\[
D_H(s)=\frac{1}{N}\sum_{j=1}^{N} d_j(s).
\]

- \(D_H(s)<0\): fixed fusion helps relative to the healthy modality.
- \(D_H(s)>0\): fixed fusion is harmful relative to the healthy modality.

Inference is performed on the signed \(D_H\). The non-negative
harmful-fusion gap, \(\max(0,D_H)\), is a presentation derivative only.

## 2. Coordinate and transform convention

All frames are right-handed and all points are column vectors. The only
transform notation permitted is

\[
T_{a\leftarrow b}p_b=p_a.
\]

Composition reads from right to left:

\[
T_{a\leftarrow c}=T_{a\leftarrow b}T_{b\leftarrow c}.
\]

The v0.1 comparison occurs in ego-frame bird's-eye-view coordinates:

- \(x\): forward,
- \(y\): left,
- \(z\): up.

nuScenes rotations are interpreted as scalar-first quaternions
\([w,x,y,z]\). Its calibrated-sensor pose maps sensor to ego, and its ego pose
maps ego to global.

## 3. Matched-center task and common support

v0.1 uses known object identities and 2D BEV centers. It intentionally isolates
estimation and fusion from detection and association.

Only object states in the common front-camera/LiDAR support are scored:

1. positive ego-forward range;
2. within the manifest's ego-forward and lateral limits;
3. within the front camera's declared field of view or calibrated image bounds;
4. eligible for both modality estimators before a fault is injected.

LiDAR's nominal 360-degree coverage is therefore restricted to the camera
support. Camera-only and LiDAR-only losses use the same object-frame
denominator.

The analytic source is conditioned on one already-eligible matched object and
operates in translation-invariant local error coordinates. Its `local-error`
origin means zero residual after subtracting an arbitrary fixed true center; it
does **not** place the object at zero ego-forward range. Analytic mode therefore
tests fusion and inference algebra, while geometry modes perform the physical
ROI eligibility calculation.

False positives, misses caused by detection, extents, classes, Hungarian
association, GOSPA, and the six-camera rig are deferred to v0.2.

Dropout is not part of the localization-crossover estimand: at full dropout,
localization is undefined. A separate availability-control mode reports
coverage, undefined-output rate, and localization loss conditional on coverage.
It retains the pre-fault eligible denominator and has no crossover. A unimodal
method is undefined when its own input is missing, and fixed fusion is undefined
when either input is missing. The diagnostic target-drop policy uses fixed
fusion at identity and the non-target modality at every non-identity dropout
probability.

Availability aggregates use different denominators because an undefined output
has no localization error. Coverage is total valid object-frames divided by
the pre-fault eligible count; undefined-output rate is one minus coverage.
Conditional matched-center MSE is total squared center error divided by the
number of valid object-frames. A paired sequence bootstrap resamples complete
sequences and recomputes these ratios. A replicate with zero valid outputs has
undefined conditional loss and is excluded, while its exclusion is counted.
A two-sided conditional-loss interval is reported only when the fraction of
defined replicates exceeds \(1-\alpha/2\); otherwise the aggregate is marked
undefined. No missing output is assigned a zero localization penalty.

## 4. Observation contract

The v0.1 base true-error model is completely specified as a zero-mean,
constant-variance, temporally IID, diagonal Cartesian Gaussian in ego-BEV
coordinates. Camera and LiDAR stochastic draws are independent. Correlation,
range dependence, non-Gaussian tails, and base bias require a later schema.

Every simulated estimator output conceptually contains:

```text
object_id
value
true_error_model
reported_covariance
timestamp_true
timestamp_reported
transform_true
transform_reported
availability
```

The actual sampling distribution and reported covariance are different
quantities. This distinction creates two scientifically different cases:

- increased noise with correctly reported uncertainty, which fixed information
  fusion can downweight;
- increased noise with nominal or underreported uncertainty, which can make the
  estimator overconfident.

All fused values and covariances must be expressed in the same frame and units.
When camera bearing/depth is introduced, its covariance is propagated into ego
BEV with a Jacobian and checked against Monte Carlo samples.

Fixed fusion uses the covariance reported to the estimator, not the covariance
that generated the error. With available estimates \(z_C,z_L\) and symmetric
positive-definite reported covariances \(R_C,R_L\),

\[
\Lambda_F=R_C^{-1}+R_L^{-1},\qquad
R_F=\Lambda_F^{-1},
\]

\[
z_F=R_F\left(R_C^{-1}z_C+R_L^{-1}z_L\right).
\]

This equation is the only fixed-fusion operator in v0.1. A missing input makes
fixed fusion undefined in availability-control mode; it does not silently
renormalize to the available sensor.

The independence literal applies to base stochastic errors. A shared
deterministic common-mode bias is a separate control and does not contradict
that base assumption.

## 5. Causal fault insertion

The physical observation is always generated with the true latent state,
physical timestamp, and physical sensor pose. A metadata fault modifies only
the metadata consumed by reconstruction or fusion.

For a camera calibration perturbation expressed in the ego frame,

\[
\widetilde T_{e\leftarrow c}=\Delta T_e T^{\mathrm{true}}_{e\leftarrow c}.
\]

The camera-space measurement remains unchanged. Reconstruction with
\(\widetilde T\) produces the misregistered ego-frame estimate. Reusing the
corrupted transform in both generation and reconstruction is prohibited because
the error would cancel.

Timing follows the same rule: a measurement is generated at
\(t_\mathrm{true}\), while the estimator receives
\(t_\mathrm{reported}=t_\mathrm{true}+\Delta t\) or explicitly omits
compensation. Correctly timestamped delayed data is not automatically a fault.

Primary single-sensor crossover axes are:

- additive estimator-output bias;
- correctly reported increased noise;
- underreported increased noise;
- camera calibration translation and yaw metadata error;
- timestamp metadata offset.

Translation, yaw, and timestamp magnitudes begin at zero and use paired positive
and negative directions. Timestamp sign is
\(\Delta t=t_\mathrm{reported}-t_\mathrm{true}\). Noise grids use standard
deviation scale \(k\), begin at \(k=1\), and apply either

\[
\sigma_\mathrm{actual}=k\sigma_0,\quad
\sigma_\mathrm{reported}=k\sigma_0
\]

for correctly reported noise, or

\[
\sigma_\mathrm{actual}=k\sigma_0,\quad
\sigma_\mathrm{reported}=\sigma_0
\]

for underreported noise.

Each crossover manifest contains exactly one physical axis and a strictly
increasing identity-first grid. Meters, radians, seconds, and standard-deviation
scales are never collapsed into one normalized severity. Calibration and
timestamp offsets are persistent per sequence; stochastic Gaussian errors
remain IID under their sequence-level configuration.

## 6. Methods and oracle definitions

Required methods are:

1. camera only;
2. LiDAR only;
3. fixed covariance-weighted information fusion.

Both analytic and geometry crossover manifests additionally fix:

- **fault-target-drop policy:** use fixed fusion at identity and drop the
  configured fault target at non-identity severity. This is a diagnostic policy,
  not an upper bound; mild faults may still benefit from fusion.
- **performance oracle:** select the lowest-loss camera-only, LiDAR-only, or
  fixed-fusion method with hindsight over a complete sequence. It is a ceiling,
  not a deployable policy.

Health-gated selection is absent from this schema until its predictor, features,
window, threshold, unknown decision, and fallback action are fully versioned.

## 7. Core and later metrics

The population fusion benefit is

\[
\mathrm{FB}(s)=
\min\left(\bar L_C(s),\bar L_L(s)\right)-\bar L_F(s),
\]

where each \(\bar L_m\) is the mean of complete-sequence losses. The minimum is
not taken per object, frame, or sequence; doing so would create an unattainable
hindsight selector.

The v1alpha1 core result vocabulary requires:

- matched-center MSE for every declared method;
- the signed fused-minus-healthy contrast for crossover experiments;
- coverage, undefined-output rate, and conditional matched-center MSE for
  availability experiments;
- a crossover record for each signed direction or increasing-noise axis.

The release-bundle validator independently recomputes every point aggregate,
method identity, paired bootstrap interval, defined-replicate count, PAVA fit,
crossover root, crossing fraction, status, and censoring field from sequence
rows plus the manifest seed. A complete but numerically contradictory bundle is
invalid. Recomputed floating-point values are reconciled with
`isclose(rel_tol=0, abs_tol=1e-12)` in the value's declared unit; categorical
decisions, integer counts, grid coordinates, and support thresholds remain
exact.

For conditional loss, sequence \(j\) publishes its valid-object-frame mean
\(\ell_j\) and valid count \(n_j\). The aggregate point is
\(\operatorname{fsum}_j(\ell_j n_j)/\sum_j n_j\); every bootstrap replicate
uses the same reconstruction after sequence resampling. A raw point can exist
while too few bootstrap replicates have a nonzero denominator. In that case the
record is `undefined` and publishes no point or interval, even though its
contributing-sequence count remains nonzero.

Fusion benefit is computed from released aggregate losses by the formula above;
it is not a separately sampled metric. NEES/NIS, clean regression, policy gaps,
and oracle-recovery fractions are planned extensions and require a result-schema
revision before they become release requirements.

## 8. Crossover and uncertainty

Crossover is predeclared separately for every single-sensor fault family,
target, direction, and physical axis:

1. Show every raw \(D_H(s)\) estimate and interval.
2. Fit a nondecreasing equal-severity-weight PAVA isotonic curve to \(D_H(s)\).
   PAVA starts with one block per ordered severity and repeatedly merges
   adjacent blocks while the left block mean is greater than the right block
   mean; every severity has weight one and a merged block receives its ordinary
   arithmetic mean.
3. Treat values with absolute magnitude at or below the manifest's numerical
   tolerance as zero. v1alpha1 fixes this tolerance at \(10^{-12}\,\mathrm{m}^2\);
   it is present in the manifest for auditability but cannot be tuned. Define
   \(s^*\) as the identity magnitude when the fitted identity value is already
   non-negative, or as the first exact zero; otherwise linearly interpolate
   between the last negative and first positive fitted values. If every fitted
   value is negative, the crossing is right-censored above the tested maximum.
4. Percentile-bootstrap complete sequences. Each replicate uses one resampled
   sequence-index vector shared across all severities and methods, then refits
   the isotonic curve.
5. Treat a replicate with no crossing as right-censored above the tested
   maximum. Let \(q\) be the fraction of replicates with a crossing and
   \(\alpha=1-\text{confidence level}\).
6. Report **observed** when the point curve crosses and
   \(q>1-\alpha/2\). The interval is the right-censored percentile interval,
   where non-crossing replicates act as \(+\infty\); this condition ensures its
   upper endpoint is finite.
7. Report **not observed** when the point curve does not cross and
   \(q<\alpha/2\), with the tested maximum as a right-censored bound.
8. Report **undetermined** otherwise, including mixed bootstrap support.

Percentiles use NumPy's `quantile(..., method="linear")` convention at
\(\alpha/2\) and \(1-\alpha/2\). Non-crossing bootstrap roots are represented as
\(+\infty\) for quantile selection. An observed result therefore has two finite
endpoints; a not-observed result explicitly records
\([\text{tested maximum},+\infty)\); an undetermined result reports no
two-sided interval. v1alpha1 fixes confidence at 95% and requires a bootstrap
replicate count divisible by 40. Together with the strict \(q>0.975\) observed
rule, this guarantees that NumPy's linearly interpolated upper endpoint is
finite.

Raw point intervals are pointwise paired-sequence percentile intervals. The
fitting rule is fixed before results, and unfitted responses remain visible,
including non-monotonic behavior. Percentile intervals are required to have
ordered endpoints but need not contain the original-sample point estimate.
A structured crossover record stores whether the point curve crossed, status,
estimate or right-censored \([\text{tested maximum},+\infty)\) bound, interval,
crossing fraction, sequence/replicate counts, confidence, and physical unit.

## 9. Paired randomness and splits

For every clean/fault severity and competing method:

- reuse the same latent sequence;
- reuse the same base stochastic observation draws;
- split by complete sequence, never frame;
- keep all variants of a latent sequence in one split;
- bootstrap the sequence identifier.

For a single-target fault, the non-target unimodal sequence record—status,
value, eligible count, and valid count—must be identical at every severity and
direction. The release-bundle validator enforces this observable consequence of
paired randomness. Common-mode controls are excluded because neither modality
is healthy.

The RNG contract fixes NumPy PCG64DXSM. Both manifest seeds are unsigned
128-bit integers. For each data stream and sequence, construct this byte string:

```text
UTF8("fusion-fault-bench/rng/v1") || 0x00
|| uint128_be(data_master_seed)
|| uint32_be(len(UTF8(stream_name))) || UTF8(stream_name)
|| uint32_be(len(UTF8(sequence_id))) || UTF8(sequence_id)
```

The PCG64DXSM seed is the unsigned big-endian integer represented by the first
16 bytes of the SHA-256 digest. The sequence ID is the exact public identifier
stored in result rows, never a local path. Within a sequence, frames are ordered
by increasing integer frame index, objects by UTF-8 byte order of object ID,
and coordinates as \(x,y\). Diagonal-Gaussian draws are independent across
coordinates, objects, frames, and modalities. Vectorized and scalar
implementations must consume the same conceptual row-major draw order.
Gaussian draws use
`Generator(PCG64DXSM(seed)).standard_normal(size=(object_frame_count, 2),
dtype=np.float64)`.

The dropout fault consumes one fault-stream uniform draw per target
modality-frame, shared by every object in that frame, with independent draws
across frames. In increasing frame-index order, it calls
`Generator.random(size=frame_count, dtype=np.float64)` once for the sequence. A
frame is dropped at probability \(p\) exactly when \(u<p\), and the same
uniform vector is reused for every declared probability. Dropout masks are
therefore nested across severity. This represents sensor-frame availability
rather than object-detection misses.

Bootstrap uses `Generator(PCG64DXSM(bootstrap_seed))` directly. It draws one
row-major \(B\times N\) integer index matrix on \([0,N)\) with
`integers(0, N, size=(B, N), dtype=np.int64)`; every method and severity reuses
that exact matrix. Adding a draw in one named data stream therefore does not
shift another stream or bootstrap inference. The lockfile pins the NumPy
implementation version.

Synthetic sequence IDs are fixed as
`analytic:{case_id}:{index:06d}` or
`procedural:{profile_id}:{split}:{index:06d}`. A nuScenes sequence ID is
`nuscenes:{scene_name}`. Zero-based indices are used. These identifiers are
part of both RNG derivation and result-bundle completeness checks. Index value
\(i\) in the bootstrap matrix selects sequence \(i\) in ascending synthetic
index order, or entry \(i\) of the manifest's ordered `scene_names` list for
nuScenes; filesystem enumeration never determines this order.

Analytic validation compares deterministic population formulas and the
contract-grid crossover with independent implementations at the manifest's
fixed \(10^{-12}\) absolute tolerances. Monte Carlo expectation checks use a
fixed six-analytic-standard-error bound rather than an arbitrary tuned
absolute tolerance. The continuous mathematical root is reported separately
from the PAVA/grid/interpolation estimand; their discretization difference is
not treated as implementation error.

Procedural generalization axes include held-out layout families, range/velocity
bins, fault families, and severity intervals. nuScenes-mini replay is an
exploratory geometry/motion shift with only ten scenes, not an independent
real-sensor validation set.

## 10. Health-feature boundary

Health scores are computed from predictions formed before current measurements.
Sensor-specific or leave-one-sensor-out predictors record which measurements
updated them.

Deployable health features may use residuals, reported covariance, missingness,
and timestamps. They may not receive the injected fault label, family,
severity, seed, sequence split, or other manifest metadata.

Health evaluation includes an `unknown/ambiguous` outcome and reports event
attribution, time to detect, false alerts per clean sequence, and recovery time.
Frame-level AUROC is secondary.

## 11. Required controls

- Identity severity for every fault.
- Increased unbiased noise with correctly reported covariance.
- Difficult but clean geometry.
- Shared/common-mode bias where modalities agree but are jointly wrong.
- Analytic Gaussian-fusion cases with closed-form expected loss.

The common-mode case uses a separate evaluation mode with absolute
ground-truth loss, no healthy modality, no target-drop or performance oracle,
and no crossover. It is an intended blind-spot test, not an ordinary
camera-versus-LiDAR attribution example.

## 12. Valid claims

v0.1 may claim only what the released evidence supports:

- behavior of declared estimator-output models;
- measured matched-center loss and uncertainty;
- persistence or non-persistence under nuScenes-mini latent geometry;
- measured CPU runtime and memory.

It cannot establish raw-sensor robustness, real fault prevalence, collision
risk, production safety, fleet generalization, or equivalence to a learned BEV
detector.
