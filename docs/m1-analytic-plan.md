# M1 Analytic Vertical Slice

Status: **pre-registered; no M1 result has been executed or released**.

## Purpose

M1 is the smallest end-to-end scientific slice of Fusion Fault Bench. It tests
whether the fixed contract can generate paired camera/LiDAR estimator outputs,
apply faults without changing the latent sample, fuse from reported
uncertainty, compute sequence-level evidence, and serialize a result that an
independent loader can reject when contradictory.

M1 remains CPU-only and contains no detector, raw image, point cloud, model
checkpoint, SE(3) replay, temporal tracker, or health classifier.

## Frozen experiments

All three manifests use the same two-dimensional Gaussian observation model,
named RNG streams, 200 sequence IDs, 2,000 paired bootstrap replicates, and 95%
pointwise intervals.

| Experiment | Fault grid | Population hypothesis |
|---|---|---|
| Signed camera \(x\)-bias | \(0,.25,.5,1,2,4,8\) m | A finite bias crossover lies inside the grid. |
| Correctly reported camera noise | \(1,1.25,1.5,2,4\) std scale | Fixed fusion remains below healthy-LiDAR population loss at every finite scale. |
| Underreported camera noise | \(1,1.25,1.5,2,4\) std scale | Nominal fusion weights create a finite overconfidence crossover. |

The pre-execution manifest identities are:

- `examples/manifests/analytic-bias-v1alpha1.json`:
  `a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e`;
- `examples/manifests/analytic-noise-correct-v1alpha1.json`:
  `3ea7ffc2949cf99f20d20ec18844f0b8dc3b3ebb81e13e926f7440b7c5084176`;
- `examples/manifests/analytic-noise-underreported-v1alpha1.json`:
  `9d26e1b33f1fd2e35b0de90703a960d2eba6bb26bd2219bce6f0bb82480f4ac4`.

Correctly reported and underreported noise reuse identical standard-normal
camera and LiDAR draws. At each matched \(k\), both experiments use the same
\(k\)-scaled actual camera error; only the camera covariance reported to fusion
differs. The grids, seeds, endpoints, and sample count will not change after
finite-sample results are observed. An observed, undetermined, or
right-censored finite-sample status is acceptable.

## Numerical implementation

- Derive every stream seed from the frozen byte-level SHA-256 contract.
- Draw one base camera and one base LiDAR standard-normal vector per sequence.
- Reuse those vectors across every severity and signed direction.
- Apply bias after sampling; never accumulate one severity into the next.
- For correctly reported noise, scale both actual target error and reported
  target standard deviation.
- For underreported noise, scale actual target error while retaining nominal
  reported standard deviation.
- Fuse float64 Cartesian estimates using reported information matrices only.
- Score \(e_x^2+e_y^2\), copy the healthy row across conditions, use fixed
  fusion at identity for target-drop, and define the performance oracle as the
  per-sequence minimum over camera-only, LiDAR-only, and fixed fusion.
- Emit identity once. Keep the population continuous root separate from the
  finite grid/PAVA estimand.

An independent scalar closed-form module must not import the production fusion,
fault, or experiment modules.

## Frozen population references

Let the nominal camera and LiDAR variance vectors be

\[
c=(9/4,9/25),\qquad \ell=(1/16,1/16),
\]

and let the nominal camera fusion weight be

\[
w=\ell/(c+\ell)=(1/37,25/169).
\]

For signed camera \(x\)-bias \(s\), reported and actual fused covariance agree
at

\[
\operatorname{diag}(9/148,9/169),
\]

and the population signed contrast is

\[
D_\text{bias}(s)=-547/50024+s^2/1369.
\]

The population contract-grid root is
\(31055/8112=3.8282790927021697\) m. The separate continuous reference is
\(\sqrt{20239/1352}=3.8690663675120667\) m.

For correctly reported noise scale \(k\), actual and reported target variance
both equal \(k^2c\), so the actual and reported fused covariance agree and

\[
D_\text{correct}(k)=
\sum_i\frac{k^2c_i\ell_i}{k^2c_i+\ell_i}-1/8<0
\]

for every finite \(k\). At the registered grid, the exact contrasts are

\[
(-547/50024,-269/36640,-2399/457888,-2113/697160,
-8377/10750664).
\]

This control has no finite grid or continuous root. The analytic-validation
record must encode a not-crossed grid right-censored above 4, a `null` grid
root, `no-finite-root` continuous status, and a `null` continuous root; JSON
infinity is prohibited.

For underreported noise, actual target variance is \(k^2c\) while reported
target variance remains \(c\). Reported fused covariance therefore remains
\(\operatorname{diag}(9/148,9/169)\), but the actual fused error covariance is

\[
\operatorname{diag}\left(w^2k^2c+(1-w)^2\ell\right).
\]

The population contrast is

\[
D_\text{under}(k)=
-547/50024+(k^2-1)(1489149/156400036).
\]

Its registered-grid root is
\(47931991/32761278=1.46306841265472\); the separate continuous reference is
\(\sqrt{6398689/2978298}=1.4657551414886731\).

## Deterministic artifact contract

The machine artifact contains exactly:

```text
manifest.json
sequence-metrics.ndjson
aggregate-metrics.ndjson
crossovers.ndjson
analytic-validation.json
payload-index.json
run.json
_SUCCESS
```

The indexed scientific payload is the first five files. `payload-index.json` is
a deterministic envelope over that payload, and byte identity covers both the
five indexed members and the envelope. `run.json` contains timestamps,
hardware, and other intentionally variable execution provenance and is outside
the byte-identity promise. `_SUCCESS` is a completion marker.

Every JSON value uses the canonical byte and record ordering defined below.

`payload-index.json` contains a fixed ordered allowlist with the five indexed
payload members' byte lengths and SHA-256 digests; it does not list itself.
`run.json.artifact_sha256` is a domain-separated SHA-256 of the exact canonical
index bytes. `run.json` is excluded to avoid self-reference.

`_SUCCESS` contains exactly the 64 lowercase hexadecimal
`run.json.artifact_sha256` value followed by one LF byte. It is written last and
must match both the run record and recomputed index digest.

The committed contract adds these strict schemas:

- `ffb.payload-index/v1alpha1` has `schema`, fixed
  `artifact_contract="ffb.scientific-payload/v1"`, `run_id`,
  `manifest_sha256`, and an ordered five-entry `files` tuple. Each file entry
  has `path`, integer `byte_length`, and lowercase `sha256`. Paths must be
  exactly `manifest.json`, `sequence-metrics.ndjson`,
  `aggregate-metrics.ndjson`, `crossovers.ndjson`, and
  `analytic-validation.json`, in that order.
- `ffb.analytic-validation/v1alpha1` has `schema`, `run_id`,
  `manifest_sha256`, fixed
  `reference_model="independent-diagonal-gaussian-closed-form-v1"`,
  fixed `variance_representation="diagonal-xy-m2"`,
  `monte_carlo_standard_error_multiplier`, ordered `population_points`,
  ordered `crossover_references`, and `all_monte_carlo_checks_passed`.

The exact analytic-validation nested records are:

```text
population point:
  severity: existing SeverityCoordinate object
    {index: int >= 0, magnitude: finite float >= 0,
     direction: identity|negative|positive|increase,
     unit: m|std-scale}
  method_id: camera-only|lidar-only|fixed-fusion
  mean_unit: "m"
  variance_unit: "m^2"
  loss_unit: "m^2"
  expected_mean_xy_m: fixed-length pair of finite floats
  expected_actual_variance_xy_m2: fixed-length pair of finite floats >= 0
  expected_reported_variance_xy_m2: fixed-length pair of finite floats > 0
  expected_mse_m2: finite float >= 0
  empirical_mse_m2: finite float >= 0
  analytic_mse_standard_error_m2: finite float > 0
  absolute_standardized_error: finite float >= 0
  monte_carlo_passed: bool

crossover reference:
  direction: negative|positive|increase
  severity_unit: m|std-scale
  tested_maximum: finite float
  grid_status: crossed|not-crossed
  grid_point_estimate: finite float|null
  grid_censoring: none|right-above-tested-maximum
  continuous_status: finite|no-finite-root
  continuous_point_estimate: finite float|null
```

Only camera-only, LiDAR-only, and fixed-fusion population points are included,
in that order for every condition. `monte_carlo_passed` is exactly
`absolute_standardized_error <=
monte_carlo_standard_error_multiplier`; the top-level boolean is the
conjunction of every point. `grid_status=crossed` requires a grid root within
the tested grid and `grid_censoring=none`; `grid_status=not-crossed` requires a
`null` grid root and right-above-tested-maximum censoring. Independently,
`continuous_status=finite` requires a non-null continuous root, while
`continuous_status=no-finite-root` requires `null`. Infinity is never
serialized.

The exact artifact digest is

```text
SHA256(
  UTF8("fusion-fault-bench/artifact/v1") || 0x00
  || uint64_be(len(payload_index_file_bytes))
  || payload_index_file_bytes
)
```

where `payload_index_file_bytes` includes its one terminal LF.

The deterministic run ID is `run:` followed by the lowercase SHA-256 hex digest
of:

```text
UTF8("fusion-fault-bench/run-id/v1") || 0x00
|| framed(UTF8(manifest_sha256))
|| framed(UTF8(git_revision))
|| framed(UTF8(lockfile_sha256))
|| framed(UTF8(package_version))
|| framed(UTF8("ffb.scientific-payload/v1"))
```

where `framed(x) = uint32_be(len(x)) || x`. It is computed before evaluation.
Byte-identical payload claims apply only when those identities are unchanged.

Canonical JSON is exactly `json.dumps(..., allow_nan=False,
ensure_ascii=False, separators=(",", ":"), sort_keys=True)` plus one LF.
Generation recursively converts every computed floating zero to positive
`0.0`. Strict loading rejects any negative-zero token, UTF-8 BOM, CR byte,
non-UTF-8 input, blank NDJSON line, missing terminal LF, extra terminal LF, or
noncanonical whitespace/key/number encoding. Every NDJSON line is one canonical
record plus LF.

Condition order is identity once, then increasing manifest-grid magnitude, with
`negative` before `positive` for signed faults and only `increase` for noise.
Sequence rows use source sequence order, then condition order, then manifest
method order. Aggregates use condition order, manifest method order, then the
fixed-fusion signed contrast. Crossovers and analytic references use
`negative`, `positive`, or `increase` as applicable.

The loader rejects noncanonical bytes, duplicate keys, wrong order, missing or
extra files, symlinks, path traversal, hash/length mismatches, an invalid
completion marker, contradictory records, and a payload digest that disagrees
with `run.json`.

## Execution safety and CLI

```bash
uv run ffb run MANIFEST --output-dir DEST
uv run ffb bundle validate DEST
```

Scientific choices have no CLI overrides. M1 accepts only analytic-crossover
manifests, refuses dirty or unavailable Git provenance, preflights CPU resource
limits, refuses any existing destination, stages in the destination's parent,
and publishes the directory atomically. It first validates the in-memory bundle,
writes and verifies every staging member except `_SUCCESS`, writes `_SUCCESS`
as the final staging mutation, strictly reloads the complete tree, and only
then renames the staging directory to the destination. It has no force or
overwrite option.

M1 execution caps are `sequence_count <= 10_000`,
`bootstrap_replicates <= 20_000`,
`sequence_count * bootstrap_replicates <= 20_000_000`, and at most 2,000,000
sequence rows. Strict loading caps each line at 1 MiB, each scientific member at
512 MiB, and the complete artifact at 1 GiB. A valid manifest outside those
operational limits fails before output creation.

Destination nonexistence is checked without following links, including dangling
links. Every existing path component from the filesystem root to the
destination parent must be a real directory and not a symlink. Missing parent
components are created one at a time and rechecked. Cleanup may remove only the
exact `mkdtemp` staging directory returned to that invocation, after verifying
its parent and reserved prefix; it never removes the requested destination or
an arbitrary path. The runner discovers both `git rev-parse --absolute-git-dir`
and the resolved `git rev-parse --git-common-dir`; the destination and staging
directory must not equal or descend from either Git metadata directory.

M1 requires the manifest to be a tracked regular file inside clean source
checkout \(A\). The logical reproduction CWD is the root of checkout \(A\) at
the recorded revision. The public run record stores:

```text
(
  "ffb",
  "run",
  <POSIX repository-relative committed manifest path>,
  "--output-dir",
  "reports/generated/<experiment>-<manifest_digest_prefix_12>"
)
```

It never stores literal local argv, an absolute path, or the artifact's copied
`manifest.json`. Running this tuple from source root \(A\) writes only beneath
the ignored `reports/generated/` path, so clean-Git enforcement remains
operational.

## Acceptance gates

- Hard-coded seed bytes and first-draw goldens pass for the renamed local-error
  sequence ID.
- Production fusion agrees with an independent two-dimensional inverse
  calculation.
- Population mean, actual covariance, reported covariance, and MSE agree with
  closed form at the manifest tolerances. Grid and continuous roots agree when
  finite; the correctly reported control must instead match the predeclared
  no-root encodings.
- Monte Carlo expectation checks use the fixed six-analytic-SE criterion;
  finite-sample values are never compared with a \(10^{-12}\) threshold.
- Bias branches, targets, axes, correct/underreported covariance, identity,
  healthy invariance, target-drop, and sequence-oracle semantics are tested.
- Crossover fixtures cover observed, not-observed, undetermined,
  non-monotonic, and exact-boundary cases independently of the released
  outcome.
- Two clean reruns produce identical scientific bytes and payload digest while
  allowing different `run.json` timestamps.
- Mutation and malformed-on-disk tests prove that strict artifact validation
  fails closed.
- Formatting, lint, strict typing, at least 90% coverage, lockfile, package
  build, isolated-wheel smoke, and three adversarial implementation reviews
  pass before results are promoted.

## Presentation commitment

The release will publish the same views regardless of whether a finite-sample
status is observed, undetermined, or right-censored:

- a bias figure containing every severity and both directions, every raw
  \(D_H\) point and interval, both PAVA curves, and the zero line;
- one uncertainty-reporting figure placing correctly reported and
  underreported noise on identical axes with every raw point, interval, PAVA
  curve, and the zero line;
- a crossover table containing every experiment/direction, population grid
  root or no-root status, continuous reference root or no-root status,
  finite-sample point root, interval, bootstrap crossing fraction \(q\), result
  status, and tested maximum;
- a claim-evidence table mapping every public quantitative sentence to its
  manifest, payload, record keys, presentation artifact, and caveat.

Raw non-monotonic points remain visible. No experiment, direction, status, or
endpoint may be omitted because it weakens the narrative.

## Release boundary

A clean source commit \(A\) generates evidence; a later commit \(B\) promotes
that evidence. The run truthfully records \(A\), not \(B\). Public claims must
map to a manifest digest, payload digest, source revision, record key,
uncertainty, CPU environment, reproduction command, and limitation.

The synthetic bias or noise scale is a controlled stress-test coordinate, not
a physical sensor tolerance, real fault prior, detector result, safety
threshold, or fleet-generalization claim.
