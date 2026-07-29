# M1 Analytic Evidence — v0.1.0

This release is the first end-to-end scientific slice of **Fusion Fault
Bench**. It asks a deliberately narrow question: under a frozen Gaussian
estimator-output model, when does fixed camera-LiDAR information fusion have
higher matched-center loss than the modality declared healthy?

The experiment does not use raw sensor data, a detector, a learned model, or a
GPU. Its purpose is to validate the benchmark's fusion, fault, inference, and
artifact machinery against independent closed-form references before adding
geometry and dataset grounding.

## Quantity being measured

For sequence \(j\) at severity \(s\), let \(L_{F,j}(s)\) be fixed-fusion
matched-center MSE and \(L_{H,j}(s)\) the MSE of the healthy modality. The
released signed estimand is

$$
D_H(s)=\frac{1}{N}\sum_{j=1}^{N}
\left[L_{F,j}(s)-L_{H,j}(s)\right].
$$

Every M1 fault targets the camera, so the healthy modality \(H\) is LiDAR.

- \(D_H(s)<0\): fixed fusion has lower benchmark loss.
- \(D_H(s)>0\): fixed fusion has higher benchmark loss.

All inference uses signed \(D_H\), not a clipped harmful-gap presentation.
Each experiment uses \(N=200\) complete sequences, \(B=2{,}000\) paired
sequence-bootstrap replicates, pointwise 95% intervals, and a preregistered
equal-weight nondecreasing PAVA crossover rule.

## Results

Bias magnitude is measured in meters. Noise severity is a configured
camera-standard-deviation scale. These axes are separate stress coordinates
and are not combined.

### Signed camera x-bias

| Direction | Population grid root (m) | Continuous population root (m) | Finite-sample root (m) | 95% crossover interval (m) | Bootstrap crossing fraction \(q\) | Status | Tested maximum (m) |
|---|---:|---:|---:|---:|---:|---|---:|
| Negative | 3.8282790927021715 | 3.869066367512064 | 3.2126457655205014 | [1.3103527307270806, 4.971038275465616] | 1.0 | observed | 8.0 |
| Positive | 3.8282790927021715 | 3.869066367512064 | 2.9641840935526584 | [1.128775473537445, 4.58019364069497] | 1.0 | observed | 8.0 |

![Signed camera x-bias fused-minus-healthy evidence](figures/bias-fused-minus-healthy.svg)

Both signed branches satisfy the preregistered `observed` rule in this finite
sample. Their point estimates differ because the same finite random draws are
paired across directions; the population model is symmetric. The intervals,
rather than the point estimates alone, show the finite-sample uncertainty.

### Camera-noise uncertainty reporting

| Reporting case | Population grid reference (std scale) | Continuous population reference (std scale) | Finite-sample root (std scale) | 95% crossover interval | Bootstrap crossing fraction \(q\) | Status | Tested maximum |
|---|---|---|---:|---|---:|---|---:|
| Correctly reported covariance | no grid root through 4.0 | no finite root | 3.539641384241362 | none; mixed bootstrap | 0.6325 (1265/2000) | **undetermined** | 4.0 |
| Underreported covariance | 1.4630684126547195 | 1.4657551414886727 | 1.2916426005640154 | [1.044728776068505, 1.5825214913938184] | 1.0 | observed | 4.0 |

> **Negative control: finite-sample UNDETERMINED.** Correct uncertainty
> reporting has no population grid root through the registered range and no
> finite continuous population root. Its fitted finite-sample point curve
> happens to cross at 3.539641384241362, but only 0.6325 of bootstrap replicates
> cross. The preregistered rule therefore publishes no two-sided crossover
> interval and does not treat that point root as an observed crossover.

At the registered maximum \(k=4.0\), the correctly reported control has
\(D_H=0.0002931315153226384\ \mathrm{m}^2\), with pointwise 95% interval
[-0.0013976621161069004, 0.002087427419543242]. This small, sign-uncertain raw
endpoint illustrates the finite-sample uncertainty; the status itself comes
from the preregistered crossing-fraction rule. It does not overturn the
independent no-finite-population-root reference.

![Correctly reported versus underreported camera-noise evidence](figures/noise-reporting-fused-minus-healthy.svg)

The contrast between these controls is the main M1 result: under this declared
model, fusion can downweight increasing camera noise when the reported
covariance tracks the actual variance, while retaining nominal fusion weights
under variance growth produces an observed controlled-stress crossover.

## Evidence identity

All three primary artifacts were produced by clean source revision
[`524c8f70ece3eca2e61796165b23ffe51baadfbc`](https://github.com/6ickomod3/fusion-fault-bench/commit/524c8f70ece3eca2e61796165b23ffe51baadfbc)
with package version `0.1.0` and lockfile SHA-256
`ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`.
The named run environment was Apple M3 Pro, arm64, 11 logical CPUs,
19,327,352,832 bytes of memory, Darwin 24.5.0, and Python 3.12.13.

| Experiment | Manifest SHA-256 | Artifact SHA-256 | Run ID |
|---|---|---|---|
| Camera x-bias | `a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e` | `3717c2b3fdce9e9f2bc43463434fde28a4d24dd9ca1d72e451b3d9d5273c2959` | `run:cc67cf35ac9b74116cbb2f39c934bbfddb041036c58194722090979549460687` |
| Correctly reported camera noise | `3ea7ffc2949cf99f20d20ec18844f0b8dc3b3ebb81e13e926f7440b7c5084176` | `51abb5043ddffd633c0fa81ea5d69dc6c1246d092185f22261f183915f911467` | `run:4d601cfe04c83839a25088a061ab9a6d4b3c29ffcec71ea4f6ce64c3343f4340` |
| Underreported camera noise | `9d26e1b33f1fd2e35b0de90703a960d2eba6bb26bd2219bce6f0bb82480f4ac4` | `8a3c2179e49cdc2ae994d9c791185a546788252710a0e80fca73cf39305165e7` | `run:a58e65ece1915d49c485f36ee478f97fce35c18763f7fa4cec2ffd44bd90b234` |

The exact record-level trace for every value above is in
[claim-evidence.md](claim-evidence.md). The release-integrity and provenance
audit is in [verification.md](verification.md). `release-index.json` is the
canonical curated-release index.

## Validate and reproduce

Validate the committed curated evidence:

```bash
uv sync --locked --group dev
uv run python tools/m1_release.py validate \
  reports/releases/m1-analytic-v0.1.0
```

Reproduce the three primary scientific artifacts from the recorded source
revision:

```bash
git checkout --detach 524c8f70ece3eca2e61796165b23ffe51baadfbc
uv sync --locked --group dev

uv run ffb run examples/manifests/analytic-bias-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-x-bias-a603d090f77a
uv run ffb run examples/manifests/analytic-noise-correct-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-noise-correctly-reported-3ea7ffc2949c
uv run ffb run examples/manifests/analytic-noise-underreported-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-noise-underreported-9d26e1b33f1f

uv run ffb bundle validate \
  reports/generated/analytic-camera-x-bias-a603d090f77a
uv run ffb bundle validate \
  reports/generated/analytic-camera-noise-correctly-reported-3ea7ffc2949c
uv run ffb bundle validate \
  reports/generated/analytic-camera-noise-underreported-9d26e1b33f1f
```

The complete fresh-rerun procedure and the distinct retained-input archival
build procedure, including the no-overwrite build command, are documented in
[verification.md](verification.md).

## Claim boundary

These results characterize one-object, two-dimensional Gaussian
estimator-output models under controlled bias and uncertainty-reporting
stress. They do not establish a physical sensor tolerance, a naturally
occurring fault distribution, raw-sensor or detector behavior, calibration or
timing-fault behavior, nuScenes transfer, collision risk, operational fallback
quality, or fleet generalization. The meter and standard-deviation-scale roots
are benchmark coordinates under the frozen model, not deployment thresholds.

See the frozen
[benchmark contract](../../../docs/benchmark-contract-v0.1.md), the
[M1 preregistration](../../../docs/m1-analytic-plan.md), and the project-wide
[limitations](../../../docs/limitations.md).
