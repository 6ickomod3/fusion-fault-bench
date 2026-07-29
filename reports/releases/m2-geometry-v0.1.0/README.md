# M2 Geometry Validation Evidence — v0.1.0

This release validates the geometry and local-data boundary used by **Fusion
Fault Bench** before temporal fault experiments begin. It asks whether the
project's frame-aware SE(3), nuScenes convention, camera projection, box,
region-of-interest, and bearing/depth covariance implementations satisfy a
frozen set of analytic, property, and local-profile checks.

M2 is an implementation-validation milestone. It does not inject a fault,
estimate a crossover, evaluate a detector, or make a claim about real sensor
noise.

## What passed

All preregistered M2 component gates passed in the released artifact:

| Validation family | Maximum observed error | Frozen tolerance or gate |
|---|---:|---:|
| Rotation composition | `1.7763568394002505e-15` | `1e-12` |
| Translation inverse composition | `2.8421709430404007e-13` m | `1e-10` m |
| Point round trip | `9.947598300641403e-14` m | `1e-10` m |
| Quaternion-sign equivalence | `0.0` | `1e-12` |
| Independent synthetic projection | `1.4779288903810084e-12` px | `1e-9` px |
| Independent synthetic depth | `1.4210854715202004e-14` m | `1e-12` m |
| Independent box corners | `4.440892098500626e-16` m | `1e-12` m |
| Covariance finite difference | `2.1762947000070199e-10` | `1e-7` |

The frozen nonlinear covariance check used 200,000 PCG64DXSM draws. Its
largest entry-wise gate ratio was `0.019455935342375528`; a ratio at or below
one passes. The `xx`, `xy`, and `yy` ratios were respectively
`0.019455935342375528`, `0.01666375422424278`, and
`0.0037162565147059053`.

![M2 normalized geometry gates and sanitized local-profile counts](figures/geometry-validation-summary.svg)

The figure is deterministically regenerated from the curated manifest and
aggregate validation record. It is not the local projection diagnostic and
contains no per-frame geometry.

## Local nuScenes-mini grounding

A clean CPU run against a user-provided tree matching the declared nuScenes
v1.0-mini profile attested all of the following:

- the fixed headline profile matched 10 scenes, 404 samples, and 18,538 sample
  annotations;
- the required 12 metadata tables and their declared links, chains, channels,
  calibrations, poses, and box records passed the structural validator;
- all 808 declared CAM_FRONT and LIDAR_TOP key-frame blob references passed the
  bounded existence check;
- production projection agreed with an independent scalar implementation on
  the deterministic local diagnostic; and
- the diagnostic was generated and visually inspected.

The public result intentionally contains only these aggregate counts and
pass/fail attestations. It contains no absolute dataset root, dataset token,
sensor/sample timestamp, dataset filename, calibration, pose, box, residual,
per-scene row, per-frame row, or diagnostic SVG. Repository-relative
code/fixture paths and execution timestamps remain in the manifest and
run-provenance record. The summary authenticates its own bytes; it does not
authenticate the upstream dataset archive or metadata tables.

## Evidence identity

The evidence was produced from clean source revision
[`cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4`](https://github.com/6ickomod3/fusion-fault-bench/commit/cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4)
with package version `0.1.0` and lockfile SHA-256
`ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`.
The named run environment was Apple M3 Pro, arm64, 11 logical CPUs,
19,327,352,832 bytes of memory, Darwin 24.5.0, and Python 3.12.13.

| Identity | Value |
|---|---|
| Manifest SHA-256 | `7bfb5427c1ea5450a795bedef327457e5959860316bcd23f930e6eaa5917a068` |
| Scientific artifact SHA-256 | `09159042ca063b50762bf4150fb275b8a1760e4317ab74da8b0d24c133f42c90` |
| Run ID | `run:697f42275ec0c2bffd91718bf6806c4e8900318e597e7f5da727643200a88ff6` |
| Primary run-record SHA-256 | `462e15bb9da0b8caa43b6040b7f147fa64d84c7cdf51264acd02a1bd2eadc50f` |
| Repeat run-record SHA-256 | `5931411b25a169533a08849593af4416dd4a4d930054b93067f53199f0d27449` |

The primary and repeat runs had byte-identical `manifest.json`,
`geometry-validation.json`, and `payload-index.json` files and the same
scientific artifact digest. Their `run.json` and `_SUCCESS` bytes differ
because timestamps and the resulting run-record digest are intentionally
volatile. The ignored diagnostic SVG was also byte-identical.

## Validate and reproduce

Validate the committed curated evidence without a dataset:

```bash
uv sync --locked --group dev
uv run python tools/m2_release.py validate \
  reports/releases/m2-geometry-v0.1.0
```

Reproduce the local check from the recorded scientific source revision after
obtaining nuScenes v1.0-mini directly from its official distributor:

```bash
git checkout --detach cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4
uv sync --locked --group dev
export NUSCENES_ROOT=/absolute/path/to/nuScenes

uv run ffb geometry validate \
  examples/validation/m2-geometry-v1.json \
  --dataset-root-env NUSCENES_ROOT \
  --output-dir reports/generated/m2-geometry
uv run ffb geometry bundle validate reports/generated/m2-geometry
```

The runner accepts only the named environment variable and records only its
name, never its value. It refuses dataset roots inside the source checkout and
publishes no source dataset or per-frame file.

## Dataset terms

The curated aggregate record is derived from **nuScenes v1.0-mini, Motional**
and is marked **CC BY-NC-SA 4.0 plus Motional Dataset Terms**. Attribution:
“nuScenes: A multimodal dataset for autonomous driving, Caesar et al., 2020.”
Motional does not sponsor, approve, or endorse Fusion Fault Bench. The
repository's Apache-2.0 license does not relicense this record.

## Claim boundary

This release supports the narrower claim that the declared geometry and
covariance implementations passed their frozen synthetic checks and that one
user-provided tree matching the declared official-mini profile passed the
published structural and projection attestations. It does not show that every
official download has the same bytes, validate natural sensor noise, exercise
raw images or point-cloud contents, establish a physical calibration
tolerance, measure localization performance under faults, or support safety,
production, or fleet claims.

See the frozen
[M2 preregistration](../../../docs/m2-geometry-plan.md), the
[benchmark contract](../../../docs/benchmark-contract-v0.1.md), the exact
[claim-evidence ledger](claim-evidence.md), and the
[verification record](verification.md).
