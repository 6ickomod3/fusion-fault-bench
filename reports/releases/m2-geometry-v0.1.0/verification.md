# M2 Release Verification

This document records how M2 evidence was admitted and which checks can be
repeated from public files. `release-index.json` is the canonical
machine-readable release index.

## Admission summary

- Frozen manifest:
  `examples/validation/m2-geometry-v1.json`.
- Scientific source revision:
  [`cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4`](https://github.com/6ickomod3/fusion-fault-bench/commit/cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4).
- Lockfile SHA-256:
  `ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`.
- Named environment: Apple M3 Pro, arm64, 11 logical CPUs,
  19,327,352,832 bytes of memory, Darwin 24.5.0, Python 3.12.13.
- Two clean local executions passed the strict five-file geometry-bundle
  loader.
- Their three stable scientific files and ignored diagnostic SVG were
  byte-identical.
- Every synthetic geometry, covariance, local structural, and artifact gate in
  the preregistration passed.
- The deterministic local diagnostic was visually inspected and remained
  untracked.
- Two independent adversarial implementation reviews passed. A separate
  post-candidate results review is the final evidence-promotion gate; its
  disposition is recorded in `release-index.json`.

The source revision's public synthetic CI run passed:
[GitHub Actions run 30437837817](https://github.com/6ickomod3/fusion-fault-bench/actions/runs/30437837817).
That CI run had `dataset_access=false`; it validates the synthetic and software
paths but is not evidence for the local-data attestations.

## Artifact and repeat identities

`artifact_sha256` addresses `manifest.json` and
`geometry-validation.json` through the exact canonical
`payload-index.json`. `run_sha256` addresses the volatile finalized
`run.json`.

| Identity | Primary | Repeat |
|---|---|---|
| Manifest SHA-256 | `7bfb5427c1ea5450a795bedef327457e5959860316bcd23f930e6eaa5917a068` | same |
| Artifact SHA-256 | `09159042ca063b50762bf4150fb275b8a1760e4317ab74da8b0d24c133f42c90` | same |
| Run ID | `run:697f42275ec0c2bffd91718bf6806c4e8900318e597e7f5da727643200a88ff6` | same |
| Run-record SHA-256 | `462e15bb9da0b8caa43b6040b7f147fa64d84c7cdf51264acd02a1bd2eadc50f` | `5931411b25a169533a08849593af4416dd4a4d930054b93067f53199f0d27449` |

Exact comparisons passed for:

1. `manifest.json`;
2. `geometry-validation.json`; and
3. `payload-index.json`.

The artifact digest therefore remained identical. `run.json` records new
timestamps on each execution, and `_SUCCESS` commits the resulting run-record
digest, so those two files are intentionally not stable. The diagnostic SVG
was compared separately, matched exactly, was visually inspected, and was not
curated.

## Public release validation

Run:

```bash
uv sync --locked --group dev
uv run python tools/m2_release.py validate \
  reports/releases/m2-geometry-v0.1.0
```

The validator:

- enforces the exact release-tree file allowlist and rejects symlinks;
- reconstructs the original five-file artifact from curated names and invokes
  the production strict geometry-bundle loader;
- recomputes canonical JSON, all member hashes and lengths, manifest,
  scientific-artifact, run-record, and success-marker identities;
- checks the frozen source revision, manifest, lockfile, package, environment,
  logical command, artifact, run ID, and primary run digest;
- recomputes every synthetic tolerance decision, covariance ratio and
  conjunction available in the sanitized record;
- verifies all release-document hashes and the exhaustive release index; and
- exactly regenerates the sanitized validation-summary SVG from the curated
  manifest and aggregate validation record; and
- confirms the fixed separate nuScenes terms, attribution, and
  non-endorsement notice.

The validator cannot independently re-execute local-data attestations from the
curated aggregate record. That requires a separately obtained nuScenes tree
and the clean source runner.

## Reproduce the source evidence

Use a fresh checkout at the scientific source revision and obtain nuScenes
v1.0-mini directly from its official distributor:

```bash
git checkout --detach cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4
uv sync --locked --group dev
uv lock --check
export NUSCENES_ROOT=/absolute/path/to/nuScenes

uv run ffb geometry validate \
  examples/validation/m2-geometry-v1.json \
  --dataset-root-env NUSCENES_ROOT \
  --output-dir reports/generated/m2-geometry
uv run ffb geometry bundle validate reports/generated/m2-geometry
```

The no-overwrite runner requires a fresh ignored destination. A second exact
run can be performed after retaining the first output outside that
destination. Compare `manifest.json`, `geometry-validation.json`, and
`payload-index.json`; do not expect volatile run timestamps to match.

The clean-source runner verifies the repository and imported package,
repository-local environment, pinned Python, locked dependency closure, clean
Git state, frozen manifest path and digest, fixed relative output path, and
dataset-root separation. Failure output is sanitized so an invalid dataset
path is not echoed.

## Privacy and data audit

The committed release contains only:

- three release documents and one canonical release index;
- one deterministic sanitized aggregate validation figure;
- the frozen repository-owned manifest;
- one sanitized aggregate validation record;
- the sanitized primary run record; and
- the source payload index and success marker.

It contains no nuScenes JSON table, image, point cloud, map, archive, token,
sensor/sample timestamp, dataset filename, calibration, pose, box,
per-frame/per-scene result, absolute dataset root, local projection residual,
or diagnostic SVG. Repository-relative code/fixture paths and execution
timestamps remain in the frozen manifest and run-provenance record. The local
artifact profile reports headline counts and check attestations, but explicitly
states that it does not authenticate dataset bytes.

The aggregate validation record, its tabular presentation, and the summary
figure are marked `CC BY-NC-SA 4.0 plus Motional Dataset Terms`, attributed to
“nuScenes: A multimodal dataset for autonomous driving, Caesar et al., 2020,”
and state that Motional does not sponsor, approve, or endorse Fusion Fault
Bench. Repository-owned software and the synthetic fixture remain under
Apache-2.0. Original explanatory prose remains Apache-2.0 except where it
presents the dataset-derived aggregate; Apache-2.0 does not relicense those
derived portions.

## Verification boundary

M2 verifies the declared frame, transform, projection, ROI, box, covariance,
local metadata-adapter, and artifact contracts. It does not read raw sensor
contents, evaluate a detector, execute temporal sequences, inject calibration
or timing faults, estimate a crossover, authenticate an upstream dataset
download, or support safety, production, or fleet conclusions.
