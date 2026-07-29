# Reproducibility

## Environment

Fusion Fault Bench pins Python 3.12.13 and uses `uv`. The base package has no
GPU, Torch, nuScenes, or plotting dependency. NumPy supplies the pinned
PCG64DXSM implementation and independent bootstrap recomputation.

```bash
uv sync --locked --group dev
uv run ffb --version
uv run pytest
```

The committed `uv.lock` is the dependency resolution used by continuous
integration. A local `.venv` is ignored.

## Manifest validation

```bash
uv run ffb manifest validate \
  examples/manifests/analytic-bias-v1alpha1.json

uv run ffb manifest digest \
  examples/manifests/analytic-bias-v1alpha1.json
```

Reordering JSON keys does not change a manifest digest. Changing experimental
intent such as a seed or severity does. Duplicate object keys, non-standard
NaN/infinity values, and negative zero at canonical identity fields are
rejected.

The CLI validator is normative because it applies cross-field scientific
semantics in addition to the committed JSON Schema. The exported schema captures
the discriminated experiment, fault, and evaluation variants and is useful for
editors and structural validation, but external JSON-Schema validation alone
does not replace CLI conformance.

## Analytic execution and artifact verification

M1 runs only from a clean Git checkout with a tracked manifest and locked
environment:

```bash
uv run ffb run \
  examples/manifests/analytic-bias-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-x-bias-a603d090f77a

uv run ffb bundle validate \
  reports/generated/analytic-camera-x-bias-a603d090f77a
```

Before evaluation, the runner verifies that its working directory and imported
package both come from that checkout, that the process uses the checkout's
`.venv`, that Python matches `.python-version`, and that every installed
runtime dependency in the `uv.lock` dependency closure has the locked version.

The output directory is created atomically and is never overwritten. Its five
scientific payload members are content-addressed by `payload-index.json`;
`run.json` records the clean source revision, lock digest, path-independent
reproduction command, timestamps, and named CPU environment. `_SUCCESS` is
written last and commits separately to both the stable scientific artifact
digest and the exact volatile `run.json` bytes. The strict loader rejects
noncanonical bytes, wrong file order, hash or length mismatches, symlinks,
unexpected files, contradictory result rows, and analytic evidence that cannot
be rebuilt from the sequence metrics and independent closed-form reference.

The deterministic payload promise excludes `run.json`, whose timestamps and
machine facts intentionally vary. Two reruns on the same named CPU and locked
software environment, with the same manifest and source revision, must have
byte-identical indexed scientific payloads and the same payload digest.
Cross-architecture byte identity is not claimed by M1.
The run-record digest detects unsynchronized mutation but is not a signature;
the public release Git commit is the authenticity boundary.

## Local data

nuScenes is optional until the geometry-grounding milestone and is never
downloaded by the package or CI.

```bash
export NUSCENES_ROOT=/absolute/path/to/nuScenes
```

Absolute dataset paths belong only in local runtime configuration. They are
excluded from manifests and public result records.

## Verification roadmap

| Level | Current status |
|---|---|
| Contract and canonicalization unit tests | Validated in M0 |
| Package build and isolated-wheel smoke test | Validated in M0 |
| Analytic RNG, fusion, fault, metric, and artifact tests | Validated in M1 |
| Geometry and temporal fault tests | Planned for M2–M3 |
| Transform and covariance property tests | Planned for M2 |
| Independent-Gaussian analytic oracles | Validated in M1 |
| nuScenes integrity and projection agreement | Planned for M2 |
| Clean-CPU release reproduction | Planned for M6 |
