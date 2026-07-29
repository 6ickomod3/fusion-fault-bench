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
| Geometry, fault, fusion, and metric tests | Planned for M1–M3 |
| Transform and covariance property tests | Planned for M2 |
| Independent-Gaussian analytic oracles | Planned for M1 |
| nuScenes integrity and projection agreement | Planned for M2 |
| Clean-CPU release reproduction | Planned for M6 |
