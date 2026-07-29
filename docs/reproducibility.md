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

## Released M1 evidence

The curated M1 release can be checked without executing an experiment:

```bash
git checkout m1-analytic-v0.1.0
uv sync --locked --group dev
uv run python tools/m1_release.py validate \
  reports/releases/m1-analytic-v0.1.0
```

The validator recomputes every curated file hash and length, manifest,
scientific-payload digest, volatile run-record digest, record provenance link,
omitted-sequence hash/count commitment, both exact SVGs from the aggregate
rows, all three document hashes, and the exhaustive release index and file
allowlist. The release records:

- scientific source revision
  `524c8f70ece3eca2e61796165b23ffe51baadfbc`;
- `uv.lock` SHA-256
  `ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`;
- Apple M3 Pro, 11 logical CPUs, 19,327,352,832 bytes of memory, arm64,
  Darwin 24.5.0, and Python 3.12.13.

To independently regenerate the scientific rows and run identities, use the
recorded clean source revision rather than the later evidence-promotion tag:

```bash
git checkout 524c8f70ece3eca2e61796165b23ffe51baadfbc
uv sync --locked --group dev

uv run ffb run \
  examples/manifests/analytic-bias-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-x-bias-a603d090f77a
uv run ffb run \
  examples/manifests/analytic-noise-correct-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-noise-correctly-reported-3ea7ffc2949c
uv run ffb run \
  examples/manifests/analytic-noise-underreported-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-noise-underreported-9d26e1b33f1f
```

Strictly validate each generated directory with `uv run ffb bundle validate
<directory>`. Run into a second ignored output root to compare scientific
bytes. The full sequence NDJSON files are intentionally omitted from Git
because they total about 14.4 MB of deterministic synthetic rows; the
release index retains each exact SHA-256, byte length, and record count.

The amendment withheld six provenance-ineligible runs from the pre-fix
revision. The one-time curator strictly loaded three retained old primary
bundles and compared them with the three replacement primaries; the earlier
repeat runs remained excluded. Those withheld artifacts are intentionally
unavailable for public rebuilding. Their role is limited to the amendment
audit recorded in
[M1 verification](../reports/releases/m1-analytic-v0.1.0/verification.md);
they are not needed to regenerate or validate the released scientific result.

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
| Geometry and temporal fault tests | Geometry pre-registered; temporal planned |
| Transform and covariance property tests | Pre-registered for M2 |
| Independent-Gaussian analytic oracles | Validated in M1 |
| nuScenes integrity and projection agreement | Pre-registered for M2 |
| Named-CPU analytic release and deterministic repeat | Released in M1 |
| Full multi-family clean-CPU report reproduction | Planned for M6 |
