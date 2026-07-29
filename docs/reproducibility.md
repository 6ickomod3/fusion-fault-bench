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

## Released M2 geometry evidence

The curated M2 release can be validated without a dataset:

```bash
uv run python tools/m2_release.py validate \
  reports/releases/m2-geometry-v0.1.0
```

The validator enforces the exact release allowlist; reconstructs and strictly
loads the sanitized five-file geometry bundle; recomputes manifest, artifact,
run, member, document, and figure identities; regenerates the release-summary
SVG; and checks the separate nuScenes-derived evidence terms. The release
records:

- scientific source revision
  `cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4`;
- manifest SHA-256
  `7bfb5427c1ea5450a795bedef327457e5959860316bcd23f930e6eaa5917a068`;
- artifact SHA-256
  `09159042ca063b50762bf4150fb275b8a1760e4317ab74da8b0d24c133f42c90`;
- the same locked environment and named Apple M3 Pro configuration as M1;
  and
- three of three stable-file comparisons and an ignored diagnostic comparison
  that were byte-identical across two clean runs.

The exact local execution requires a separately obtained mini tree:

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

The deterministic scientific comparison covers `manifest.json`,
`geometry-validation.json`, and `payload-index.json`. `run.json` and
`_SUCCESS` change with timestamps and the resulting run-record digest.
Cross-architecture byte identity is not claimed.

The public loader recomputes all sanitized synthetic numeric gates and
conjunctions. It cannot recompute local metadata, key-frame-reference,
projection, diagnostic-generation, or visual-inspection attestations without
the separately obtained dataset and local run. The public CI record explicitly
has `dataset_access=false`.

## Local data

nuScenes is optional for M2 local grounding and planned M5 replay. It is never
downloaded by the package or CI.

```bash
export NUSCENES_ROOT=/absolute/path/to/nuScenes
```

Absolute dataset paths belong only in local runtime configuration. They are
excluded from manifests and public result records.

## Released M3 temporal procedural evidence

The aggregate-only M3 release is dataset-free and can be validated offline
from a clean checkout:

```bash
git checkout m3-procedural-v0.1.0
uv sync --locked --group dev
uv run python tools/m3_release.py validate-release \
  reports/releases/m3-procedural-v0.1.0 \
  --official-identity examples/release-identities/m3-procedural-v0.1.0.json
```

The standalone validator checks the exact 88-file allowlist, official
Git-bound identity, independent review bytes, all 429 aggregate and 10
crossover rows, 71,700 omitted-row commitments, source payload indexes,
primary/repeat run graphs, summaries, documentation, and three regenerated
SVG figures.

The scientific rows originate from revision
`e8595fe428bcb9dfb269069e4b02972aff10f4ee`. Full regeneration executes the
frozen matrix twice:

```bash
git checkout --detach e8595fe428bcb9dfb269069e4b02972aff10f4ee
uv sync --locked --group dev
uv run python tools/m3_release.py execute \
  examples/matrices/m3-procedural-v1.json \
  --first-output-dir reports/generated/m3-reproduction-first \
  --second-output-dir reports/generated/m3-reproduction-second \
  --evidence-dir reports/generated/m3-reproduction-evidence
uv run python tools/m3_release.py validate \
  examples/matrices/m3-procedural-v1.json \
  --first-output-dir reports/generated/m3-reproduction-first \
  --second-output-dir reports/generated/m3-reproduction-second \
  --evidence-dir reports/generated/m3-reproduction-evidence
```

The two released runs measured 1218.376 and 1258.861 seconds with peak RSS
369,000,448 and 389,431,296 bytes on an Apple M3 Pro. These are observations
self-reported by the tracked child-process driver, not independently
recomputable facts. All 48 indexed scientific member pairs were byte
identical; cross-architecture byte identity is not claimed.

## Released M4 observable-health evidence

The strict M4 aggregate release is dataset-free:

```bash
uv sync --locked --group dev
uv run python tools/m4_release.py validate-release \
  reports/releases/m4-health-v0.1.0
```

The validator enforces the exact curated allowlist; authenticates both fit
copies and both evaluation provenance envelopes; recomputes scientific member
commitments, aggregate structure, repeat equality, resource sidecars, numeric
claim projections, privacy scans, release index, and success marker; and
rejects unexpected sequence payloads or local paths.

Scientific regeneration starts from revision
`a829a9f3af541c1b92b89d051b7c8b7003dc5a15`:

```bash
git checkout --detach a829a9f3af541c1b92b89d051b7c8b7003dc5a15
uv sync --locked --group dev

uv run ffb health fit \
  --output-dir reports/generated/m4-reproduction-fit-primary
uv run ffb health fit \
  --output-dir reports/generated/m4-reproduction-fit-repeat

uv run ffb health evaluate \
  reports/generated/m4-reproduction-fit-primary \
  --output-dir reports/generated/m4-reproduction-eval-primary
uv run ffb health evaluate \
  reports/generated/m4-reproduction-fit-primary \
  --output-dir reports/generated/m4-reproduction-eval-repeat
```

Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `1`
for the named single-thread resource profile. Strictly validate each fit and
evaluation with:

```bash
uv run ffb health bundle fit validate <fit>
uv run ffb health bundle evaluation validate <evaluation> \
  --fit-artifact reports/generated/m4-reproduction-fit-primary
```

The released fits measured `279.62` and `286.17` seconds with peak RSS
`121,585,664` and `123,486,208` bytes. Evaluations measured `592.01` and
`559.82` seconds with peak RSS `169,869,312` and `168,116,224` bytes on an
Apple M3 Pro. The four raw Darwin `/usr/bin/time -l` records are retained and
reparsed, but remain operator-recorded, self-reported evidence.

Both fits matched on seven indexed scientific members; both evaluations
matched on nine. The public release retains all 11,515 aggregate rows but
omits and commits 433,700 sequence-level rows. Aggregate claims and figures
for scientific outcomes are reproducible from those rows; fit-selection and
resource tables use the other retained release evidence. Independently
recomputing bootstrap intervals requires full regeneration because the
sequence rows are not public.

## Verification roadmap

| Level | Current status |
|---|---|
| Contract and canonicalization unit tests | Validated in M0 |
| Package build and isolated-wheel smoke test | Validated in M0 |
| Analytic RNG, fusion, fault, metric, and artifact tests | Validated in M1 |
| Geometry and temporal fault tests | Geometry foundation released in M2; temporal matrix released in M3 |
| Transform and covariance property tests | Validated and released in M2 |
| Independent-Gaussian analytic oracles | Validated in M1 |
| Local nuScenes profile integrity and scalar projection cross-check | Attested and released in M2 |
| Named-CPU analytic release and deterministic repeat | Released in M1 |
| Named-CPU geometry release and deterministic stable-file repeat | Released in M2 |
| Full multi-family clean-CPU procedural fault report | Released in M3 |
| Observable health-aware fallback and deterministic repeat | Released in M4 |
| nuScenes latent-scene replay | Planned for M5 |
