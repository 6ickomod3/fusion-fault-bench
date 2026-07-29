# M1 Release Verification

This document records how the M1 evidence was admitted, what is independently
checkable from the repository, and what remains intentionally unpublished.
`release-index.json` is the canonical machine-readable release index; this file
is its human-readable audit trail.

## Admission summary

- Frozen manifests: three analytic crossover experiments.
- Scientific source revision:
  [`524c8f70ece3eca2e61796165b23ffe51baadfbc`](https://github.com/6ickomod3/fusion-fault-bench/commit/524c8f70ece3eca2e61796165b23ffe51baadfbc).
- Lockfile SHA-256:
  `ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`.
- Named execution environment: Apple M3 Pro, arm64, 11 logical CPUs,
  19,327,352,832 bytes of memory, Darwin 24.5.0, Python 3.12.13.
- Six source bundles—three primary and three repeat—passed strict bundle
  validation.
- The three within-revision primary/repeat pairs produced 18/18
  byte-identical stable-file comparisons.
- Every independent analytic-validation record passed its frozen
  six-standard-error Monte Carlo gate.

The source revision's public CI run passed:
[GitHub Actions run 30427718867](https://github.com/6ickomod3/fusion-fault-bench/actions/runs/30427718867).

## Artifact and repeat identities

`artifact_sha256` addresses the indexed scientific payload. `run_sha256`
addresses the exact volatile primary `run.json`; the repeat run digest differs
because timestamps are intentionally variable.

| Experiment | Manifest SHA-256 | Artifact SHA-256 | Primary run SHA-256 | Repeat run SHA-256 |
|---|---|---|---|---|
| Camera x-bias | `a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e` | `3717c2b3fdce9e9f2bc43463434fde28a4d24dd9ca1d72e451b3d9d5273c2959` | `6cc2cefdb0a37aad5c58da8b55462729d294d5e99bb707b866c9d54b2e8ad63c` | `bb37c2f21acdc8a1e542c06a19f776bbd92cab1682e7355d41437e011bd816f4` |
| Correctly reported camera noise | `3ea7ffc2949cf99f20d20ec18844f0b8dc3b3ebb81e13e926f7440b7c5084176` | `51abb5043ddffd633c0fa81ea5d69dc6c1246d092185f22261f183915f911467` | `085d928b68a785d59b41f1879d4d262030a855ef0677771866d2d91efe6268e3` | `5bda8ece678016346b57857d55a46db8eed7d1eabfb24c6cbb9b078273841585` |
| Underreported camera noise | `9d26e1b33f1fd2e35b0de90703a960d2eba6bb26bd2219bce6f0bb82480f4ac4` | `8a3c2179e49cdc2ae994d9c791185a546788252710a0e80fca73cf39305165e7` | `7f1a2fa90eeb2e83267c76c1863eb23c707ecc893604cb025166a2f00da7f594` | `dd54fb25500c7f47fc7e9c573b9aba8da976cc4cc4afbd77d9382189240503f3` |

The stable comparison allowlist was, for each experiment:
`manifest.json`, `sequence-metrics.ndjson`,
`aggregate-metrics.ndjson`, `crossovers.ndjson`,
`analytic-validation.json`, and `payload-index.json`. The first five are the
indexed scientific payload; `payload-index.json` is the deterministic envelope
that commits them and is the sixth stable file. Across three primary/repeat
pairs, all 18 comparisons were byte-identical. `run.json` and `_SUCCESS` are
excluded from stable-byte comparison because their timestamps and run-record
digests are deliberately volatile; each was nevertheless strictly validated.

## Independent analytic checks

The production evaluator was checked against an independent diagonal-Gaussian
closed-form module. All population points passed the frozen
`absolute_standardized_error <= 6` rule.

| Experiment | Maximum absolute standardized error | Population crossover reference |
|---|---:|---|
| Camera x-bias | 1.0144140839301548 | Grid root 3.8282790927021715 m; continuous root 3.869066367512064 m |
| Correctly reported camera noise | 0.9162214715685516 | No grid root through 4.0; no finite continuous root |
| Underreported camera noise | 1.0842355904362782 | Grid root 1.4630684126547195 std scale; continuous root 1.4657551414886727 std scale |

These checks validate the declared Gaussian estimator-output algebra. They do
not validate transfer to a raw-sensor noise process.

## Provenance amendment and excluded execution

The original preregistration required a named hardware run. The first
execution at source revision
[`1649dec5de387dd8b408a14678fa0acad0818735`](https://github.com/6ickomod3/fusion-fault-bench/commit/1649dec5de387dd8b408a14678fa0acad0818735)
recorded `cpu_model="arm"`. That is an architecture family, not a named
processor. Although all six original bundles were scientifically valid, their
hardware provenance failed the release gate.

Before replacement execution and before any result release, commit
[`c381d1cb67b4d60724d57e4431c1edddbbd2d874`](https://github.com/6ickomod3/fusion-fault-bench/commit/c381d1cb67b4d60724d57e4431c1edddbbd2d874)
amended the preregistration to:

- exclude all six old artifacts from quantitative public evidence;
- require Darwin discovery to record a specific processor/chip name or fail
  closed;
- freeze all scientific settings and manifest digests;
- require three primary and three repeat replacement runs; and
- require exact old/new scientific equality after excluding only the
  provenance-bound `run_id`.

The replacement runs recorded `cpu_model="Apple M3 Pro"`. The curation audit
strictly loaded the three withheld primary comparison bundles, then compared
five scientific file views per experiment—manifest, sequence rows, aggregate
rows, crossovers, and analytic validation. All 15/15 file comparisons and all
23,148 normalized records were exactly equal after recursively excluding only
`run_id`. A new source revision necessarily changes the deterministic run ID
and therefore changes payload-envelope identities even when numerical values
are equal.

The `1649dec` artifacts remain excluded and are not committed in this release.
Consequently, a public clone can verify the recorded equivalence counts in
`release-index.json` but cannot independently rerun that old/new comparison
without the separately retained withheld bundles. No quantitative result in
this release cites an old artifact.

## Curated-row omission

The aggregate evidence, crossover records, independent analytic references,
manifests, provenance, source payload indexes, and completion records are
committed. Reproducible synthetic `sequence-metrics.ndjson` intermediates are
omitted from Git history:

| Experiment | Omitted rows | Byte length | SHA-256 |
|---|---:|---:|---|
| Camera x-bias | 13,000 | 8,033,267 | `82af401303cb7783db46ff6c769cba285bc08486930e2847dfae40349e623000` |
| Correctly reported camera noise | 5,000 | 3,195,026 | `33b888922c7af9341198a47dc576a903468055df0ca7643fa4919d78f55b3887` |
| Underreported camera noise | 5,000 | 3,169,944 | `7427ae25dddc60c55d10a10c548de8288422efe5d60db69b0693f1e250615478` |

For each omission, `release-index.json` retains the row count, exact byte
length, and SHA-256, while the curated `source-payload-index.json` retains the
original scientific member commitment. The commands below regenerate the
rows.

## Public validation

From the tagged release checkout:

```bash
uv sync --locked --group dev
uv lock --check
uv run python tools/m1_release.py validate \
  reports/releases/m1-analytic-v0.1.0
```

The release validator:

- canonical-loads every JSON and NDJSON record;
- verifies every curated record, figure, and document length and SHA-256 from
  `release-index.json`;
- recomputes artifact and primary run-record digests;
- cross-binds record, manifest, payload, run, environment, source revision,
  lockfile, package version, and logical command identity;
- reconciles curated scientific bytes with each source payload index;
- checks omitted sequence member commitments; and
- exactly regenerates both SVG byte strings from the curated aggregate rows;
- exhaustively regenerates the expected release-index content; and
- enforces the frozen official primary and repeat identities.

## Reproduce the source evidence

Use a fresh checkout at source revision
`524c8f70ece3eca2e61796165b23ffe51baadfbc`. All destinations below are under
the ignored `reports/generated/` tree, preserving clean-source enforcement.

```bash
git checkout --detach 524c8f70ece3eca2e61796165b23ffe51baadfbc
uv sync --locked --group dev
uv lock --check

uv run ffb run examples/manifests/analytic-bias-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-x-bias-a603d090f77a
uv run ffb run examples/manifests/analytic-noise-correct-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-noise-correctly-reported-3ea7ffc2949c
uv run ffb run examples/manifests/analytic-noise-underreported-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-noise-underreported-9d26e1b33f1f

uv run ffb run examples/manifests/analytic-bias-v1alpha1.json \
  --output-dir reports/generated/repeat/analytic-camera-x-bias-a603d090f77a
uv run ffb run examples/manifests/analytic-noise-correct-v1alpha1.json \
  --output-dir reports/generated/repeat/analytic-camera-noise-correctly-reported-3ea7ffc2949c
uv run ffb run examples/manifests/analytic-noise-underreported-v1alpha1.json \
  --output-dir reports/generated/repeat/analytic-camera-noise-underreported-9d26e1b33f1f
```

Strictly validate all six:

```bash
uv run ffb bundle validate \
  reports/generated/analytic-camera-x-bias-a603d090f77a
uv run ffb bundle validate \
  reports/generated/analytic-camera-noise-correctly-reported-3ea7ffc2949c
uv run ffb bundle validate \
  reports/generated/analytic-camera-noise-underreported-9d26e1b33f1f
uv run ffb bundle validate \
  reports/generated/repeat/analytic-camera-x-bias-a603d090f77a
uv run ffb bundle validate \
  reports/generated/repeat/analytic-camera-noise-correctly-reported-3ea7ffc2949c
uv run ffb bundle validate \
  reports/generated/repeat/analytic-camera-noise-underreported-9d26e1b33f1f
```

These fresh runs can reproduce the stable scientific results and can be
strictly validated. Their new timestamps produce new `run.json` digests, so
they cannot rebuild this exact archival release: the official curator
deliberately freezes the original primary and repeat identities. Exact
scientific-byte reproduction is claimed only for the locked, same-environment
deterministic reruns; cross-architecture byte identity is not claimed.

To rebuild the exact curated release—including the one-time withheld-execution
equivalence audit—run the release tool from the tagged release revision with
the retained original three primary bundles, three repeat bundles, three
withheld comparison bundles, and final document source:

```bash
export FFB_PRIMARY_ROOT=reports/generated/retained-primary
export FFB_REPEAT_ROOT=reports/generated/retained-repeat
export FFB_WITHHELD_ROOT=reports/generated/retained-withheld
export FFB_DOCUMENTS_ROOT=reports/releases/m1-analytic-v0.1.0
export FFB_REBUILT_RELEASE=reports/generated/rebuilt-m1-release

uv run python tools/m1_release.py build \
  --primary-root "${FFB_PRIMARY_ROOT}" \
  --repeat-root "${FFB_REPEAT_ROOT}" \
  --withheld-root "${FFB_WITHHELD_ROOT}" \
  --documents-root "${FFB_DOCUMENTS_ROOT}" \
  --output-dir "${FFB_REBUILT_RELEASE}"

uv run python tools/m1_release.py validate "${FFB_REBUILT_RELEASE}"
```

The builder has no overwrite mode. The example uses repository-relative,
ignored artifact paths and records no local absolute path. The retained
primary/repeat bundles are required only for their frozen archival identities;
the withheld input is required only for the old/new equivalence audit. None of
those volatile or excluded inputs changes a released scientific result value.
A public clone can validate the committed release and reproduce its stable
science, but it cannot reconstruct the exact archival package without those
retained local inputs.

## Verification boundary

The release verifies deterministic CPU execution, serialization, analytic
fusion/fault algebra, complete-sequence bootstrap inference, and the
preregistered crossover presentation for the declared synthetic models. It
does not verify raw-image or point-cloud behavior, detector performance,
real-fault prevalence, calibration or timing faults, nuScenes grounding,
closed-loop outcomes, or fleet behavior.
