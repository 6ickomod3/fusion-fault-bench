# M5 release-tooling WIP handoff

Status: **work in progress; not an implementation-review or release
checkpoint**.

Date: 2026-07-29.

This note preserves an interrupted, pre-dataset implementation state so work
can resume without treating partially reviewed code as released evidence. The
authoritative public branch before this work is revision `da5e040`, which
contains the frozen M5 release-pipeline plan and its adversarial plan review.

No nuScenes file was accessed, no replay was run, no outcome was inspected, and
no generated or private evidence is included in this checkpoint.

## Implemented so far

- `ReplayFigureSourceBindingV1`, the exact five figure definitions, typed
  source links, descriptor-coordinate source IDs, rendered-byte commitments,
  and aggregate-artifact integration.
- Outcome-independent source selection for the fixed 100-row persistent,
  10-row crossover, 43-row health, 67-row descriptor, and complete
  primary/control cluster-sensitivity figure grids.
- Canonical figure specifications and deterministic SVG generation.
- A draft `replay_release` module containing implementation/software review
  attestations, exact 34-file candidate construction/loading, check-specific
  validation-evidence derivation, results-review attestation, five-input
  regeneration, exact 41-file package construction/loading, reviewed-evidence
  synchronization, and package/publication validation.
- A repository release tool exposing the ten preregistered commands.
- A descriptor-relative timed replay wrapper with exclusive `0600` timing-log
  creation, no-follow checks, parent reauthentication, fsync, child-output
  suppression, exit propagation, and failed-attempt preservation.
- Installed `ffb replay release validate` delegation with path-free success and
  sanitized failure output.

The existing strict 14-file aggregate artifact now treats the 17 validation
digests as opaque commitments. The enclosing release validator is responsible
for independently deriving all 17 check-specific authorities; it must receive
comprehensive tamper tests before release use.

## Verification completed at this WIP point

- `ruff format --check` over `src/`, `tools/`, and `tests/`: passed, 185 files.
- `ruff check` over `src/`, `tools/`, and `tests/`: passed.
- `pyright`: passed with 0 errors and 0 warnings.
- Current tool, installed-CLI, and figure tests: 30 passed.
- An earlier combined figure/artifact/contract run passed 46 tests; after the
  final validation-boundary adjustment, its directly affected record-link test
  was rerun and passed.

The complete Pytest suite has **not** been run against the final WIP tree.
`tests/test_replay_release.py` does not exist yet, so the draft candidate and
final-package APIs have no dedicated integration, transaction, or tamper suite.

## Unresolved adversarial findings

The partial independent figure review found no P0 finding and the following
release blockers:

1. **P1 — Visible figure semantics are incomplete.** SVG output currently
   shows labels, status, point, and interval but omits required hypothesis
   distinction, persistence/sign partitions, nonpositive-control
   interpretation, common-mode ambiguity wording, and visible crossover
   censoring.
2. **P1 — Projection identifiers are outcome-dependent.** The projection ID
   includes the source-record digest, so an estimate change changes an ID that
   is required to remain stable for a fixed source coordinate.
3. **P1 — Crossover labels are opaque.** The visible label uses a truncated
   hash instead of a human-readable physical fault-axis identity.
4. **P1 — Exact figure grids are not enforced by the aggregate-only artifact
   validator.** This is acceptable only if the enclosing release validator
   regenerates and byte-compares the complete exact bundle, with tests proving
   that incomplete 100/10/43/67/cluster grids fail closed.

The same review recorded two P2 gaps:

- native-unit facets need visible axes, domains, ticks, and labels that prevent
  invalid cross-unit comparison; and
- tests need exact ordered-coordinate and visible-semantic assertions, not only
  counts and byte determinism.

The release-core handoff also identified these unresolved items:

- make isolated wheel smoke explicitly offline and remove `NUSCENES_ROOT` from
  its environment;
- add tamper coverage for every one of the 17 validation authorities;
- adversarially review `sync_reviewed_evidence` failure atomicity when its
  second exclusive copy fails;
- test exact candidate/final allowlists, regeneration, privacy, no-overwrite,
  rollback, sidecar digests, package limits, and standalone offline loading;
  and
- independently review the final release-core, CLI, and transaction code. The
  interrupted figure review did not cover those areas.

## Resume sequence

1. Work from this WIP branch and inspect the complete diff from `da5e040`.
2. Resolve every P1 and P2 finding above without reading replay outcomes.
3. Add `tests/test_replay_release.py` with positive synthetic construction and
   exhaustive fail-closed/tamper/transaction tests.
4. Rerun the focused figure, artifact, release-tool, and CLI suites.
5. Run the full release checks:

   ```bash
   UV_CACHE_DIR=/private/tmp/ffb-m5-release-uv-cache \
     uv run --frozen --no-sync ruff format --check .
   UV_CACHE_DIR=/private/tmp/ffb-m5-release-uv-cache \
     uv run --frozen --no-sync ruff check .
   UV_CACHE_DIR=/private/tmp/ffb-m5-release-uv-cache \
     uv run --frozen --no-sync pyright
   UV_CACHE_DIR=/private/tmp/ffb-m5-release-uv-cache \
     uv run --frozen --no-sync pytest
   UV_CACHE_DIR=/private/tmp/ffb-m5-release-uv-cache \
     uv build --no-sources
   ```

6. Obtain a fresh independent adversarial implementation review of the exact
   complete source snapshot. Resolve all P0, P1, and P2 findings.
7. Only after the reviewed tooling and its canonical attestation are committed,
   pushed, and remotely verified may authoritative primary/repeat nuScenes
   execution begin.

Do not describe this branch or commit as a passed M5 implementation review, a
validated release pipeline, an M5 result, or a release candidate.
