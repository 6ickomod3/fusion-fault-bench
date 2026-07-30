# M5 adversarial implementation review

Status: **PASS for clean execution; dataset replay and results not reviewed**.

Date: 2026-07-29.

This review covers the CPU-only implementation of the frozen
[`m5-nuscenes-mini-replay-v1.json`](../../examples/replay/m5-nuscenes-mini-replay-v1.json)
intent and [`m5-replay-plan.md`](../m5-replay-plan.md), including the
pre-outcome
[`m5-resource-scope-amendment.md`](../m5-resource-scope-amendment.md).
It does not approve a dataset profile, numerical result, resource result,
persistence claim, or release. No M5 nuScenes data was read while closing this
implementation checkpoint.

## Review scope

Independent adversarial passes covered:

- full-frame geometry, calibration and timing causality, and analytic oracles;
- replay identity, exact grids, support, scene-first inference, and crossover
  semantics;
- pre-update health causality, apply-only M4 transfer, and prohibited-feature
  leakage;
- strict typed reload, exact eight-member scientific repeatability, and
  curation from persisted bytes;
- no-overwrite publication, failure atomicity, path substitution, hard links,
  symbolic links, and source-revision races;
- metadata-only access, raw-payload read accounting, Torch/CUDA boundaries,
  and scientific-worker process scope;
- externally imported resource evidence, exact command authority, and public
  cross-bindings; and
- privacy, licensing, public role allowlists, limitations, and claim scope.

## Release-blocking findings and resolutions

### Failure-atomic local publication

The initial transaction could leave a loadable success marker when a later
gate failed. A pending local artifact now has no `_SUCCESS`; strict reload and
identity checks occur before finalization, and a post-finalization failure
removes the marker. Mutation tests cover failed publication, failed final
reload, destination substitution, and rollback ownership.

### Persisted-byte curation authority

The first curation path relied too heavily on in-memory benchmark objects.
Every scientific member is now reloaded as canonical JSON or NDJSON, decoded
through its typed contract, and reassembled into the exact
`ReplayBenchmarkEvidence` grid before curation. The curation entry point
freshly verifies both separately created artifacts and rechecks the same clean
source snapshot after curation.

### Exact scientific repeatability

The public source-commitment contract accepts exactly eight ordered roles:

1. descriptor aggregates;
2. health population metrics;
3. health sequence contrasts;
4. health sequence events;
5. health sequence results;
6. persistent crossovers;
7. persistent population metrics; and
8. persistent scene evaluations.

Primary and repeat executions must match byte-for-byte on those members.
Arbitrary roles cannot satisfy the gate. Each local artifact also
content-addresses its own `resources.json`; therefore volatile elapsed time and
RSS may make the enclosing local artifact digests differ without weakening
scientific equality. Both artifact and run digests remain retained and
cross-bound to their corresponding resource records.

### Metadata, path, and raw-read integrity

The adapter and local loaders open bounded regular files with no-follow
semantics, reject hard links, compare initial/open/final fingerprints, and
reject replacement or rewrite races. The scoped read guard counts attempted
raw-payload access before rejecting it. Successful replay loading requires the
exact metadata-table read count and zero blocked raw-payload reads.

### Resource and current-source authority

Both the live runner and standalone curated-artifact validator require the
exact logical command `ffb replay run --output-dir <safe generated path>`.
Strict loaders reject coercible booleans or strings in numeric resource fields.
Two distinct Darwin `/usr/bin/time -l` logs are safely imported, parsed with
the frozen grammar, and bound to the exact local artifact, run, environment,
logical command, and internal diagnostics.

The one-process preregistration is recorded as one scientific replay worker
with no benchmark multiprocessing. Only sequential provenance, environment,
and resource-measurement helpers are permitted. Verification and curation
authenticate the current clean revision, lock digest, package version,
environment, relative output command, and unchanged source before returning.

## Verification evidence

The stable implementation checkpoint reported:

- 250 focused runner, resource, artifact, curation, CLI, and adapter tests
  passed;
- the independent reviewer reran 235 selected adversarial tests and returned
  no P0, P1, or P2 blocker;
- 1,542 repository tests passed and one optional local-data test skipped;
- branch-aware repository coverage was 91.88% against the 90% gate;
- Ruff format and lint passed, and strict Pyright reported zero errors;
- the source distribution and wheel built successfully; and
- an isolated Python 3.12 environment imported version `0.1.0`, exposed the M5
  replay CLI, and rendered the replay validation and resource-evidence
  schemas.

The tracked-file privacy audit found no dataset payload, generated raw output,
private interview material, credential, or real local dataset path in the
checkpoint. Synthetic private-path and credential strings are retained only
inside negative privacy tests.

## Residual limitations

External timing evidence is operator-recorded and self-reported. Its exact log
bytes are committed and structurally cross-bound, but that is not a
cryptographic proof that `/usr/bin/time` launched the claimed command or
observed its exit.

The process-spawn and raw-read controls are scoped Python-level guards plus
hardened loaders and mutation tests, not operating-system syscall attestation.
Dataset bytes remain unauthenticated. Distinct paths, inodes, run hashes, and
exact member equality provide consistency evidence, not proof of independent
physical executions.

## Verdict

No P0, P1, or P2 implementation blocker remains. M5 may proceed from a clean
committed source revision to two separately timed nuScenes-mini executions.
Local-data checks, resource caps, numerical acceptance gates, curated
aggregates and figures, adversarial results and claims review, and the M5
release commit/tag remain mandatory. This verdict advances none of them.
