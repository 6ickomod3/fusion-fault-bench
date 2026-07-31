# M5 whole-revision implementation review

Independent adversarial implementation review of the M5 release revision,
covering the seven required areas: runner, release-builder, validators,
claim-projections, figures, privacy-boundary, and failure-and-rollback.
Reviewer identity scope: operator-recorded, not cryptographically
authenticated. Disposition: **pass with non-blocking findings**.

## Scope

The whole M5 implementation revision on branch `codex/m5-release-pipeline`.
The review exercised the sandboxed metadata-only runner, the two-phase
review-candidate and final-package builders, the offline validators, the
deterministic claim projections and five-figure contract, the role-aware
privacy boundary, and the failure-atomic publication and rollback paths. It
also examined two changes made while completing the revision: the completion of
the ninth closeout-document (dashboard) projection with its re-pinned frozen
methodology digest, and a runner reload fix.

## Assessment by area

**Runner.** The local artifact reload was corrected to validate NDJSON members
in JSON mode rather than strict Python mode, so JSON arrays are accepted for the
`tuple` schedule fields that a real all-ten-scene run produces. The relaxation
is confined to the array/tuple axis; every other strict type check remains, and
the pre-existing exact canonical round-trip check still fails closed on any
value-changing coercion. The metadata guard uses per-component no-follow opens,
full-stat identity binding, time-of-check/time-of-use re-checks, and
symlink/hard-link alias rejection.

**Release-builder.** The candidate and final builders reload persisted bytes,
require the exact ordered scientific members, and reconstruct the reviewed
candidate rather than trusting supplied digests. The dashboard projection is
deterministic and restricted to validated reviewed values plus fixed static
text, replacing only the single frozen marker region.

**Validators, claim-projections, figures.** The offline release and publication
validators re-derive every digest from its named authority. The claim registry
is the sole numeric source, and the five figure specs regenerate their rendered
output byte-for-byte from spec and aggregate bytes.

**Privacy-boundary.** Every member is scanned on build and load. The frozen
methodology digest grants the role-aware scan bypass only for the exact tracked
methodology bytes; a stale or tampered digest fails closed. The re-pin of that
digest to the amended pre-outcome plan was verified internally consistent across
both copies of the constant.

**Failure-and-rollback.** Publication is failure-atomic: exclusive no-follow
writes with before/after identity verification, no-replace renames, and cleanup
limited to process-owned staging. The pending and clean source-state gates are
exact and content-fingerprinted.

## Findings

- **frozen-digest-direct-test (P2, resolved).** The security-critical copy of
  the frozen methodology digest (the one granting the scan bypass) previously
  had only an indirect test. A direct byte-binding test was added that pins that
  copy to the tracked methodology bytes and asserts it stays identical to the
  builder copy. Runtime behaviour was already fail-safe; the gap was
  test-coverage only.

No P0 or P1 findings. No unresolved findings.

## Disposition

Pass with non-blocking findings. The revision is release-permitting: the
implementation is correct, the reload fix and digest re-pin are sound and
guarded, the privacy and atomicity boundaries hold, and the single P2 finding is
resolved. This attestation records the reviewer disposition only; it is not
execution proof and must be re-affirmed against the exact pushed scientific
revision before the authoritative runs.
