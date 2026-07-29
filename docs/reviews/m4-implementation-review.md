# M4 adversarial implementation review

Status: **PASS for clean execution; results not yet reviewed**.

Post-release note (2026-07-29): numerical evidence was subsequently evaluated
in the separate
[`m4-health-v0.1.0` adversarial results review](m4-results-review.md). The
historical scope statement below is retained to keep implementation approval
separate from results approval.

This review covers the implementation of the frozen
[`m4-health-v1.json`](../../examples/health/m4-health-v1.json) intent and
[`m4-health-plan.md`](../m4-health-plan.md). It does not approve a numerical
result or a public claim. Those require two clean executions, measured resource
evidence, release curation, and a separate results review.

## Review scope

Independent adversarial passes exercised:

- observable-feature leakage and pre-update causality;
- event, attribution, recovery, and false-alert semantics;
- exact matrix membership, denominators, method support, and dropout nesting;
- common-support policy contrasts and frame-oracle comparisons;
- fit authentication and apply-only test evaluation;
- canonical artifact ordering, mutation resistance, and aggregate
  recomputation;
- no-overwrite publication, symlink/hard-link rejection, path substitution,
  source mutation, and cleanup;
- bounded-memory generation, serialization, and strict reload; and
- release curation, omitted-row commitments, resource evidence, privacy, and
  quantitative-claim binding.

## Release-blocking findings and resolutions

### Retained contrast evidence

The first implementation retained method losses and aggregate policy gaps but
not enough sequence-level common-support information to reconstruct every
fixed-policy, target-drop, and oracle comparison. M4 now retains canonical
`ffb.health-sequence-contrast/v1` rows with counts, loss sums, and
domain-separated support-mask digests. The strict loader independently
recomputes every aggregate from loss, contrast, and event rows.

Reverse implications are enforced: equal pairwise support commitments require
equal full counts and full retained loss sums. Unequal-support dropout and
abstention comparisons remain explicitly distinct.

### Exact event and support closure

Mutation tests now reject:

- nonfrozen score, event, recovery, or valid-support denominators;
- impossible subset sums or common-support counts;
- inconsistent nonabstaining, target-drop, oracle, full-dropout, or unrealized
  dropout support;
- fabricated detection, latch, attribution, early-clear, recovery, false-alert,
  state-occupancy, or action-occupancy relations; and
- censored latencies outside their schedule windows.

The exact evaluation matrix is 47 conditions, 8,900 sequence-condition pairs,
264,600 loss rows, 133,500 contrast rows, and 35,600 event rows.

### Authenticated apply-only boundary

The public test entry point accepts only a strictly loaded fit artifact. Intent,
profiles, ECDF arrays, selected thresholds, populations, cases, seeds, and
policy semantics are reloaded or derived from that artifact. Scientific
override parameters are absent. The transaction snapshots and reauthenticates
the fit before publication and after final reload.

### Bounded-memory artifact path

Materializing all retained rows plus canonical byte buffers and two reloads
would have exceeded the pre-registered 1 GiB peak-RSS gate. The exact runner now
evaluates one canonical condition at a time and streams four NDJSON members
through exclusive file descriptors while incrementally computing byte counts,
record counts, SHA-256 commitments, and caps.

Strict loading performs a four-way condition-group merge and retains only one
condition plus the curated aggregate matrix. Exact record-count mismatches fail
before sequence-row parsing. The loaded handle explicitly records whether
sequence rows are materialized.

Measured peak RSS remains a results-release gate; this implementation decision
does not claim that the gate has already passed.

### Transaction and path integrity

Executed adversarial harnesses attempted:

- staging-directory substitution after strict validation;
- destination-parent replacement before and after rename;
- fit-path disappearance after transaction construction;
- alternate valid-artifact substitution during final reauthentication; and
- near-exact count rebinding intended to force full materialization.

The final transaction rejects each case. It pins parent and artifact
descriptors, verifies expected artifact/run identities, reasserts descriptors
after every potentially lengthy authentication step, and removes its own
published tree when a post-rename check fails. It never accepts or returns an
attacker-substituted lexical path.

### Curated release integrity

The release layer retains both complete fit artifacts, one exact copy of each
small evaluation scientific member, both evaluation provenance envelopes, four
raw Darwin `/usr/bin/time -l` logs, and exact commitments for the three omitted
sequence members. It:

- strictly revalidates both retained fits;
- validates the canonical 47-condition aggregate structure;
- cross-binds intent, profiles, fit reference, artifact, run, and success
  identities;
- reparses resource logs and checks their identity-bound sidecars;
- validates every public quantitative claim against retained rows; and
- rejects local paths, dataset content indicators, and common secret patterns.

The resource logs are operator-recorded, self-reported evidence, not
cryptographic proof of execution. Because sequence rows are omitted from the
sub-50 MiB curated release, third parties can regenerate claims and figures
from aggregates but cannot independently rerun the sequence bootstrap. The
release records this limitation literally.

The repository-local release driver additionally requires all four artifact
run records to match the current clean source revision, lock digest, package
version, and clean-source flag. It rejects reused resource paths or inodes,
symlinked generated-input components, and resource paths that change while an
`O_NOFOLLOW` descriptor is open. Adversarial stale-source, cross-phase,
same-log, symlink-parent, and path-substitution counterexamples fail closed.

## Verification evidence

Final implementation verification reported:

- 1,134 repository tests passed and one optional local-data test skipped;
- branch-aware repository coverage was 90.67% against the 90% gate;
- 56 transaction, runner, and CLI tests passed after the final path-race fix;
- 16 release-curation tests passed, including a semantically valid 11,037-row
  aggregate fixture, plus 11 focused release-validator mutation tests;
- 19 focused artifact-security tests passed;
- Ruff format and lint passed; and
- Pyright reported zero errors;
- the source distribution and wheel built successfully; and
- the isolated wheel imported as version `0.1.0` and exposed the M4 schema CLI.

Clean repeat executions, resource measurements, results review, and public CI
remain mandatory results-release steps.

## Verdict

No P0 or P1 implementation blocker remains. M4 may proceed to clean execution
and results review without changing the frozen intent, scientific estimands, or
resource caps.
