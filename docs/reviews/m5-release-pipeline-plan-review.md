# M5 release-pipeline adversarial plan review

Status: **PASS after outcome-blind amendment; whole-revision implementation,
replay outcomes, and release not reviewed**.

Original review date: 2026-07-29.

Amendment review date: 2026-07-30.

Reviewed plan byte SHA-256:
`0a645323d346668707442eb2e9cd76bac221f8a0c9ff48c4baad5bf078ce946d`.

This was an independent, outcome-blind review of
[`m5-release-pipeline-plan.md`](../m5-release-pipeline-plan.md) against the
frozen
[`m5-replay-plan.md`](../m5-replay-plan.md),
[`benchmark-contract-v0.1.md`](../benchmark-contract-v0.1.md), the
outcome-blind
[`m5-resource-scope-amendment.md`](../m5-resource-scope-amendment.md), and the
current replay contracts, curation, artifact, provenance, resource, and runner
code.

No nuScenes data, generated M5 replay output, fault outcome, descriptor,
persistence result, crossover result, health-transfer result, resource result,
or private/interview file was inspected. This verdict approves only the
pre-execution release-pipeline design.

## Disposition

- P0 findings: **0**
- unresolved P1 findings: **0**
- unresolved P2 findings: **0**
- disposition: **PASS for implementation**

Release tooling, synthetic tests, a whole-revision adversarial implementation
review, two authoritative executions, exact repeat verification, an
independent results-and-claims review, release validation, commit, CI, and tag
remain mandatory. This review advances none of those gates.

## Resolved adversarial findings

The following issues were blocking or materially ambiguous in earlier plan
bytes and are resolved in the reviewed digest above.

### P1-1: candidate and package arithmetic

The candidate is now unambiguous: 34 files total, with
`candidate-index.json` indexing the other 33. The final package is also
unambiguous: 14 strict-artifact files, 26 sidecars indexed by
`release-sidecar-index.json`, and that self-excluded index, for 41 files total.
No stale 29-member or 27-indexed-sidecar count remains.

### P1-2: candidate-to-final authority

Final construction no longer treats reviewed candidate bytes as a sufficient
scientific source. It must reopen the original two local artifacts, both raw
resource logs, and the software-verification attestation; re-curate and
regenerate every scientific member, claim projection, figure, template,
evidence record, and candidate index; and require byte identity with the
reviewed candidate before building the final artifact. The candidate is an
immutable review target, not a substitute for the original authority.

### P1-3: timing-log overwrite and process scope

The original direct `time -o` shape could truncate an existing path. The
frozen `run-replay` wrapper now reserves the log descriptor-relative with
exclusive, no-follow creation and mode `0600`, passes the owned descriptor to
Darwin `/usr/bin/time -l`, measures the actual `ffb replay run` process after
environment preparation, separates child stderr from the strict resource
block, propagates failure, and never deletes or overwrites the attempt.

This remains consistent with the resource-scope amendment: there is one
scientific replay worker, while the sequential wrapper and timing process are
measurement helpers rather than additional scientific workers.

### P1-4: retry semantics

The lifecycle now permits exactly one primary `r1` followed by exactly one
repeat `r1` at a scientific revision; no caller-selected ordinal remains.
Revision-specific software verification must exist before primary execution.
Each run receives an owner-only success receipt only after zero exit, a
complete timing log, and unchanged postflight authority. Candidate and final
construction reauthenticate both receipts and reject a partial, sibling,
out-of-order, retried, or non-`r1` attempt.

A failed outcome-bearing run therefore cannot be replaced at the same
revision. Any authoritative failure blocks that revision. Resumption requires
a public protocol or incident amendment, a new committed scientific revision,
and fresh primary and repeat `r1` runs. This agrees with the frozen no-retry
and no-selection rules.

### P1-5: source-revision and implementation-review authority

The implementation review now binds a domain-separated content snapshot rather
than a self-referential Git commit. Its exact path/byte table closes over code,
tools, tests, configuration, CI, the replay intent, the controlling M3 matrix
and its exact manifests/profiles, the complete authenticated M4 release
allowlist, and the controlling public methodology.

The implementation-review report and canonical attestation are excluded from
that content snapshot, allowing both to be committed before execution without
self-reference. A reviewer-authored decision input, rather than the builder,
controls disposition and finding counts. Candidate preparation recomputes the
snapshot and requires zero unresolved P0/P1 findings.

Outcome-updated public README material is correctly outside the implementation
snapshot and is instead checked by the publication projection validator.

Execution authority now additionally requires the process working directory
to equal the authenticated real source root, binds stable inode/metadata and
SHA-256 fingerprints for the exact locked `ffb` and Darwin `/usr/bin/time`
executables, and binds the revision-specific software-verification digest.
Every dataset/cache directory component is opened without following links, so
an intermediate redirect cannot substitute a runtime authority. Each preflight
and postflight requires local `HEAD`, its remote-tracking ref, and one exact
live branch response to agree. The live query is bounded and noninteractive,
uses SSH batch mode and strict known-host checking when applicable, and fails
without reflecting remote URLs or credentials.

The frozen workflow also creates the previously absent revision-specific UV
cache exactly once with mode `0700` before any authoritative `uv run`; an
existing, redirected, nonprivate, overlapping, or replacement cache fails
closed.

### P1-6: final-build clean-source semantics

Ordinary replay and candidate operations still require the exact clean
checkout. Final construction has a narrowly defined exception for only its
owned staging or release destination. It verifies all tracked bytes and index
entries against `HEAD`, rechecks the implementation snapshot, and permits no
other untracked non-ignored path. This avoids the contradiction in requiring a
non-ignored release output while simultaneously rejecting every untracked
path.

### P1-7: publication transaction durability

Candidate and release publication now fsync every file and directory, perform
the no-replace rename, and fsync the destination parent after the rename.
Post-rename fsync failure is classified as indeterminate durability; the
complete destination is left untouched and is not declared published.
Cleanup is limited to the process-owned staging tree.

### P1-8: reviewed-evidence synchronization

Package validation and repository publication validation are now separate.
The package-only validator is self-contained. A no-overwrite
`sync-reviewed-evidence` step copies the exact reviewed report and attestation
from the package to their fixed public documentation paths before the
publication validator requires equality. This removes the earlier
build-then-validate ordering contradiction.

The pair synchronization is now crash recoverable. A fixed owner-only staging
directory retains exact package-bound members across an interruption; an exact
staged or published partial pair resumes by publishing only the missing
member, and an exact complete pair is accepted without rewriting it. Any
mismatched, extra, linked, non-regular, or unsafe staged or public member fails
closed.

Repository closeout is no longer token-only. Each of the eight public
documents must equal its scientific-revision blob with exactly one frozen
placeholder replaced by the same deterministic package/claim digest and
fixed-order H5 result/role projection. Publication validation accepts only the
exact fingerprinted pre-stage pending state or a clean descendant state. In
both states it revalidates one unchanged package digest and requires the
packaged implementation-review bytes to equal current tracked authority and
validate against the current implementation snapshot.

### P1-9: methodology privacy exceptions

The candidate includes exact frozen methodology and review documents, so a
blanket ban on every path-looking string or authored date would reject its own
required evidence. The final rule is role-aware: generated machine,
presentation, specification, and figure members remain free of runtime paths
and timestamps; hashed methodology may retain authored dates, fixed system
command paths, and abstract angle-bracket placeholders.

Realized machine-local roots, dataset paths, private cache paths, dataset
filenames, tokens, and path variants remain forbidden, with explicit negative
tests. The exception therefore enables reproducible methodology without
weakening the dataset/privacy boundary.

### P2-1: explicit offline proof

Offline validation is no longer only a prose claim. The plan requires an
already provisioned environment, dataset-root removal, network-disabled
execution, frozen/no-sync package and publication validation, and an isolated
installed-wheel CLI check. Dependency installation is explicitly outside the
offline proof.

## Scientific-selector audit

The proposed public projection matches the frozen code-level coordinate
contracts.

### M5-A

The contract contains exactly 33 M5-A hypothesis coordinates and the plan
retains all of them:

- 16 directional H5-A1 through H5-A4 coordinates;
- four H5-A5 full-dropout coordinates; and
- 13 H5-A6 common-mode coordinates.

The 100-row persistent-panel figure is fixed before outcomes:

- 54 ordinary-fault fused-minus-healthy rows;
- six dropout selectors times four declared availability/loss rows, or 24;
  and
- 11 common-mode selectors times two declared absolute-loss/disagreement
  rows, or 22.

The dropout projection includes all six probabilities, the four full-dropout
hypothesis coordinates, and the mechanically derived nesting evidence. It can
therefore report zero fixed-fusion coverage, undefined conditional loss, and
the target-drop versus LiDAR-only coverage comparison without selecting a
favorable row.

The common-mode projection includes the identity absolute-loss baseline,
absolute-loss rows at both signed maximum endpoints, and disagreement at all
11 severities. It therefore supplies the fixed comparisons required for
H5-A6 rather than choosing a baseline after outcomes.

All ten predeclared crossover records remain separate by physical axis,
direction, and unit. Dropout and common-mode controls do not acquire an
unauthorized crossover.

### M5-B

The plan retains one combined-health-gate event-window policy-gain row for
each of all 43 frozen health selectors. The 11 hypotheses remain explicitly
marked within that complete set:

- seven H5-B1 positive-transport coordinates;
- one H5-B2 negative counterexample;
- two H5-B3 nonpositive controls; and
- one H5-B4 common-mode diagnostic without a healthy-target interpretation.

The remaining 32 rows stay visible as fixed context. There is no top-\(k\),
magnitude ordering, favorable family selection, or healthy-target
reinterpretation.

### Sensitivity and descriptors

The cluster figure retains all 26 aggregate sources that require sensitivity:
16 M5-A directional sources, eight M5-B directional sources, and two M5-B
nonpositive controls. For committed log-group count \(G\), it retains every
`26 * (10 + G)` leave-one-scene and leave-one-log-group record, including
undefined leave-outs.

The descriptor figure arithmetic is also exact: 42 paired replay/M3 rows,
24 replay-only rows, and one distinct-log-group row, for 67 fixed rows. It is
descriptive and makes no cross-milestone inferential claim.

## Seventeen-check authority audit

The plan preserves the exact order of
`M5_RELEASE_VALIDATION_CHECK_IDS`. Each check now names evidence that the
builder must derive and the standalone validator must independently
reconstruct:

| Check | Review conclusion |
|---|---|
| `intent-freeze` | Binds frozen intent plus exact plan, plan-review, and resource-amendment bytes. |
| `fixed-scene-population` | Binds literal profile counts and frozen identity/selector commitments. |
| `base-support` | Binds the profile support attestation and authenticated support/descriptor commitments. |
| `health-schedules` | Binds the schedule attestation and complete health coordinate commitments. |
| `transform-and-timing-oracles` | Binds named software-oracle entries and the implementation-review attestation. |
| `eligibility-and-fault-causality` | Binds named support, pairing, calibration, timestamp, mutation, and dropout-nesting checks. |
| `health-feature-leakage` | Binds pre-update, future-mutation, and prohibited-feature tests. |
| `persistent-panel-completeness` | Binds exact aggregate bytes, 71 selectors, 464 coordinates, and all 33 M5-A coordinates. |
| `health-panel-completeness` | Binds exact aggregate bytes, 43 selectors, 14,988 coordinates, and all 11 M5-B coordinates. |
| `scene-bootstrap-and-cluster-sensitivity` | Binds bootstrap fields and complete LOSO/LOLO bytes. |
| `repeat-scientific-members` | Binds repeat verification and all eight ordered source-member commitments. |
| `cpu-and-memory-caps` | Binds the ordered primary/repeat external resource records. |
| `no-raw-payload-reads` | Binds zero-read profile state, local read accounting, and privacy evidence. |
| `privacy-and-dataset-license` | Binds role-aware scans, terms, attribution/non-endorsement, and the privacy/license attestation. |
| `implementation-review` | Binds the exact whole-snapshot review report and canonical attestation. |
| `results-and-claims-review` | Remains pending until a reviewer-authored attestation binds the exact candidate. |
| `software-verification` | Binds the canonical verification attestation for the clean scientific revision. |

`validation-inputs.json` cannot claim the final conjunction while the results
review is pending. Final construction inserts that authority, derives all 17
check-specific digests, and only then creates `ReplayValidationV1`. The public
scope correctly distinguishes content consistency and operator attestation
from dataset authentication, syscall proof, or cryptographically proven
independent execution.

## Digest and byte-authority audit

The candidate digest has no self-reference: its canonical core indexes the
other 33 members and excludes only `candidate_sha256`. The review attestation
then binds both that semantic digest and the ordinary hash of the complete
index bytes.

The sidecar digest is similarly acyclic: its core excludes the two derived
digest fields, and the sidecar index is excluded from its own entry list. The
package digest binds the strict machine-artifact digest and reconstructed
sidecar-set digest. The strict success marker separately closes the machine
artifact/run relationship.

The four Markdown substitutions are downstream-only identities: machine
artifact digest, run digest, results-review-attestation digest, and machine
artifact byte length. None is an input to the object whose identity it
records, so no substitution creates a hash cycle. Scientific, claim, spec,
SVG, and non-placeholder presentation bytes remain candidate-identical.

## Figure-source bridge audit

The plan correctly rejects using one `ReplayFigureRecordV1` per public figure:
that existing type carries one source and requires a unique figure ID, so it
cannot faithfully bind five multi-source figures.

`ReplayFigureSourceBindingV1` removes that false carrier abstraction. Its
unique key is `(figure_id, mark_ordinal)`; every plotted source has one ordered
binding to the exact source record, canonical spec, rendered SVG path, SVG
hash, and SVG length. Descriptor sources need no invented replay identity.
Non-descriptor identity coverage is derived through authenticated source
records and must equal the exact 22-identity replay set.

The dependency direction is acyclic:

1. aggregate records determine fixed spec marks;
2. spec plus aggregate bytes deterministically render the SVG;
3. source-binding records commit source, spec, and SVG bytes; and
4. the outer sidecar index commits the actual spec/SVG files.

The standalone 14-file machine artifact can therefore validate aggregate and
binding integrity without requiring sidecar files, while the 41-file package
validator closes the rendered-byte link.

## Software, privacy, and transaction conclusions

The software-verification attestation is produced before replay, without
overwrite, at the same clean scientific revision. It freezes named format,
lint, type, test, oracle, build, wheel-smoke, and privacy checks while
excluding volatile durations and local paths. Candidate and final builders
reload it rather than accepting a Boolean assertion.

The release remains CPU-only and metadata-only. The plan introduces no Torch,
CUDA, detector, raw-sensor generation, or learned-method work. Resource
records remain operator-recorded self-reports, and the report must not elevate
them to independent execution proof.

Privacy scans cover candidate, package, public copies, tracked/staged files,
filenames, process output, and exception text. Public scene identifiers remain
only the already frozen exception. Raw timing logs, local sequence rows,
tokens, timestamps, poses, calibration, coordinates, metadata tables,
credentials, and private interview material remain excluded.

Every public aggregate and presentation surface carries the dataset terms,
attribution, and non-endorsement boundary. The repository license does not
relicense the replay evidence. Claims remain limited to matched-center proxy
loss, persistence or non-persistence across the fixed mini population,
apply-only M4 transport, finite cluster sensitivity, and measured CPU
resources.

No-overwrite, no-follow, hard-link rejection, stable-tree reload, staged
publication, post-rename durability, bounded cleanup, and separate package
versus publication validation collectively provide a coherent failure-closed
release transaction.

## 2026-07-30 outcome-blind amendment review

This amendment independently reviewed the current plan bytes and only the M5
production and synthetic-test changes needed to implement the repaired
authority boundaries. No nuScenes data, generated replay output, descriptor,
persistence, crossover, health-transfer, resource result, or outcome-bearing
review artifact was inspected.

The review confirmed:

- software verification is revision-specific, canonical, and mandatory before
  either replay;
- exact executable content/metadata fingerprints, a real source-root working
  directory, no-follow dataset/cache components, and bounded noninteractive
  live-upstream equality are bound into replay authority;
- the only lifecycle is primary `r1` then repeat `r1`, with exclusive success
  receipts required again by candidate and release construction;
- the UV cache is created owner-only before the first authoritative `uv run`;
- reviewed-evidence synchronization resumes exact staged or published partial
  pairs without overwriting a valid member;
- all eight closeout documents are deterministic projections of the validated
  reviewed package, with no independent quantitative claim surface;
- pending publication authority fixes both Git path sets and stable file
  contents, while clean authority requires a descendant of the scientific
  revision; and
- both publication states bind the package's implementation review to current
  tracked report, attestation, and implementation-snapshot bytes.

The outcome-blind M5 test set passed on these bytes. No P0, P1, or P2 amendment
blocker remains. The amendment disposition is **PASS** for the pre-execution
release-pipeline plan. It does not replace the required whole-revision
implementation review, software-verification attestation, authoritative
executions, results-and-claims review, release validation, CI, or tag gates.

## Final verdict

No P0, P1, or P2 release-plan blocker remains in the reviewed plan bytes. The
design may proceed to its whole-revision implementation review and remaining
pre-execution gates.

This is not permission to run the dataset from an uncommitted or unreviewed
implementation. Before authoritative execution, the complete release tooling,
tests, CI path, plan, and this review must be committed; the whole
implementation snapshot must receive an independent adversarial review and
canonical attestation; software verification must pass; and the clean
revision must be pushed. M5 outcomes and release claims remain entirely
unreviewed.
