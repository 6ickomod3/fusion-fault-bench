# M5 release-pipeline pre-registration

Status: **frozen before M5 replay descriptor, fault-outcome, persistence,
crossover, health-transfer, or resource-result inspection**.

Date: 2026-07-29.

This document freezes the evidence-publication workflow for M5. It does not
change the scientific intent in
[`m5-replay-plan.md`](m5-replay-plan.md), the machine-readable replay intent,
the outcome-blind
[`m5-resource-scope-amendment.md`](m5-resource-scope-amendment.md), any M3
source manifest, or the released M4 fit. No M5 outcome was inspected to choose
the files, selectors, figures, formatting rules, review boundary, or
acceptance gates below.

This is a release-process plan, not a release verdict. The results review,
validation records, release package, release commit, tag, and CI result do not
yet exist.

## 1. Purpose and non-goals

The current replay runner can produce two ignored local artifacts, verify
their exact scientific-member equality, import two external resource logs, and
reconstruct curated aggregate objects. A public release additionally needs an
immutable object for independent outcome review and a non-circular path from
that review to a complete tracked package.

The pipeline has exactly two publication phases:

1. build one ignored, immutable **review candidate** containing the complete
   scientific aggregate bytes, fixed claim projection, deterministic figures,
   presentation templates, and every pre-review validation input; then
2. after an independent review of that exact candidate, build one
   no-overwrite **final release package** containing the strict machine
   artifact and all review, figure, claim, verification, and package-index
   sidecars.

This plan does not permit:

- changing an estimand, threshold, severity, seed, method, policy, population,
  hypothesis, support rule, or inference rule;
- choosing figures, claims, precision, ordering, or review scope from outcome
  magnitude or favorability;
- publishing local sequence rows, raw timing logs, dataset metadata, tokens,
  filenames, paths, poses, calibration, timestamps, or coordinates;
- treating the independent review as cryptographic execution proof; or
- calling a candidate, a failed review, or a machine artifact without its
  sidecars an M5 release.

## 2. Source-revision and execution authority

The release-pipeline implementation, its synthetic tests, this plan, and the
frozen scientific intent must be committed before the authoritative primary
and repeat runs.

After that implementation is complete, an independent adversarial
implementation review must cover the runner, release builder, validators,
claim projections, figures, privacy boundary, and failure/rollback behavior at
the exact implementation content snapshot. Its resolved public report and
canonical attestation are committed before either run and are copied to
`evidence/implementation-review.md` and
`evidence/implementation-review-attestation.json`. The earlier
runner-checkpoint review remains useful evidence but does not substitute for
this whole-revision review. Any blocker fix changes the implementation
snapshot and requires the affected review checks to be repeated.

The implementation snapshot is a domain-separated digest over an exact
canonical path/byte table containing:

- every tracked path under `src/`, `tools/`, and `tests/`;
- `examples/replay/m5-nuscenes-mini-replay-v1.json`;
- `examples/matrices/m3-procedural-v1.json` plus exactly the eight manifest
  and three profile paths/digests referenced by that canonical matrix;
- the complete exact M4 `HEALTH_RELEASE_ARTIFACT_PATHS` allowlist under
  `reports/releases/m4-health-v0.1.0/`, including its authenticated index and
  success marker;
- `pyproject.toml`, `.python-version`, `uv.lock`, `LICENSE`,
  `DATA_AND_MODEL_TERMS.md`, and `.github/workflows/ci.yml`; and
- the benchmark contract, M5 replay plan and review, resource amendment, this
  release plan, and its plan review.

Expansion uses the strict M3 matrix and M4 release loaders and rejects a
missing, extra, redirected, or digest-mismatched controlling input. The
snapshot excludes only the implementation review report and attestation
themselves, later outcome documents, and release outputs, avoiding a review
self-reference. A repository command
`attest-implementation-review` accepts the reviewer-authored report and a
small ignored reviewer decision input, but it only canonicalizes:

- implementation-snapshot SHA-256;
- exact report-byte SHA-256;
- reviewer identity scope
  `operator-recorded-not-cryptographically-authenticated`;
- reviewed areas and finding identifiers;
- P0, P1, and P2 counts plus unresolved finding identifiers; and
- the reviewer-authored disposition.

It writes the public attestation without overwrite and never chooses or
rewrites the disposition. Candidate preparation reloads the exact committed
report and attestation, recomputes the implementation snapshot and report
digest, and requires a release-permitting disposition with zero unresolved P0
or P1 findings.

Both local runs, candidate preparation, and final machine-artifact
construction must authenticate the same:

- clean Git revision;
- tracked replay intent bytes and canonical digest;
- lockfile digest and locked environment;
- package version;
- named Darwin CPU environment;
- exact logical command
  `ffb replay run --output-dir <safe reports/generated path>`; and
- unchanged source snapshot before and after each authoritative operation.

Here, a clean scientific source means that every tracked file and index entry
equals `HEAD`, the implementation-snapshot digest is unchanged, and no
untracked path exists except ignored local evidence plus the single
operation-owned staging/destination path. Candidate preparation requires the
ordinary clean state because its destination is ignored. Final construction
checks the same state before staging and immediately before publication; after
the no-replace rename it permits only the newly created exact release
destination in addition to ignored local evidence. No tracked source or index
change is permitted. Later public documentation/review copies are a separate
closeout phase and cannot alter the authenticated implementation snapshot.

Any change to code, contracts, tests, lockfile, intent, this release plan, or
release tooling after either run invalidates both runs for release. Both runs
must then be repeated from the new clean revision. Outcome documents and the
review report are instead created under ignored `reports/generated/` after
the runs; they may not modify the authenticated scientific source while
candidate preparation or final machine construction is in progress.

The final release commit may differ from the scientific run revision because
it adds only the already reviewed release package and public documentation.
The strict artifact retains the exact scientific revision in `run.json`.

## 3. Authoritative local inputs

The candidate builder accepts exactly five local inputs:

1. one complete primary local replay artifact;
2. one complete repeat local replay artifact;
3. the primary replay-process Darwin `/usr/bin/time -l` log;
4. the repeat replay-process Darwin `/usr/bin/time -l` log; and
5. one canonical software-verification attestation produced without overwrite
   at the same clean scientific revision before either replay.

The builder freshly reloads both artifacts, verifies current-source authority,
requires the exact eight ordered scientific-member commitments, requires zero
mismatches and distinct paths, inodes, and run records, and imports the two
resource logs through the strict parser. It never accepts hashes copied from
CLI stdout as authority.

The logs must be distinct, bounded, private regular files without symbolic or
hard links. They remain ignored and local. Only their SHA-256, byte length,
parsed elapsed time and peak RSS, and exact artifact/run/environment/command
bindings enter the candidate.

The software-verification attestation names the exact format, lint, type,
unit/property/oracle/integration, build, wheel-smoke, and privacy checks;
records their stable command and required-test identifiers, exit status, and
output digest; and binds the clean revision, lockfile, package version, and
tooling revision. It excludes wall-clock durations, temporary paths, current
time, and other nondeterministic text. Candidate preparation strictly reloads
the attestation and reruns its cheap structural and source-binding checks; it
does not accept a caller-supplied Boolean verdict.

## 4. Exact review-candidate tree

The candidate root is a no-overwrite directory under
`reports/generated/m5-review-candidate-<scientific-revision>/`. It contains
exactly the following 34 files in this canonical path order:

1. `candidate-index.json`;
2. `machine/intent.json`;
3. `machine/replay-profile-summary.json`;
4. `machine/descriptor-aggregates.ndjson`;
5. `machine/persistent-panel-aggregates.ndjson`;
6. `machine/persistent-panel-crossovers.ndjson`;
7. `machine/health-panel-aggregates.ndjson`;
8. `machine/leave-one-cluster-sensitivity.ndjson`;
9. `machine/repeat-verification.json`;
10. `machine/figure-records.ndjson`;
11. `machine/source-member-commitments.ndjson`;
12. `evidence/release-pipeline-plan.md`;
13. `evidence/release-pipeline-plan-review.md`;
14. `evidence/resource-scope-amendment.md`;
15. `evidence/implementation-review.md`;
16. `evidence/validation-inputs.json`;
17. `evidence/implementation-review-attestation.json`;
18. `evidence/software-verification.json`;
19. `evidence/privacy-license-attestation.json`;
20. `figures/m5-persistent-panel-summary.spec.json`;
21. `figures/m5-persistent-panel-summary.svg`;
22. `figures/m5-crossovers.spec.json`;
23. `figures/m5-crossovers.svg`;
24. `figures/m5-health-transfer.spec.json`;
25. `figures/m5-health-transfer.svg`;
26. `figures/m5-descriptor-comparison.spec.json`;
27. `figures/m5-descriptor-comparison.svg`;
28. `figures/m5-cluster-sensitivity.spec.json`;
29. `figures/m5-cluster-sensitivity.svg`;
30. `presentation/README.md`;
31. `presentation/claim-evidence.md`;
32. `presentation/verification.md`;
33. `presentation/release-summary.json`; and
34. `presentation/public-claim-projections.json`.

No other file or directory is allowed. In particular, the candidate excludes
`validation.json`, `release-index.json`, final `run.json`, `_SUCCESS`, a
results-review report, a results-review attestation, raw resource logs, and
local source rows.

The ten `machine/` payload members after the candidate index are the exact
bytes later supplied to the strict artifact writer. They may be reordered only
by their already frozen canonical record order. Candidate and final bytes for
these members must be identical.

The three Markdown presentation files use only these literal identity
placeholders:

- `@M5_RELEASE_ARTIFACT_SHA256@`;
- `@M5_RELEASE_RUN_SHA256@`;
- `@M5_RESULTS_REVIEW_ATTESTATION_SHA256@`; and
- `@M5_MACHINE_ARTIFACT_BYTES@`.

No outcome text may change between candidate review and final publication.
Final construction replaces only those four placeholders, once each at every
declared location, with lower-case canonical values. All other presentation
bytes must remain identical.

## 5. Candidate index and digest boundary

All JSON uses the repository canonical JSON contract: UTF-8, sorted keys,
compact separators, finite numbers, positive zero normalization, and one final
LF. NDJSON uses one such record per line. Generated presentation Markdown,
SVG, and JSON-spec files are UTF-8 with LF endings and contain no runtime
timestamp, absolute path, random identifier, software-generated comment, or
host-dependent metadata.

Exact frozen methodology/review Markdown copies may retain their authored
preregistration/review dates and the literal system command paths
`/usr/bin/time` and `/dev/fd/<fd>`. Angle-bracket path placeholders are also
permitted, but a realized machine-local or dataset path is not. Candidate
privacy scanning is therefore role-aware: it permits only those exact frozen
literals in hashed methodology evidence while keeping generated claims,
templates, specs, SVGs, and machine records path-free. Negative tests must
still reject macOS home paths, a realized `/Users/<name>/.../nuScenes` root,
private temporary/cache paths, dataset filenames, tokens, and path-like
variants in every member.

`candidate-index.json` has schema
`ffb.m5-release-review-candidate-index/v1` and contains:

- release ID `m5-nuscenes-replay-v0.1.0`;
- scientific Git revision, lockfile SHA-256, package version, run ID, replay
  intent SHA-256, and replay identity-set SHA-256;
- primary and repeat local artifact and run SHA-256 values;
- a literal `results_review_status: "pending"`;
- the exact ordered list of the other 33 files; and
- `candidate_sha256`.

Each file entry contains exact `path`, `role`, `byte_length`, and `sha256`.
NDJSON entries additionally contain exact `record_count`. The allowed roles
are:

- `reviewed-scientific-aggregate`;
- `reviewed-repeat-or-provenance`;
- `pre-review-validation-input`;
- `frozen-public-methodology`;
- `independent-review-evidence`;
- `deterministic-figure-spec`;
- `deterministic-rendered-figure`;
- `reviewed-presentation-template`; and
- `reviewed-claim-projection`.

Let `candidate_core` be the complete candidate-index mapping without
`candidate_sha256`. The candidate digest is:

```text
SHA256(
  b"fusion-fault-bench/m5-review-candidate/v1\x00"
  || uint64_be(len(canonical_json_bytes(candidate_core)))
  || canonical_json_bytes(candidate_core)
)
```

The stored `candidate_sha256` must equal that value. The later review
attestation binds both this semantic candidate digest and the ordinary SHA-256
of the exact `candidate-index.json` bytes. The index itself is not a member of
its own file list, so no self-reference exists.

Every candidate load reopens and rehashes all 33 indexed members, validates exact
allowlists, types, counts, canonical bytes, regular-file properties, tree
stability, internal record links, and the candidate digest. Review may begin
only after this reload passes.

## 6. Validation-evidence authority

The final `validation.json` retains the 17 checks and exact order already
declared by `M5_RELEASE_VALIDATION_CHECK_IDS`. A valid-looking arbitrary digest
is never sufficient. The release builder and full release validator derive
each `evidence_sha256` with a check-specific domain from the following exact
authority:

| Check | Authoritative evidence |
|---|---|
| `intent-freeze` | frozen intent bytes and canonical digest plus the exact candidate copies of the release plan, plan review, and outcome-blind resource amendment |
| `fixed-scene-population` | profile scene and experiment counts plus the frozen identity and selector-set commitments |
| `base-support` | profile base-support attestation and exact descriptor/support commitments from the authenticated local artifacts |
| `health-schedules` | profile schedule attestation and complete frozen health selector/coordinate commitments |
| `transform-and-timing-oracles` | named oracle-test entries in `software-verification.json` and the implementation-review attestation |
| `eligibility-and-fault-causality` | named support, pairing, calibration, timestamp, fault-mutation, and mechanically derived dropout-nesting entries in `software-verification.json` |
| `health-feature-leakage` | named pre-update, future-mutation, and prohibited-feature tests in `software-verification.json` |
| `persistent-panel-completeness` | exact persistent aggregate bytes, 71-selector commitment, 464-coordinate commitment, and all 33 M5-A claim coordinates |
| `health-panel-completeness` | exact health aggregate bytes, 43-selector commitment, 14,988-coordinate commitment, and all 11 M5-B claim coordinates |
| `scene-bootstrap-and-cluster-sensitivity` | exact aggregate bootstrap fields and the complete leave-one-scene/log-group sensitivity bytes |
| `repeat-scientific-members` | repeat-verification bytes and all eight exact ordered source-member commitments |
| `cpu-and-memory-caps` | canonical ordered primary/repeat external resource-evidence records |
| `no-raw-payload-reads` | profile zero-read field, local read-accounting attestation, and privacy attestation |
| `privacy-and-dataset-license` | deterministic candidate scan, fixed aggregate terms, attribution/non-endorsement fields, and `privacy-license-attestation.json` |
| `implementation-review` | exact candidate copy of the tracked implementation-review bytes and `implementation-review-attestation.json` |
| `results-and-claims-review` | exact canonical results-review attestation bytes created after candidate review |
| `software-verification` | exact `software-verification.json` bytes for the clean scientific revision |

`evidence/validation-inputs.json` contains the first 16 derived input records
except `results-and-claims-review`, in final check order with that one position
explicitly marked pending. It may not assert `all_checks_passed`. Final
construction inserts the reviewed attestation, derives all 17 digests again,
and creates `ReplayValidationV1`. The standalone full release validator
independently repeats every derivation.

Where the public package retains only an operator attestation rather than the
underlying local rows, the evidence scope says so. Digest consistency is not
described as independent dataset authentication, execution proof, or syscall
attestation.

## 7. Predeclared public claim registry

`presentation/public-claim-projections.json` is the only registry from which
numeric public M5 prose, tables, captions, and figure annotations may be
generated. It is selected before outcomes and contains:

1. all 33 exact M5-A hypothesis coordinates, count `33`, set SHA-256
   `22c9e55602c9a8faefa6fea7e0f4c1fe8185f8b4160729072b091801ede36ab3`;
2. all 11 exact M5-B hypothesis coordinates, count `11`, set SHA-256
   `6d73453c8cef65e90bc6d4cb1fe972bce976a43a924046dcaafdbdc57b7f7cb8`;
3. the complete fixed 100-row M5-A figure projection in Section 8.1, with the
   33 hypothesis coordinates marked by membership in item 1;
4. all ten predeclared physical-axis crossover records in frozen identity,
   direction, and axis order;
5. the fixed common-mode healthy-selector absolute-loss baseline within that
   100-row projection, needed to
   interpret H5-A6 without inferring a baseline after observing outcomes;
6. the mechanically derived dropout-nesting evidence needed to interpret
   H5-A5;
7. all 43 fixed M5-B
   `combined-health-gate / policy-gain-vs-fixed / event / m^2 /
   equal-scene-mean` rows, with the 11 hypothesis rows explicitly marked and
   the other 32 retained as predeclared context;
8. all 26 primary/control aggregate sources and every corresponding
   `26 * (10 + G)` leave-one-scene/log-group sensitivity row in Section 8.5;
9. both resource records and their exact elapsed/RSS maxima;
10. repeat mismatch count, scientific-member count, scene count, experiment
   counts, and distinct log-group count; and
11. the following fixed descriptor comparison selectors:
   - `sample-count`;
   - `eligible-object-frame-count`;
   - `eligible-track-length-q50`;
   - `ego-range-q50`;
   - `ego-bearing-q50`;
   - `finite-difference-speed-q50`;
   - `support-all-annotations`;
   - `support-roi-pass`;
   - `support-camera-center-pass`;
   - `support-lidar-points-positive`;
   - `support-final-eligible`;
   - `unique-eligible-track-count`;
   - `reference-time-delta-q50`;
   - `camera-minus-lidar-acquisition-offset-q50`;
   - `zero-order-hold-velocity-fraction`; and
   - `distinct-log-group-count`.

The seven shared descriptor IDs `sample-count`,
`eligible-object-frame-count`, `eligible-track-length-q50`, `ego-range-q50`,
`ego-bearing-q50`, `finite-difference-speed-q50`, and
`reference-time-delta-q50` include minimum, median, and maximum for both
`nuscenes-mini-replay` and `m3-main-test-comparator`, giving 42 rows. The five
support-waterfall IDs, `unique-eligible-track-count`,
`zero-order-hold-velocity-fraction`, and
`camera-minus-lidar-acquisition-offset-q50` use those three statistics for
replay only, giving 24 rows. `distinct-log-group-count` contributes its single
replay count row. The descriptor figure therefore has exactly 67 fixed rows.
No category is chosen after observing its frequency. Complete categorical and
other descriptor rows remain in the machine artifact and claim ledger even
when not plotted.

The registry also predeclares a finalization-metadata selector for strict
machine-artifact byte length. Its candidate value is literal `null`, its
source is fixed to final `artifact/release-index.json`, and it may populate
only `@M5_MACHINE_ARTIFACT_BYTES@` after the reviewed scientific and
presentation bytes are frozen. It is verification metadata, not a scientific
outcome or an outcome-selectable claim.

Each registry entry names exact source member, selector fields, permitted
projected fields, unit, status behavior, figure mapping, and stable public
claim ID. The claim ledger renders machine numeric tokens exactly. Overview
and figure finite floats use Python's outcome-independent `.6g` formatter;
elapsed time uses `.2f`; byte and count values remain exact integers.
Undefined, not-applicable, censored, and positive-infinity states use literal
labels rather than invented numeric values.

There is no magnitude ranking, top-\(k\), favorable-family selection, or
omission of a failed hypothesis. The global M3-mechanism wording follows the
already frozen rule: it is global only if every H5-A1 through H5-A4 check is
robustly persistent; otherwise the wording is partial or non-persistence.
H5-B1 through H5-B4 are always reported together. H5-B3 remains a
nonpositive control, and H5-B4 never gains a healthy-target interpretation.

No quantitative sentence outside this registry is allowed in the release
overview, claim ledger, figure text, README, results page, benchmark card, or
resume evidence.

## 8. Deterministic five-figure contract

Exactly five SVG figures are generated, in the fixed order below. Plot and
legend order follows frozen selector order, never estimate order.

### 8.1 Persistent panel summary

`m5-persistent-panel-summary.svg` contains exactly 100 fixed source rows:

- all 54 ordinary-fault selectors using
  `fixed-fusion / fused-minus-healthy / full / m^2 / equal-scene-mean`;
- all six dropout selectors for fixed-fusion coverage, fixed-fusion
  conditional matched-center MSE, fault-target-drop coverage, and LiDAR-only
  coverage, giving 24 rows; and
- all 11 common-mode selectors for fixed-fusion absolute matched-center MSE
  and camera-LiDAR disagreement MSE, giving 22 rows.

These rows contain all 33 M5-A claim coordinates and the fixed auxiliary
context needed for H5-A5 and H5-A6. Defined inference rows show point and
pointwise interval plus the frozen persistence label and ten-scene sign
partition. Undefined values remain visibly undefined. No source row is chosen
from its sign, interval, status, persistence label, or magnitude.

### 8.2 Crossovers

`m5-crossovers.svg` contains all ten predeclared crossover axes and directions.
Meters, radians, seconds, and standard-deviation scale remain separate facets.
Observed, not-observed, and undetermined records retain their exact censoring
semantics; no mixed-unit ordering or synthetic combined severity is allowed.

### 8.3 Health transfer

`m5-health-transfer.svg` contains exactly one fixed combined-health-gate event
gain row for each of the 43 health selectors, faceted in the frozen
14-experiment order. It visually distinguishes, but does not exclusively
select, the exact 11 M5-B claim coordinates in H5-B1, H5-B2, H5-B3, H5-B4
order. It shows pointwise intervals, status, and persistence/control
interpretation without sorting by gain. The common-mode row is labeled as
lacking a uniquely faulty target.

### 8.4 Descriptor comparison

`m5-descriptor-comparison.svg` contains exactly the 67 descriptor rows in
Section 7. Shared replay/M3 descriptors are paired descriptively; replay-only
quantities are explicitly marked unavailable in the M3 comparator. It makes
no inferential cross-milestone difference claim.

### 8.5 Cluster sensitivity

`m5-cluster-sensitivity.svg` contains the exact 26 aggregate sources whose
role is `primary-directional` or `nonpositive-control`—16 M5-A, eight M5-B
directional, and two M5-B controls—and every corresponding
leave-one-scene-out and leave-one-log-group-out record. With `G` committed
distinct log groups, it therefore contains all `26 * (10 + G)` sensitivity
rows. It includes undefined leave-outs and uses opaque scene/log-group
ordinals only.

Each `.spec.json` has schema `ffb.m5-figure-spec/v1` and declares:

- fixed figure file, kind, dimensions, fonts, colors, units, axis facets, and
  caption boundary;
- every ordered mark with exact source member, result/descriptor/crossover
  identifier, source-record SHA-256, projected fields, and stable public
  projection ID;
- a fixed renderer/version identifier; and
- the aggregate terms and non-endorsement footer.

The existing `ReplayFigureRecordV1` is not used for this release because its
one-source/unique-figure-ID shape cannot represent five multi-source figures
or authenticate rendered bytes. Before authoritative execution, the release
tooling introduces `ReplayFigureSourceBindingV1` for the existing
`figure-records.ndjson` path. Each ordered row binds:

- one of the exact five public `figure_id` values and its fixed kind;
- a zero-based mark ordinal unique within that figure;
- a discriminated source kind, stable source identifier, source-record
  SHA-256, and optional replay identity binding;
- the exact canonical spec SHA-256; and
- the exact rendered SVG path, SHA-256, and byte length.

The unique key is `(figure_id, mark_ordinal)`, not `figure_id` alone. Exactly
one binding exists for every plotted source, in frozen figure/mark order.
Descriptor bindings have no synthetic experiment identity. The union of
non-descriptor bindings must cover the exact 22 replay identities, and the
validator derives that coverage from source records rather than a carrier
identity. The five spec/SVG pairs remain outer release sidecars so the strict
machine artifact stays aggregate-only; the binding hashes and sidecar index
close the machine-to-rendered-figure link.

The SVG renderer is deterministic: no current time, filesystem path, random
SVG ID, platform font discovery, embedded raster, script, external resource,
or hidden metadata. The candidate validator regenerates every SVG from its
spec and aggregate bytes and requires byte identity. The final validator does
the same. The release sidecar index commits both spec and SVG bytes.

## 9. Independent results-review attestation

The independent reviewer receives only a strictly validated candidate root and
its two digests. The reviewer checks:

- all preregistered positive, negative, control, undefined, and
  not-applicable outcomes;
- sign, interval, support, persistence, crossover, and censoring semantics;
- complete-scene and cluster-sensitivity interpretation;
- M4 apply-only transport and absence of refitting or favorable selection;
- figure/claim selector completeness and exact source bindings;
- counterexamples and common-mode limitations;
- resource and execution-evidence scope;
- dataset/license/privacy language; and
- every proposed public quantitative statement.

The reviewer authors a Markdown report and a small decision input. A
repository tool canonicalizes, but does not choose, the decision into
`ffb.m5-results-review-attestation/v1`. The attestation contains:

- candidate semantic SHA-256;
- exact candidate-index byte SHA-256;
- scientific-member-set, claim-projection, figure-spec-set, rendered-figure-set,
  and presentation-template-set SHA-256 values;
- review-report byte SHA-256;
- reviewer identity scope
  `operator-recorded-not-cryptographically-authenticated`;
- counts of P0, P1, and P2 findings;
- explicit declarations that negative/undefined results and limitations were
  reviewed and retained; and
- the reviewer-authored disposition.

Release requires zero unresolved P0 or P1 findings and a disposition that
permits release. This plan does not supply that disposition.

The review report may name the candidate digest but not the future attestation
or final artifact digest. The attestation binds the report and candidate. The
final `results-and-claims-review` validation evidence binds the attestation.
The final artifact therefore depends on the review without requiring the
review to hash itself or predict the final artifact.

Any scientific, claim, figure-spec, SVG, or non-placeholder presentation-byte
change after review creates a new candidate digest and requires a new
independent review.

## 10. Exact final release package

The tracked package root is exactly:

```text
reports/releases/m5-nuscenes-replay-v0.1.0/
```

It is published atomically as one no-overwrite directory and contains exactly:

```text
artifact/
  intent.json
  replay-profile-summary.json
  descriptor-aggregates.ndjson
  persistent-panel-aggregates.ndjson
  persistent-panel-crossovers.ndjson
  health-panel-aggregates.ndjson
  leave-one-cluster-sensitivity.ndjson
  validation.json
  repeat-verification.json
  figure-records.ndjson
  source-member-commitments.ndjson
  release-index.json
  run.json
  _SUCCESS
README.md
claim-evidence.md
verification.md
release-summary.json
release-sidecar-index.json
figures/
  m5-persistent-panel-summary.spec.json
  m5-persistent-panel-summary.svg
  m5-crossovers.spec.json
  m5-crossovers.svg
  m5-health-transfer.spec.json
  m5-health-transfer.svg
  m5-descriptor-comparison.spec.json
  m5-descriptor-comparison.svg
  m5-cluster-sensitivity.spec.json
  m5-cluster-sensitivity.svg
evidence/
  release-pipeline-plan.md
  release-pipeline-plan-review.md
  resource-scope-amendment.md
  implementation-review.md
  review-candidate-index.json
  validation-inputs.json
  implementation-review-attestation.json
  software-verification.json
  privacy-license-attestation.json
  public-claim-projections.json
  results-review.md
  results-review-attestation.json
```

The `artifact/` subtree is the existing exact 14-file
`ffb.replay-curated-payload/v1` machine artifact. Its strict loader remains
usable without the dataset. Candidate scientific members, figure records, and
source commitments must enter it byte-for-byte. The final validation, release
index, finalized run, and success marker are created only after the review
attestation is authenticated. The outer package contains exactly 41 files:
the 14 machine-artifact files, 26 indexed sidecars, and
`release-sidecar-index.json`, which is excluded from its own entry list.

The candidate index is copied byte-for-byte to
`evidence/review-candidate-index.json`. All other evidence files are exact
candidate or review bytes. The plan, plan review, resource amendment, and
implementation review are exact copies of their public tracked authorities.
Figure specs and SVGs are exact candidate bytes.
The three Markdown files are the reviewed templates with only the four allowed
identity substitutions. `release-summary.json` and
`public-claim-projections.json` are exact candidate bytes.

The review report and attestation are also copied byte-for-byte to
`docs/reviews/m5-results-review.md` and
`docs/reviews/m5-results-review-attestation.json` before the release commit.
The package-only `validate-release` command remains self-contained and does
not depend on those repository paths. A separate `validate-publication`
command requires both public copies to equal the package and checks the
release-specific documentation projection. This separation lets an isolated
wheel validate the 41-file package while CI and closeout validate the complete
repository publication.

## 11. Sidecar index and package identity

`release-sidecar-index.json` has schema
`ffb.m5-release-sidecar-index/v1` and contains:

- release ID;
- reviewed candidate and result-review-attestation SHA-256 values;
- strict machine artifact and run SHA-256 values;
- exact scientific Git revision;
- exact ordered entries for every package file outside `artifact/` and outside
  `release-sidecar-index.json`;
- exact machine-artifact and indexed-sidecar payload byte lengths;
- `sidecar_set_sha256`; and
- `release_package_sha256`.

Each sidecar entry contains path, role, byte length, SHA-256, and NDJSON record
count when applicable. `sidecar_core` is the mapping without
`sidecar_set_sha256` and `release_package_sha256`.

```text
sidecar_set_sha256 =
  SHA256(
    b"fusion-fault-bench/m5-release-sidecars/v1\x00"
    || uint64_be(len(canonical_json_bytes(sidecar_core)))
    || canonical_json_bytes(sidecar_core)
  )

release_package_sha256 =
  SHA256(
    b"fusion-fault-bench/m5-release-package/v1\x00"
    || bytes.fromhex(machine_artifact_sha256)
    || bytes.fromhex(sidecar_set_sha256)
  )
```

The index itself is excluded from its file list. The final validator
reconstructs both digests. The entire package, not only `artifact/`, must
remain below 50 MiB. The indexed-sidecar payload length excludes
`release-sidecar-index.json`, avoiding a length self-reference; the validator
still measures every package byte, including the index, for the cap.

## 12. Privacy, licensing, and public claim boundary

Candidate preparation, review, final build, validation, documentation sync,
staging, and CI apply the same bounded privacy scan to all JSON, NDJSON,
Markdown, SVG, filenames, stdout, stderr, and exception messages.

The candidate and final package must contain none of:

- archives, maps, images, point clouds, or nuScenes metadata tables;
- sample, annotation, instance, calibration, pose, or log tokens;
- dataset filenames or absolute/relative dataset paths;
- per-frame timestamps, poses, calibrations, coordinates, or local sequence
  rows;
- raw `/usr/bin/time` logs;
- credentials, private interview material, or generated local outputs; or
- undeclared files, symbolic links, hard links, devices, sockets, or FIFOs.

The frozen public scene names remain the only intentional dataset identifiers.
Opaque scene and log-group ordinals remain non-authenticating labels.

Every aggregate, figure spec, figure, release summary, overview, and claim
ledger states `CC BY-NC-SA 4.0 plus Motional Dataset Terms`, attribution, and
non-endorsement. The repository Apache-2.0 license does not relicense the
aggregate replay evidence.

Public claims remain limited to matched-center estimator-output loss under the
declared proxy, persistence or non-persistence across the ten fixed mini
scenes, apply-only transport of the frozen M4 rule, finite scene/log-group
sensitivity, and measured named-CPU resources. No result is described as raw
sensor robustness, detector performance, a physical tolerance, fleet
generalization, planning/collision benefit, production readiness, or safety.

## 13. No-overwrite, staging, and rollback

Candidate and release publication use descriptor-relative, no-follow,
regular-file operations and owned random staging directories adjacent to the
destination. Existing destinations, dangling links, aliases, and equal or
nested input/output paths are rejected.

Candidate publication:

1. validates current source and all five local inputs;
2. writes all candidate members except the index into a private staging
   directory;
3. computes and writes `candidate-index.json` last;
4. strictly reloads and regenerates the complete candidate;
5. fsyncs every file, directory, and the staging parent;
6. atomically renames the candidate root without replacement; and
7. fsyncs the destination parent after the rename.

Final publication:

1. freshly reloads the candidate, report, and attestation;
2. re-verifies the original local inputs and unchanged current source;
3. deterministically re-curates the original primary/repeat artifacts and
   resource logs, regenerates all ten machine inputs, claim projections, five
   spec/SVG pairs, presentation templates, evidence records, and candidate
   index in memory, and requires every regenerated member plus both candidate
   digests to equal the reviewed candidate byte-for-byte;
4. creates the strict artifact from those regenerated machine bytes under a
   private package staging root;
5. materializes exact candidate/review sidecars and allowed substitutions;
6. writes the machine `_SUCCESS` only after its indexed members validate;
7. writes `release-sidecar-index.json` only after every other package member
   validates;
8. strictly reloads the complete package, regenerates claims and figures, and
   verifies both package digests;
9. fsyncs every file, directory, and the staging parent;
10. atomically renames the complete package root without replacement; and
11. fsyncs the destination parent after the rename.

The reviewed candidate is a comparison target, never the authority from which
the final scientific payload is blindly copied.

On failure, cleanup removes only the process-owned staging tree. It never
removes or rewrites an existing candidate, release, report, dataset file, raw
log, local artifact, tracked file, or user directory. A package is not valid
without both the machine `_SUCCESS` and valid sidecar index. A failure of the
post-rename parent fsync is reported as indeterminate durability: the complete
new destination is left untouched, is not declared published, and cannot be
retried at the same path.

## 14. Frozen command workflow

The following command shape is fixed; concrete output suffixes use the clean
scientific revision and an operator run ordinal chosen before execution.
After release tooling implementation and its independent review, the
reviewer-authored report and ignored decision input are first canonicalized:

```bash
uv run --frozen --no-sync python tools/m5_release.py \
  attest-implementation-review \
  --review-report docs/reviews/m5-release-implementation-review.md \
  --decision \
    reports/generated/m5-release-implementation-review-decision.json \
  --output \
    docs/reviews/m5-release-implementation-review-attestation.json
```

The report and attestation then pass the full source/privacy suite and are
committed and pushed with no implementation-snapshot change. Only from that
clean, remotely synchronized scientific revision does execution begin:

```bash
git status --short --untracked-files=all
uv lock --check
uv sync --locked --group dev

export NUSCENES_ROOT=<user-provided-nuscenes-root>
export UV_CACHE_DIR=<private-temporary-directory>/ffb-m5-uv-cache-<revision>
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

uv run --frozen --no-sync python tools/m5_release.py verify-software \
  --output reports/generated/m5-software-verification-<revision>.json

uv run --frozen --no-sync python tools/m5_release.py run-replay \
  --run-label primary \
  --output-dir reports/generated/m5-replay-primary-<revision>-r1 \
  --time-l-output \
    reports/generated/m5-replay-primary-<revision>-r1.time-l.txt

uv run --frozen --no-sync python tools/m5_release.py run-replay \
  --run-label repeat \
  --output-dir reports/generated/m5-replay-repeat-<revision>-r1 \
  --time-l-output \
    reports/generated/m5-replay-repeat-<revision>-r1.time-l.txt

uv run --frozen --no-sync ffb replay local validate \
  reports/generated/m5-replay-primary-<revision>-r1
uv run --frozen --no-sync ffb replay local validate \
  reports/generated/m5-replay-repeat-<revision>-r1

uv run --frozen --no-sync ffb replay verify-repeat \
  --primary-artifact reports/generated/m5-replay-primary-<revision>-r1 \
  --repeat-artifact reports/generated/m5-replay-repeat-<revision>-r1 \
  --primary-time-log \
    reports/generated/m5-replay-primary-<revision>-r1.time-l.txt \
  --repeat-time-log \
    reports/generated/m5-replay-repeat-<revision>-r1.time-l.txt

uv run --frozen --no-sync python tools/m5_release.py prepare-review \
  --primary-artifact reports/generated/m5-replay-primary-<revision>-r1 \
  --repeat-artifact reports/generated/m5-replay-repeat-<revision>-r1 \
  --primary-time-l \
    reports/generated/m5-replay-primary-<revision>-r1.time-l.txt \
  --repeat-time-l \
    reports/generated/m5-replay-repeat-<revision>-r1.time-l.txt \
  --software-verification \
    reports/generated/m5-software-verification-<revision>.json \
  --output-dir reports/generated/m5-review-candidate-<revision>

uv run --frozen --no-sync python tools/m5_release.py validate-review-candidate \
  reports/generated/m5-review-candidate-<revision>
```

`run-replay` validates real parents and absent output/log destinations, then
reserves the private timing log with descriptor-relative
`O_CREAT | O_EXCL | O_NOFOLLOW` and mode `0600`. It passes that already owned
descriptor to Darwin `/usr/bin/time -l -o /dev/fd/<fd>` and launches exactly
`ffb replay run --output-dir <declared-output>`. Thus `/usr/bin/time` measures
the actual replay process after the locked environment is prepared, writes
only its resource block to the reserved log, and cannot truncate a preexisting
path or mix dependency-tool/child stderr into the strict parser. The wrapper
propagates the replay exit status, fsyncs a complete log, and never retries,
deletes, or overwrites a failed attempt. Primary and repeat run sequentially
with fresh predeclared destinations. Any failure blocks authoritative
execution at that revision; it does not authorize a new ordinal or another
outcome-bearing run. Resumption requires an explicit public protocol/incident
amendment, a new committed scientific revision, and fresh primary and repeat
runs under that revision.

The independent reviewer then reviews that exact ignored candidate and authors
the report and decision input without changing candidate bytes. The
attestation command only canonicalizes the reviewer-authored disposition:

```bash
uv run --frozen --no-sync python tools/m5_release.py attest-results-review \
  --candidate reports/generated/m5-review-candidate-<revision> \
  --review-report reports/generated/m5-results-review-<revision>.md \
  --decision reports/generated/m5-results-review-decision-<revision>.json \
  --output reports/generated/m5-results-review-attestation-<revision>.json

uv run --frozen --no-sync python tools/m5_release.py build-release \
  --candidate reports/generated/m5-review-candidate-<revision> \
  --results-review reports/generated/m5-results-review-<revision>.md \
  --results-review-attestation \
    reports/generated/m5-results-review-attestation-<revision>.json \
  --primary-artifact reports/generated/m5-replay-primary-<revision>-r1 \
  --repeat-artifact reports/generated/m5-replay-repeat-<revision>-r1 \
  --primary-time-l \
    reports/generated/m5-replay-primary-<revision>-r1.time-l.txt \
  --repeat-time-l \
    reports/generated/m5-replay-repeat-<revision>-r1.time-l.txt \
  --software-verification \
    reports/generated/m5-software-verification-<revision>.json \
  --output-dir reports/releases/m5-nuscenes-replay-v0.1.0

uv run --frozen --no-sync python tools/m5_release.py sync-reviewed-evidence \
  --release reports/releases/m5-nuscenes-replay-v0.1.0 \
  --review-report-output docs/reviews/m5-results-review.md \
  --review-attestation-output \
    docs/reviews/m5-results-review-attestation.json

uv run --frozen --no-sync python tools/m5_release.py validate-release \
  reports/releases/m5-nuscenes-replay-v0.1.0
uv run --frozen --no-sync ffb replay bundle validate \
  reports/releases/m5-nuscenes-replay-v0.1.0/artifact
```

`sync-reviewed-evidence` copies only the exact package review bytes to the two
fixed absent destinations using the same exclusive, no-follow, fsync, and
post-rename-parent-fsync rules. It refuses existing paths and performs no
scientific regeneration. The build refuses an unresolved review blocker, but
no command infers or rewrites the reviewer disposition.

## 15. Release closeout

After the package is built, the release-specific sections of `README.md`,
`docs/results.md`, `docs/benchmark-card.md`, `docs/limitations.md`,
`docs/reproducibility.md`, `docs/project-plan.md`,
`docs/dataset-preparation.md`, and `docs/m5-technical-walkthrough.md` are
updated only from the reviewed public-claim projection and package sidecars.
They may summarize but may not introduce another quantitative selector.

The complete closeout suite is:

```bash
uv lock --check
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
uv run --locked pytest
uv build --no-sources
uv run --locked python tools/m5_release.py validate-release \
  reports/releases/m5-nuscenes-replay-v0.1.0
uv run --locked python tools/m5_release.py validate-publication \
  --release reports/releases/m5-nuscenes-replay-v0.1.0 \
  --source-root .
```

After the environment is provisioned, the explicit offline proof is:

```bash
env -u NUSCENES_ROOT UV_OFFLINE=1 \
  uv run --frozen --no-sync python tools/m5_release.py validate-release \
  reports/releases/m5-nuscenes-replay-v0.1.0
env -u NUSCENES_ROOT UV_OFFLINE=1 \
  uv run --frozen --no-sync python tools/m5_release.py validate-publication \
  --release reports/releases/m5-nuscenes-replay-v0.1.0 \
  --source-root .
```

The release API and package-only validator are also exposed through the
installed `ffb replay release validate` CLI. An already provisioned isolated
Python 3.12 wheel environment, with network disabled and `NUSCENES_ROOT`
absent, must import the package, print the version, expose the replay CLI,
render the replay-validation and replay-resource schemas, and validate the
tracked M5 package. Dependency installation is not part of the offline proof.

Before staging and again before pushing:

- inspect tracked, untracked, and staged names;
- require no tracked `interview/`, `reports/generated/`, dataset, raw-log, or
  local-artifact member;
- run the bounded tracked/staged privacy and secret audit;
- run `git diff --check` and the release and publication validators; and
- verify that the package and review copies are byte-identical and below the
  cap.

CI adds the same `UV_OFFLINE=1`, `--frozen`, `--no-sync`, dataset-unset
`validate-release` and `validate-publication` steps after dependency
provisioning. The release commit is pushed first. Only after GitHub CI passes
on that exact commit may the annotated tag
`m5-nuscenes-replay-v0.1.0` be created and pushed. Branch, tag, and remote
commit resolution must agree exactly.

## 16. Final acceptance gates

M5 release acceptance requires all of the following:

1. this release workflow was frozen before outcome inspection;
2. release tooling and tests were committed before the authoritative runs;
3. both runs bind the same clean source and pass every local-data gate;
4. all eight scientific members match exactly with zero mismatch;
5. both external resource records pass their strict command, environment,
   dominance, wall-time, and RSS gates;
6. the exact 34-file review candidate validates and remains unchanged;
7. all 44 preregistered claim coordinates, all ten crossovers, fixed
   descriptors, resources, repeat facts, and unsupported states remain
   visible;
8. all five figure specs and SVGs regenerate byte-for-byte with complete
   predeclared source coverage;
9. every validation digest is derived from its named authority;
10. the independent results-and-claims review binds the exact candidate and
    has no unresolved P0 or P1 blocker;
11. the strict 14-file machine artifact and complete sidecar package validate
    offline without the dataset;
12. methodology, limitations, reproducibility, results, claim ledger,
    technical walkthrough, and figures agree;
13. the full software, build, wheel, privacy, license, size, and no-overwrite
    gates pass;
14. no dataset/private/generated local payload is tracked or staged;
15. the release commit passes GitHub CI; and
16. the pushed annotated tag resolves to that exact release commit.

A failed, unsupported, contradictory, undefined, not-applicable, or
non-persistent scientific result does not by itself block release. Invalid
base support, incomplete evidence, favorable selection, review blockers,
privacy leakage, failed software verification, or an unauthenticated release
package does.
