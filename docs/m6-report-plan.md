# M6 Public Report Pre-registration

Status: **frozen before any M6 report prose, figure curation, evidence-table
value, or claims-audit verdict is produced.** Revised once after an independent
adversarial plan review (see
[`reviews/m6-report-plan-review.md`](reviews/m6-report-plan-review.md)); the
review's blockers are resolved in this frozen text.

Date: 2026-07-31.

This document freezes the M6 scope, public and private deliverables, the
claim-source registry, the report structure, the curated figure set, the
one-command reproduction path, the resume-ready evidence table, the adversarial
claims-audit criteria, the acceptance gates, and the claim boundary **before any
M6 artifact is written**. M6 is a curation-and-reporting milestone: it produces
**no new experiment, estimand, manifest, method, or result.** It only reports,
reproduces, and audits evidence that has **already been released** under the
frozen per-milestone contracts. Freezing the report scope before writing the
prose prevents outcome-driven selection, emphasis, or omission in the report
itself.

M6 does not modify the [benchmark contract v0.1](benchmark-contract-v0.1.md),
the [project plan](project-plan.md), any released manifest, or any released
aggregate. Where this document is broader than the benchmark contract, the
contract controls; where a unit or boundary rule below is narrower than the
contract, the contract still controls.

## 1. Purpose and non-goals

Purpose: produce a concise, reproducible, adversarially audited public report of
the Fusion Fault Bench CPU-only benchmark, plus a private engineering and
interview-preparation companion, such that every public headline statement
traces to released evidence and a stated validity boundary.

M6 does **not**:

- run a new experiment, add a severity/seed/method/policy/population, or change
  an estimand or inference rule;
- introduce a v0.2 method (set prediction, Hungarian association, robust fusion,
  learned health classification) — that requires separate preregistration and
  fresh evaluation data;
- promote any result that is not already released under its milestone contract;
- restate external work (nuScenes, MultiCorrupt, CalibRobustBEV) as new;
- make a real-world fault-prevalence, fault-prior, or naturally-occurring
  fault-distribution claim (contract §1, §5, §12); or
- make a raw-sensor, detector, physical-tolerance, fleet-generalization,
  planning/collision, production-safety, or certification claim.

## 2. Evidence in scope (released only)

The public report draws headline evidence **only** from released milestones:

- **M1 analytic** — `reports/releases/m1-analytic-v0.1.0/` (fault-performance
  estimator-output result; named CPU Apple M3 Pro).
- **M2 geometry** — `reports/releases/m2-geometry-v0.1.0/` (implementation and
  local-grounding **validation** only; explicitly **not** a fault-performance
  result).
- **M3 procedural** — `reports/releases/m3-procedural-v0.1.0/` (fault-performance
  result; the fixed procedural matrix; named CPU Apple M3 Pro).
- **M4 health** — `reports/releases/m4-health-v0.1.0/` (fault-performance and
  observable-fallback result, and the only released **resource** benchmark;
  named CPU Apple M3 Pro).

Each milestone's role is carried verbatim; M2 is never presented as a
fault-performance, crossover, or estimand result (enforced by §11 and §14.9).

## 3. Handling of M5 (prepared, not released)

M5 (nuScenes latent replay) is **prepared and reviewed but not released** (no
`reports/releases/m5-nuscenes-replay-v0.1.0/`, no tag). Therefore:

- M5 contributes **no public headline number** to the M6 report;
- the report may state, in a clearly labeled status/roadmap section, that M5 is
  *prepared, pending an authoritative release*, with no quantitative replay
  claim; and
- incorporating any M5 number after this freeze requires **all** of: (a) M5's
  own contract release, (b) a documented §15 amendment, and (c) a repeated
  independent adversarial claims audit over the added M5 registry entries,
  including honest reporting of the mixed-hardware provenance (M1–M4 on Apple M3
  Pro; M5 on Apple M4 Max). Until all three occur, only the "prepared, pending"
  status appears.

## 4. Public and private deliverables

**Public (tracked, released with M6):**

1. a concise technical report `docs/m6-report.md` (the existing `docs/results.md`
   remains the results page);
2. the curated aggregate figure set of §7, each figure tracing to a released
   member;
3. the one-command procedural reproduction path of §9;
4. the resume-ready evidence table of §10; and
5. the machine-readable claim-source registry of §5.

**Private (ignored, never released):** a code tour, derivations, a question
bank, project narratives, and failure postmortems, kept **only** under the
frozen path `interview/m6/`, which is already covered by `.gitignore`
(`/interview/`). No private material, secret, or dataset payload enters any
tracked or released file; §13 and the §14.10 verification procedure enforce
this.

## 5. Claim-source registry (the only source of public numbers)

Every quantitative token in the public report, figures, evidence table, README,
results page, benchmark card, and limitations is generated from a single
machine-readable **claim-source registry**. Membership is **bound, not
author-chosen**:

- the registry's headline entries are the **exact union of the released
  per-milestone quantitative-claim ledgers**, keyed by each ledger's stable
  claim IDs: `reports/releases/m1-analytic-v0.1.0/claim-evidence.md`,
  `reports/releases/m3-procedural-v0.1.0/claim-evidence.md`,
  `reports/releases/m4-health-v0.1.0/quantitative-claims.ndjson`, plus the M2
  validation claims in `reports/releases/m2-geometry-v0.1.0/claim-evidence.md`;
- every headline-eligible claim ID in those ledgers MUST appear in the registry.
  Any exclusion requires an explicit, itemized §15 amendment stating a
  non-outcome reason; outcome-driven omission is prohibited;
- released **negative, undetermined, not-applicable, and non-persistent**
  results are therefore included by construction (e.g. M1 correctly-reported
  noise undetermined; M3 correctly-reported-noise counterexample and common-mode
  blind spot; M4 negative policy gain under underreported LiDAR noise, the
  recovery-latch cost, and clean-condition false alerts).

Each registry entry binds: a stable public claim ID; the released source member
(release path + member) and its SHA-256 as recorded in that release's index; the
selector and permitted projected fields; the unit and the outcome-independent
rendering rule (finite floats `.6g`; elapsed time `.2f`; byte/count values exact
integers; undefined / not-applicable / censored / positive-infinity use literal
labels, never invented numbers); a `validation-only` flag for M2 entries; and a
one-sentence validity boundary.

Physical-unit axes are never collapsed: meters, radians, seconds, and
standard-deviation scale (contract §5) are kept separate; no two physical-unit
axes are pooled or normalized into one severity anywhere. No quantitative
sentence outside the registry is permitted in any public M6 document, and no
magnitude ranking, top-k, or favorable-family selection occurs.

## 6. Fixed report structure

Fixed section order; no outcome-driven reordering:

1. Question and estimator-output abstraction (estimator-output, not raw sensor).
2. Method: paired counterfactuals, signed healthy-modality delta, predeclared
   crossover, matched-center ROI, baselines/oracles.
3. Released results by milestone (M1, M3, M4 fault-performance; M2 validation,
   explicitly labeled), each headline rendered from the §5 registry with its
   validity boundary; the complete bound ledger set is reported, positives and
   negatives together.
4. Required negative controls and negative/undetermined results (from the bound
   ledgers).
5. Reproducibility and the CPU benchmark (§8, §9).
6. Limitations and related work (§13; the three named external works cited and
   distinguished with non-endorsement).
7. Status and roadmap (M5 prepared/pending per §3; the optional neural-detector
   extension noted as out of scope and non-required).

## 7. Curated figure set

The public figure set is **complete and fixed here** — every listed figure is
included and no other figure is added; omitting any listed figure requires a
§15 amendment. Figures are copied byte-for-byte from already-released bundles,
never regenerated with new data; each declares its released source member and
SHA-256, and its caption text is drawn only from the §5 registry:

- M1 signed camera x-bias and noise-reporting curves;
- M3 signed fusion-delta curves, dropout controls, and common-mode control;
- M4 observable health policy outcomes.

No figure is selected or omitted by outcome magnitude or favorability.

## 8. Fixed experiment matrix and CPU benchmark

The **fixed experiment matrix** is the already-released procedural matrix,
cited by its released member
`reports/releases/m3-procedural-v0.1.0/intent/matrix.json` with canonical
`matrix_sha256 7162a01bb766a38e8f7196cba9eb2c6c546f5866bc640f220252bb6e7aab6f9b`
(recorded in that release's `release-summary.json`). The reproduction input
`examples/matrices/m3-procedural-v1.json` is the same matrix by canonical
digest; it is the input, not a released artifact. M6 adds no matrix entry.

The **CPU benchmark** (elapsed, peak RSS) is rendered **only** from the M4
released resource members that actually publish addressable resource claims
(`reports/releases/m4-health-v0.1.0/quantitative-claims.ndjson`
`resource-*` IDs and the SHA-256-indexed `*-time-l.txt` sidecars), on the named
Apple M3 Pro. M3 publishes **no** addressable resource member — only volatile
run-record timestamps, which the registry must never render — so no M3 resource
token is claimed. CPU-only reproduction is a design property, not a research
claim.

## 9. One-command procedural reproduction path (frozen command)

M6 provides a single command that, from a clean checkout with no dataset and no
GPU, reproduces the **procedural (M3)** aggregates and validates them against the
released bundle:

- **Command:** `uv run --frozen --no-sync python tools/m6_reproduce.py`
  (implementation wires it to the existing procedural runner and M3 release
  loaders; no new science).
- **Input:** `examples/matrices/m3-procedural-v1.json` (canonical digest equals
  the released `matrix_sha256` above).
- **Compared members:** per-experiment `aggregate-metrics.ndjson` and
  `crossovers.ndjson`, plus the committed **429 aggregate / 10 crossover /
  71,700 sequence** row hash/length/count from the M3
  `release-summary.json` member commitments.
- **Excluded from comparison:** all volatile run-provenance (`run.json`
  timestamps, environment, git revision, lockfile, wall time, RSS), matching the
  M3 `verification.md` scoping.
- **Source pinning:** the reproduction is validated against the M3
  official-identity **source revision
  `e8595fe428bcb9dfb269069e4b02972aff10f4ee`**; the command asserts the
  procedural scientific source path is byte-unchanged from that revision (or is
  run at that revision), so "reproduces the released M3 aggregates" is a
  code-pinned claim, not only a dependency-pinned one.
- **Environment scope of exact equality:** byte-exact hash/length/count equality
  is claimed and gated **only** on the named reference environment (arm64 macOS
  with `uv sync --locked` toolchain matching the M3 official identity), mirroring
  `reports/releases/m3-procedural-v0.1.0/verification.md`, which disclaims
  cross-architecture byte identity. On any other clean environment the command
  performs a **portable equivalence check** (relative tolerance 0, absolute
  tolerance `1e-12` in each declared unit; exact for counts, categorical, and
  grid values) and does not exit nonzero on legitimate floating-point-level
  differences.
- **Contract:** exits nonzero on any comparison failure within the applicable
  mode above.

## 10. Resume-ready evidence table

A resume-ready table containing **only measured values**, each a §5 registry
claim with its released source and validity boundary. It excludes any projected,
rounded-for-effect, hypothetical, or unreleased value (in particular no M5
number until M5 is released per §3). Meters, radians, seconds, and
standard-deviation scale remain separate axes and are never combined.

## 11. Adversarial claims audit

Before release, an **independent adversarial claims audit** verifies that:

- every public headline statement maps to a §5 registry entry whose SHA-256
  matches the released member, with a stated validity boundary;
- registry **coverage** is complete: every headline-eligible claim ID in the
  bound released ledgers (§5) is present, with zero un-amended omissions;
- no public quantitative token exists outside the registry;
- every validation-only claim (in particular all M2 quantities) is labeled
  validation and is never presented as a fault-performance, crossover, or
  estimand result;
- estimator-output simulation is distinguished from raw-sensor simulation
  everywhere;
- negative, undetermined, not-applicable, and non-persistent results are
  retained, not dropped or softened;
- the four physical-unit axes are never collapsed and no favorable selection
  occurs;
- the three named external works are cited and distinguished with
  non-endorsement; and
- no real-world fault-prevalence/prior claim and no
  raw-sensor/detector/physical-tolerance/fleet/planning/safety claim appears.

The audit is recorded with reviewer-identity scope
`operator-recorded-not-cryptographically-authenticated` and must reach a
release-permitting disposition with zero unresolved blocking findings.

## 12. Reproducibility, determinism, and dependency gates

- A clean environment reproduces the procedural aggregates via the §9 command
  under the applicable equality mode.
- Every public figure and number traces to a released versioned member by
  SHA-256, sourced only from the §5 registry.
- The report build is deterministic: no runtime timestamp, absolute path, random
  identifier, software-generated comment, or host-dependent metadata in tracked
  public output.
- M6 report generation and figure curation use **only already-locked
  dependencies plus the Python standard library** (SVGs copied byte-for-byte
  from released bundles; no new plotting, markdown, or templating package), so
  `uv.lock` is unchanged by construction — preserving the frozen-lockfile
  assumptions M1–M5 depend on.

## 13. Privacy, license, and public claim boundary

- Every public aggregate, figure, table, and claim states the applicable terms
  (project code Apache-2.0; any nuScenes-derived aggregate — only if M5 is later
  released — carries `CC BY-NC-SA 4.0 plus Motional Dataset Terms`, attribution,
  and non-endorsement).
- No dataset payload, token, path, secret, private interview material, or
  generated-local output enters any tracked or released M6 file.
- Public claims remain limited to matched-center estimator-output loss under the
  declared proxy faults and the released milestones' stated boundaries.

## 14. Acceptance gates

M6 acceptance requires all of the following:

1. this report scope was frozen before outcome curation (this document);
2. a clean environment reproduces the procedural aggregates via the single §9
   command — byte-exact on the named reference environment, portable-tolerance
   elsewhere — validated against the M3 official-identity source revision
   `e8595fe…`;
3. every public figure and number traces to a released versioned member by
   SHA-256, sourced only from the §5 registry;
4. the report distinguishes estimator-output simulation from raw-sensor
   simulation;
5. limitations and related work are explicit, and nuScenes, MultiCorrupt, and
   CalibRobustBEV are each cited, distinguished (no taxonomy/model claimed as
   new), and marked non-endorsement;
6. the independent adversarial claims audit passes with zero unresolved blocking
   findings, and every headline statement has released evidence plus a validity
   boundary;
7. no unreleased result (in particular no M5 number) is presented as released;
8. registry coverage against the bound released ledgers (§5) has zero un-amended
   omissions;
9. every validation-only claim (all M2 quantities) is labeled validation and is
   never rendered as a fault-performance/crossover/estimand result;
10. the privacy/no-leak boundary is verified by a concrete procedure: a fresh
    clean clone contains no files under ignored data/dataset/interview/generated
    paths and no secrets; the exact tracked+released M6 file set is byte-scanned
    for dataset tokens, nuScenes sample/scene tokens, absolute local paths, and
    secrets with zero hits; `interview/m6/` is in `.gitignore` and `git ls-files`
    returns zero entries under it; the check exits nonzero on any violation;
11. every public aggregate/figure/table/claim states its applicable license/
    attribution terms per §13; and
12. no dataset/private/generated-local payload is tracked or staged, and
    `uv.lock` is unchanged.

A negative, undetermined, or non-persistent released result does not block M6;
misattributed, unreleased, unaudited, favorably selected, incompletely covered,
or privacy-leaking content does.

## 15. Preregistration integrity

This plan is frozen before any M6 artifact is produced and received an
independent adversarial plan review
([`reviews/m6-report-plan-review.md`](reviews/m6-report-plan-review.md)) whose
blockers are resolved in this text, following the same preregister →
adversarial-plan-review → build → results/claims-review discipline used for
M1–M5. Any change to scope, the claim-source membership rule, the reproduction
contract, the unit/claim boundaries, or the acceptance gates after freeze
requires a documented amendment and a repeated review of the affected checks;
incorporating M5 is such a change (§3).
