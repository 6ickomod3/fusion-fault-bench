# M6 adversarial plan review

Status: **PASS after revision**.

Date: 2026-07-31. Reviewer-identity scope:
`operator-recorded-not-cryptographically-authenticated` (independent adversarial
agents; no author context).

Reviewed artifact: [`docs/m6-report-plan.md`](../m6-report-plan.md), the M6
public-report pre-registration. M6 has no separate machine-readable intent JSON
at preregistration: the claim-source registry is an implementation artifact
whose *membership rule* is frozen in the plan (§5), so this review binds the
plan bytes only.

- pre-revision plan byte SHA-256:
  `cdd0ff877200b85531b7085bb69f0b9bbc6ae44c97b0713e020d7715363a466d`
- post-revision plan byte SHA-256:
  `702a1dda205eed7afc9ddad5026b6ca3d258509a622403e34f9a3bd934a99532`

## Method

Three independent, outcome-blind adversarial reviewers examined the frozen
pre-revision plan across diverse lenses — scope/anti-selection,
reproducibility/evidence-traceability, and privacy/claim-boundary/completeness —
each recomputing the plan digest and cross-checking against
`docs/project-plan.md` §11 (M6), `docs/benchmark-contract-v0.1.md`, and the
released M1–M4 evidence. A fourth independent verifier confirmed the revision on
the post-revision bytes. The panel returned **2 P0, 6 P1, and 8 P2** findings
(all three lenses: *revise*); every finding is resolved in the post-revision
text below.

## Release-blocking (P0) blockers and resolutions

1. **Registry membership not bound to a canonical released denominator**
   (outcome-driven omission of a released negative/undetermined result was
   possible within the letter of every gate). **Resolved:** §5 binds the
   registry's headline entries to the *exact union* of the released
   per-milestone claim ledgers (M1/M3 `claim-evidence.md`, M4
   `quantitative-claims.ndjson`, M2 validation claims), keyed by stable claim
   IDs; §11 audits coverage by ID; any exclusion needs an itemized §15 amendment
   with a non-outcome reason; §14.8 gates zero un-amended omissions.

2. **Exact-hash reproduction on a generic "clean CPU" contradicted M3's released
   byte-identity boundary** (`m3 verification.md` disclaims cross-architecture
   byte identity). **Resolved:** §9 scopes byte-exact hash/length/count equality
   to the named reference environment (arm64 macOS + `uv sync --locked`
   toolchain) and uses a portable equivalence check (abs tol `1e-12` per declared
   unit; exact for counts/categorical/grid) elsewhere; §14.2 mirrors this.

## Major (P1) findings and resolutions

3. **M2 validation-vs-fault-performance distinction was unenforced.**
   **Resolved:** §11 and gate §14.9 require every validation-only claim (all M2
   quantities) to be labeled validation and never rendered as a
   fault-performance/crossover/estimand result; §5 adds a `validation-only` flag.
4. **CPU-benchmark resource evidence wrongly attributed to M3** (only M4
   publishes addressable `resource-*` members). **Resolved:** §8 renders resource
   claims only from the M4 members and states M3 has no addressable resource
   token.
5. **Reproduction pinned dependencies and matrix but not the M3 source
   revision.** **Resolved:** §9 pins the reproduction to the M3 official-identity
   source revision `e8595fe428bcb9dfb269069e4b02972aff10f4ee`.
6. **§9 command/inputs/compared-members were declared "frozen" but unstated.**
   **Resolved:** §9 now names the command, the input matrix, the exact compared
   members (per-experiment `aggregate-metrics.ndjson` + `crossovers.ndjson` +
   the committed 429/10/71,700 hash/length/count), and the excluded volatile
   provenance.
7. **Privacy/no-leak gate had no testable procedure.** **Resolved:** §14.10 adds
   a concrete clean-clone + tracked/released byte-scan + `.gitignore`/`git
   ls-files` check that exits nonzero on any violation.
8. **Units rule was narrower than the contract** (omitted radians/seconds).
   **Resolved:** §5 and §10 enumerate all four axes (meters, radians, seconds,
   standard-deviation scale) and forbid pooling any two.

## Minor (P2) findings — all addressed

§3 now requires M5 incorporation to have a release **plus** a §15 amendment and a
repeated independent audit (with honest M3-Pro/M4-Max mixed-hardware
disclosure); §7 drops "candidate" and fixes the complete public figure set; §8
cites the matrix via its released member and canonical
`matrix_sha256 7162a01b…`; §12 restricts M6 to already-locked dependencies plus
the standard library so `uv.lock` is unchanged by construction; §1 adds the
fault-prevalence/prior/distribution non-goal; §14.5 requires the three named
external works to be cited, distinguished, and marked non-endorsement; §14.11
gates per-artifact license/attribution; and §4 freezes the private path to
`interview/m6/`.

## Verifier's new finding and resolution

The post-revision verifier raised one new P1: the plan's Status header and §15
referenced this review-of-record in the past tense before it existed.
**Resolved:** this file is committed at the cited path
`docs/reviews/m6-report-plan-review.md`, completing the
preregister → review → build chain.

## Non-blocking implementation notes

- Only M4's ledger is machine-keyed (`claim_id`); the M1/M2/M3 ledgers are
  markdown tables (M1 B/C/U rows, M2/M3 Q-/family rows). Implementation must
  define the exact per-ledger keying used for the §5 coverage audit; the
  membership *rule* is nonetheless frozen and enumerable.
- §8's phrase "volatile run-record timestamps" for M3 refers to the run-record
  wall-time/RSS/timestamps generally; the substantive rule (no M3 resource token
  in the registry) is correct.

## Final verdict

All 2 P0 and 6 P1 blockers and all 8 P2 findings are resolved in the
post-revision plan (`702a1dda…`), and the verifier's new P1 is resolved by this
committed review. The plan is approved with **zero unresolved P0/P1 blocker**.
This review approves the M6 *plan* only; it is not an implementation, results, or
claims-audit approval, which occur under their own later reviews.
