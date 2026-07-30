# M5 exploratory implementation review

Status: **independent adversarial review of the M5 revision at an offline
exploratory checkpoint**. Reviewer identity scope:
`operator-recorded-not-cryptographically-authenticated`.

This is a checkpoint review performed against a local, network-isolated
environment. It is **not** the authoritative whole-revision implementation
review required by `docs/m5-release-pipeline-plan.md` §2, which must be
authored, canonicalized with `attest-implementation-review`, and committed on a
live-upstream-synchronized revision before the authoritative replay runs. It
records the substance of an independent review so the authoritative reviewer can
confirm it on the final pushed revision.

## Scope

M5 release revision at `codex/m5-release-pipeline`, commit
`e99c097` (runner strict-reload fix) atop `661f65e` (dashboard 9th-closeout
projection + frozen-plan-digest re-pin). Files examined: `replay_runner.py`
(metadata guard, `_decode_contract_row` / `_decode_dataclass_row`, local NDJSON
reload), `replay_release.py`, `replay_release_package.py`,
`replay_release_workflow.py`, `replay_release_software.py`,
`replay_publication.py`, `replay_publication_authority.py`,
`replay_artifacts.py` loaders, `contracts/common.py`, `contracts/replay_health_v1.py`,
`docs/dashboard.html`, `docs/m5-release-pipeline-plan.md`, and the M5 test
suite. Focused pytest subsets were run and the privacy scan was evaluated
against the real plan doc.

## Disposition: permit-release (0 P0, 0 P1, 1 P2)

### Decode fix (commit e99c097)

The runner reloaded freshly written local NDJSON members via strict Python-mode
validation (`model_validate` / `validate_python` over a `json.loads` dict).
`ContractModel` is `strict=True` + `extra=forbid`, so strict Python mode
genuinely rejects JSON arrays for the `tuple[int, int]` schedule fields of
`ReplayHealthSequenceEventV1` — a full all-ten-scene replay computed
successfully but failed at artifact publication on real nuScenes-mini data. The
fix decodes in JSON mode (`model_validate_json` / `adapter.validate_json` over
`canonical_json_bytes(payload)`), matching the release/curation loaders in
`replay_artifacts.py`. Verified:

- the relaxation is **only** JSON `list` → `tuple`; JSON mode still enforces all
  other strict types;
- the pre-existing exact canonical round-trip check
  (`canonical_json_bytes(decoded) != canonical_json_bytes(payload)`) still runs
  and remains the real guarantor — any value-changing coercion re-serializes to
  different canonical bytes and fails closed;
- `_strict_json_mapping` already forces each line to equal its canonical bytes
  (unique keys, finite numbers) before the typed decode, so re-canonicalizing
  cannot admit non-canonical input.

No silent data corruption is possible.

### Frozen-methodology re-pin (commit 661f65e)

Both `_FROZEN_METHODOLOGY_SHA256` maps (`replay_release.py`,
`replay_release_package.py`) are byte-identical and equal the committed doc
bytes (plan `60d1335…`, plan-review `3a881c4…`, amendment `f7eb19e…`); the
working tree is clean. The plan-doc diff is purely methodological (8 → 9
closeout documents, 51 → 52 pending files, exact 9-modified/43-added blob
accounting, descriptor-relative no-follow reading requirement). No dataset
bytes, paths, tokens, coordinates, or outcome-dependent choices are smuggled.
The frozen digest is a privacy-scan bypass, so the re-pin is security-critical;
it is legitimate and safe because (a) the pinned bytes are the reviewed tracked
doc, (b) tampering is caught by `test_methodology_tampering_cannot_use_frozen_privacy_exception`,
and (c) a stale digest fails **closed** (the doc trips the scan without the
bypass). This is a valid pre-outcome methodology amendment.

### Dashboard projection, privacy boundary, atomicity

`render_dashboard_projection` is deterministic and built only from validated
reviewed values (SHA-256s constrained to `^[0-9a-f]{64}$`, result/role slugs to
`^[a-z0-9-]+$`) plus fixed static text, then re-scanned by
`_require_public_projection_safe`; the marker region is the only mutable span.
The runner metadata guard uses per-component `O_NOFOLLOW` opens, full-stat
identity binding, TOCTOU re-checks, and symlink/hardlink alias rejection. Every
release-package member is privacy-scanned on build and load; publication uses
`O_CREAT|O_EXCL|O_NOFOLLOW` exclusive writes with before/after dev/ino/size/nlink
verification and no-replace renames; the pending/clean git-state gates are exact
and content-fingerprinted.

## Findings

### P2-1 — release-module frozen digest lacked a direct byte-binding test

`_FROZEN_METHODOLOGY_SHA256` is defined twice; only the package-module copy had
a direct hash-vs-tracked-doc assertion. The security-critical `replay_release.py`
copy (which grants the scan bypass) was only indirectly guarded. Runtime
behavior is fail-safe (a stale digest fails closed), but a future edit that both
desynced the release-module digest and removed the scan-tripping literals from
the doc would fail opaquely without a test catching it.

**Resolution (applied at this checkpoint):**
`tests/test_m5_release_package.py::test_release_module_frozen_methodology_digests_match_exact_tracked_authorities`
now binds the `replay_release.py` copy directly to the tracked doc bytes and
asserts the two frozen maps are byte-identical.

## Note for the authoritative reviewer

This review covered `e99c097`; the P2-1 test was added afterward (a non-blocking
hardening of the reviewer's own finding). The authoritative implementation
review must be re-affirmed against the final pushed revision per
`m5-release-pipeline-plan.md` §2.
