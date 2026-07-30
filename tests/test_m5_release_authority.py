"""Synthetic authority and validation tests for the M5 release pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_RELEASE_VALIDATION_CHECK_IDS,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_IMPLEMENTATION_REVIEW_AREAS,
    M5_SOFTWARE_VERIFICATION_CATEGORIES,
    ReplayImplementationReviewDecisionV1,
    ReplayResultsReviewDecisionV1,
    ReplayReviewFindingV1,
    ReplaySoftwareVerificationCheckV1,
)
from fusion_fault_bench.replay_release_authority import (
    M5_IMPLEMENTATION_SNAPSHOT_DOMAIN,
    ImplementationSnapshot,
    ReplayReleaseAuthorityError,
    build_implementation_snapshot,
    implementation_snapshot_from_files,
)
from fusion_fault_bench.replay_release_validation import (
    M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK,
    M5_SOFTWARE_VERIFICATION_CHECK_IDS,
    M5_TRANSFORM_TIMING_ORACLE_TEST_IDS,
    M5_VALIDATION_AUTHORITY_PARTS,
    ReplayReleaseValidationError,
    ResultsReviewBindings,
    build_implementation_review_attestation,
    build_privacy_license_attestation,
    build_results_review_attestation,
    build_software_verification,
    derive_final_replay_validation,
    derive_pre_review_validation_inputs,
    derive_validation_evidence_sha256,
    load_final_replay_validation,
    load_implementation_review_attestation,
    load_implementation_review_decision,
    load_pre_review_validation_inputs,
    load_privacy_license_attestation,
    load_results_review_attestation,
    load_results_review_decision,
    load_software_verification,
    software_verification_test_subset_bytes,
)

REVISION = "a" * 40
EVIDENCE_COMMIT_REVISION = "d" * 40
LOCKFILE_SHA256 = "b" * 64
RUN_ID = "m5-run-1"


def _snapshot(
    *,
    license_bytes: bytes = b"repository license\n",
    revision: str = REVISION,
) -> ImplementationSnapshot:
    return implementation_snapshot_from_files(
        {
            "src/fusion_fault_bench/example.py": b"VALUE = 1\n",
            "LICENSE": license_bytes,
            "DATA_AND_MODEL_TERMS.md": b"dataset terms\n",
        },
        scientific_git_revision=revision,
    )


def _manual_snapshot_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256(M5_IMPLEMENTATION_SNAPSHOT_DOMAIN)
    ordered = sorted(files.items(), key=lambda item: item[0].encode("utf-8"))
    digest.update(len(ordered).to_bytes(8, "big"))
    for path, value in ordered:
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def test_implementation_snapshot_uses_exact_sorted_path_byte_framing() -> None:
    files = {"z.txt": b"z\n", "a.txt": b"a\n"}
    snapshot = implementation_snapshot_from_files(files, scientific_git_revision=REVISION)

    assert snapshot.sha256 == _manual_snapshot_digest(files)
    assert tuple(entry.path for entry in snapshot.entries) == ("a.txt", "z.txt")
    assert snapshot.file_count == 2
    assert snapshot == implementation_snapshot_from_files(
        dict(reversed(tuple(files.items()))),
        scientific_git_revision=REVISION,
    )
    assert (
        snapshot.sha256
        != implementation_snapshot_from_files(
            {"z.txt": b"z\n", "a.txt": b"changed\n"},
            scientific_git_revision=REVISION,
        ).sha256
    )


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../b", "a\\b", "a//b"])
def test_implementation_snapshot_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ReplayReleaseAuthorityError, match="normalized"):
        implementation_snapshot_from_files({path: b"value"}, scientific_git_revision=REVISION)


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def test_implementation_snapshot_expands_exact_strict_authorities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fusion_fault_bench import replay_release_authority as authority

    manifests = tuple(f"examples/manifests/m{index}.json" for index in range(8))
    profiles = tuple(f"examples/profiles/p{index}.json" for index in range(3))
    paths = (
        "src/package.py",
        "tools/release.py",
        "tests/test_release.py",
        "control.txt",
        "matrix.json",
        *manifests,
        *profiles,
        "m4/release.json",
    )
    files = {path: f"{path}\n".encode() for path in paths}
    for path, value in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)

    matrix_record = SimpleNamespace(
        execution_order=tuple(SimpleNamespace(manifest=path) for path in manifests),
        profiles=tuple(SimpleNamespace(profile=path) for path in profiles),
    )
    matrix = SimpleNamespace(
        matrix_sha256=authority.M3_PROCEDURAL_MATRIX_SHA256,
        matrix=matrix_record,
    )
    health = SimpleNamespace(release_artifact_sha256="c" * 64)
    loader_counts = {"matrix": 0, "health": 0}

    def load_matrix(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        loader_counts["matrix"] += 1
        return matrix

    def load_health(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        loader_counts["health"] += 1
        return health

    entries = {path: ("100644", _git_blob_sha1(value)) for path, value in files.items()}
    monkeypatch.setattr(authority, "_IMPLEMENTATION_STATIC_PATHS", ("control.txt", "matrix.json"))
    monkeypatch.setattr(authority, "M5_PERSISTENT_MATRIX_PATH", Path("matrix.json"))
    monkeypatch.setattr(authority, "M4_RELEASE_RELATIVE_PATH", Path("m4"))
    monkeypatch.setattr(authority, "HEALTH_RELEASE_ARTIFACT_PATHS", ("release.json",))
    monkeypatch.setattr(authority, "load_experiment_matrix", load_matrix)
    monkeypatch.setattr(authority, "load_health_release", load_health)
    monkeypatch.setattr(
        authority,
        "_git_head_and_index",
        lambda _root: (REVISION, entries, entries.copy()),
    )
    monkeypatch.setattr(authority, "_git_bytes", lambda *_args: b"")

    observed = build_implementation_snapshot(tmp_path)

    assert observed == implementation_snapshot_from_files(
        files,
        scientific_git_revision=REVISION,
    )
    assert loader_counts == {"matrix": 2, "health": 2}


def _implementation_decision(
    *,
    findings: tuple[ReplayReviewFindingV1, ...] = (),
    disposition: str = "pass",
) -> ReplayImplementationReviewDecisionV1:
    return ReplayImplementationReviewDecisionV1.model_validate(
        {
            "schema": "ffb.m5-implementation-review-decision/v1",
            "reviewer_identity": "independent-reviewer",
            "reviewed_areas": M5_IMPLEMENTATION_REVIEW_AREAS,
            "findings": findings,
            "disposition": disposition,
        }
    )


def test_implementation_review_is_canonicalized_and_source_bound() -> None:
    snapshot = _snapshot()
    post_commit_snapshot = _snapshot(revision=EVIDENCE_COMMIT_REVISION)
    assert post_commit_snapshot.sha256 == snapshot.sha256
    assert post_commit_snapshot.scientific_git_revision != snapshot.scientific_git_revision

    finding = ReplayReviewFindingV1(
        finding_id="wording-1",
        severity="p2",
        status="unresolved",
    )
    decision = _implementation_decision(
        findings=(finding,),
        disposition="pass-with-nonblocking-findings",
    )
    decision_bytes = canonical_json_bytes(decision)
    assert load_implementation_review_decision(decision_bytes) == decision

    report = b"# Independent implementation review\n\nNo blockers.\n"
    attestation = build_implementation_review_attestation(
        decision,
        review_report=report,
        snapshot=snapshot,
    )
    assert (attestation.p0_count, attestation.p1_count, attestation.p2_count) == (0, 0, 1)
    assert attestation.unresolved_finding_ids == ("wording-1",)
    assert attestation.disposition == decision.disposition
    assert "scientific_git_revision" not in attestation.model_dump(mode="json", by_alias=True)
    assert (
        load_implementation_review_attestation(
            canonical_json_bytes(attestation),
            review_report=report,
            snapshot=post_commit_snapshot,
        )
        == attestation
    )

    with pytest.raises(ReplayReleaseValidationError, match="stale source or report"):
        load_implementation_review_attestation(
            canonical_json_bytes(attestation),
            review_report=b"# Changed report\n",
            snapshot=snapshot,
        )

    with pytest.raises(ReplayReleaseValidationError, match="stale source or report"):
        load_implementation_review_attestation(
            canonical_json_bytes(attestation),
            review_report=report,
            snapshot=_snapshot(
                license_bytes=b"changed repository license\n",
                revision=EVIDENCE_COMMIT_REVISION,
            ),
        )


def test_blocked_implementation_review_is_preserved_but_not_release_permitting() -> None:
    finding = ReplayReviewFindingV1(
        finding_id="blocker-1",
        severity="p1",
        status="unresolved",
    )
    decision = _implementation_decision(findings=(finding,), disposition="block")
    report = b"# Review\n\nRelease is blocked.\n"
    attestation = build_implementation_review_attestation(
        decision,
        review_report=report,
        snapshot=_snapshot(),
    )
    assert attestation.disposition == "block"
    with pytest.raises(ReplayReleaseValidationError, match="does not permit release"):
        load_implementation_review_attestation(
            canonical_json_bytes(attestation),
            review_report=report,
            snapshot=_snapshot(),
        )
    assert (
        load_implementation_review_attestation(
            canonical_json_bytes(attestation),
            review_report=report,
            snapshot=_snapshot(),
            require_release_permitting=False,
        )
        == attestation
    )


def _results_bindings() -> ResultsReviewBindings:
    return ResultsReviewBindings(
        candidate_sha256="1" * 64,
        candidate_index_sha256="2" * 64,
        scientific_member_set_sha256="3" * 64,
        claim_projection_sha256="4" * 64,
        figure_spec_set_sha256="5" * 64,
        rendered_figure_set_sha256="6" * 64,
        presentation_template_set_sha256="7" * 64,
    )


def _results_decision() -> ReplayResultsReviewDecisionV1:
    return ReplayResultsReviewDecisionV1(
        schema="ffb.m5-results-review-decision/v1",
        reviewer_identity="independent-results-reviewer",
        findings=(
            ReplayReviewFindingV1(
                finding_id="wording-2",
                severity="p2",
                status="resolved",
            ),
        ),
        negative_and_undefined_results_reviewed_and_retained=True,
        limitations_reviewed_and_retained=True,
        disposition="pass",
    )


def test_results_review_canonicalization_binds_every_candidate_digest() -> None:
    decision = _results_decision()
    assert load_results_review_decision(canonical_json_bytes(decision)) == decision
    report = b"# Results review\n\nAll preregistered results retained.\n"
    bindings = _results_bindings()
    attestation = build_results_review_attestation(
        decision,
        review_report=report,
        scientific_git_revision=REVISION,
        bindings=bindings,
    )
    assert attestation.candidate_sha256 == bindings.candidate_sha256
    assert attestation.p2_count == 1
    assert (
        load_results_review_attestation(
            canonical_json_bytes(attestation),
            review_report=report,
            scientific_git_revision=REVISION,
            bindings=bindings,
        )
        == attestation
    )

    stale = ResultsReviewBindings(
        candidate_sha256="8" * 64,
        candidate_index_sha256=bindings.candidate_index_sha256,
        scientific_member_set_sha256=bindings.scientific_member_set_sha256,
        claim_projection_sha256=bindings.claim_projection_sha256,
        figure_spec_set_sha256=bindings.figure_spec_set_sha256,
        rendered_figure_set_sha256=bindings.rendered_figure_set_sha256,
        presentation_template_set_sha256=bindings.presentation_template_set_sha256,
    )
    with pytest.raises(ReplayReleaseValidationError, match="stale candidate or report"):
        load_results_review_attestation(
            canonical_json_bytes(attestation),
            review_report=report,
            scientific_git_revision=REVISION,
            bindings=stale,
        )


@pytest.mark.parametrize(
    "field",
    (
        "negative_and_undefined_results_reviewed_and_retained",
        "limitations_reviewed_and_retained",
    ),
)
def test_results_review_canonicalization_never_overrides_retention_decision(field: str) -> None:
    decision = _results_decision().model_copy(update={field: False})

    with pytest.raises(
        ReplayReleaseValidationError,
        match="results review decision cannot be canonicalized",
    ):
        build_results_review_attestation(
            decision,
            review_report=b"# Results review\n\nRetention was not confirmed.\n",
            scientific_git_revision=REVISION,
            bindings=_results_bindings(),
        )


def _software_checks() -> tuple[ReplaySoftwareVerificationCheckV1, ...]:
    return tuple(
        ReplaySoftwareVerificationCheckV1(
            check_id=check_id,
            category=category,
            command=("release-check", check_id),
            required_test_ids=M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[check_id],
            exit_status=0,
            output_sha256=hashlib.sha256(check_id.encode()).hexdigest(),
            output_normalization=("stable-command-output-with-runtime-paths-and-durations-removed"),
        )
        for check_id, category in zip(
            M5_SOFTWARE_VERIFICATION_CHECK_IDS,
            M5_SOFTWARE_VERIFICATION_CATEGORIES,
            strict=True,
        )
    )


def test_software_verification_envelope_reloads_exact_named_checks() -> None:
    snapshot = _snapshot()
    verification = build_software_verification(
        _software_checks(),
        snapshot=snapshot,
        lockfile_sha256=LOCKFILE_SHA256,
        package_version="0.1.0",
    )
    verification_bytes = canonical_json_bytes(verification)
    assert (
        load_software_verification(
            verification_bytes,
            snapshot=snapshot,
            lockfile_sha256=LOCKFILE_SHA256,
            package_version="0.1.0",
        )
        == verification
    )

    subset = software_verification_test_subset_bytes(
        verification,
        M5_TRANSFORM_TIMING_ORACLE_TEST_IDS,
    )
    assert b"rigid-transform-oracle" in subset
    with pytest.raises(ReplayReleaseValidationError, match="missing a required named test"):
        software_verification_test_subset_bytes(verification, ("invented-test",))
    with pytest.raises(ReplayReleaseValidationError, match="stale source authority"):
        load_software_verification(
            verification_bytes,
            snapshot=snapshot,
            lockfile_sha256="c" * 64,
            package_version="0.1.0",
        )


def test_software_verification_rejects_missing_required_test_authority() -> None:
    checks = list(_software_checks())
    checks[3] = checks[3].model_copy(update={"required_test_ids": ()})
    with pytest.raises(ReplayReleaseValidationError, match="required-test authority"):
        build_software_verification(
            checks,
            snapshot=_snapshot(),
            lockfile_sha256=LOCKFILE_SHA256,
            package_version="0.1.0",
        )


def test_privacy_license_attestation_binds_tracked_license_bytes() -> None:
    snapshot = _snapshot()
    attestation = build_privacy_license_attestation(snapshot=snapshot, run_id=RUN_ID)
    assert attestation.forbidden_match_count == 0
    assert attestation.raw_sensor_payload_reads == 0
    assert (
        load_privacy_license_attestation(
            canonical_json_bytes(attestation),
            snapshot=snapshot,
            run_id=RUN_ID,
        )
        == attestation
    )

    with pytest.raises(ReplayReleaseValidationError, match="stale source or run"):
        load_privacy_license_attestation(
            canonical_json_bytes(attestation),
            snapshot=_snapshot(license_bytes=b"different license\n"),
            run_id=RUN_ID,
        )


def _validation_evidence(*, include_results: bool) -> dict[str, dict[str, bytes]]:
    result: dict[str, dict[str, bytes]] = {}
    for check_id in M5_RELEASE_VALIDATION_CHECK_IDS:
        if check_id == "results-and-claims-review" and not include_results:
            continue
        result[check_id] = {
            part: f"{check_id}/{part}\n".encode()
            for part in M5_VALIDATION_AUTHORITY_PARTS[check_id]
        }
    return result


def test_seventeen_check_domains_and_pre_review_pending_slot_are_exact() -> None:
    pre_evidence = _validation_evidence(include_results=False)
    inputs = derive_pre_review_validation_inputs(
        run_id=RUN_ID,
        evidence_by_check=pre_evidence,
    )
    assert tuple(check.check_id for check in inputs.checks) == M5_RELEASE_VALIDATION_CHECK_IDS
    pending = inputs.checks[M5_RELEASE_VALIDATION_CHECK_IDS.index("results-and-claims-review")]
    assert pending.status == "pending"
    assert pending.passed is None
    assert pending.evidence_sha256 is None
    assert (
        load_pre_review_validation_inputs(
            canonical_json_bytes(inputs),
            run_id=RUN_ID,
            evidence_by_check=pre_evidence,
        )
        == inputs
    )

    result_digest = derive_validation_evidence_sha256(
        "results-and-claims-review",
        {"results-review-attestation-bytes": b"same payload"},
    )
    software_digest = derive_validation_evidence_sha256(
        "software-verification",
        {"software-verification-bytes": b"same payload"},
    )
    assert result_digest != software_digest


def test_final_validation_rederives_all_seventeen_checks() -> None:
    pre_evidence = _validation_evidence(include_results=False)
    pre_review = derive_pre_review_validation_inputs(
        run_id=RUN_ID,
        evidence_by_check=pre_evidence,
    )
    final_evidence = _validation_evidence(include_results=True)
    validation = derive_final_replay_validation(
        run_id=RUN_ID,
        evidence_by_check=final_evidence,
        pre_review_inputs=pre_review,
    )

    assert validation.all_checks_passed
    assert len(validation.checks) == 17
    assert tuple(check.check_id for check in validation.checks) == M5_RELEASE_VALIDATION_CHECK_IDS
    assert (
        load_final_replay_validation(
            canonical_json_bytes(validation),
            run_id=RUN_ID,
            evidence_by_check=final_evidence,
            pre_review_inputs=pre_review,
        )
        == validation
    )

    final_evidence["software-verification"]["software-verification-bytes"] = b"changed\n"
    with pytest.raises(ReplayReleaseValidationError, match="reviewed pre-review inputs"):
        load_final_replay_validation(
            canonical_json_bytes(validation),
            run_id=RUN_ID,
            evidence_by_check=final_evidence,
            pre_review_inputs=pre_review,
        )


def test_validation_derivation_rejects_missing_extra_and_stale_authority() -> None:
    evidence = _validation_evidence(include_results=False)
    missing = dict(evidence)
    del missing["intent-freeze"]
    with pytest.raises(ReplayReleaseValidationError, match="exact required check set"):
        derive_pre_review_validation_inputs(run_id=RUN_ID, evidence_by_check=missing)

    intent_parts = dict(evidence["intent-freeze"])
    intent_parts["invented"] = b"valid-looking but unauthorized\n"
    with pytest.raises(ReplayReleaseValidationError, match="missing, extra"):
        derive_validation_evidence_sha256("intent-freeze", intent_parts)

    pre_review = derive_pre_review_validation_inputs(run_id=RUN_ID, evidence_by_check=evidence)
    first = pre_review.checks[0].model_copy(update={"evidence_sha256": "f" * 64})
    stale = pre_review.model_copy(update={"checks": (first, *pre_review.checks[1:])})
    with pytest.raises(ReplayReleaseValidationError, match="named authority"):
        load_pre_review_validation_inputs(
            canonical_json_bytes(stale),
            run_id=RUN_ID,
            evidence_by_check=evidence,
        )
