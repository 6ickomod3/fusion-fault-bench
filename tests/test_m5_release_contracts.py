from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_IDENTITY_SET_SHA256,
    REPLAY_ARTIFACT_PATHS,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_FIGURE_IDS,
    M5_FIGURE_PATHS,
    M5_IMPLEMENTATION_REVIEW_AREAS,
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_PACKAGE_PATHS,
    M5_RELEASE_SIDECAR_INDEXED_PATHS,
    M5_RELEASE_SIDECAR_ROLE_BY_PATH,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
    M5_REVIEW_CANDIDATE_PATHS,
    M5_REVIEW_CANDIDATE_ROLE_BY_PATH,
    M5_SOFTWARE_VERIFICATION_CATEGORIES,
    ReplayFigureMarkV1,
    ReplayFigureSourceBindingV1,
    ReplayFigureSpecV1,
    ReplayImplementationReviewAttestationV1,
    ReplayImplementationReviewDecisionV1,
    ReplayPublicClaimProjectionsV1,
    ReplayReleaseSidecarFileEntryV1,
    ReplayReleaseSidecarIndexV1,
    ReplayResultsReviewAttestationV1,
    ReplayResultsReviewDecisionV1,
    ReplayReviewCandidateFileEntryV1,
    ReplayReviewCandidateIndexV1,
    ReplaySoftwareVerificationV1,
    ReplayValidationInputsV1,
    compute_replay_release_package_sha256,
    compute_replay_release_sidecar_set_sha256,
    compute_replay_review_candidate_sha256,
    replay_release_contract_json_schemas,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_SHA256,
    expected_replay_identities,
    replay_experiment_identity_sha256,
)

_DIGEST = "a" * 64
_GIT_REVISION = "b" * 40

_EXPECTED_CANDIDATE_INDEXED_PATHS = (
    "machine/intent.json",
    "machine/replay-profile-summary.json",
    "machine/descriptor-aggregates.ndjson",
    "machine/persistent-panel-aggregates.ndjson",
    "machine/persistent-panel-crossovers.ndjson",
    "machine/health-panel-aggregates.ndjson",
    "machine/leave-one-cluster-sensitivity.ndjson",
    "machine/repeat-verification.json",
    "machine/figure-records.ndjson",
    "machine/source-member-commitments.ndjson",
    "evidence/release-pipeline-plan.md",
    "evidence/release-pipeline-plan-review.md",
    "evidence/resource-scope-amendment.md",
    "evidence/implementation-review.md",
    "evidence/validation-inputs.json",
    "evidence/implementation-review-attestation.json",
    "evidence/software-verification.json",
    "evidence/privacy-license-attestation.json",
    "figures/m5-persistent-panel-summary.spec.json",
    "figures/m5-persistent-panel-summary.svg",
    "figures/m5-crossovers.spec.json",
    "figures/m5-crossovers.svg",
    "figures/m5-health-transfer.spec.json",
    "figures/m5-health-transfer.svg",
    "figures/m5-descriptor-comparison.spec.json",
    "figures/m5-descriptor-comparison.svg",
    "figures/m5-cluster-sensitivity.spec.json",
    "figures/m5-cluster-sensitivity.svg",
    "presentation/README.md",
    "presentation/claim-evidence.md",
    "presentation/verification.md",
    "presentation/release-summary.json",
    "presentation/public-claim-projections.json",
)

_EXPECTED_CANDIDATE_ROLES = (
    *("reviewed-scientific-aggregate" for _ in range(7)),
    *("reviewed-repeat-or-provenance" for _ in range(3)),
    "frozen-public-methodology",
    "independent-review-evidence",
    "frozen-public-methodology",
    "independent-review-evidence",
    "pre-review-validation-input",
    "independent-review-evidence",
    "pre-review-validation-input",
    "pre-review-validation-input",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    *("reviewed-presentation-template" for _ in range(4)),
    "reviewed-claim-projection",
)

_EXPECTED_SIDECAR_INDEXED_PATHS = (
    "README.md",
    "claim-evidence.md",
    "verification.md",
    "release-summary.json",
    "figures/m5-persistent-panel-summary.spec.json",
    "figures/m5-persistent-panel-summary.svg",
    "figures/m5-crossovers.spec.json",
    "figures/m5-crossovers.svg",
    "figures/m5-health-transfer.spec.json",
    "figures/m5-health-transfer.svg",
    "figures/m5-descriptor-comparison.spec.json",
    "figures/m5-descriptor-comparison.svg",
    "figures/m5-cluster-sensitivity.spec.json",
    "figures/m5-cluster-sensitivity.svg",
    "evidence/release-pipeline-plan.md",
    "evidence/release-pipeline-plan-review.md",
    "evidence/resource-scope-amendment.md",
    "evidence/implementation-review.md",
    "evidence/review-candidate-index.json",
    "evidence/validation-inputs.json",
    "evidence/implementation-review-attestation.json",
    "evidence/software-verification.json",
    "evidence/privacy-license-attestation.json",
    "evidence/public-claim-projections.json",
    "evidence/results-review.md",
    "evidence/results-review-attestation.json",
)

_EXPECTED_SIDECAR_ROLES = (
    *("final-presentation" for _ in range(4)),
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "frozen-public-methodology",
    "independent-review-evidence",
    "frozen-public-methodology",
    "independent-review-evidence",
    "reviewed-candidate-index",
    "pre-review-validation-input",
    "independent-review-evidence",
    "pre-review-validation-input",
    "pre-review-validation-input",
    "reviewed-claim-projection",
    "independent-review-evidence",
    "independent-review-evidence",
)


def _candidate_entry(path: str, ordinal: int) -> dict[str, Any]:
    return {
        "path": path,
        "role": M5_REVIEW_CANDIDATE_ROLE_BY_PATH[path],
        "byte_length": ordinal + 1,
        "sha256": f"{ordinal + 1:064x}",
        "record_count": 1 if path.endswith(".ndjson") else None,
    }


def _candidate_mapping() -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema": "ffb.m5-release-review-candidate-index/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "scientific_git_revision": _GIT_REVISION,
        "lockfile_sha256": "c" * 64,
        "package_version": "0.1.0",
        "run_id": "run-primary",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "primary_local_artifact_sha256": "1" * 64,
        "repeat_local_artifact_sha256": "2" * 64,
        "primary_local_run_sha256": "3" * 64,
        "repeat_local_run_sha256": "4" * 64,
        "results_review_status": "pending",
        "files": tuple(
            _candidate_entry(path, ordinal)
            for ordinal, path in enumerate(M5_REVIEW_CANDIDATE_INDEXED_PATHS)
        ),
    }
    return {**core, "candidate_sha256": compute_replay_review_candidate_sha256(core)}


def _sidecar_entry(path: str, ordinal: int) -> dict[str, Any]:
    return {
        "path": path,
        "role": M5_RELEASE_SIDECAR_ROLE_BY_PATH[path],
        "byte_length": ordinal + 11,
        "sha256": f"{ordinal + 101:064x}",
        "record_count": None,
    }


def _sidecar_mapping() -> dict[str, Any]:
    files = tuple(
        _sidecar_entry(path, ordinal)
        for ordinal, path in enumerate(M5_RELEASE_SIDECAR_INDEXED_PATHS)
    )
    core: dict[str, Any] = {
        "schema": "ffb.m5-release-sidecar-index/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "reviewed_candidate_sha256": "1" * 64,
        "results_review_attestation_sha256": "2" * 64,
        "machine_artifact_sha256": "3" * 64,
        "machine_run_sha256": "4" * 64,
        "scientific_git_revision": _GIT_REVISION,
        "files": files,
        "machine_artifact_byte_length": 1234,
        "indexed_sidecar_payload_byte_length": sum(row["byte_length"] for row in files),
    }
    sidecar_digest = compute_replay_release_sidecar_set_sha256(core)
    return {
        **core,
        "sidecar_set_sha256": sidecar_digest,
        "release_package_sha256": compute_replay_release_package_sha256(
            core["machine_artifact_sha256"],
            sidecar_digest,
        ),
    }


def _finding(
    finding_id: str,
    severity: str = "p2",
    status: str = "unresolved",
) -> dict[str, Any]:
    return {"finding_id": finding_id, "severity": severity, "status": status}


def _implementation_attestation_mapping(
    findings: tuple[dict[str, Any], ...],
    *,
    disposition: str,
) -> dict[str, Any]:
    return {
        "schema": "ffb.m5-implementation-review-attestation/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "implementation_snapshot_sha256": "1" * 64,
        "implementation_snapshot_file_count": 17,
        "review_report_sha256": "2" * 64,
        "reviewer_identity": "independent-reviewer",
        "reviewer_identity_scope": "operator-recorded-not-cryptographically-authenticated",
        "reviewed_areas": M5_IMPLEMENTATION_REVIEW_AREAS,
        "findings": findings,
        "p0_count": sum(row["severity"] == "p0" for row in findings),
        "p1_count": sum(row["severity"] == "p1" for row in findings),
        "p2_count": sum(row["severity"] == "p2" for row in findings),
        "unresolved_finding_ids": tuple(
            row["finding_id"] for row in findings if row["status"] == "unresolved"
        ),
        "disposition": disposition,
    }


def _results_attestation_mapping(
    findings: tuple[dict[str, Any], ...],
    *,
    disposition: str,
) -> dict[str, Any]:
    return {
        "schema": "ffb.m5-results-review-attestation/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "scientific_git_revision": _GIT_REVISION,
        "candidate_sha256": "1" * 64,
        "candidate_index_sha256": "2" * 64,
        "scientific_member_set_sha256": "3" * 64,
        "claim_projection_sha256": "4" * 64,
        "figure_spec_set_sha256": "5" * 64,
        "rendered_figure_set_sha256": "6" * 64,
        "presentation_template_set_sha256": "7" * 64,
        "review_report_sha256": "8" * 64,
        "reviewer_identity": "results-reviewer",
        "reviewer_identity_scope": "operator-recorded-not-cryptographically-authenticated",
        "findings": findings,
        "p0_count": sum(row["severity"] == "p0" for row in findings),
        "p1_count": sum(row["severity"] == "p1" for row in findings),
        "p2_count": sum(row["severity"] == "p2" for row in findings),
        "unresolved_finding_ids": tuple(
            row["finding_id"] for row in findings if row["status"] == "unresolved"
        ),
        "negative_and_undefined_results_reviewed_and_retained": True,
        "limitations_reviewed_and_retained": True,
        "disposition": disposition,
    }


def _software_mapping() -> dict[str, Any]:
    checks = tuple(
        {
            "check_id": f"verify-{category}",
            "category": category,
            "command": ("uv", "run", category),
            "required_test_ids": (f"test-{ordinal}",),
            "exit_status": 0,
            "output_sha256": f"{ordinal + 1:064x}",
            "output_normalization": (
                "stable-command-output-with-runtime-paths-and-durations-removed"
            ),
        }
        for ordinal, category in enumerate(M5_SOFTWARE_VERIFICATION_CATEGORIES)
    )
    return {
        "schema": "ffb.m5-software-verification/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "scientific_git_revision": _GIT_REVISION,
        "lockfile_sha256": "1" * 64,
        "package_version": "0.1.0",
        "implementation_snapshot_sha256": "2" * 64,
        "tooling_revision": _GIT_REVISION,
        "checks": checks,
    }


def _validation_inputs_mapping() -> dict[str, Any]:
    checks = tuple(
        {
            "check_id": check_id,
            "status": "pending" if check_id == "results-and-claims-review" else "ready",
            "passed": None if check_id == "results-and-claims-review" else True,
            "evidence_sha256": None if check_id == "results-and-claims-review" else _DIGEST,
            "evidence_scope": (
                "pending-independent-results-review"
                if check_id == "results-and-claims-review"
                else "direct-content-and-contract-recomputation"
            ),
        }
        for check_id in M5_RELEASE_VALIDATION_CHECK_IDS
    )
    return {
        "schema": "ffb.m5-validation-inputs/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "run_id": "run-primary",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "checks": checks,
    }


def _figure_mark(ordinal: int = 0) -> dict[str, Any]:
    return {
        "mark_ordinal": ordinal,
        "projection_id": f"projection-{ordinal}",
        "source_member": "machine/persistent-panel-crossovers.ndjson",
        "source_kind": "persistent-crossover",
        "source_identifier": f"crossover-{ordinal}",
        "source_record_sha256": _DIGEST,
        "projected_fields": ("estimate",),
        "replay_identity_sha256": None,
    }


def _figure_spec_mapping() -> dict[str, Any]:
    return {
        "schema": "ffb.m5-figure-spec/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "run_id": "run-primary",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "figure_id": "m5-crossovers",
        "figure_kind": "crossovers",
        "figure_file": "figures/m5-crossovers.svg",
        "width_px": 1400,
        "height_px": 900,
        "font_families": ("Arial", "Helvetica", "sans-serif"),
        "colors": ("#112233", "#AABBCC"),
        "units": ("m^2",),
        "axis_facets": ("experiment",),
        "caption_boundary": "registry-projected-values-and-literal-statuses-only",
        "renderer_id": "ffb.m5-deterministic-svg/v1",
        "marks": (_figure_mark(),),
        "tracked_aggregate_terms": "CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms",
        "non_endorsement_footer": (
            "CC BY-NC-SA 4.0 plus Motional Dataset Terms; attribution required; no endorsement."
        ),
    }


def _descriptor_source_binding_mapping() -> dict[str, Any]:
    return {
        "schema": "ffb.replay-figure-source-binding/v1",
        "run_id": "run-primary",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "figure_id": "m5-descriptor-comparison",
        "figure_kind": "descriptor-comparison",
        "mark_ordinal": 0,
        "source_kind": "descriptor-aggregate",
        "source_identifier": "descriptor-0",
        "source_record_sha256": "1" * 64,
        "identity": None,
        "replay_identity_sha256": None,
        "figure_spec_sha256": "2" * 64,
        "rendered_svg_path": "figures/m5-descriptor-comparison.svg",
        "rendered_svg_sha256": "3" * 64,
        "rendered_svg_byte_length": 256,
        "tracked_aggregate_terms": "CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms",
    }


def _identity_source_binding_mapping() -> dict[str, Any]:
    identity = expected_replay_identities()[0]
    return {
        "schema": "ffb.replay-figure-source-binding/v1",
        "run_id": "run-primary",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "figure_id": "m5-persistent-panel-summary",
        "figure_kind": "persistent-panel-summary",
        "mark_ordinal": 0,
        "source_kind": "persistent-aggregate",
        "source_identifier": "aggregate-0",
        "source_record_sha256": "1" * 64,
        "identity": identity,
        "replay_identity_sha256": replay_experiment_identity_sha256(identity),
        "figure_spec_sha256": "2" * 64,
        "rendered_svg_path": "figures/m5-persistent-panel-summary.svg",
        "rendered_svg_sha256": "3" * 64,
        "rendered_svg_byte_length": 256,
        "tracked_aggregate_terms": "CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms",
    }


def _projection(
    group: str,
    ordinal: int,
    *,
    hypothesis_id: str | None = None,
    finalization: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "ffb.m5-public-claim-projection/v1",
        "projection_id": f"projection-{group}-{ordinal}",
        "public_claim_id": f"claim-{group}-{ordinal}",
        "projection_group": group,
        "source_member": (
            "release-sidecar-index.json"
            if finalization
            else "machine/persistent-panel-aggregates.ndjson"
        ),
        "source_kind": "release-index" if finalization else "persistent-aggregate",
        "source_identifier": f"source-{group}-{ordinal}",
        "source_record_sha256": None if finalization else _DIGEST,
        "selector_fields": (),
        "projected_fields": (
            {
                "field": "value",
                "value": None if finalization else 1.0,
                "rendering": "exact-integer" if finalization else ".6g",
            },
        ),
        "unit": "unitless",
        "status_behavior": (
            "finalization-null-then-exact-integer" if finalization else "defined-numeric"
        ),
        "figure_ids": (),
        "hypothesis_id": hypothesis_id,
    }


def _claim_registry_mapping() -> dict[str, Any]:
    projections: list[dict[str, Any]] = []
    groups = (
        ("persistent-panel", 100),
        ("crossovers", 10),
        ("health-transfer", 43),
        ("cluster-sensitivity", 286),
        ("descriptor-comparison", 67),
        ("resources", 2),
    )
    hypothesis_ordinal = 0
    for group, count in groups:
        for ordinal in range(count):
            hypothesis_id = None
            if hypothesis_ordinal < 44:
                hypothesis_id = f"hypothesis-{hypothesis_ordinal}"
                hypothesis_ordinal += 1
            projections.append(_projection(group, ordinal, hypothesis_id=hypothesis_id))
    projections.append(_projection("finalization-metadata", 0, finalization=True))
    return {
        "schema": "ffb.m5-public-claim-projections/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "run_id": "run-primary",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "persistent_hypothesis_count": 33,
        "persistent_hypothesis_coordinate_set_sha256": "1" * 64,
        "health_hypothesis_count": 11,
        "health_hypothesis_coordinate_set_sha256": "2" * 64,
        "persistent_figure_projection_count": 100,
        "crossover_projection_count": 10,
        "health_figure_projection_count": 43,
        "sensitivity_source_count": 26,
        "distinct_log_group_count": 1,
        "sensitivity_projection_count": 286,
        "descriptor_figure_projection_count": 67,
        "resource_record_count": 2,
        "projections": tuple(projections),
    }


def test_candidate_and_package_paths_have_exact_arithmetic_order_and_roles() -> None:
    assert M5_REVIEW_CANDIDATE_INDEXED_PATHS == _EXPECTED_CANDIDATE_INDEXED_PATHS
    assert (
        "candidate-index.json",
        *_EXPECTED_CANDIDATE_INDEXED_PATHS,
    ) == M5_REVIEW_CANDIDATE_PATHS
    assert (len(M5_REVIEW_CANDIDATE_PATHS), len(M5_REVIEW_CANDIDATE_INDEXED_PATHS)) == (
        34,
        33,
    )
    candidate_roles = tuple(
        M5_REVIEW_CANDIDATE_ROLE_BY_PATH[path] for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS
    )
    assert candidate_roles == _EXPECTED_CANDIDATE_ROLES
    assert Counter(candidate_roles) == {
        "reviewed-scientific-aggregate": 7,
        "reviewed-repeat-or-provenance": 3,
        "pre-review-validation-input": 3,
        "frozen-public-methodology": 2,
        "independent-review-evidence": 3,
        "deterministic-figure-spec": 5,
        "deterministic-rendered-figure": 5,
        "reviewed-presentation-template": 4,
        "reviewed-claim-projection": 1,
    }

    assert M5_RELEASE_SIDECAR_INDEXED_PATHS == _EXPECTED_SIDECAR_INDEXED_PATHS
    assert (len(REPLAY_ARTIFACT_PATHS), len(M5_RELEASE_SIDECAR_INDEXED_PATHS)) == (14, 26)
    assert (
        *(f"artifact/{path}" for path in REPLAY_ARTIFACT_PATHS),
        *_EXPECTED_SIDECAR_INDEXED_PATHS,
        "release-sidecar-index.json",
    ) == M5_RELEASE_PACKAGE_PATHS
    assert len(M5_RELEASE_PACKAGE_PATHS) == len(set(M5_RELEASE_PACKAGE_PATHS)) == 41
    sidecar_roles = tuple(
        M5_RELEASE_SIDECAR_ROLE_BY_PATH[path] for path in M5_RELEASE_SIDECAR_INDEXED_PATHS
    )
    assert sidecar_roles == _EXPECTED_SIDECAR_ROLES
    assert Counter(sidecar_roles) == {
        "final-presentation": 4,
        "deterministic-figure-spec": 5,
        "deterministic-rendered-figure": 5,
        "frozen-public-methodology": 2,
        "independent-review-evidence": 5,
        "reviewed-candidate-index": 1,
        "pre-review-validation-input": 3,
        "reviewed-claim-projection": 1,
    }
    assert M5_FIGURE_IDS == (
        "m5-persistent-panel-summary",
        "m5-crossovers",
        "m5-health-transfer",
        "m5-descriptor-comparison",
        "m5-cluster-sensitivity",
    )
    assert len(M5_FIGURE_PATHS) == 10
    assert len(M5_PRESENTATION_PLACEHOLDERS) == 4


def test_candidate_and_sidecar_digests_are_deterministic_and_domain_separated() -> None:
    core = {"schema": "fixture/v1", "value": 7}
    candidate_digest = compute_replay_review_candidate_sha256(core)

    assert candidate_digest == compute_replay_review_candidate_sha256(dict(reversed(core.items())))
    assert candidate_digest != compute_replay_release_sidecar_set_sha256(core)
    assert len(candidate_digest) == 64

    package_digest = compute_replay_release_package_sha256("1" * 64, "2" * 64)
    assert package_digest == compute_replay_release_package_sha256("1" * 64, "2" * 64)
    assert package_digest != compute_replay_release_package_sha256("2" * 64, "1" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        compute_replay_release_package_sha256("A" * 64, "2" * 64)


def test_candidate_index_accepts_exact_files_and_recomputed_digest() -> None:
    candidate = ReplayReviewCandidateIndexV1.model_validate(_candidate_mapping())

    assert tuple(row.path for row in candidate.files) == M5_REVIEW_CANDIDATE_INDEXED_PATHS
    assert sum(row.record_count is not None for row in candidate.files) == 7
    assert candidate.candidate_sha256 == compute_replay_review_candidate_sha256(
        candidate.model_dump(mode="json", by_alias=True, exclude={"candidate_sha256"})
    )


def test_candidate_entries_and_index_reject_role_order_digest_and_nonindependence() -> None:
    wrong_role = _candidate_entry(M5_REVIEW_CANDIDATE_INDEXED_PATHS[0], 0)
    wrong_role["role"] = "reviewed-claim-projection"
    with pytest.raises(ValidationError, match="role disagrees"):
        ReplayReviewCandidateFileEntryV1.model_validate(wrong_role)

    wrong_record_count = _candidate_entry(M5_REVIEW_CANDIDATE_INDEXED_PATHS[0], 0)
    wrong_record_count["record_count"] = 1
    with pytest.raises(ValidationError, match="exactly for NDJSON"):
        ReplayReviewCandidateFileEntryV1.model_validate(wrong_record_count)

    wrong_digest = _candidate_mapping()
    wrong_digest["candidate_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="semantic digest"):
        ReplayReviewCandidateIndexV1.model_validate(wrong_digest)

    wrong_order = _candidate_mapping()
    files = list(wrong_order["files"])
    files[0], files[1] = files[1], files[0]
    wrong_order["files"] = tuple(files)
    with pytest.raises(ValidationError, match="frozen independent inputs"):
        ReplayReviewCandidateIndexV1.model_validate(wrong_order)

    repeated_artifact = _candidate_mapping()
    repeated_artifact["repeat_local_artifact_sha256"] = repeated_artifact[
        "primary_local_artifact_sha256"
    ]
    with pytest.raises(ValidationError, match="frozen independent inputs"):
        ReplayReviewCandidateIndexV1.model_validate(repeated_artifact)


def test_sidecar_index_accepts_exact_files_and_package_binding() -> None:
    sidecars = ReplayReleaseSidecarIndexV1.model_validate(_sidecar_mapping())

    assert tuple(row.path for row in sidecars.files) == M5_RELEASE_SIDECAR_INDEXED_PATHS
    assert sidecars.indexed_sidecar_payload_byte_length == sum(
        row.byte_length for row in sidecars.files
    )
    assert sidecars.release_package_sha256 == compute_replay_release_package_sha256(
        sidecars.machine_artifact_sha256,
        sidecars.sidecar_set_sha256,
    )


def test_sidecar_entries_and_index_reject_role_order_bytes_and_digest_tampering() -> None:
    wrong_role = _sidecar_entry(M5_RELEASE_SIDECAR_INDEXED_PATHS[0], 0)
    wrong_role["role"] = "reviewed-candidate-index"
    with pytest.raises(ValidationError, match="role disagrees"):
        ReplayReleaseSidecarFileEntryV1.model_validate(wrong_role)

    wrong_order = _sidecar_mapping()
    files = list(wrong_order["files"])
    files[-1], files[-2] = files[-2], files[-1]
    wrong_order["files"] = tuple(files)
    with pytest.raises(ValidationError, match="fixed path order"):
        ReplayReleaseSidecarIndexV1.model_validate(wrong_order)

    wrong_bytes = _sidecar_mapping()
    wrong_bytes["indexed_sidecar_payload_byte_length"] += 1
    with pytest.raises(ValidationError, match="exact member sum"):
        ReplayReleaseSidecarIndexV1.model_validate(wrong_bytes)

    wrong_digest = _sidecar_mapping()
    wrong_digest["sidecar_set_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="sidecar or package digest"):
        ReplayReleaseSidecarIndexV1.model_validate(wrong_digest)


def test_implementation_review_decision_and_attestation_enforce_scope_counts_and_blockers() -> None:
    decision = ReplayImplementationReviewDecisionV1.model_validate(
        {
            "schema": "ffb.m5-implementation-review-decision/v1",
            "reviewer_identity": "reviewer",
            "reviewed_areas": M5_IMPLEMENTATION_REVIEW_AREAS,
            "findings": (_finding("finding-p2"),),
            "disposition": "pass-with-nonblocking-findings",
        }
    )
    assert decision.reviewed_areas == M5_IMPLEMENTATION_REVIEW_AREAS

    wrong_scope = decision.model_dump(mode="python", by_alias=True)
    wrong_scope["reviewed_areas"] = M5_IMPLEMENTATION_REVIEW_AREAS[:-1]
    with pytest.raises(ValidationError, match="areas are incomplete"):
        ReplayImplementationReviewDecisionV1.model_validate(wrong_scope)

    duplicate = decision.model_dump(mode="python", by_alias=True)
    duplicate["findings"] = (duplicate["findings"][0], duplicate["findings"][0])
    with pytest.raises(ValidationError, match="finding IDs must be unique"):
        ReplayImplementationReviewDecisionV1.model_validate(duplicate)

    nonblocking = _implementation_attestation_mapping(
        (_finding("finding-p2"),),
        disposition="pass-with-nonblocking-findings",
    )
    attestation = ReplayImplementationReviewAttestationV1.model_validate(nonblocking)
    assert (attestation.p0_count, attestation.p1_count, attestation.p2_count) == (0, 0, 1)
    assert "scientific_git_revision" not in attestation.model_dump(mode="json", by_alias=True)
    assert attestation.unresolved_finding_ids == ("finding-p2",)

    wrong_count = copy.deepcopy(nonblocking)
    wrong_count["p2_count"] = 0
    with pytest.raises(ValidationError, match="internally inconsistent"):
        ReplayImplementationReviewAttestationV1.model_validate(wrong_count)

    unresolved_blocker = _implementation_attestation_mapping(
        (_finding("finding-p1", severity="p1"),),
        disposition="pass",
    )
    with pytest.raises(ValidationError, match="retains a blocker"):
        ReplayImplementationReviewAttestationV1.model_validate(unresolved_blocker)
    unresolved_blocker["disposition"] = "block"
    ReplayImplementationReviewAttestationV1.model_validate(unresolved_blocker)


def test_results_review_decision_and_attestation_enforce_counts_and_blockers() -> None:
    decision_mapping = {
        "schema": "ffb.m5-results-review-decision/v1",
        "reviewer_identity": "results-reviewer",
        "findings": (_finding("result-p2"),),
        "negative_and_undefined_results_reviewed_and_retained": True,
        "limitations_reviewed_and_retained": True,
        "disposition": "pass-with-nonblocking-findings",
    }
    decision = ReplayResultsReviewDecisionV1.model_validate(decision_mapping)
    assert decision.negative_and_undefined_results_reviewed_and_retained

    duplicate = copy.deepcopy(decision_mapping)
    duplicate["findings"] = (duplicate["findings"][0], duplicate["findings"][0])
    with pytest.raises(ValidationError, match="finding IDs must be unique"):
        ReplayResultsReviewDecisionV1.model_validate(duplicate)

    nonblocking = _results_attestation_mapping(
        (_finding("result-p2"),),
        disposition="pass-with-nonblocking-findings",
    )
    attestation = ReplayResultsReviewAttestationV1.model_validate(nonblocking)
    assert (attestation.p0_count, attestation.p1_count, attestation.p2_count) == (0, 0, 1)

    wrong_count = copy.deepcopy(nonblocking)
    wrong_count["unresolved_finding_ids"] = ()
    with pytest.raises(ValidationError, match="internally inconsistent"):
        ReplayResultsReviewAttestationV1.model_validate(wrong_count)

    unresolved_blocker = _results_attestation_mapping(
        (_finding("result-p0", severity="p0"),),
        disposition="pass",
    )
    with pytest.raises(ValidationError, match="retains a blocker"):
        ReplayResultsReviewAttestationV1.model_validate(unresolved_blocker)
    unresolved_blocker["disposition"] = "block"
    ReplayResultsReviewAttestationV1.model_validate(unresolved_blocker)


def test_software_verification_requires_all_seven_ordered_categories() -> None:
    software = ReplaySoftwareVerificationV1.model_validate(_software_mapping())

    assert tuple(row.category for row in software.checks) == (
        "format",
        "lint",
        "type-check",
        "unit-property-oracle-integration",
        "build",
        "wheel-smoke",
        "privacy",
    )

    wrong_order = _software_mapping()
    checks = list(wrong_order["checks"])
    checks[0], checks[1] = checks[1], checks[0]
    wrong_order["checks"] = tuple(checks)
    with pytest.raises(ValidationError, match="categories are incomplete or reordered"):
        ReplaySoftwareVerificationV1.model_validate(wrong_order)

    duplicate = _software_mapping()
    checks = list(duplicate["checks"])
    checks[1]["check_id"] = checks[0]["check_id"]
    duplicate["checks"] = tuple(checks)
    with pytest.raises(ValidationError, match="check IDs must be unique"):
        ReplaySoftwareVerificationV1.model_validate(duplicate)

    wrong_revision = _software_mapping()
    wrong_revision["tooling_revision"] = "c" * 40
    with pytest.raises(ValidationError, match="must equal scientific HEAD"):
        ReplaySoftwareVerificationV1.model_validate(wrong_revision)


def test_validation_inputs_require_exact_17_slots_and_one_pending_results_review() -> None:
    validation = ReplayValidationInputsV1.model_validate(_validation_inputs_mapping())

    assert len(validation.checks) == len(M5_RELEASE_VALIDATION_CHECK_IDS) == 17
    assert tuple(row.check_id for row in validation.checks) == M5_RELEASE_VALIDATION_CHECK_IDS
    pending = tuple(row for row in validation.checks if row.status == "pending")
    assert len(pending) == 1
    assert pending[0].check_id == "results-and-claims-review"
    assert pending[0].passed is None and pending[0].evidence_sha256 is None

    wrong_pending = _validation_inputs_mapping()
    checks = list(wrong_pending["checks"])
    checks[0] = {
        **checks[0],
        "status": "pending",
        "passed": None,
        "evidence_sha256": None,
        "evidence_scope": "pending-independent-results-review",
    }
    wrong_pending["checks"] = tuple(checks)
    with pytest.raises(ValidationError, match="only results-and-claims-review"):
        ReplayValidationInputsV1.model_validate(wrong_pending)

    finalized_too_early = _validation_inputs_mapping()
    checks = list(finalized_too_early["checks"])
    results_index = M5_RELEASE_VALIDATION_CHECK_IDS.index("results-and-claims-review")
    checks[results_index] = {
        **checks[results_index],
        "status": "ready",
        "passed": True,
        "evidence_sha256": _DIGEST,
        "evidence_scope": "direct-content-and-contract-recomputation",
    }
    finalized_too_early["checks"] = tuple(checks)
    with pytest.raises(ValidationError, match="only results-and-claims-review"):
        ReplayValidationInputsV1.model_validate(finalized_too_early)

    wrong_order = _validation_inputs_mapping()
    checks = list(wrong_order["checks"])
    checks[0], checks[1] = checks[1], checks[0]
    wrong_order["checks"] = tuple(checks)
    with pytest.raises(ValidationError, match="frozen M5 slots"):
        ReplayValidationInputsV1.model_validate(wrong_order)


def test_figure_spec_requires_frozen_metadata_and_contiguous_unique_marks() -> None:
    figure = ReplayFigureSpecV1.model_validate(_figure_spec_mapping())

    assert (figure.width_px, figure.height_px) == (1400, 900)
    assert figure.font_families == ("Arial", "Helvetica", "sans-serif")

    wrong_dimensions = _figure_spec_mapping()
    wrong_dimensions["width_px"] = 1401
    with pytest.raises(ValidationError, match="frozen rendering contract"):
        ReplayFigureSpecV1.model_validate(wrong_dimensions)

    wrong_ordinal = _figure_spec_mapping()
    wrong_ordinal["marks"] = (_figure_mark(1),)
    with pytest.raises(ValidationError, match="frozen rendering contract"):
        ReplayFigureSpecV1.model_validate(wrong_ordinal)

    duplicate_projection = _figure_spec_mapping()
    second = _figure_mark(1)
    second["projection_id"] = "projection-0"
    duplicate_projection["marks"] = (_figure_mark(), second)
    with pytest.raises(ValidationError, match="frozen rendering contract"):
        ReplayFigureSpecV1.model_validate(duplicate_projection)


def test_figure_source_binding_separates_descriptor_and_identity_bound_sources() -> None:
    descriptor = ReplayFigureSourceBindingV1.model_validate(_descriptor_source_binding_mapping())
    assert descriptor.identity is None and descriptor.replay_identity_sha256 is None

    identity_bound = ReplayFigureSourceBindingV1.model_validate(_identity_source_binding_mapping())
    assert identity_bound.identity is not None
    assert identity_bound.replay_identity_sha256 == replay_experiment_identity_sha256(
        identity_bound.identity
    )

    descriptor_with_identity = _descriptor_source_binding_mapping()
    identity = expected_replay_identities()[0]
    descriptor_with_identity["identity"] = identity
    descriptor_with_identity["replay_identity_sha256"] = replay_experiment_identity_sha256(identity)
    with pytest.raises(ValidationError, match="invalid global or source shape"):
        ReplayFigureSourceBindingV1.model_validate(descriptor_with_identity)

    identity_without_binding = _identity_source_binding_mapping()
    identity_without_binding["identity"] = None
    identity_without_binding["replay_identity_sha256"] = None
    with pytest.raises(ValidationError, match="invalid global or source shape"):
        ReplayFigureSourceBindingV1.model_validate(identity_without_binding)

    wrong_identity_digest = _identity_source_binding_mapping()
    wrong_identity_digest["replay_identity_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="invalid replay identity digest"):
        ReplayFigureSourceBindingV1.model_validate(wrong_identity_digest)


def test_claim_projection_registry_enforces_all_group_and_hypothesis_counts() -> None:
    mapping = _claim_registry_mapping()
    registry = ReplayPublicClaimProjectionsV1.model_validate(mapping)

    assert len(registry.projections) == 509
    assert Counter(row.projection_group for row in registry.projections) == {
        "persistent-panel": 100,
        "crossovers": 10,
        "health-transfer": 43,
        "cluster-sensitivity": 286,
        "descriptor-comparison": 67,
        "resources": 2,
        "finalization-metadata": 1,
    }
    assert sum(row.hypothesis_id is not None for row in registry.projections) == 44

    wrong_group_count = copy.deepcopy(mapping)
    projections = list(wrong_group_count["projections"])
    projections[0]["projection_group"] = "resources"
    wrong_group_count["projections"] = tuple(projections)
    with pytest.raises(ValidationError, match="group count"):
        ReplayPublicClaimProjectionsV1.model_validate(wrong_group_count)

    wrong_sensitivity_arithmetic = copy.deepcopy(mapping)
    wrong_sensitivity_arithmetic["sensitivity_projection_count"] = 287
    with pytest.raises(ValidationError, match="incomplete or duplicated"):
        ReplayPublicClaimProjectionsV1.model_validate(wrong_sensitivity_arithmetic)

    missing_hypothesis = copy.deepcopy(mapping)
    projections = list(missing_hypothesis["projections"])
    projections[0]["hypothesis_id"] = None
    missing_hypothesis["projections"] = tuple(projections)
    with pytest.raises(ValidationError, match="all 44 hypotheses"):
        ReplayPublicClaimProjectionsV1.model_validate(missing_hypothesis)


def test_release_contract_schema_exports_are_stable_and_strict() -> None:
    schemas = replay_release_contract_json_schemas()

    assert tuple(schemas) == (
        "replay-review-candidate-index",
        "replay-implementation-review-attestation",
        "replay-software-verification",
        "replay-privacy-license-attestation",
        "replay-validation-inputs",
        "replay-public-claim-projections",
        "replay-figure-spec",
        "replay-figure-source-binding",
        "replay-results-review-attestation",
        "replay-release-sidecar-index",
    )
    assert schemas == replay_release_contract_json_schemas()
    assert all(schema["additionalProperties"] is False for schema in schemas.values())


def test_strict_contracts_reject_bool_values_for_integer_fields() -> None:
    candidate_entry = _candidate_entry(M5_REVIEW_CANDIDATE_INDEXED_PATHS[0], 0)
    candidate_entry["byte_length"] = True
    with pytest.raises(ValidationError):
        ReplayReviewCandidateFileEntryV1.model_validate(candidate_entry)

    mark = _figure_mark()
    mark["mark_ordinal"] = False
    with pytest.raises(ValidationError):
        ReplayFigureMarkV1.model_validate(mark)
