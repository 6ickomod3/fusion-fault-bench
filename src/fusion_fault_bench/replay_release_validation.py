"""Canonical review, software, privacy, and validation authority for M5.

The functions here operate only on bounded synthetic or aggregate evidence.
They never open a dataset and never infer a reviewer's disposition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ValidationError

from fusion_fault_bench.artifacts import canonical_json_bytes, strict_json_object_body
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    M5_TRACKED_AGGREGATE_TERMS,
    ReplayValidationCheckV1,
    ReplayValidationV1,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_IMPLEMENTATION_REVIEW_AREAS,
    M5_SOFTWARE_VERIFICATION_CATEGORIES,
    ReplayImplementationReviewAttestationV1,
    ReplayImplementationReviewDecisionV1,
    ReplayPrivacyLicenseAttestationV1,
    ReplayResultsReviewAttestationV1,
    ReplayResultsReviewDecisionV1,
    ReplayReviewFindingV1,
    ReplaySoftwareVerificationCheckV1,
    ReplaySoftwareVerificationV1,
    ReplayValidationEvidenceInputV1,
    ReplayValidationInputsV1,
)
from fusion_fault_bench.contracts.replay_v1 import M5_REPLAY_INTENT_SHA256
from fusion_fault_bench.replay_release_authority import ImplementationSnapshot

M5_REVIEWER_IDENTITY_SCOPE = "operator-recorded-not-cryptographically-authenticated"
M5_VALIDATION_PART_MAX_BYTES = 50 * 1024 * 1024
M5_VALIDATION_TOTAL_MAX_BYTES = 128 * 1024 * 1024

M5_TRANSFORM_TIMING_ORACLE_TEST_IDS = (
    "rigid-transform-oracle",
    "camera-timing-oracle",
)
M5_ELIGIBILITY_CAUSALITY_TEST_IDS = (
    "base-support-gate",
    "paired-observation-gate",
    "calibration-application-gate",
    "timestamp-application-gate",
    "fault-mutation-causality",
    "dropout-nesting-derivation",
)
M5_HEALTH_LEAKAGE_TEST_IDS = (
    "health-evidence-pre-update",
    "future-mutation-invariance",
    "prohibited-health-feature-rejection",
)
M5_LOCAL_READ_ACCOUNTING_TEST_IDS = (
    "local-read-accounting",
    "role-aware-public-byte-scan",
)
M5_WHEEL_SMOKE_TEST_IDS = ("built-wheel-replay-release-smoke",)

M5_SOFTWARE_VERIFICATION_CHECK_IDS = (
    "ruff-format",
    "ruff-lint",
    "pyright",
    "pytest-release-authority",
    "distribution-build",
    "built-wheel-smoke",
    "privacy-audit",
)
M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ruff-format": (),
        "ruff-lint": (),
        "pyright": (),
        "pytest-release-authority": (
            *M5_TRANSFORM_TIMING_ORACLE_TEST_IDS,
            *M5_ELIGIBILITY_CAUSALITY_TEST_IDS,
            *M5_HEALTH_LEAKAGE_TEST_IDS,
        ),
        "distribution-build": (),
        "built-wheel-smoke": M5_WHEEL_SMOKE_TEST_IDS,
        "privacy-audit": M5_LOCAL_READ_ACCOUNTING_TEST_IDS,
    }
)

M5_VALIDATION_AUTHORITY_PARTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "intent-freeze": (
            "frozen-intent-bytes",
            "frozen-intent-canonical-digest",
            "release-pipeline-plan-bytes",
            "release-pipeline-plan-review-bytes",
            "resource-scope-amendment-bytes",
        ),
        "fixed-scene-population": (
            "profile-scene-and-experiment-counts",
            "replay-identity-set-commitment",
            "persistent-selector-set-commitment",
            "health-selector-set-commitment",
        ),
        "base-support": (
            "profile-base-support-attestation",
            "authenticated-descriptor-commitments",
            "authenticated-support-commitments",
        ),
        "health-schedules": (
            "profile-schedule-attestation",
            "health-selector-set-commitment",
            "health-coordinate-set-commitment",
        ),
        "transform-and-timing-oracles": (
            "software-transform-and-timing-oracle-entries",
            "implementation-review-attestation-bytes",
        ),
        "eligibility-and-fault-causality": (
            "software-eligibility-and-causality-entries",
            "dropout-nesting-evidence",
        ),
        "health-feature-leakage": ("software-health-leakage-entries",),
        "persistent-panel-completeness": (
            "persistent-aggregate-bytes",
            "persistent-71-selector-commitment",
            "persistent-464-coordinate-commitment",
            "persistent-33-claim-coordinate-commitment",
        ),
        "health-panel-completeness": (
            "health-aggregate-bytes",
            "health-43-selector-commitment",
            "health-14988-coordinate-commitment",
            "health-11-claim-coordinate-commitment",
        ),
        "scene-bootstrap-and-cluster-sensitivity": (
            "aggregate-bootstrap-fields",
            "complete-cluster-sensitivity-bytes",
        ),
        "repeat-scientific-members": (
            "repeat-verification-bytes",
            "ordered-source-member-commitments-bytes",
        ),
        "cpu-and-memory-caps": ("ordered-primary-repeat-resource-evidence",),
        "no-raw-payload-reads": (
            "profile-zero-read-field",
            "local-read-accounting-attestation",
            "privacy-license-attestation-bytes",
        ),
        "privacy-and-dataset-license": (
            "deterministic-candidate-scan",
            "fixed-aggregate-terms",
            "attribution-and-non-endorsement-fields",
            "privacy-license-attestation-bytes",
        ),
        "implementation-review": (
            "implementation-review-report-bytes",
            "implementation-review-attestation-bytes",
        ),
        "results-and-claims-review": ("results-review-attestation-bytes",),
        "software-verification": ("software-verification-bytes",),
    }
)

M5_VALIDATION_EVIDENCE_DOMAINS: Mapping[str, bytes] = MappingProxyType(
    {
        check_id: f"fusion-fault-bench/m5-validation/{check_id}/v1\x00".encode("ascii")
        for check_id in M5_RELEASE_VALIDATION_CHECK_IDS
    }
)

_OPERATOR_ATTESTED_CHECKS = frozenset(
    {
        "transform-and-timing-oracles",
        "eligibility-and-fault-causality",
        "health-feature-leakage",
        "cpu-and-memory-caps",
        "no-raw-payload-reads",
        "privacy-and-dataset-license",
        "implementation-review",
        "results-and-claims-review",
        "software-verification",
    }
)


class ReplayReleaseValidationError(ValueError):
    """M5 release evidence failed canonical or authority validation."""


@dataclass(frozen=True, slots=True)
class ResultsReviewBindings:
    """Candidate-derived digests that an independent results review binds."""

    candidate_sha256: str
    candidate_index_sha256: str
    scientific_member_set_sha256: str
    claim_projection_sha256: str
    figure_spec_set_sha256: str
    rendered_figure_set_sha256: str
    presentation_template_set_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("candidate", self.candidate_sha256),
            ("candidate index", self.candidate_index_sha256),
            ("scientific member set", self.scientific_member_set_sha256),
            ("claim projection", self.claim_projection_sha256),
            ("figure spec set", self.figure_spec_set_sha256),
            ("rendered figure set", self.rendered_figure_set_sha256),
            ("presentation template set", self.presentation_template_set_sha256),
        ):
            _require_digest(value, label=label)


def _require_digest(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReplayReleaseValidationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _load_canonical_model[ModelT: BaseModel](
    value: bytes,
    *,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    try:
        body = strict_json_object_body(value, label=label)
        model = model_type.model_validate_json(body)
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseValidationError(f"{label} violates its strict contract") from error
    if canonical_json_bytes(model) != value:
        raise ReplayReleaseValidationError(f"{label} is not canonical JSON")
    return model


def _report_sha256(value: bytes, *, label: str) -> str:
    if (
        not value
        or len(value) > 1024 * 1024
        or value.startswith(b"\xef\xbb\xbf")
        or b"\r" in value
        or b"\x00" in value
        or not value.endswith(b"\n")
    ):
        raise ReplayReleaseValidationError(f"{label} is not bounded canonical UTF-8 text")
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReplayReleaseValidationError(
            f"{label} is not bounded canonical UTF-8 text"
        ) from error
    return hashlib.sha256(value).hexdigest()


def _review_counts(
    findings: Sequence[ReplayReviewFindingV1],
) -> tuple[int, int, int, tuple[str, ...]]:
    return (
        sum(row.severity == "p0" for row in findings),
        sum(row.severity == "p1" for row in findings),
        sum(row.severity == "p2" for row in findings),
        tuple(row.finding_id for row in findings if row.status == "unresolved"),
    )


def _require_release_permitting(
    findings: Sequence[ReplayReviewFindingV1],
    disposition: str,
    *,
    label: str,
) -> None:
    blocking = any(row.status == "unresolved" and row.severity in {"p0", "p1"} for row in findings)
    if disposition not in {"pass", "pass-with-nonblocking-findings"} or blocking:
        raise ReplayReleaseValidationError(f"{label} does not permit release")


def load_implementation_review_decision(
    value: bytes,
) -> ReplayImplementationReviewDecisionV1:
    """Load the reviewer-authored implementation decision without changing it."""

    return _load_canonical_model(
        value,
        model_type=ReplayImplementationReviewDecisionV1,
        label="implementation review decision",
    )


def build_implementation_review_attestation(
    decision: ReplayImplementationReviewDecisionV1,
    *,
    review_report: bytes,
    snapshot: ImplementationSnapshot,
) -> ReplayImplementationReviewAttestationV1:
    """Canonicalize, but never choose, the implementation-review disposition."""

    if decision.reviewed_areas != M5_IMPLEMENTATION_REVIEW_AREAS:
        raise ReplayReleaseValidationError("implementation review areas are incomplete")
    p0, p1, p2, unresolved = _review_counts(decision.findings)
    try:
        return ReplayImplementationReviewAttestationV1(
            schema="ffb.m5-implementation-review-attestation/v1",
            release_id=M5_REPLAY_RELEASE_ID,
            implementation_snapshot_sha256=snapshot.sha256,
            implementation_snapshot_file_count=snapshot.file_count,
            review_report_sha256=_report_sha256(
                review_report,
                label="implementation review report",
            ),
            reviewer_identity=decision.reviewer_identity,
            reviewer_identity_scope=M5_REVIEWER_IDENTITY_SCOPE,
            reviewed_areas=decision.reviewed_areas,
            findings=decision.findings,
            p0_count=p0,
            p1_count=p1,
            p2_count=p2,
            unresolved_finding_ids=unresolved,
            disposition=decision.disposition,
        )
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseValidationError(
            "implementation review decision cannot be canonicalized"
        ) from error


def load_implementation_review_attestation(
    value: bytes,
    *,
    review_report: bytes,
    snapshot: ImplementationSnapshot,
    require_release_permitting: bool = True,
) -> ReplayImplementationReviewAttestationV1:
    """Strictly reload and bind an implementation-review attestation."""

    attestation = _load_canonical_model(
        value,
        model_type=ReplayImplementationReviewAttestationV1,
        label="implementation review attestation",
    )
    if (
        attestation.release_id != M5_REPLAY_RELEASE_ID
        or attestation.implementation_snapshot_sha256 != snapshot.sha256
        or attestation.implementation_snapshot_file_count != snapshot.file_count
        or attestation.review_report_sha256
        != _report_sha256(review_report, label="implementation review report")
    ):
        raise ReplayReleaseValidationError(
            "implementation review attestation has stale source or report authority"
        )
    if require_release_permitting:
        _require_release_permitting(
            attestation.findings,
            attestation.disposition,
            label="implementation review attestation",
        )
    return attestation


def load_results_review_decision(value: bytes) -> ReplayResultsReviewDecisionV1:
    """Load the reviewer-authored results decision without changing it."""

    return _load_canonical_model(
        value,
        model_type=ReplayResultsReviewDecisionV1,
        label="results review decision",
    )


def build_results_review_attestation(
    decision: ReplayResultsReviewDecisionV1,
    *,
    review_report: bytes,
    scientific_git_revision: str,
    bindings: ResultsReviewBindings,
) -> ReplayResultsReviewAttestationV1:
    """Bind a reviewer-authored disposition to the exact immutable candidate."""

    if not (
        decision.negative_and_undefined_results_reviewed_and_retained
        and decision.limitations_reviewed_and_retained
    ):
        raise ReplayReleaseValidationError(
            "results review decision cannot be canonicalized without retention declarations"
        )
    p0, p1, p2, unresolved = _review_counts(decision.findings)
    try:
        return ReplayResultsReviewAttestationV1(
            schema="ffb.m5-results-review-attestation/v1",
            release_id=M5_REPLAY_RELEASE_ID,
            scientific_git_revision=scientific_git_revision,
            candidate_sha256=bindings.candidate_sha256,
            candidate_index_sha256=bindings.candidate_index_sha256,
            scientific_member_set_sha256=bindings.scientific_member_set_sha256,
            claim_projection_sha256=bindings.claim_projection_sha256,
            figure_spec_set_sha256=bindings.figure_spec_set_sha256,
            rendered_figure_set_sha256=bindings.rendered_figure_set_sha256,
            presentation_template_set_sha256=bindings.presentation_template_set_sha256,
            review_report_sha256=_report_sha256(review_report, label="results review report"),
            reviewer_identity=decision.reviewer_identity,
            reviewer_identity_scope=M5_REVIEWER_IDENTITY_SCOPE,
            findings=decision.findings,
            p0_count=p0,
            p1_count=p1,
            p2_count=p2,
            unresolved_finding_ids=unresolved,
            negative_and_undefined_results_reviewed_and_retained=True,
            limitations_reviewed_and_retained=True,
            disposition=decision.disposition,
        )
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseValidationError(
            "results review decision cannot be canonicalized"
        ) from error


def load_results_review_attestation(
    value: bytes,
    *,
    review_report: bytes,
    scientific_git_revision: str,
    bindings: ResultsReviewBindings,
    require_release_permitting: bool = True,
) -> ReplayResultsReviewAttestationV1:
    """Strictly reload and authenticate the independent results review."""

    attestation = _load_canonical_model(
        value,
        model_type=ReplayResultsReviewAttestationV1,
        label="results review attestation",
    )
    expected = (
        M5_REPLAY_RELEASE_ID,
        scientific_git_revision,
        bindings.candidate_sha256,
        bindings.candidate_index_sha256,
        bindings.scientific_member_set_sha256,
        bindings.claim_projection_sha256,
        bindings.figure_spec_set_sha256,
        bindings.rendered_figure_set_sha256,
        bindings.presentation_template_set_sha256,
        _report_sha256(review_report, label="results review report"),
    )
    actual = (
        attestation.release_id,
        attestation.scientific_git_revision,
        attestation.candidate_sha256,
        attestation.candidate_index_sha256,
        attestation.scientific_member_set_sha256,
        attestation.claim_projection_sha256,
        attestation.figure_spec_set_sha256,
        attestation.rendered_figure_set_sha256,
        attestation.presentation_template_set_sha256,
        attestation.review_report_sha256,
    )
    if actual != expected:
        raise ReplayReleaseValidationError(
            "results review attestation has stale candidate or report authority"
        )
    if require_release_permitting:
        _require_release_permitting(
            attestation.findings,
            attestation.disposition,
            label="results review attestation",
        )
    return attestation


def _require_software_check_authority(
    checks: Sequence[ReplaySoftwareVerificationCheckV1],
) -> None:
    if (
        tuple(row.check_id for row in checks) != M5_SOFTWARE_VERIFICATION_CHECK_IDS
        or tuple(row.category for row in checks) != M5_SOFTWARE_VERIFICATION_CATEGORIES
    ):
        raise ReplayReleaseValidationError(
            "software verification checks are incomplete or reordered"
        )
    for row in checks:
        if row.required_test_ids != M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[row.check_id]:
            raise ReplayReleaseValidationError(
                "software verification required-test authority is incomplete"
            )


def build_software_verification(
    checks: Sequence[ReplaySoftwareVerificationCheckV1],
    *,
    snapshot: ImplementationSnapshot,
    lockfile_sha256: str,
    package_version: str,
) -> ReplaySoftwareVerificationV1:
    """Build the source-bound software envelope from exact successful checks."""

    _require_software_check_authority(checks)
    try:
        return ReplaySoftwareVerificationV1(
            schema="ffb.m5-software-verification/v1",
            release_id=M5_REPLAY_RELEASE_ID,
            scientific_git_revision=snapshot.scientific_git_revision,
            lockfile_sha256=_require_digest(lockfile_sha256, label="lockfile"),
            package_version=package_version,
            implementation_snapshot_sha256=snapshot.sha256,
            tooling_revision=snapshot.scientific_git_revision,
            checks=tuple(checks),
        )
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseValidationError(
            "software verification envelope violates its strict contract"
        ) from error


def load_software_verification(
    value: bytes,
    *,
    snapshot: ImplementationSnapshot,
    lockfile_sha256: str,
    package_version: str,
) -> ReplaySoftwareVerificationV1:
    """Reload software evidence and recheck cheap structural/source bindings."""

    verification = _load_canonical_model(
        value,
        model_type=ReplaySoftwareVerificationV1,
        label="software verification",
    )
    _require_software_check_authority(verification.checks)
    if (
        verification.release_id != M5_REPLAY_RELEASE_ID
        or verification.scientific_git_revision != snapshot.scientific_git_revision
        or verification.tooling_revision != snapshot.scientific_git_revision
        or verification.implementation_snapshot_sha256 != snapshot.sha256
        or verification.lockfile_sha256 != _require_digest(lockfile_sha256, label="lockfile")
        or verification.package_version != package_version
    ):
        raise ReplayReleaseValidationError("software verification has stale source authority")
    return verification


def software_verification_test_subset_bytes(
    verification: ReplaySoftwareVerificationV1,
    required_test_ids: Sequence[str],
) -> bytes:
    """Project named test entries without accepting an unbound Boolean verdict."""

    if not required_test_ids or len(set(required_test_ids)) != len(required_test_ids):
        raise ReplayReleaseValidationError("software test subset is empty or duplicated")
    locations: dict[str, ReplaySoftwareVerificationCheckV1] = {}
    for check in verification.checks:
        for test_id in check.required_test_ids:
            if test_id in locations:
                raise ReplayReleaseValidationError(
                    "software required-test identifier is ambiguously attested"
                )
            locations[test_id] = check
    try:
        selected = tuple(locations[test_id] for test_id in required_test_ids)
    except KeyError as error:
        raise ReplayReleaseValidationError(
            "software verification is missing a required named test"
        ) from error
    return canonical_json_bytes(
        {
            "schema": "ffb.m5-software-verification-test-subset/v1",
            "required_test_ids": list(required_test_ids),
            "checks": [row.model_dump(mode="json", by_alias=True) for row in selected],
        }
    )


def _snapshot_digest(snapshot: ImplementationSnapshot, path: str) -> str:
    matches = tuple(entry.sha256 for entry in snapshot.entries if entry.path == path)
    if len(matches) != 1:
        raise ReplayReleaseValidationError(
            "implementation snapshot is missing a privacy/license authority"
        )
    return matches[0]


def build_privacy_license_attestation(
    *,
    snapshot: ImplementationSnapshot,
    run_id: str,
) -> ReplayPrivacyLicenseAttestationV1:
    """Create the fixed zero-leakage/license assertion bound to tracked terms."""

    try:
        return ReplayPrivacyLicenseAttestationV1(
            schema="ffb.m5-privacy-license-attestation/v1",
            release_id=M5_REPLAY_RELEASE_ID,
            scientific_git_revision=snapshot.scientific_git_revision,
            run_id=run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            scan_contract="ffb.m5-role-aware-privacy-scan/v1",
            scan_scope=(
                "deterministic-candidate-and-package-members-plus-filenames-and-sanitized-messages"
            ),
            forbidden_match_count=0,
            raw_sensor_payload_reads=0,
            dataset_root_serialized=False,
            local_sequence_rows_serialized=False,
            tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
            repository_license_sha256=_snapshot_digest(snapshot, "LICENSE"),
            data_and_model_terms_sha256=_snapshot_digest(
                snapshot,
                "DATA_AND_MODEL_TERMS.md",
            ),
            attribution_present=True,
            non_endorsement_present=True,
            evidence_scope=(
                "deterministic-content-scan-and-operator-license-attestation-not-dataset-"
                "authentication"
            ),
        )
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseValidationError(
            "privacy/license attestation violates its strict contract"
        ) from error


def load_privacy_license_attestation(
    value: bytes,
    *,
    snapshot: ImplementationSnapshot,
    run_id: str,
) -> ReplayPrivacyLicenseAttestationV1:
    """Reload the privacy/license assertion and bind tracked authority bytes."""

    attestation = _load_canonical_model(
        value,
        model_type=ReplayPrivacyLicenseAttestationV1,
        label="privacy/license attestation",
    )
    if (
        attestation.release_id != M5_REPLAY_RELEASE_ID
        or attestation.scientific_git_revision != snapshot.scientific_git_revision
        or attestation.run_id != run_id
        or attestation.repository_license_sha256 != _snapshot_digest(snapshot, "LICENSE")
        or attestation.data_and_model_terms_sha256
        != _snapshot_digest(snapshot, "DATA_AND_MODEL_TERMS.md")
    ):
        raise ReplayReleaseValidationError(
            "privacy/license attestation has stale source or run authority"
        )
    return attestation


def derive_validation_evidence_sha256(
    check_id: str,
    evidence: Mapping[str, bytes],
) -> str:
    """Derive one digest from the exact named authority for its fixed check."""

    expected_parts = M5_VALIDATION_AUTHORITY_PARTS.get(check_id)
    domain = M5_VALIDATION_EVIDENCE_DOMAINS.get(check_id)
    if expected_parts is None or domain is None:
        raise ReplayReleaseValidationError("validation check ID is not preregistered")
    if set(evidence) != set(expected_parts):
        raise ReplayReleaseValidationError(
            "validation evidence parts are missing, extra, or assigned to the wrong check"
        )
    digest = hashlib.sha256(domain)
    digest.update(len(expected_parts).to_bytes(8, "big"))
    total = 0
    for part_name in expected_parts:
        value = evidence[part_name]
        if not value:
            raise ReplayReleaseValidationError("validation evidence parts must be nonempty bytes")
        if len(value) > M5_VALIDATION_PART_MAX_BYTES:
            raise ReplayReleaseValidationError("validation evidence part exceeds its byte cap")
        total += len(value)
        if total > M5_VALIDATION_TOTAL_MAX_BYTES:
            raise ReplayReleaseValidationError("validation evidence exceeds its total byte cap")
        encoded_name = part_name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _validation_scope(
    check_id: str,
) -> Literal[
    "direct-content-and-contract-recomputation",
    "operator-attested-with-content-binding-not-execution-proof",
]:
    if check_id in _OPERATOR_ATTESTED_CHECKS:
        return "operator-attested-with-content-binding-not-execution-proof"
    return "direct-content-and-contract-recomputation"


def _require_evidence_check_set(
    evidence_by_check: Mapping[str, Mapping[str, bytes]],
    *,
    include_results_review: bool,
) -> None:
    expected = set(M5_RELEASE_VALIDATION_CHECK_IDS)
    if not include_results_review:
        expected.remove("results-and-claims-review")
    if set(evidence_by_check) != expected:
        raise ReplayReleaseValidationError(
            "validation evidence does not contain the exact required check set"
        )


def derive_pre_review_validation_inputs(
    *,
    run_id: str,
    evidence_by_check: Mapping[str, Mapping[str, bytes]],
) -> ReplayValidationInputsV1:
    """Derive 16 ready inputs and one explicit independent-review pending slot."""

    _require_evidence_check_set(evidence_by_check, include_results_review=False)
    checks: list[ReplayValidationEvidenceInputV1] = []
    for check_id in M5_RELEASE_VALIDATION_CHECK_IDS:
        if check_id == "results-and-claims-review":
            checks.append(
                ReplayValidationEvidenceInputV1(
                    check_id=check_id,
                    status="pending",
                    passed=None,
                    evidence_sha256=None,
                    evidence_scope="pending-independent-results-review",
                )
            )
        else:
            checks.append(
                ReplayValidationEvidenceInputV1(
                    check_id=check_id,
                    status="ready",
                    passed=True,
                    evidence_sha256=derive_validation_evidence_sha256(
                        check_id,
                        evidence_by_check[check_id],
                    ),
                    evidence_scope=_validation_scope(check_id),
                )
            )
    try:
        return ReplayValidationInputsV1(
            schema="ffb.m5-validation-inputs/v1",
            release_id=M5_REPLAY_RELEASE_ID,
            run_id=run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            checks=tuple(checks),
        )
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseValidationError(
            "pre-review validation inputs violate their strict contract"
        ) from error


def load_pre_review_validation_inputs(
    value: bytes,
    *,
    run_id: str,
    evidence_by_check: Mapping[str, Mapping[str, bytes]],
) -> ReplayValidationInputsV1:
    """Reload and independently reconstruct every pre-review evidence digest."""

    observed = _load_canonical_model(
        value,
        model_type=ReplayValidationInputsV1,
        label="pre-review validation inputs",
    )
    expected = derive_pre_review_validation_inputs(
        run_id=run_id,
        evidence_by_check=evidence_by_check,
    )
    if observed != expected:
        raise ReplayReleaseValidationError(
            "pre-review validation inputs disagree with named authority"
        )
    return observed


def derive_final_replay_validation(
    *,
    run_id: str,
    evidence_by_check: Mapping[str, Mapping[str, bytes]],
    pre_review_inputs: ReplayValidationInputsV1 | None = None,
) -> ReplayValidationV1:
    """Insert results-review authority and freshly derive all 17 final checks."""

    _require_evidence_check_set(evidence_by_check, include_results_review=True)
    pre_review_evidence = {
        check_id: evidence
        for check_id, evidence in evidence_by_check.items()
        if check_id != "results-and-claims-review"
    }
    expected_pre_review = derive_pre_review_validation_inputs(
        run_id=run_id,
        evidence_by_check=pre_review_evidence,
    )
    if pre_review_inputs is not None and pre_review_inputs != expected_pre_review:
        raise ReplayReleaseValidationError(
            "final validation disagrees with the reviewed pre-review inputs"
        )
    checks = tuple(
        ReplayValidationCheckV1(
            check_id=check_id,
            passed=True,
            evidence_sha256=derive_validation_evidence_sha256(
                check_id,
                evidence_by_check[check_id],
            ),
        )
        for check_id in M5_RELEASE_VALIDATION_CHECK_IDS
    )
    try:
        return ReplayValidationV1(
            schema="ffb.replay-validation/v1",
            run_id=run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            checks=checks,
            scene_count=10,
            replay_experiment_count=22,
            raw_sensor_payload_reads=0,
            all_checks_passed=True,
        )
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseValidationError(
            "final replay validation violates its strict contract"
        ) from error


def load_final_replay_validation(
    value: bytes,
    *,
    run_id: str,
    evidence_by_check: Mapping[str, Mapping[str, bytes]],
    pre_review_inputs: ReplayValidationInputsV1 | None = None,
) -> ReplayValidationV1:
    """Reload final validation and independently repeat all 17 derivations."""

    observed = _load_canonical_model(
        value,
        model_type=ReplayValidationV1,
        label="final replay validation",
    )
    expected = derive_final_replay_validation(
        run_id=run_id,
        evidence_by_check=evidence_by_check,
        pre_review_inputs=pre_review_inputs,
    )
    if observed != expected:
        raise ReplayReleaseValidationError("final replay validation disagrees with named authority")
    return observed
