"""Strict contracts for M5 review-candidate and release-package evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    REPLAY_ARTIFACT_PATHS,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_SHA256,
    ReplayExperimentIdentityV1,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, GitRevision, Identifier

_CANDIDATE_DOMAIN = b"fusion-fault-bench/m5-review-candidate/v1\x00"
_SIDECAR_DOMAIN = b"fusion-fault-bench/m5-release-sidecars/v1\x00"
_PACKAGE_DOMAIN = b"fusion-fault-bench/m5-release-package/v1\x00"
M5_IMPLEMENTATION_SNAPSHOT_DOMAIN = b"fusion-fault-bench/m5-implementation-snapshot/v1\x00"
M5_RELEASE_PACKAGE_BYTE_CAP = 50 * 1024 * 1024
M5_RELEASE_DESTINATION_PATH = f"reports/releases/{M5_REPLAY_RELEASE_ID}"

M5_REVIEW_CANDIDATE_INDEXED_PATHS = (
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
M5_REVIEW_CANDIDATE_PATHS = (
    "candidate-index.json",
    *M5_REVIEW_CANDIDATE_INDEXED_PATHS,
)

type ReviewCandidateRole = Literal[
    "reviewed-scientific-aggregate",
    "reviewed-repeat-or-provenance",
    "pre-review-validation-input",
    "frozen-public-methodology",
    "independent-review-evidence",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "reviewed-presentation-template",
    "reviewed-claim-projection",
]

_CANDIDATE_ROLE_PAIRS: tuple[tuple[str, ReviewCandidateRole], ...] = (
    *((path, "reviewed-scientific-aggregate") for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[:7]),
    *((path, "reviewed-repeat-or-provenance") for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[7:10]),
    ("evidence/release-pipeline-plan.md", "frozen-public-methodology"),
    ("evidence/release-pipeline-plan-review.md", "independent-review-evidence"),
    ("evidence/resource-scope-amendment.md", "frozen-public-methodology"),
    ("evidence/implementation-review.md", "independent-review-evidence"),
    ("evidence/validation-inputs.json", "pre-review-validation-input"),
    ("evidence/implementation-review-attestation.json", "independent-review-evidence"),
    ("evidence/software-verification.json", "pre-review-validation-input"),
    ("evidence/privacy-license-attestation.json", "pre-review-validation-input"),
    *((path, "deterministic-figure-spec") for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[18:28:2]),
    *(
        (path, "deterministic-rendered-figure")
        for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[19:28:2]
    ),
    *(
        (path, "reviewed-presentation-template")
        for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[28:32]
    ),
    ("presentation/public-claim-projections.json", "reviewed-claim-projection"),
)
M5_REVIEW_CANDIDATE_ROLE_BY_PATH: Mapping[str, ReviewCandidateRole] = MappingProxyType(
    dict(_CANDIDATE_ROLE_PAIRS)
)

M5_FIGURE_IDS = (
    "m5-persistent-panel-summary",
    "m5-crossovers",
    "m5-health-transfer",
    "m5-descriptor-comparison",
    "m5-cluster-sensitivity",
)
M5_FIGURE_PATHS = tuple(
    path
    for figure_id in M5_FIGURE_IDS
    for path in (f"figures/{figure_id}.spec.json", f"figures/{figure_id}.svg")
)
M5_PRESENTATION_PLACEHOLDERS = (
    "@M5_RELEASE_ARTIFACT_SHA256@",
    "@M5_RELEASE_RUN_SHA256@",
    "@M5_RESULTS_REVIEW_ATTESTATION_SHA256@",
    "@M5_MACHINE_ARTIFACT_BYTES@",
)

M5_RELEASE_SIDECAR_INDEXED_PATHS = (
    "README.md",
    "claim-evidence.md",
    "verification.md",
    "release-summary.json",
    *M5_FIGURE_PATHS,
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
M5_RELEASE_PACKAGE_PATHS = (
    *(f"artifact/{path}" for path in REPLAY_ARTIFACT_PATHS),
    *M5_RELEASE_SIDECAR_INDEXED_PATHS,
    "release-sidecar-index.json",
)

type ReleaseSidecarRole = Literal[
    "final-presentation",
    "deterministic-figure-spec",
    "deterministic-rendered-figure",
    "frozen-public-methodology",
    "independent-review-evidence",
    "reviewed-candidate-index",
    "pre-review-validation-input",
    "reviewed-claim-projection",
]

_SIDECAR_ROLE_PAIRS: tuple[tuple[str, ReleaseSidecarRole], ...] = (
    *((path, "final-presentation") for path in M5_RELEASE_SIDECAR_INDEXED_PATHS[:4]),
    *((path, "deterministic-figure-spec") for path in M5_FIGURE_PATHS[::2]),
    *((path, "deterministic-rendered-figure") for path in M5_FIGURE_PATHS[1::2]),
    ("evidence/release-pipeline-plan.md", "frozen-public-methodology"),
    ("evidence/release-pipeline-plan-review.md", "independent-review-evidence"),
    ("evidence/resource-scope-amendment.md", "frozen-public-methodology"),
    ("evidence/implementation-review.md", "independent-review-evidence"),
    ("evidence/review-candidate-index.json", "reviewed-candidate-index"),
    ("evidence/validation-inputs.json", "pre-review-validation-input"),
    ("evidence/implementation-review-attestation.json", "independent-review-evidence"),
    ("evidence/software-verification.json", "pre-review-validation-input"),
    ("evidence/privacy-license-attestation.json", "pre-review-validation-input"),
    ("evidence/public-claim-projections.json", "reviewed-claim-projection"),
    ("evidence/results-review.md", "independent-review-evidence"),
    ("evidence/results-review-attestation.json", "independent-review-evidence"),
)
M5_RELEASE_SIDECAR_ROLE_BY_PATH: Mapping[str, ReleaseSidecarRole] = MappingProxyType(
    dict(_SIDECAR_ROLE_PAIRS)
)

type ReviewSeverity = Literal["p0", "p1", "p2"]
type ReviewFindingStatus = Literal["resolved", "unresolved"]
type ReviewDisposition = Literal["pass", "pass-with-nonblocking-findings", "block"]
type SoftwareVerificationCategory = Literal[
    "format",
    "lint",
    "type-check",
    "unit-property-oracle-integration",
    "build",
    "wheel-smoke",
    "privacy",
]
M5_SOFTWARE_VERIFICATION_CATEGORIES: tuple[SoftwareVerificationCategory, ...] = (
    "format",
    "lint",
    "type-check",
    "unit-property-oracle-integration",
    "build",
    "wheel-smoke",
    "privacy",
)
M5_IMPLEMENTATION_REVIEW_AREAS = (
    "runner",
    "release-builder",
    "validators",
    "claim-projections",
    "figures",
    "privacy-boundary",
    "failure-and-rollback",
)

type FigureId = Literal[
    "m5-persistent-panel-summary",
    "m5-crossovers",
    "m5-health-transfer",
    "m5-descriptor-comparison",
    "m5-cluster-sensitivity",
]
type FigureKind = Literal[
    "persistent-panel-summary",
    "crossovers",
    "health-transfer",
    "descriptor-comparison",
    "cluster-sensitivity",
]
type FigureSourceKind = Literal[
    "persistent-aggregate",
    "persistent-crossover",
    "health-aggregate",
    "cluster-sensitivity",
    "descriptor-aggregate",
]
type ClaimProjectionGroup = Literal[
    "persistent-panel",
    "crossovers",
    "health-transfer",
    "cluster-sensitivity",
    "descriptor-comparison",
    "resources",
    "release-facts",
    "dropout-nesting",
    "finalization-metadata",
]
type ClaimSourceKind = Literal[
    "persistent-aggregate",
    "persistent-crossover",
    "health-aggregate",
    "cluster-sensitivity",
    "descriptor-aggregate",
    "execution-resource",
    "repeat-verification",
    "profile-summary",
    "software-verification",
    "release-index",
]
type ProjectedScalar = str | int | FiniteFloat | bool | None


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("release path must be normalized repository-relative POSIX")
    return value


def _safe_public_text(value: str) -> str:
    if not value or any(ord(character) < 32 for character in value) or "\\" in value:
        raise ValueError("public release text is empty or unsafe")
    return value


def _domain_digest(domain: bytes, payload: bytes) -> str:
    return hashlib.sha256(domain + len(payload).to_bytes(8, "big") + payload).hexdigest()


def compute_replay_review_candidate_sha256(candidate_core: BaseModel | Mapping[str, Any]) -> str:
    """Compute the semantic review-candidate digest over its canonical core."""

    return _domain_digest(_CANDIDATE_DOMAIN, canonical_json_bytes(candidate_core))


def compute_replay_release_sidecar_set_sha256(
    sidecar_core: BaseModel | Mapping[str, Any],
) -> str:
    """Compute the semantic digest over the self-excluded sidecar-index core."""

    return _domain_digest(_SIDECAR_DOMAIN, canonical_json_bytes(sidecar_core))


def compute_replay_release_package_sha256(
    machine_artifact_sha256: str,
    sidecar_set_sha256: str,
) -> str:
    """Bind the strict machine artifact to the complete indexed sidecar set."""

    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in (machine_artifact_sha256, sidecar_set_sha256)
    ):
        raise ValueError("release package digests must be lowercase SHA-256 values")
    return hashlib.sha256(
        _PACKAGE_DOMAIN + bytes.fromhex(machine_artifact_sha256) + bytes.fromhex(sidecar_set_sha256)
    ).hexdigest()


class ReplayReviewCandidateFileEntryV1(ContractModel):
    path: str
    role: ReviewCandidateRole
    byte_length: Annotated[int, Field(ge=1, le=M5_RELEASE_PACKAGE_BYTE_CAP)]
    sha256: Digest
    record_count: Annotated[int, Field(ge=1, le=500_000)] | None = None

    @field_validator("path")
    @classmethod
    def require_candidate_path(cls, value: str) -> str:
        if value not in M5_REVIEW_CANDIDATE_INDEXED_PATHS:
            raise ValueError("candidate file path is not allowlisted")
        return value

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if self.role != M5_REVIEW_CANDIDATE_ROLE_BY_PATH[self.path]:
            raise ValueError("candidate file role disagrees with its fixed path")
        if self.path.endswith(".ndjson") != (self.record_count is not None):
            raise ValueError("candidate record count is required exactly for NDJSON")
        return self


class ReplayReviewCandidateIndexV1(ContractModel):
    schema_id: Literal["ffb.m5-release-review-candidate-index/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    scientific_git_revision: GitRevision
    lockfile_sha256: Digest
    package_version: Identifier
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    primary_local_artifact_sha256: Digest
    repeat_local_artifact_sha256: Digest
    primary_local_run_sha256: Digest
    repeat_local_run_sha256: Digest
    results_review_status: Literal["pending"]
    files: Annotated[
        tuple[ReplayReviewCandidateFileEntryV1, ...],
        Field(
            min_length=len(M5_REVIEW_CANDIDATE_INDEXED_PATHS),
            max_length=len(M5_REVIEW_CANDIDATE_INDEXED_PATHS),
        ),
    ]
    candidate_sha256: Digest

    @model_validator(mode="after")
    def require_exact_candidate(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or tuple(entry.path for entry in self.files) != M5_REVIEW_CANDIDATE_INDEXED_PATHS
            or self.primary_local_artifact_sha256 == self.repeat_local_artifact_sha256
            or self.primary_local_run_sha256 == self.repeat_local_run_sha256
        ):
            raise ValueError("candidate index does not bind the frozen independent inputs")
        core = self.model_dump(mode="json", by_alias=True, exclude={"candidate_sha256"})
        if self.candidate_sha256 != compute_replay_review_candidate_sha256(core):
            raise ValueError("candidate semantic digest is invalid")
        return self


class ReplayReviewFindingV1(ContractModel):
    finding_id: Identifier
    severity: ReviewSeverity
    status: ReviewFindingStatus


class ReplayImplementationReviewDecisionV1(ContractModel):
    schema_id: Literal["ffb.m5-implementation-review-decision/v1"] = Field(alias="schema")
    reviewer_identity: Annotated[str, Field(min_length=1, max_length=256)]
    reviewed_areas: tuple[Identifier, ...]
    findings: tuple[ReplayReviewFindingV1, ...]
    disposition: ReviewDisposition

    @field_validator("reviewer_identity")
    @classmethod
    def require_safe_identity(cls, value: str) -> str:
        return _safe_public_text(value)

    @model_validator(mode="after")
    def require_decision(self) -> Self:
        if self.reviewed_areas != M5_IMPLEMENTATION_REVIEW_AREAS:
            raise ValueError("implementation review areas are incomplete or reordered")
        if len({row.finding_id for row in self.findings}) != len(self.findings):
            raise ValueError("implementation review finding IDs must be unique")
        return self


def _review_counts(
    findings: tuple[ReplayReviewFindingV1, ...],
) -> tuple[int, int, int, tuple[str, ...]]:
    return (
        sum(row.severity == "p0" for row in findings),
        sum(row.severity == "p1" for row in findings),
        sum(row.severity == "p2" for row in findings),
        tuple(row.finding_id for row in findings if row.status == "unresolved"),
    )


class ReplayImplementationReviewAttestationV1(ContractModel):
    schema_id: Literal["ffb.m5-implementation-review-attestation/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    implementation_snapshot_sha256: Digest
    implementation_snapshot_file_count: Annotated[int, Field(ge=1, le=100_000)]
    review_report_sha256: Digest
    reviewer_identity: Annotated[str, Field(min_length=1, max_length=256)]
    reviewer_identity_scope: Literal["operator-recorded-not-cryptographically-authenticated"]
    reviewed_areas: tuple[Identifier, ...]
    findings: tuple[ReplayReviewFindingV1, ...]
    p0_count: Annotated[int, Field(ge=0)]
    p1_count: Annotated[int, Field(ge=0)]
    p2_count: Annotated[int, Field(ge=0)]
    unresolved_finding_ids: tuple[Identifier, ...]
    disposition: ReviewDisposition

    @model_validator(mode="after")
    def require_attested_decision(self) -> Self:
        p0, p1, p2, unresolved = _review_counts(self.findings)
        if (
            self.reviewed_areas != M5_IMPLEMENTATION_REVIEW_AREAS
            or (self.p0_count, self.p1_count, self.p2_count) != (p0, p1, p2)
            or self.unresolved_finding_ids != unresolved
            or len({row.finding_id for row in self.findings}) != len(self.findings)
        ):
            raise ValueError("implementation review attestation is internally inconsistent")
        blocking = any(
            row.status == "unresolved" and row.severity in {"p0", "p1"} for row in self.findings
        )
        if self.disposition != "block" and blocking:
            raise ValueError("release-permitting implementation review retains a blocker")
        return self


class ReplaySoftwareVerificationCheckV1(ContractModel):
    check_id: Identifier
    category: SoftwareVerificationCategory
    command: Annotated[tuple[str, ...], Field(min_length=1)]
    required_test_ids: tuple[Identifier, ...]
    exit_status: Literal[0]
    output_sha256: Digest
    output_normalization: Literal["stable-command-output-with-runtime-paths-and-durations-removed"]

    @field_validator("command")
    @classmethod
    def require_safe_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument or argument.startswith(("/", "file:")) for argument in value):
            raise ValueError("software verification command contains a local absolute path")
        return value


class ReplaySoftwareVerificationV1(ContractModel):
    schema_id: Literal["ffb.m5-software-verification/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    scientific_git_revision: GitRevision
    lockfile_sha256: Digest
    package_version: Identifier
    implementation_snapshot_sha256: Digest
    tooling_revision: GitRevision
    checks: Annotated[
        tuple[ReplaySoftwareVerificationCheckV1, ...],
        Field(
            min_length=len(M5_SOFTWARE_VERIFICATION_CATEGORIES),
            max_length=len(M5_SOFTWARE_VERIFICATION_CATEGORIES),
        ),
    ]

    @model_validator(mode="after")
    def require_check_order(self) -> Self:
        if tuple(row.category for row in self.checks) != M5_SOFTWARE_VERIFICATION_CATEGORIES:
            raise ValueError("software verification categories are incomplete or reordered")
        if len({row.check_id for row in self.checks}) != len(self.checks):
            raise ValueError("software verification check IDs must be unique")
        if self.tooling_revision != self.scientific_git_revision:
            raise ValueError("software verification tooling revision must equal scientific HEAD")
        return self


class ReplayPrivacyLicenseAttestationV1(ContractModel):
    schema_id: Literal["ffb.m5-privacy-license-attestation/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    scientific_git_revision: GitRevision
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    scan_contract: Literal["ffb.m5-role-aware-privacy-scan/v1"]
    scan_scope: Literal[
        "deterministic-candidate-and-package-members-plus-filenames-and-sanitized-messages"
    ]
    forbidden_match_count: Literal[0]
    raw_sensor_payload_reads: Literal[0]
    dataset_root_serialized: Literal[False]
    local_sequence_rows_serialized: Literal[False]
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]
    repository_license_sha256: Digest
    data_and_model_terms_sha256: Digest
    attribution_present: Literal[True]
    non_endorsement_present: Literal[True]
    evidence_scope: Literal[
        "deterministic-content-scan-and-operator-license-attestation-not-dataset-authentication"
    ]

    @model_validator(mode="after")
    def require_frozen_binding(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
        ):
            raise ValueError("privacy attestation does not bind frozen M5 intent")
        return self


class ReplayValidationEvidenceInputV1(ContractModel):
    check_id: Identifier
    status: Literal["ready", "pending"]
    passed: Literal[True] | None
    evidence_sha256: Digest | None
    evidence_scope: Literal[
        "direct-content-and-contract-recomputation",
        "operator-attested-with-content-binding-not-execution-proof",
        "pending-independent-results-review",
    ]

    @model_validator(mode="after")
    def require_slot_shape(self) -> Self:
        pending = self.check_id == "results-and-claims-review"
        if pending != (self.status == "pending"):
            raise ValueError("only results-and-claims-review may remain pending")
        if pending:
            if self.passed is not None or self.evidence_sha256 is not None:
                raise ValueError("pending validation input cannot claim evidence")
            if self.evidence_scope != "pending-independent-results-review":
                raise ValueError("pending validation input has the wrong scope")
        elif self.passed is not True or self.evidence_sha256 is None:
            raise ValueError("ready validation input requires passing bound evidence")
        return self


class ReplayValidationInputsV1(ContractModel):
    schema_id: Literal["ffb.m5-validation-inputs/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    checks: Annotated[
        tuple[ReplayValidationEvidenceInputV1, ...],
        Field(
            min_length=len(M5_RELEASE_VALIDATION_CHECK_IDS),
            max_length=len(M5_RELEASE_VALIDATION_CHECK_IDS),
        ),
    ]

    @model_validator(mode="after")
    def require_exact_slots(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or tuple(row.check_id for row in self.checks) != M5_RELEASE_VALIDATION_CHECK_IDS
        ):
            raise ValueError("validation inputs do not use the frozen M5 slots")
        return self


class ReplayClaimSelectorFieldV1(ContractModel):
    field: Identifier
    value: Annotated[str, Field(min_length=1, max_length=256)]


class ReplayProjectedFieldV1(ContractModel):
    field: Identifier
    value: ProjectedScalar
    rendering: Literal["machine-token", ".6g", ".2f", "exact-integer", "literal-status"]


class ReplayPublicClaimProjectionV1(ContractModel):
    schema_id: Literal["ffb.m5-public-claim-projection/v1"] = Field(alias="schema")
    projection_id: Identifier
    public_claim_id: Identifier
    projection_group: ClaimProjectionGroup
    source_member: Annotated[str, Field(min_length=1, max_length=128)]
    source_kind: ClaimSourceKind
    source_identifier: Annotated[str, Field(min_length=1, max_length=256)]
    source_record_sha256: Digest | None
    selector_fields: tuple[ReplayClaimSelectorFieldV1, ...]
    projected_fields: Annotated[tuple[ReplayProjectedFieldV1, ...], Field(min_length=1)]
    unit: Annotated[str, Field(min_length=1, max_length=64)]
    status_behavior: Literal[
        "defined-numeric",
        "literal-undefined",
        "literal-not-applicable",
        "literal-censored",
        "literal-positive-infinity",
        "finalization-null-then-exact-integer",
    ]
    figure_ids: tuple[FigureId, ...]
    hypothesis_id: Identifier | None = None

    @field_validator("source_member")
    @classmethod
    def require_safe_source_member(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def require_projection_shape(self) -> Self:
        finalization = self.projection_group == "finalization-metadata"
        if finalization != (self.status_behavior == "finalization-null-then-exact-integer"):
            raise ValueError("finalization metadata uses a unique status behavior")
        if finalization:
            if self.source_kind != "release-index" or self.source_record_sha256 is not None:
                raise ValueError("candidate finalization metadata must reference a future index")
            if any(field.value is not None for field in self.projected_fields):
                raise ValueError("candidate finalization metadata value must be null")
        elif self.source_record_sha256 is None:
            raise ValueError("scientific claim projection requires a source-record digest")
        if len({row.field for row in self.selector_fields}) != len(self.selector_fields):
            raise ValueError("claim selector fields must be unique")
        if len({row.field for row in self.projected_fields}) != len(self.projected_fields):
            raise ValueError("claim projected fields must be unique")
        return self


class ReplayPublicClaimProjectionsV1(ContractModel):
    schema_id: Literal["ffb.m5-public-claim-projections/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    persistent_hypothesis_count: Literal[33]
    persistent_hypothesis_coordinate_set_sha256: Digest
    health_hypothesis_count: Literal[11]
    health_hypothesis_coordinate_set_sha256: Digest
    persistent_figure_projection_count: Literal[100]
    crossover_projection_count: Literal[10]
    health_figure_projection_count: Literal[43]
    sensitivity_source_count: Literal[26]
    distinct_log_group_count: Annotated[int, Field(ge=1, le=10)]
    sensitivity_projection_count: Annotated[int, Field(ge=286, le=520)]
    descriptor_figure_projection_count: Literal[67]
    resource_record_count: Literal[2]
    projections: Annotated[tuple[ReplayPublicClaimProjectionV1, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_projection_registry(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or self.sensitivity_projection_count
            != self.sensitivity_source_count * (10 + self.distinct_log_group_count)
            or len({row.projection_id for row in self.projections}) != len(self.projections)
        ):
            raise ValueError("public claim projection registry is incomplete or duplicated")
        groups = tuple(row.projection_group for row in self.projections)
        expected_counts = {
            "persistent-panel": self.persistent_figure_projection_count,
            "crossovers": self.crossover_projection_count,
            "health-transfer": self.health_figure_projection_count,
            "cluster-sensitivity": self.sensitivity_projection_count,
            "descriptor-comparison": self.descriptor_figure_projection_count,
            "resources": self.resource_record_count,
            "finalization-metadata": 1,
        }
        if any(groups.count(group) != count for group, count in expected_counts.items()):
            raise ValueError("public projection group count disagrees with the frozen registry")
        if sum(row.hypothesis_id is not None for row in self.projections) != 44:
            raise ValueError("public projection registry must mark all 44 hypotheses")
        return self


class ReplayFigureMarkV1(ContractModel):
    mark_ordinal: Annotated[int, Field(ge=0)]
    projection_id: Identifier
    source_member: Annotated[str, Field(min_length=1, max_length=128)]
    source_kind: FigureSourceKind
    source_identifier: Annotated[str, Field(min_length=1, max_length=256)]
    source_record_sha256: Digest
    projected_fields: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    replay_identity_sha256: Digest | None = None

    @field_validator("source_member")
    @classmethod
    def require_safe_member(cls, value: str) -> str:
        return _safe_relative_path(value)


_FIGURE_META: Mapping[FigureId, tuple[FigureKind, str, tuple[int, int]]] = MappingProxyType(
    {
        "m5-persistent-panel-summary": (
            "persistent-panel-summary",
            "figures/m5-persistent-panel-summary.svg",
            (1600, 1800),
        ),
        "m5-crossovers": ("crossovers", "figures/m5-crossovers.svg", (1400, 900)),
        "m5-health-transfer": (
            "health-transfer",
            "figures/m5-health-transfer.svg",
            (1600, 1800),
        ),
        "m5-descriptor-comparison": (
            "descriptor-comparison",
            "figures/m5-descriptor-comparison.svg",
            (1500, 1600),
        ),
        "m5-cluster-sensitivity": (
            "cluster-sensitivity",
            "figures/m5-cluster-sensitivity.svg",
            (1600, 1800),
        ),
    }
)


class ReplayFigureSpecV1(ContractModel):
    schema_id: Literal["ffb.m5-figure-spec/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    figure_id: FigureId
    figure_kind: FigureKind
    figure_file: str
    width_px: Annotated[int, Field(ge=640, le=4096)]
    height_px: Annotated[int, Field(ge=480, le=4096)]
    font_families: tuple[Literal["Arial", "Helvetica", "sans-serif"], ...]
    colors: tuple[Annotated[str, Field(pattern=r"^#[0-9A-F]{6}$")], ...]
    units: tuple[Annotated[str, Field(min_length=1, max_length=32)], ...]
    axis_facets: tuple[Identifier, ...]
    caption_boundary: Literal["registry-projected-values-and-literal-statuses-only"]
    renderer_id: Literal["ffb.m5-deterministic-svg/v1"]
    marks: Annotated[tuple[ReplayFigureMarkV1, ...], Field(min_length=1)]
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]
    non_endorsement_footer: Literal[
        "CC BY-NC-SA 4.0 plus Motional Dataset Terms; attribution required; no endorsement."
    ]

    @field_validator("figure_file")
    @classmethod
    def require_safe_figure_file(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def require_figure_shape(self) -> Self:
        expected_kind, expected_file, dimensions = _FIGURE_META[self.figure_id]
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or self.figure_kind != expected_kind
            or self.figure_file != expected_file
            or (self.width_px, self.height_px) != dimensions
            or self.font_families != ("Arial", "Helvetica", "sans-serif")
            or tuple(row.mark_ordinal for row in self.marks) != tuple(range(len(self.marks)))
            or len({row.projection_id for row in self.marks}) != len(self.marks)
        ):
            raise ValueError("figure specification differs from the frozen rendering contract")
        return self


class ReplayFigureSourceBindingV1(ContractModel):
    schema_id: Literal["ffb.replay-figure-source-binding/v1"] = Field(alias="schema")
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    figure_id: FigureId
    figure_kind: FigureKind
    mark_ordinal: Annotated[int, Field(ge=0)]
    source_kind: FigureSourceKind
    source_identifier: Annotated[str, Field(min_length=1, max_length=256)]
    source_record_sha256: Digest
    identity: ReplayExperimentIdentityV1 | None = None
    replay_identity_sha256: Digest | None = None
    figure_spec_sha256: Digest
    rendered_svg_path: str
    rendered_svg_sha256: Digest
    rendered_svg_byte_length: Annotated[int, Field(ge=1, le=M5_RELEASE_PACKAGE_BYTE_CAP)]
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]

    @field_validator("rendered_svg_path")
    @classmethod
    def require_safe_svg_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        expected_kind, expected_path, _ = _FIGURE_META[self.figure_id]
        descriptor = self.source_kind == "descriptor-aggregate"
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or self.figure_kind != expected_kind
            or self.rendered_svg_path != expected_path
            or descriptor != (self.identity is None)
            or descriptor != (self.replay_identity_sha256 is None)
        ):
            raise ValueError("figure source binding has an invalid global or source shape")
        if self.identity is not None and self.replay_identity_sha256 != (
            replay_experiment_identity_sha256(self.identity)
        ):
            raise ValueError("figure source binding has an invalid replay identity digest")
        return self


class ReplayResultsReviewDecisionV1(ContractModel):
    schema_id: Literal["ffb.m5-results-review-decision/v1"] = Field(alias="schema")
    reviewer_identity: Annotated[str, Field(min_length=1, max_length=256)]
    findings: tuple[ReplayReviewFindingV1, ...]
    negative_and_undefined_results_reviewed_and_retained: bool
    limitations_reviewed_and_retained: bool
    disposition: ReviewDisposition

    @field_validator("reviewer_identity")
    @classmethod
    def require_safe_identity(cls, value: str) -> str:
        return _safe_public_text(value)

    @model_validator(mode="after")
    def require_unique_findings(self) -> Self:
        if len({row.finding_id for row in self.findings}) != len(self.findings):
            raise ValueError("results review finding IDs must be unique")
        return self


class ReplayResultsReviewAttestationV1(ContractModel):
    schema_id: Literal["ffb.m5-results-review-attestation/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    scientific_git_revision: GitRevision
    candidate_sha256: Digest
    candidate_index_sha256: Digest
    scientific_member_set_sha256: Digest
    claim_projection_sha256: Digest
    figure_spec_set_sha256: Digest
    rendered_figure_set_sha256: Digest
    presentation_template_set_sha256: Digest
    review_report_sha256: Digest
    reviewer_identity: Annotated[str, Field(min_length=1, max_length=256)]
    reviewer_identity_scope: Literal["operator-recorded-not-cryptographically-authenticated"]
    findings: tuple[ReplayReviewFindingV1, ...]
    p0_count: Annotated[int, Field(ge=0)]
    p1_count: Annotated[int, Field(ge=0)]
    p2_count: Annotated[int, Field(ge=0)]
    unresolved_finding_ids: tuple[Identifier, ...]
    negative_and_undefined_results_reviewed_and_retained: Literal[True]
    limitations_reviewed_and_retained: Literal[True]
    disposition: ReviewDisposition

    @model_validator(mode="after")
    def require_review_conjunction(self) -> Self:
        p0, p1, p2, unresolved = _review_counts(self.findings)
        if (
            (self.p0_count, self.p1_count, self.p2_count) != (p0, p1, p2)
            or self.unresolved_finding_ids != unresolved
            or len({row.finding_id for row in self.findings}) != len(self.findings)
        ):
            raise ValueError("results review attestation is internally inconsistent")
        blocking = any(
            row.status == "unresolved" and row.severity in {"p0", "p1"} for row in self.findings
        )
        if self.disposition != "block" and blocking:
            raise ValueError("release-permitting results review retains a blocker")
        return self


class ReplayReleaseSidecarFileEntryV1(ContractModel):
    path: str
    role: ReleaseSidecarRole
    byte_length: Annotated[int, Field(ge=1, le=M5_RELEASE_PACKAGE_BYTE_CAP)]
    sha256: Digest
    record_count: Annotated[int, Field(ge=1, le=500_000)] | None = None

    @field_validator("path")
    @classmethod
    def require_sidecar_path(cls, value: str) -> str:
        if value not in M5_RELEASE_SIDECAR_INDEXED_PATHS:
            raise ValueError("release sidecar path is not allowlisted")
        return value

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if self.role != M5_RELEASE_SIDECAR_ROLE_BY_PATH[self.path]:
            raise ValueError("release sidecar role disagrees with its fixed path")
        if self.path.endswith(".ndjson") != (self.record_count is not None):
            raise ValueError("sidecar record count is required exactly for NDJSON")
        return self


class ReplayReleaseSidecarIndexV1(ContractModel):
    schema_id: Literal["ffb.m5-release-sidecar-index/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    reviewed_candidate_sha256: Digest
    results_review_attestation_sha256: Digest
    machine_artifact_sha256: Digest
    machine_run_sha256: Digest
    scientific_git_revision: GitRevision
    files: Annotated[
        tuple[ReplayReleaseSidecarFileEntryV1, ...],
        Field(
            min_length=len(M5_RELEASE_SIDECAR_INDEXED_PATHS),
            max_length=len(M5_RELEASE_SIDECAR_INDEXED_PATHS),
        ),
    ]
    machine_artifact_byte_length: Annotated[
        int,
        Field(ge=1, le=M5_RELEASE_PACKAGE_BYTE_CAP),
    ]
    indexed_sidecar_payload_byte_length: Annotated[
        int,
        Field(ge=1, le=M5_RELEASE_PACKAGE_BYTE_CAP),
    ]
    sidecar_set_sha256: Digest
    release_package_sha256: Digest

    @model_validator(mode="after")
    def require_exact_index(self) -> Self:
        if tuple(entry.path for entry in self.files) != M5_RELEASE_SIDECAR_INDEXED_PATHS:
            raise ValueError("release sidecar index does not use the fixed path order")
        if self.indexed_sidecar_payload_byte_length != sum(
            entry.byte_length for entry in self.files
        ):
            raise ValueError("indexed sidecar byte length is not the exact member sum")
        core = self.model_dump(
            mode="json",
            by_alias=True,
            exclude={"sidecar_set_sha256", "release_package_sha256"},
        )
        expected_sidecars = compute_replay_release_sidecar_set_sha256(core)
        expected_package = compute_replay_release_package_sha256(
            self.machine_artifact_sha256,
            expected_sidecars,
        )
        if (
            self.sidecar_set_sha256 != expected_sidecars
            or self.release_package_sha256 != expected_package
        ):
            raise ValueError("release sidecar or package digest is invalid")
        return self


def replay_release_contract_json_schemas() -> dict[str, dict[str, Any]]:
    """Return the release-only schemas under stable CLI-facing names."""

    models: Mapping[str, type[BaseModel]] = {
        "replay-review-candidate-index": ReplayReviewCandidateIndexV1,
        "replay-implementation-review-attestation": ReplayImplementationReviewAttestationV1,
        "replay-software-verification": ReplaySoftwareVerificationV1,
        "replay-privacy-license-attestation": ReplayPrivacyLicenseAttestationV1,
        "replay-validation-inputs": ReplayValidationInputsV1,
        "replay-public-claim-projections": ReplayPublicClaimProjectionsV1,
        "replay-figure-spec": ReplayFigureSpecV1,
        "replay-figure-source-binding": ReplayFigureSourceBindingV1,
        "replay-results-review-attestation": ReplayResultsReviewAttestationV1,
        "replay-release-sidecar-index": ReplayReleaseSidecarIndexV1,
    }
    return {name: model.model_json_schema(by_alias=True) for name, model in models.items()}
