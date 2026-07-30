"""Outcome-blind review and publication boundary for the frozen M5 replay.

This module deliberately separates three authorities:

* the five local inputs from which aggregate evidence is regenerated;
* the immutable 34-file object reviewed by an independent reviewer; and
* the self-contained 41-file public package constructed only after that review.

The public loaders never need nuScenes and never accept caller-supplied verdict
booleans or unchecked digests as evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    absolute_artifact_path,
    assert_directory_descriptor_matches_path,
    atomic_rename_directory_no_replace_at,
    canonical_json_bytes,
    create_staging_directory_at,
    discover_git_metadata_dirs,
    entry_exists_at,
    open_or_create_real_directory,
    reject_directory_descriptor_in_git_metadata,
    reject_git_metadata_destination,
    strict_json_object_body,
    write_exclusive_file_at,
)
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    REPLAY_ARTIFACT_PATHS,
    REPLAY_MAX_ARTIFACT_BYTES,
    REPLAY_MAX_NDJSON_RECORDS,
    REPLAY_MAX_RECORD_BYTES,
    ReplayFigureSourceBindingV1,
    ReplayValidationCheckV1,
    ReplayValidationV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_SHA256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, GitRevision, Identifier

if TYPE_CHECKING:
    from fusion_fault_bench.replay_curation import ReplayCuratedAggregateEvidence

M5_REVIEW_CANDIDATE_SCHEMA = "ffb.m5-release-review-candidate-index/v1"
M5_RELEASE_SIDECAR_SCHEMA = "ffb.m5-release-sidecar-index/v1"
M5_IMPLEMENTATION_REVIEW_SCHEMA = "ffb.m5-implementation-review-attestation/v1"
M5_SOFTWARE_VERIFICATION_SCHEMA = "ffb.m5-software-verification/v1"
M5_PRIVACY_LICENSE_SCHEMA = "ffb.m5-privacy-license-attestation/v1"
M5_VALIDATION_INPUTS_SCHEMA = "ffb.m5-validation-inputs/v1"
M5_RESULTS_REVIEW_SCHEMA = "ffb.m5-results-review-attestation/v1"
M5_PUBLIC_CLAIMS_SCHEMA = "ffb.m5-public-claim-projections/v1"

M5_CANDIDATE_DOMAIN = b"fusion-fault-bench/m5-review-candidate/v1\x00"
M5_SIDECAR_DOMAIN = b"fusion-fault-bench/m5-release-sidecars/v1\x00"
M5_PACKAGE_DOMAIN = b"fusion-fault-bench/m5-release-package/v1\x00"
M5_IMPLEMENTATION_SNAPSHOT_DOMAIN = b"fusion-fault-bench/m5-implementation-snapshot/v1\x00"
M5_VALIDATION_EVIDENCE_DOMAIN = b"fusion-fault-bench/m5-validation-evidence/v1\x00"
M5_CANDIDATE_SET_DOMAIN = b"fusion-fault-bench/m5-candidate-set/v1\x00"

M5_RELEASE_PACKAGE_MAX_BYTES = 50 * 1024 * 1024
M5_RELEASE_MEMBER_MAX_BYTES = 32 * 1024 * 1024
M5_RELEASE_READ_CHUNK_BYTES = 1024 * 1024

M5_MACHINE_CANDIDATE_PATHS = (
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
)
M5_CANDIDATE_EVIDENCE_PATHS = (
    "evidence/release-pipeline-plan.md",
    "evidence/release-pipeline-plan-review.md",
    "evidence/resource-scope-amendment.md",
    "evidence/implementation-review.md",
    "evidence/validation-inputs.json",
    "evidence/implementation-review-attestation.json",
    "evidence/software-verification.json",
    "evidence/privacy-license-attestation.json",
)
M5_FIGURE_STEMS = (
    "m5-persistent-panel-summary",
    "m5-crossovers",
    "m5-health-transfer",
    "m5-descriptor-comparison",
    "m5-cluster-sensitivity",
)
M5_CANDIDATE_FIGURE_PATHS = tuple(
    path
    for stem in M5_FIGURE_STEMS
    for path in (f"figures/{stem}.spec.json", f"figures/{stem}.svg")
)
M5_CANDIDATE_PRESENTATION_PATHS = (
    "presentation/README.md",
    "presentation/claim-evidence.md",
    "presentation/verification.md",
    "presentation/release-summary.json",
    "presentation/public-claim-projections.json",
)
M5_CANDIDATE_MEMBER_PATHS = (
    *M5_MACHINE_CANDIDATE_PATHS,
    *M5_CANDIDATE_EVIDENCE_PATHS,
    *M5_CANDIDATE_FIGURE_PATHS,
    *M5_CANDIDATE_PRESENTATION_PATHS,
)
M5_REVIEW_CANDIDATE_PATHS = ("candidate-index.json", *M5_CANDIDATE_MEMBER_PATHS)

M5_FINAL_SIDECAR_PATHS = (
    "README.md",
    "claim-evidence.md",
    "verification.md",
    "release-summary.json",
    *M5_CANDIDATE_FIGURE_PATHS,
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
M5_FINAL_RELEASE_PATHS = (
    *(f"artifact/{path}" for path in REPLAY_ARTIFACT_PATHS),
    *M5_FINAL_SIDECAR_PATHS,
    "release-sidecar-index.json",
)

if len(M5_REVIEW_CANDIDATE_PATHS) != 34 or len(set(M5_REVIEW_CANDIDATE_PATHS)) != 34:
    raise RuntimeError("M5 review-candidate allowlist must contain exactly 34 files")
if len(M5_FINAL_RELEASE_PATHS) != 41 or len(set(M5_FINAL_RELEASE_PATHS)) != 41:
    raise RuntimeError("M5 final-package allowlist must contain exactly 41 files")
if len(M5_FINAL_SIDECAR_PATHS) != 26:
    raise RuntimeError("M5 sidecar index must cover exactly 26 files")

type CandidateRole = Literal[
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
type ReviewDisposition = Literal["pass", "fail"]
type ValidationInputStatus = Literal["passed", "pending"]

M5_PRESENTATION_PLACEHOLDERS = (
    "@M5_RELEASE_ARTIFACT_SHA256@",
    "@M5_RELEASE_RUN_SHA256@",
    "@M5_RESULTS_REVIEW_ATTESTATION_SHA256@",
    "@M5_MACHINE_ARTIFACT_BYTES@",
)

_CANDIDATE_ROLE_LOOKUP: dict[str, CandidateRole] = {
    **{path: "reviewed-scientific-aggregate" for path in M5_MACHINE_CANDIDATE_PATHS[1:7]},
    "machine/intent.json": "reviewed-repeat-or-provenance",
    "machine/repeat-verification.json": "reviewed-repeat-or-provenance",
    "machine/figure-records.ndjson": "reviewed-repeat-or-provenance",
    "machine/source-member-commitments.ndjson": "reviewed-repeat-or-provenance",
    "evidence/release-pipeline-plan.md": "frozen-public-methodology",
    "evidence/release-pipeline-plan-review.md": "independent-review-evidence",
    "evidence/resource-scope-amendment.md": "frozen-public-methodology",
    "evidence/implementation-review.md": "independent-review-evidence",
    "evidence/validation-inputs.json": "pre-review-validation-input",
    "evidence/implementation-review-attestation.json": ("pre-review-validation-input"),
    "evidence/software-verification.json": "pre-review-validation-input",
    "evidence/privacy-license-attestation.json": ("pre-review-validation-input"),
    **{f"figures/{stem}.spec.json": "deterministic-figure-spec" for stem in M5_FIGURE_STEMS},
    **{f"figures/{stem}.svg": "deterministic-rendered-figure" for stem in M5_FIGURE_STEMS},
    "presentation/README.md": "reviewed-presentation-template",
    "presentation/claim-evidence.md": "reviewed-presentation-template",
    "presentation/verification.md": "reviewed-presentation-template",
    "presentation/release-summary.json": "reviewed-claim-projection",
    "presentation/public-claim-projections.json": ("reviewed-claim-projection"),
}
_CANDIDATE_ROLE_BY_PATH: Mapping[str, CandidateRole] = MappingProxyType(
    {path: _CANDIDATE_ROLE_LOOKUP[path] for path in M5_CANDIDATE_MEMBER_PATHS}
)
if tuple(_CANDIDATE_ROLE_BY_PATH) != M5_CANDIDATE_MEMBER_PATHS:
    raise RuntimeError("M5 candidate roles must follow the exact member allowlist")

_FROZEN_METHODOLOGY_ROLES = frozenset({"frozen-public-methodology", "independent-review-evidence"})
_PRIVATE_PATH_PATTERN = re.compile(
    rb"(?i)(?:file:(?://)?|/(?:Users|home|private|tmp|Volumes)/|[A-Z]:[\\/])"
)
_DATASET_PATH_PATTERN = re.compile(
    rb"(?i)(?:^|[/\\])(?:nuscenes|v1\.0-mini|samples|sweeps|maps)(?:[/\\]|$)"
)
_SECRET_PATTERN = re.compile(
    rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    rb"(?:[\"'\s]*[:=][\"'\s]*|_)[A-Za-z0-9+/=_-]{8,}"
)
_RAW_PAYLOAD_PATTERN = re.compile(
    rb"(?i)[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp|bmp|gif|pcd|ply|las|laz|"
    rb"bin|npy|npz|bag|db3|mcap|tar|tgz|zip|7z|gz|bz2|xz)"
    rb"(?:[\"'\s]|$)"
)
_TOKEN_PATTERN = re.compile(
    rb"(?i)(?:sample|annotation|instance|calibrated_sensor|ego_pose|log)_token"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "annotation_token",
        "calibrated_sensor_token",
        "calibration",
        "coordinates",
        "dataset_path",
        "dataset_root",
        "ego_pose_token",
        "file_name",
        "filename",
        "filepath",
        "image_path",
        "instance_token",
        "log_token",
        "point_cloud_path",
        "pose",
        "rotation",
        "sample_token",
        "timestamp",
        "timestamp_us",
        "translation",
    }
)
_ALLOWED_METHODOLOGY_PATH_LITERALS = (
    b"/usr/bin/time",
    b"/dev/fd/<fd>",
    b"/Users/<name>/.../nuScenes",
    b"dataset/private/generated",
)


class CandidateFileEntryV1(ContractModel):
    """One exact reviewed candidate member."""

    path: str
    role: CandidateRole
    byte_length: Annotated[int, Field(ge=1, le=M5_RELEASE_MEMBER_MAX_BYTES)]
    sha256: Digest
    record_count: Annotated[int, Field(ge=1, le=REPLAY_MAX_NDJSON_RECORDS)] | None = None

    @model_validator(mode="after")
    def require_exact_shape(self) -> Self:
        if self.path not in M5_CANDIDATE_MEMBER_PATHS:
            raise ValueError("candidate entry path is not allowlisted")
        if self.role != _CANDIDATE_ROLE_BY_PATH[self.path]:
            raise ValueError("candidate entry role is not canonical")
        if self.path.endswith(".ndjson") != (self.record_count is not None):
            raise ValueError("record_count must exist exactly for candidate NDJSON")
        return self


class ReviewCandidateIndexV1(ContractModel):
    """Self-excluding index for the exact immutable 34-file review object."""

    schema_id: Literal["ffb.m5-release-review-candidate-index/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    scientific_git_revision: GitRevision
    implementation_snapshot_sha256: Digest
    lockfile_sha256: Digest
    package_version: Identifier
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    primary_local_artifact_sha256: Digest
    repeat_local_artifact_sha256: Digest
    primary_run_sha256: Digest
    repeat_run_sha256: Digest
    results_review_status: Literal["pending"]
    files: Annotated[
        tuple[CandidateFileEntryV1, ...],
        Field(min_length=33, max_length=33),
    ]
    candidate_sha256: Digest

    @model_validator(mode="after")
    def require_exact_index(self) -> Self:
        if (
            self.release_id != M5_REPLAY_RELEASE_ID
            or self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or tuple(entry.path for entry in self.files) != M5_CANDIDATE_MEMBER_PATHS
            or self.primary_local_artifact_sha256 == self.repeat_local_artifact_sha256
            or self.primary_run_sha256 == self.repeat_run_sha256
        ):
            raise ValueError("candidate index does not bind the frozen M5 review object")
        if self.candidate_sha256 != compute_candidate_digest_from_index(self):
            raise ValueError("candidate semantic digest is invalid")
        return self


class ImplementationSnapshotEntryV1(ContractModel):
    """One content byte binding in the reviewed implementation snapshot."""

    path: str
    byte_length: Annotated[int, Field(ge=0, le=REPLAY_MAX_ARTIFACT_BYTES)]
    sha256: Digest


class ImplementationReviewAttestationV1(ContractModel):
    """Canonicalized reviewer-authored implementation disposition."""

    schema_id: Literal["ffb.m5-implementation-review-attestation/v1"] = Field(alias="schema")
    implementation_snapshot_sha256: Digest
    implementation_snapshot_entry_count: Annotated[int, Field(ge=1)]
    review_report_sha256: Digest
    reviewer: Annotated[str, Field(min_length=1, max_length=128)]
    reviewer_identity_scope: Literal["operator-recorded-not-cryptographically-authenticated"]
    reviewed_areas: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    finding_ids: tuple[Identifier, ...]
    p0_count: Annotated[int, Field(ge=0)]
    p1_count: Annotated[int, Field(ge=0)]
    p2_count: Annotated[int, Field(ge=0)]
    unresolved_finding_ids: tuple[Identifier, ...]
    disposition: ReviewDisposition

    @model_validator(mode="after")
    def require_release_disposition_consistency(self) -> Self:
        if len(set(self.reviewed_areas)) != len(self.reviewed_areas):
            raise ValueError("implementation review areas must be unique")
        if len(set(self.finding_ids)) != len(self.finding_ids):
            raise ValueError("implementation review finding IDs must be unique")
        if not set(self.unresolved_finding_ids).issubset(self.finding_ids):
            raise ValueError("unresolved implementation findings must be declared")
        if self.disposition == "pass" and (
            self.p0_count != 0 or self.p1_count != 0 or self.unresolved_finding_ids
        ):
            raise ValueError("passing implementation review cannot retain blockers")
        return self


class SoftwareVerificationCheckV1(ContractModel):
    """One deterministic, path-free software verification record."""

    check_id: Identifier
    command: Annotated[tuple[str, ...], Field(min_length=1)]
    required_test_ids: tuple[str, ...]
    exit_status: int
    normalized_output_sha256: Digest
    passed: bool

    @model_validator(mode="after")
    def require_exit_conjunction(self) -> Self:
        if self.passed != (self.exit_status == 0):
            raise ValueError("software check status contradicts its exit status")
        if any(
            not value
            or value.startswith("/")
            or "\\" in value
            or ".." in PurePosixPath(value).parts
            for value in self.required_test_ids
        ):
            raise ValueError("software required-test IDs must be stable public labels")
        return self


M5_SOFTWARE_CHECK_ORDER = (
    "lockfile",
    "format",
    "lint",
    "type",
    "tests",
    "build",
    "wheel-smoke",
    "privacy",
)


class SoftwareVerificationV1(ContractModel):
    """Canonical clean-revision software-verification attestation."""

    schema_id: Literal["ffb.m5-software-verification/v1"] = Field(alias="schema")
    scientific_git_revision: GitRevision
    implementation_snapshot_sha256: Digest
    lockfile_sha256: Digest
    package_version: Identifier
    checks: Annotated[
        tuple[SoftwareVerificationCheckV1, ...],
        Field(min_length=8, max_length=8),
    ]
    all_checks_passed: bool
    evidence_scope: Literal["operator-executed-command-status-and-normalized-output-commitments"]

    @model_validator(mode="after")
    def require_exact_checks(self) -> Self:
        if tuple(check.check_id for check in self.checks) != M5_SOFTWARE_CHECK_ORDER:
            raise ValueError("software verification checks do not use frozen order")
        if self.all_checks_passed != all(check.passed for check in self.checks):
            raise ValueError("software verification is not the exact conjunction")
        return self


class PrivacyLicenseAttestationV1(ContractModel):
    """Deterministic bounded scan and public evidence-use boundary."""

    schema_id: Literal["ffb.m5-privacy-license-attestation/v1"] = Field(alias="schema")
    candidate_scanned_path_count: Literal[33]
    raw_sensor_payload_reads: Literal[0]
    dataset_root_serialized: Literal[False]
    local_sequence_rows_published: Literal[False]
    raw_resource_logs_published: Literal[False]
    private_or_credential_material_found: Literal[False]
    dataset_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]
    attribution_required: Literal[True]
    non_endorsement_required: Literal[True]
    repository_license_does_not_relicense_evidence: Literal[True]
    all_checks_passed: Literal[True]
    evidence_scope: Literal["deterministic-bounded-content-scan-and-operator-read-accounting"]


class ValidationInputRecordV1(ContractModel):
    """One check-specific pre-review evidence derivation."""

    check_id: Identifier
    status: ValidationInputStatus
    authority_sha256: Digest | None
    evidence_scope: Annotated[str, Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def require_status_shape(self) -> Self:
        if (self.status == "passed") != (self.authority_sha256 is not None):
            raise ValueError("validation input status contradicts its authority digest")
        if self.check_id == "results-and-claims-review":
            if self.status != "pending":
                raise ValueError("candidate results review must remain pending")
        elif self.status != "passed":
            raise ValueError("all other candidate validation inputs must pass")
        return self


class ValidationInputsV1(ContractModel):
    """The first sixteen authorities plus one explicit review placeholder."""

    schema_id: Literal["ffb.m5-validation-inputs/v1"] = Field(alias="schema")
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    records: Annotated[
        tuple[ValidationInputRecordV1, ...],
        Field(min_length=17, max_length=17),
    ]

    @model_validator(mode="after")
    def require_exact_order(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or tuple(record.check_id for record in self.records) != M5_RELEASE_VALIDATION_CHECK_IDS
        ):
            raise ValueError("validation inputs do not bind the frozen M5 checks")
        return self


class ResultsReviewAttestationV1(ContractModel):
    """Canonicalized independent review of one exact candidate."""

    schema_id: Literal["ffb.m5-results-review-attestation/v1"] = Field(alias="schema")
    candidate_sha256: Digest
    candidate_index_sha256: Digest
    scientific_member_set_sha256: Digest
    claim_projection_set_sha256: Digest
    figure_spec_set_sha256: Digest
    rendered_figure_set_sha256: Digest
    presentation_template_set_sha256: Digest
    review_report_sha256: Digest
    reviewer: Annotated[str, Field(min_length=1, max_length=128)]
    reviewer_identity_scope: Literal["operator-recorded-not-cryptographically-authenticated"]
    p0_count: Annotated[int, Field(ge=0)]
    p1_count: Annotated[int, Field(ge=0)]
    p2_count: Annotated[int, Field(ge=0)]
    unresolved_finding_ids: tuple[Identifier, ...]
    negative_and_undefined_results_reviewed_and_retained: bool
    limitations_reviewed_and_retained: bool
    disposition: ReviewDisposition

    @model_validator(mode="after")
    def require_release_disposition_consistency(self) -> Self:
        if self.disposition == "pass" and (
            self.p0_count != 0
            or self.p1_count != 0
            or self.unresolved_finding_ids
            or not self.negative_and_undefined_results_reviewed_and_retained
            or not self.limitations_reviewed_and_retained
        ):
            raise ValueError("passing results review cannot retain blockers or omissions")
        return self


class ReleaseSidecarEntryV1(ContractModel):
    """One exact non-machine member in the final package."""

    path: str
    role: CandidateRole | Literal["results-review-evidence", "reviewed-candidate-index"]
    byte_length: Annotated[int, Field(ge=1, le=M5_RELEASE_MEMBER_MAX_BYTES)]
    sha256: Digest
    record_count: Annotated[int, Field(ge=1, le=REPLAY_MAX_NDJSON_RECORDS)] | None = None

    @model_validator(mode="after")
    def require_exact_shape(self) -> Self:
        if self.path not in M5_FINAL_SIDECAR_PATHS:
            raise ValueError("release sidecar path is not allowlisted")
        if self.path.endswith(".ndjson") != (self.record_count is not None):
            raise ValueError("record_count must exist exactly for sidecar NDJSON")
        return self


class ReleaseSidecarIndexV1(ContractModel):
    """Self-excluding identity for all final-package sidecars."""

    schema_id: Literal["ffb.m5-release-sidecar-index/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    reviewed_candidate_sha256: Digest
    results_review_attestation_sha256: Digest
    machine_artifact_sha256: Digest
    machine_run_sha256: Digest
    scientific_git_revision: GitRevision
    files: Annotated[
        tuple[ReleaseSidecarEntryV1, ...],
        Field(min_length=26, max_length=26),
    ]
    machine_artifact_payload_bytes: Annotated[int, Field(ge=1)]
    indexed_sidecar_payload_bytes: Annotated[int, Field(ge=1)]
    sidecar_set_sha256: Digest
    release_package_sha256: Digest

    @model_validator(mode="after")
    def require_exact_index(self) -> Self:
        if tuple(
            entry.path for entry in self.files
        ) != M5_FINAL_SIDECAR_PATHS or self.indexed_sidecar_payload_bytes != sum(
            entry.byte_length for entry in self.files
        ):
            raise ValueError("release sidecar index is incomplete")
        sidecar_sha256 = compute_sidecar_set_digest_from_index(self)
        if self.sidecar_set_sha256 != sidecar_sha256:
            raise ValueError("release sidecar digest is invalid")
        expected_package = compute_release_package_digest(
            machine_artifact_sha256=self.machine_artifact_sha256,
            sidecar_set_sha256=sidecar_sha256,
        )
        if self.release_package_sha256 != expected_package:
            raise ValueError("release package digest is invalid")
        return self


@dataclass(frozen=True, slots=True)
class ImplementationSnapshot:
    """Exact path/byte table reviewed before authoritative replay."""

    entries: tuple[ImplementationSnapshotEntryV1, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class M5ReleaseLocalInputs:
    """Exactly five local inputs authorized to cross the curation boundary."""

    primary_artifact: Path
    repeat_artifact: Path
    primary_time_l: Path
    repeat_time_l: Path
    software_verification: Path

    def __post_init__(self) -> None:
        values = (
            self.primary_artifact,
            self.repeat_artifact,
            self.primary_time_l,
            self.repeat_time_l,
            self.software_verification,
        )
        if len(values) != 5:
            raise ValueError("M5 release requires exactly five local inputs")
        lexical = tuple(absolute_artifact_path(path) for path in values)
        if len(set(lexical)) != len(lexical):
            raise ValueError("M5 local input paths must be distinct")
        for left_index, left in enumerate(lexical):
            for right in lexical[left_index + 1 :]:
                if left in right.parents or right in left.parents:
                    raise ValueError("M5 local inputs must not contain one another")


@dataclass(frozen=True, slots=True)
class CandidateMaterial:
    """Complete in-memory candidate regenerated from the five local inputs."""

    scientific_git_revision: str
    implementation_snapshot_sha256: str
    lockfile_sha256: str
    package_version: str
    run_id: str
    primary_local_artifact_sha256: str
    repeat_local_artifact_sha256: str
    primary_run_sha256: str
    repeat_run_sha256: str
    members: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class LoadedReviewCandidate:
    """One stable, strictly reloaded review candidate."""

    path: Path
    index: ReviewCandidateIndexV1
    index_bytes: bytes
    members: Mapping[str, bytes]
    candidate_sha256: str
    index_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedM5Release:
    """One stable, self-contained final M5 package."""

    path: Path
    sidecar_index: ReleaseSidecarIndexV1
    sidecar_index_bytes: bytes
    files: Mapping[str, bytes]
    package_sha256: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_digest(domain: bytes, payload: bytes) -> str:
    return hashlib.sha256(domain + len(payload).to_bytes(8, "big") + payload).hexdigest()


def _index_core_bytes(
    index: ReviewCandidateIndexV1 | ReleaseSidecarIndexV1,
    *,
    omitted_fields: frozenset[str],
) -> bytes:
    value = index.model_dump(mode="json", by_alias=True)
    for field in omitted_fields:
        value.pop(field)
    return canonical_json_bytes(value)


def compute_candidate_digest_from_index(index: ReviewCandidateIndexV1) -> str:
    """Compute the domain-separated semantic candidate digest."""

    return _domain_digest(
        M5_CANDIDATE_DOMAIN,
        _index_core_bytes(index, omitted_fields=frozenset({"candidate_sha256"})),
    )


def compute_sidecar_set_digest_from_index(index: ReleaseSidecarIndexV1) -> str:
    """Compute the domain-separated final sidecar-set digest."""

    return _domain_digest(
        M5_SIDECAR_DOMAIN,
        _index_core_bytes(
            index,
            omitted_fields=frozenset({"sidecar_set_sha256", "release_package_sha256"}),
        ),
    )


def compute_release_package_digest(
    *,
    machine_artifact_sha256: str,
    sidecar_set_sha256: str,
) -> str:
    """Bind the strict machine artifact and exact sidecar set."""

    try:
        payload = bytes.fromhex(machine_artifact_sha256) + bytes.fromhex(sidecar_set_sha256)
    except ValueError as error:
        raise ArtifactValidationError("M5 package digest input is malformed") from error
    if len(payload) != 64:
        raise ArtifactValidationError("M5 package digest input is malformed")
    return hashlib.sha256(M5_PACKAGE_DOMAIN + payload).hexdigest()


def derive_validation_evidence_sha256(
    check_id: str,
    authorities: Sequence[bytes],
) -> str:
    """Derive one check-specific digest from exact ordered authority bytes."""

    if check_id not in M5_RELEASE_VALIDATION_CHECK_IDS:
        raise ArtifactValidationError("unknown M5 release validation check")
    payload = bytearray(M5_VALIDATION_EVIDENCE_DOMAIN)
    encoded_id = check_id.encode("ascii")
    payload.extend(len(encoded_id).to_bytes(4, "big"))
    payload.extend(encoded_id)
    payload.extend(len(authorities).to_bytes(4, "big"))
    for authority in authorities:
        payload.extend(len(authority).to_bytes(8, "big"))
        payload.extend(authority)
    return hashlib.sha256(payload).hexdigest()


def compute_member_set_digest(
    *,
    label: str,
    paths: Sequence[str],
    members: Mapping[str, bytes],
) -> str:
    """Hash an ordered path/byte set without relying on filesystem names."""

    payload = bytearray(M5_CANDIDATE_SET_DOMAIN)
    label_bytes = label.encode("ascii")
    payload.extend(len(label_bytes).to_bytes(4, "big"))
    payload.extend(label_bytes)
    payload.extend(len(paths).to_bytes(4, "big"))
    for path in paths:
        if path not in members:
            raise ArtifactValidationError("candidate set member is missing")
        path_bytes = path.encode("utf-8")
        value = members[path]
        payload.extend(len(path_bytes).to_bytes(4, "big"))
        payload.extend(path_bytes)
        payload.extend(len(value).to_bytes(8, "big"))
        payload.extend(value)
    return hashlib.sha256(payload).hexdigest()


def _ndjson_record_count(value: bytes, *, label: str) -> int:
    if not value or len(value) > M5_RELEASE_MEMBER_MAX_BYTES:
        raise ArtifactValidationError("M5 NDJSON member is empty or oversized")
    count = 0
    for line in value.splitlines(keepends=True):
        count += 1
        if len(line) > REPLAY_MAX_RECORD_BYTES or not line.endswith(b"\n") or b"\r" in line:
            raise ArtifactValidationError("M5 NDJSON member is not canonical")
        try:
            body = strict_json_object_body(line, label=label)
            parsed = cast(object, json.loads(body))
        except (ArtifactValidationError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactValidationError("M5 NDJSON member is not canonical") from error
        if not isinstance(parsed, dict):
            raise ArtifactValidationError("M5 NDJSON rows must be objects")
        if canonical_json_bytes(cast("dict[str, Any]", parsed)) != line:
            raise ArtifactValidationError("M5 NDJSON member is not canonical")
        if count > REPLAY_MAX_NDJSON_RECORDS:
            raise ArtifactValidationError("M5 NDJSON member exceeds its record cap")
    if count == 0:
        raise ArtifactValidationError("M5 NDJSON member must not be empty")
    return count


def _strict_json_mapping(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        body = strict_json_object_body(value, label=label)
        parsed = cast(object, json.loads(body))
    except (ArtifactValidationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("M5 JSON member is not canonical") from error
    if not isinstance(parsed, dict):
        raise ArtifactValidationError("M5 JSON member must be an object")
    mapping = cast("dict[str, Any]", parsed)
    if canonical_json_bytes(mapping) != value:
        raise ArtifactValidationError("M5 JSON member is not canonical")
    return mapping


def _scan_public_value(value: object) -> None:
    if isinstance(value, dict):
        for key, item in cast("dict[object, object]", value).items():
            if not isinstance(key, str):
                raise ArtifactValidationError("public M5 mapping key is not text")
            if key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise ArtifactValidationError("public M5 data contains a forbidden field")
            _scan_public_value(item)
    elif isinstance(value, list):
        for item in cast("list[object]", value):
            _scan_public_value(item)
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        if (
            _PRIVATE_PATH_PATTERN.search(encoded)
            or _DATASET_PATH_PATTERN.search(encoded)
            or _RAW_PAYLOAD_PATTERN.search(encoded)
            or _TOKEN_PATTERN.search(encoded)
        ):
            raise ArtifactValidationError("public M5 data contains private source identity")


def scan_public_member(
    path: str,
    value: bytes,
    *,
    role: CandidateRole | Literal["results-review-evidence", "reviewed-candidate-index"],
) -> None:
    """Apply the bounded, role-aware privacy and canonical-text boundary."""

    if (
        not value
        or len(value) > M5_RELEASE_MEMBER_MAX_BYTES
        or b"\x00" in value
        or b"\r" in value
        or not value.endswith(b"\n")
        or _SECRET_PATTERN.search(value)
    ):
        raise ArtifactValidationError("M5 public member fails bounded privacy scan")
    if role in _FROZEN_METHODOLOGY_ROLES:
        scrubbed = value
        for literal in _ALLOWED_METHODOLOGY_PATH_LITERALS:
            scrubbed = scrubbed.replace(literal, b"")
        scrubbed = re.sub(rb"<[A-Za-z0-9._:/ -]+>", b"", scrubbed)
        if (
            _PRIVATE_PATH_PATTERN.search(scrubbed)
            or _DATASET_PATH_PATTERN.search(scrubbed)
            or _TOKEN_PATTERN.search(scrubbed)
        ):
            raise ArtifactValidationError("M5 methodology contains a realized private path")
    elif (
        _PRIVATE_PATH_PATTERN.search(value)
        or _DATASET_PATH_PATTERN.search(value)
        or _TOKEN_PATTERN.search(value)
        or _RAW_PAYLOAD_PATTERN.search(value)
        or b"interview/" in value.lower()
    ):
        raise ArtifactValidationError("M5 generated member contains private source identity")

    if path.endswith(".ndjson"):
        for line in value.splitlines(keepends=True):
            mapping = _strict_json_mapping(line, label=path)
            _scan_public_value(mapping)
    elif path.endswith(".json"):
        _scan_public_value(_strict_json_mapping(value, label=path))
    else:
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactValidationError("M5 textual member is not UTF-8") from error


def _candidate_entries(members: Mapping[str, bytes]) -> tuple[CandidateFileEntryV1, ...]:
    if tuple(members) != M5_CANDIDATE_MEMBER_PATHS:
        raise ArtifactValidationError("M5 candidate members are missing or out of order")
    entries: list[CandidateFileEntryV1] = []
    total = 0
    for path in M5_CANDIDATE_MEMBER_PATHS:
        value = members[path]
        role = _CANDIDATE_ROLE_BY_PATH[path]
        scan_public_member(path, value, role=role)
        total += len(value)
        entries.append(
            CandidateFileEntryV1(
                path=path,
                role=role,
                byte_length=len(value),
                sha256=_sha256(value),
                record_count=(
                    _ndjson_record_count(value, label=path) if path.endswith(".ndjson") else None
                ),
            )
        )
    if total > M5_RELEASE_PACKAGE_MAX_BYTES:
        raise ArtifactValidationError("M5 candidate exceeds the 50 MiB cap")
    return tuple(entries)


def build_review_candidate_index(material: CandidateMaterial) -> ReviewCandidateIndexV1:
    """Create the self-excluding candidate index from regenerated bytes."""

    entries = _candidate_entries(material.members)
    base: dict[str, object] = {
        "schema": M5_REVIEW_CANDIDATE_SCHEMA,
        "release_id": M5_REPLAY_RELEASE_ID,
        "scientific_git_revision": material.scientific_git_revision,
        "implementation_snapshot_sha256": material.implementation_snapshot_sha256,
        "lockfile_sha256": material.lockfile_sha256,
        "package_version": material.package_version,
        "run_id": material.run_id,
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "primary_local_artifact_sha256": material.primary_local_artifact_sha256,
        "repeat_local_artifact_sha256": material.repeat_local_artifact_sha256,
        "primary_run_sha256": material.primary_run_sha256,
        "repeat_run_sha256": material.repeat_run_sha256,
        "results_review_status": "pending",
        "files": [entry.model_dump(mode="json", by_alias=True) for entry in entries],
        "candidate_sha256": "0" * 64,
    }
    provisional = ReviewCandidateIndexV1.model_construct(
        schema_id=M5_REVIEW_CANDIDATE_SCHEMA,
        release_id=M5_REPLAY_RELEASE_ID,
        scientific_git_revision=material.scientific_git_revision,
        implementation_snapshot_sha256=material.implementation_snapshot_sha256,
        lockfile_sha256=material.lockfile_sha256,
        package_version=material.package_version,
        run_id=material.run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        primary_local_artifact_sha256=material.primary_local_artifact_sha256,
        repeat_local_artifact_sha256=material.repeat_local_artifact_sha256,
        primary_run_sha256=material.primary_run_sha256,
        repeat_run_sha256=material.repeat_run_sha256,
        results_review_status="pending",
        files=entries,
        candidate_sha256="0" * 64,
    )
    base["candidate_sha256"] = compute_candidate_digest_from_index(provisional)
    return ReviewCandidateIndexV1.model_validate(base)


class _ImplementationReviewDecisionV1(ContractModel):
    schema_id: Literal["ffb.m5-implementation-review-decision/v1"] = Field(alias="schema")
    reviewer: Annotated[str, Field(min_length=1, max_length=128)]
    reviewed_areas: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    finding_ids: tuple[Identifier, ...]
    p0_count: Annotated[int, Field(ge=0)]
    p1_count: Annotated[int, Field(ge=0)]
    p2_count: Annotated[int, Field(ge=0)]
    unresolved_finding_ids: tuple[Identifier, ...]
    disposition: ReviewDisposition


class _ResultsReviewDecisionV1(ContractModel):
    schema_id: Literal["ffb.m5-results-review-decision/v1"] = Field(alias="schema")
    reviewer: Annotated[str, Field(min_length=1, max_length=128)]
    p0_count: Annotated[int, Field(ge=0)]
    p1_count: Annotated[int, Field(ge=0)]
    p2_count: Annotated[int, Field(ge=0)]
    unresolved_finding_ids: tuple[Identifier, ...]
    negative_and_undefined_results_reviewed_and_retained: bool
    limitations_reviewed_and_retained: bool
    disposition: ReviewDisposition


_IMPLEMENTATION_DECISION_ADAPTER = TypeAdapter(_ImplementationReviewDecisionV1)
_RESULTS_DECISION_ADAPTER = TypeAdapter(_ResultsReviewDecisionV1)

_IMPLEMENTATION_FIXED_PATHS = (
    "examples/replay/m5-nuscenes-mini-replay-v1.json",
    "examples/matrices/m3-procedural-v1.json",
    "pyproject.toml",
    ".python-version",
    "uv.lock",
    "LICENSE",
    "DATA_AND_MODEL_TERMS.md",
    ".github/workflows/ci.yml",
    "docs/benchmark-contract-v0.1.md",
    "docs/m5-replay-plan.md",
    "docs/reviews/m5-replay-plan-review.md",
    "docs/m5-resource-scope-amendment.md",
    "docs/m5-release-pipeline-plan.md",
    "docs/reviews/m5-release-pipeline-plan-review.md",
)
_M4_RELEASE_ROOT = PurePosixPath("reports/releases/m4-health-v0.1.0")


def _git_output_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", os.fspath(source_root), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ArtifactValidationError("M5 Git source authority is unavailable")
    return result.stdout


def _tracked_paths(source_root: Path) -> tuple[str, ...]:
    raw = _git_output_bytes(source_root, "ls-files", "-z")
    paths: list[str] = []
    for value in raw.split(b"\x00"):
        if not value:
            continue
        try:
            path = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactValidationError("M5 tracked path is not UTF-8") from error
        if PurePosixPath(path).as_posix() != path:
            raise ArtifactValidationError("M5 tracked path is not canonical")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ArtifactValidationError("M5 tracked path set contains duplicates")
    return tuple(paths)


def _strict_source_file(
    source_root: Path,
    relative: str,
    *,
    allow_empty: bool = False,
) -> bytes:
    path = source_root / relative
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ArtifactValidationError("M5 implementation source is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (metadata.st_size == 0 and not allow_empty)
        or metadata.st_size > REPLAY_MAX_ARTIFACT_BYTES
    ):
        raise ArtifactValidationError("M5 implementation source is not a private regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != metadata.st_dev
            or opened.st_ino != metadata.st_ino
            or opened.st_size != metadata.st_size
        ):
            raise ArtifactValidationError("M5 implementation source changed during read")
        chunks: list[bytes] = []
        remaining = REPLAY_MAX_ARTIFACT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(M5_RELEASE_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) != metadata.st_size or len(value) > REPLAY_MAX_ARTIFACT_BYTES:
            raise ArtifactValidationError("M5 implementation source changed during read")
        return value
    finally:
        os.close(descriptor)


def _matrix_reference_paths(source_root: Path) -> tuple[str, ...]:
    matrix_path = "examples/matrices/m3-procedural-v1.json"
    matrix = _strict_json_mapping(
        _strict_source_file(source_root, matrix_path),
        label=matrix_path,
    )
    execution = matrix.get("execution_order")
    profiles = matrix.get("profiles")
    if not isinstance(execution, list) or not isinstance(profiles, list):
        raise ArtifactValidationError("M3 matrix reference table is malformed")
    manifest_paths: list[str] = []
    profile_paths: list[str] = []
    for item in execution:
        if not isinstance(item, dict) or not isinstance(item.get("manifest"), str):
            raise ArtifactValidationError("M3 matrix manifest reference is malformed")
        path = cast(str, item["manifest"])
        value = _strict_source_file(source_root, path)
        if _sha256(value) != cast(str, item.get("manifest_sha256")):
            # Matrix digests are canonical-object digests in the source contracts, not
            # ordinary byte hashes. The authenticated matrix loader is the authority;
            # byte inclusion here closes the implementation snapshot.
            pass
        manifest_paths.append(path)
    for item in profiles:
        if not isinstance(item, dict) or not isinstance(item.get("profile"), str):
            raise ArtifactValidationError("M3 matrix profile reference is malformed")
        profile_paths.append(cast(str, item["profile"]))
    if len(manifest_paths) != 8 or len(set(manifest_paths)) != 8:
        raise ArtifactValidationError("M3 matrix must reference exactly eight manifests")
    if len(profile_paths) != 3 or len(set(profile_paths)) != 3:
        raise ArtifactValidationError("M3 matrix must reference exactly three profiles")
    # Full typed loaders verify the embedded canonical digests. Import lazily to keep
    # this package-only module lightweight at import time.
    from fusion_fault_bench.contracts.matrix_v1 import load_experiment_matrix

    loaded = load_experiment_matrix(Path(matrix_path), source_root=source_root)
    if tuple(entry.manifest for entry in loaded.matrix.execution_order) != tuple(
        manifest_paths
    ) or tuple(entry.profile for entry in loaded.matrix.profiles) != tuple(profile_paths):
        raise ArtifactValidationError("M3 matrix typed expansion changed")
    return (*manifest_paths, *profile_paths)


def implementation_snapshot_paths(source_root: Path) -> tuple[str, ...]:
    """Expand the exact source set covered by the whole-revision review."""

    tracked = _tracked_paths(source_root)
    code_paths = tuple(path for path in tracked if path.startswith(("src/", "tools/", "tests/")))
    from fusion_fault_bench.health_release import HEALTH_RELEASE_ARTIFACT_PATHS

    m4_paths = tuple((_M4_RELEASE_ROOT / path).as_posix() for path in HEALTH_RELEASE_ARTIFACT_PATHS)
    paths = tuple(
        sorted(
            {
                *code_paths,
                *_IMPLEMENTATION_FIXED_PATHS,
                *_matrix_reference_paths(source_root),
                *m4_paths,
            },
            key=lambda path: path.encode("utf-8"),
        )
    )
    if not paths or not set(paths).issubset(tracked):
        raise ArtifactValidationError(
            "M5 implementation snapshot contains an untracked or missing authority"
        )
    return paths


def compute_implementation_snapshot(source_root: Path) -> ImplementationSnapshot:
    """Compute the frozen path/byte implementation-review authority."""

    root = source_root.resolve(strict=True)
    entries: list[ImplementationSnapshotEntryV1] = []
    payload = bytearray(M5_IMPLEMENTATION_SNAPSHOT_DOMAIN)
    paths = implementation_snapshot_paths(root)
    payload.extend(len(paths).to_bytes(4, "big"))
    for relative in paths:
        value = _strict_source_file(root, relative)
        path_bytes = relative.encode("utf-8")
        payload.extend(len(path_bytes).to_bytes(4, "big"))
        payload.extend(path_bytes)
        payload.extend(len(value).to_bytes(8, "big"))
        payload.extend(value)
        entries.append(
            ImplementationSnapshotEntryV1(
                path=relative,
                byte_length=len(value),
                sha256=_sha256(value),
            )
        )
    return ImplementationSnapshot(
        entries=tuple(entries),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _read_stable_public_file(
    path: Path,
    *,
    byte_cap: int = M5_RELEASE_MEMBER_MAX_BYTES,
) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ArtifactValidationError("M5 public input is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > byte_cap
    ):
        raise ArtifactValidationError("M5 public input must be a bounded private regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
        ):
            raise ArtifactValidationError("M5 public input changed during read")
        chunks: list[bytes] = []
        remaining = byte_cap + 1
        while remaining:
            chunk = os.read(descriptor, min(M5_RELEASE_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) != metadata.st_size or len(value) > byte_cap:
            raise ArtifactValidationError("M5 public input changed during read")
        return value
    finally:
        os.close(descriptor)


def _load_canonical_model[ModelT: ContractModel](
    value: bytes,
    *,
    label: str,
    adapter: TypeAdapter[ModelT],
) -> ModelT:
    try:
        strict_json_object_body(value, label=label)
        model = adapter.validate_json(value)
    except (ArtifactValidationError, ValidationError, ValueError) as error:
        raise ArtifactValidationError("M5 attestation input is not canonical") from error
    if canonical_json_bytes(model) != value:
        raise ArtifactValidationError("M5 attestation input is not canonical")
    return model


def _exclusive_file(path: Path, value: bytes) -> None:
    """Write one exact file without following or replacing any path."""

    target = absolute_artifact_path(path)
    if os.path.lexists(target):
        raise FileExistsError("M5 evidence destination already exists")
    parent_fd = open_or_create_real_directory(target.parent)
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            target.parent,
            label="M5 evidence parent",
        )
        if entry_exists_at(parent_fd, target.name):
            raise FileExistsError("M5 evidence destination already exists")
        write_exclusive_file_at(parent_fd, target.name, value)
        os.fsync(parent_fd)
        assert_directory_descriptor_matches_path(
            parent_fd,
            target.parent,
            label="M5 evidence parent",
        )
    finally:
        os.close(parent_fd)


def attest_implementation_review(
    *,
    source_root: Path,
    review_report: Path,
    decision: Path,
    output: Path,
) -> ImplementationReviewAttestationV1:
    """Canonicalize, but never choose, a whole-snapshot review disposition."""

    snapshot = compute_implementation_snapshot(source_root)
    report_bytes = _read_stable_public_file(review_report)
    scan_public_member(
        "evidence/implementation-review.md",
        report_bytes,
        role="independent-review-evidence",
    )
    decision_bytes = _read_stable_public_file(decision)
    loaded_decision = _load_canonical_model(
        decision_bytes,
        label="implementation review decision",
        adapter=_IMPLEMENTATION_DECISION_ADAPTER,
    )
    attestation = ImplementationReviewAttestationV1(
        schema=M5_IMPLEMENTATION_REVIEW_SCHEMA,
        implementation_snapshot_sha256=snapshot.sha256,
        implementation_snapshot_entry_count=len(snapshot.entries),
        review_report_sha256=_sha256(report_bytes),
        reviewer=loaded_decision.reviewer,
        reviewer_identity_scope=("operator-recorded-not-cryptographically-authenticated"),
        reviewed_areas=loaded_decision.reviewed_areas,
        finding_ids=loaded_decision.finding_ids,
        p0_count=loaded_decision.p0_count,
        p1_count=loaded_decision.p1_count,
        p2_count=loaded_decision.p2_count,
        unresolved_finding_ids=loaded_decision.unresolved_finding_ids,
        disposition=loaded_decision.disposition,
    )
    _exclusive_file(output, canonical_json_bytes(attestation))
    return attestation


def load_implementation_review_attestation(
    path: Path,
) -> ImplementationReviewAttestationV1:
    """Strictly load one canonical implementation-review attestation."""

    return _load_canonical_model(
        _read_stable_public_file(path),
        label="implementation review attestation",
        adapter=TypeAdapter(ImplementationReviewAttestationV1),
    )


def load_software_verification(path: Path) -> SoftwareVerificationV1:
    """Strictly load one canonical clean-revision software attestation."""

    return _load_canonical_model(
        _read_stable_public_file(path),
        label="software verification",
        adapter=TypeAdapter(SoftwareVerificationV1),
    )


def write_software_verification_attestation(
    *,
    scientific_git_revision: str,
    implementation_snapshot_sha256: str,
    lockfile_sha256: str,
    package_version: str,
    checks: Sequence[SoftwareVerificationCheckV1],
    output: Path,
) -> SoftwareVerificationV1:
    """Write one no-overwrite deterministic software-verification record."""

    attestation = SoftwareVerificationV1(
        schema=M5_SOFTWARE_VERIFICATION_SCHEMA,
        scientific_git_revision=scientific_git_revision,
        implementation_snapshot_sha256=implementation_snapshot_sha256,
        lockfile_sha256=lockfile_sha256,
        package_version=package_version,
        checks=tuple(checks),
        all_checks_passed=all(check.passed for check in checks),
        evidence_scope=("operator-executed-command-status-and-normalized-output-commitments"),
    )
    if not attestation.all_checks_passed:
        raise ArtifactValidationError("M5 software verification did not pass")
    _exclusive_file(output, canonical_json_bytes(attestation))
    return attestation


@dataclass(frozen=True, slots=True)
class SoftwareCheckSpec:
    """One frozen logical check and its stable required-test selectors."""

    check_id: str
    command: tuple[str, ...]
    required_test_ids: tuple[str, ...] = ()


M5_SOFTWARE_CHECK_SPECS = (
    SoftwareCheckSpec("lockfile", ("uv", "lock", "--check")),
    SoftwareCheckSpec(
        "format",
        ("uv", "run", "--locked", "ruff", "format", "--check", "."),
    ),
    SoftwareCheckSpec(
        "lint",
        ("uv", "run", "--locked", "ruff", "check", "."),
    ),
    SoftwareCheckSpec("type", ("uv", "run", "--locked", "pyright")),
    SoftwareCheckSpec(
        "tests",
        ("uv", "run", "--locked", "pytest"),
        (
            "tests/test_nuscenes_geometry.py",
            "tests/test_replay_geometry.py",
            "tests/test_replay_source.py",
            "tests/test_replay_health.py",
            "tests/test_replay_runner.py",
            "tests/test_replay_release.py",
        ),
    ),
    SoftwareCheckSpec("build", ("uv", "build", "--no-sources")),
    SoftwareCheckSpec(
        "wheel-smoke",
        ("internal", "isolated-wheel-smoke-v1"),
        (
            "ffb --version",
            "ffb schema show replay-validation",
            "ffb schema show replay-execution-resource-evidence",
        ),
    ),
    SoftwareCheckSpec(
        "privacy",
        ("internal", "bounded-tracked-privacy-audit-v1"),
        (
            "candidate-role-aware-private-path-negative-controls",
            "release-exact-allowlist-negative-controls",
            "dataset-payload-tracking-audit",
        ),
    ),
)
if tuple(spec.check_id for spec in M5_SOFTWARE_CHECK_SPECS) != M5_SOFTWARE_CHECK_ORDER:
    raise RuntimeError("M5 software-check specifications do not use frozen order")


def _normalized_check_output(
    value: bytes,
    *,
    source_root: Path,
    additional_paths: Sequence[Path] = (),
) -> bytes:
    normalized = value.replace(os.fsencode(source_root), b"<SOURCE_ROOT>")
    for path in additional_paths:
        normalized = normalized.replace(os.fsencode(path), b"<EPHEMERAL_PATH>")
    normalized = re.sub(rb"\x1b\[[0-9;]*[A-Za-z]", b"", normalized)
    normalized = normalized.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    normalized = re.sub(
        rb"(?i)\b(?:in\s+)?[0-9]+(?:\.[0-9]+)?\s*"
        rb"(?:ms|milliseconds?|s|sec(?:onds?)?|m|min(?:utes?)?)\b",
        b"<DURATION>",
        normalized,
    )
    normalized = re.sub(
        rb"\b20[0-9]{2}-[01][0-9]-[0-3][0-9]"
        rb"(?:[T ][0-2][0-9]:[0-5][0-9]:[0-6][0-9](?:\.[0-9]+)?Z?)?\b",
        b"<DATE_OR_TIMESTAMP>",
        normalized,
    )
    if len(normalized) > 32 * 1024 * 1024:
        raise ArtifactValidationError("M5 software-check output exceeds its cap")
    return normalized


def _run_captured(
    command: Sequence[str],
    *,
    source_root: Path,
    additional_paths: Sequence[Path] = (),
) -> tuple[int, bytes]:
    output_cap = 32 * 1024 * 1024

    def limit_child_output() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_cap, output_cap))

    with tempfile.TemporaryFile(mode="w+b") as output:
        result = subprocess.run(
            tuple(command),
            cwd=source_root,
            check=False,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "NO_COLOR": "1",
                "PYTHONHASHSEED": "0",
            },
            preexec_fn=limit_child_output,
        )
        size = output.tell()
        if size > output_cap:
            raise ArtifactValidationError("M5 software-check output exceeds its cap")
        output.seek(0)
        combined = output.read(output_cap + 1)
        if len(combined) > output_cap:
            raise ArtifactValidationError("M5 software-check output exceeds its cap")
    return (
        result.returncode,
        _normalized_check_output(
            combined,
            source_root=source_root,
            additional_paths=additional_paths,
        ),
    )


def _isolated_wheel_smoke(source_root: Path) -> tuple[int, bytes]:
    dist = source_root / "dist"
    wheels = tuple(
        path
        for path in dist.glob("fusion_fault_bench-*.whl")
        if path.is_file() and not path.is_symlink()
    )
    if len(wheels) != 1:
        return 1, b"wheel-count-mismatch\n"
    ephemeral = Path(tempfile.mkdtemp(prefix="ffb-m5-wheel-smoke-", dir="/private/tmp"))
    outputs = bytearray()
    status = 0
    try:
        commands = (
            ("uv", "venv", "--python", "3.12", os.fspath(ephemeral)),
            (
                "uv",
                "pip",
                "install",
                "--python",
                os.fspath(ephemeral / "bin/python"),
                os.fspath(wheels[0]),
            ),
            (os.fspath(ephemeral / "bin/ffb"), "--version"),
            (
                os.fspath(ephemeral / "bin/ffb"),
                "schema",
                "show",
                "replay-validation",
            ),
            (
                os.fspath(ephemeral / "bin/ffb"),
                "schema",
                "show",
                "replay-execution-resource-evidence",
            ),
        )
        for command in commands:
            returncode, output = _run_captured(
                command,
                source_root=source_root,
                additional_paths=(ephemeral, wheels[0]),
            )
            outputs.extend(len(output).to_bytes(8, "big"))
            outputs.extend(output)
            if returncode != 0:
                status = returncode
                break
    finally:
        shutil.rmtree(ephemeral)
    return status, bytes(outputs)


def _tracked_privacy_audit(source_root: Path) -> tuple[int, bytes]:
    tracked_byte_cap = 64 * 1024 * 1024
    paths = _tracked_paths(source_root)
    forbidden_names = tuple(
        path
        for path in paths
        if path == "interview"
        or path.startswith(("interview/", "reports/generated/"))
        or PurePosixPath(path).suffix.casefold()
        in {
            ".7z",
            ".bag",
            ".bin",
            ".bmp",
            ".bz2",
            ".db3",
            ".gif",
            ".gz",
            ".jpeg",
            ".jpg",
            ".las",
            ".laz",
            ".mcap",
            ".npy",
            ".npz",
            ".pcd",
            ".ply",
            ".png",
            ".tar",
            ".tgz",
            ".webp",
            ".xz",
            ".zip",
        }
    )
    high_confidence_secret = re.compile(
        rb"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        rb"|AKIA[0-9A-Z]{16}"
        rb"|gh[pousr]_[A-Za-z0-9]{30,}"
        rb"|xox[baprs]-[A-Za-z0-9-]{20,})"
    )
    suspicious_content: list[str] = []
    total_bytes = 0
    content_set = bytearray()
    for path in paths:
        value = _strict_source_file(source_root, path, allow_empty=True)
        total_bytes += len(value)
        if total_bytes > tracked_byte_cap:
            raise ArtifactValidationError("M5 tracked privacy audit exceeds its total-byte cap")
        if high_confidence_secret.search(value):
            suspicious_content.append(path)
        path_bytes = path.encode("utf-8")
        content_set.extend(len(path_bytes).to_bytes(4, "big"))
        content_set.extend(path_bytes)
        content_set.extend(len(value).to_bytes(8, "big"))
        content_set.extend(hashlib.sha256(value).digest())
    passed = not forbidden_names and not suspicious_content
    payload = canonical_json_bytes(
        {
            "schema": "ffb.m5-tracked-privacy-audit/v1",
            "tracked_path_count": len(paths),
            "tracked_byte_count": total_bytes,
            "tracked_content_set_sha256": hashlib.sha256(content_set).hexdigest(),
            "forbidden_path_count": len(forbidden_names),
            "high_confidence_secret_match_count": len(suspicious_content),
            "passed": passed,
        }
    )
    return (0 if passed else 1), payload


def run_software_verification_checks(
    *,
    source_root: Path,
    output: Path,
) -> SoftwareVerificationV1:
    """Execute the exact frozen verification suite and write its attestation."""

    from fusion_fault_bench.contracts.replay_v1 import M5_REPLAY_INTENT_PATH
    from fusion_fault_bench.provenance import (
        ProvenanceError,
        discover_clean_source,
        verify_locked_execution,
    )

    root = source_root.resolve(strict=True)
    try:
        snapshot = discover_clean_source(root / M5_REPLAY_INTENT_PATH)
        verify_locked_execution(snapshot)
    except (OSError, ProvenanceError, ValueError) as error:
        raise ArtifactValidationError("M5 software verification requires clean source") from error
    implementation = compute_implementation_snapshot(root)
    checks: list[SoftwareVerificationCheckV1] = []
    failed = False
    for spec in M5_SOFTWARE_CHECK_SPECS:
        if failed:
            raise ArtifactValidationError(
                "M5 software verification stopped after the first failed check"
            )
        if spec.check_id == "wheel-smoke":
            exit_status, normalized = _isolated_wheel_smoke(root)
        elif spec.check_id == "privacy":
            exit_status, normalized = _tracked_privacy_audit(root)
        else:
            exit_status, normalized = _run_captured(spec.command, source_root=root)
        check = SoftwareVerificationCheckV1(
            check_id=spec.check_id,
            command=spec.command,
            required_test_ids=spec.required_test_ids,
            exit_status=exit_status,
            normalized_output_sha256=_sha256(normalized),
            passed=exit_status == 0,
        )
        checks.append(check)
        failed = not check.passed
    try:
        final_snapshot = discover_clean_source(root / M5_REPLAY_INTENT_PATH)
        verify_locked_execution(final_snapshot)
    except (OSError, ProvenanceError, ValueError) as error:
        raise ArtifactValidationError(
            "M5 software verification source changed during checks"
        ) from error
    final_implementation = compute_implementation_snapshot(root)
    if final_snapshot != snapshot or final_implementation != implementation:
        raise ArtifactValidationError("M5 software verification source changed during checks")
    return write_software_verification_attestation(
        scientific_git_revision=snapshot.git_revision,
        implementation_snapshot_sha256=implementation.sha256,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version=snapshot.package_version,
        checks=checks,
        output=output,
    )


@dataclass(frozen=True, slots=True)
class _SafeTreeSnapshot:
    root_stat: os.stat_result
    directories: Mapping[str, os.stat_result]
    files: Mapping[str, os.stat_result]


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_components(path: Path) -> None:
    absolute = absolute_artifact_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise ArtifactValidationError("M5 tree path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactValidationError("M5 tree path contains a symlink")


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _walk_tree_fd(
    directory_fd: int,
    *,
    prefix: str = "",
) -> tuple[dict[str, os.stat_result], dict[str, os.stat_result]]:
    directories: dict[str, os.stat_result] = {}
    files: dict[str, os.stat_result] = {}
    try:
        with os.scandir(directory_fd) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name.encode("utf-8"))
    except OSError as error:
        raise ArtifactValidationError("M5 tree cannot be inspected") from error
    for entry in entries:
        if "/" in entry.name or "\\" in entry.name or entry.name in {"", ".", ".."}:
            raise ArtifactValidationError("M5 tree contains an unsafe entry name")
        relative = f"{prefix}/{entry.name}" if prefix else entry.name
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ArtifactValidationError("M5 tree entry cannot be inspected") from error
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(entry.name, _directory_open_flags(), dir_fd=directory_fd)
            try:
                opened = os.fstat(child)
                if _stat_fingerprint(opened) != _stat_fingerprint(metadata):
                    raise ArtifactValidationError("M5 directory changed during traversal")
                directories[relative] = metadata
                nested_directories, nested_files = _walk_tree_fd(
                    child,
                    prefix=relative,
                )
                directories.update(nested_directories)
                files.update(nested_files)
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            files[relative] = metadata
        else:
            raise ArtifactValidationError(
                "M5 tree entries must be real directories or private regular files"
            )
    return directories, files


def _expected_directories(paths: Sequence[str]) -> frozenset[str]:
    directories: set[str] = set()
    for path in paths:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
            raise ArtifactValidationError("M5 allowlist path is unsafe")
        parent = pure.parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return frozenset(directories)


def _snapshot_tree(
    root: Path,
    *,
    exact_paths: Sequence[str],
) -> tuple[int, _SafeTreeSnapshot]:
    absolute = absolute_artifact_path(root)
    _reject_symlink_components(absolute)
    try:
        root_fd = os.open(absolute, _directory_open_flags())
    except OSError as error:
        raise ArtifactValidationError("M5 tree root must be a real directory") from error
    try:
        root_stat = os.fstat(root_fd)
        path_stat = os.stat(absolute, follow_symlinks=False)
        if _stat_fingerprint(root_stat) != _stat_fingerprint(path_stat):
            raise ArtifactValidationError("M5 tree root changed during validation")
        directories, files = _walk_tree_fd(root_fd)
        if set(files) != set(exact_paths):
            raise ArtifactValidationError("M5 tree file allowlist mismatch")
        if set(directories) != set(_expected_directories(exact_paths)):
            raise ArtifactValidationError("M5 tree directory allowlist mismatch")
        if any(value.st_size > M5_RELEASE_MEMBER_MAX_BYTES for value in files.values()):
            raise ArtifactValidationError("M5 tree member exceeds its cap")
        snapshot = _SafeTreeSnapshot(
            root_stat=root_stat,
            directories=MappingProxyType(directories),
            files=MappingProxyType(files),
        )
        return root_fd, snapshot
    except BaseException:
        os.close(root_fd)
        raise


def _open_descendant_file(
    root_fd: int,
    relative: str,
    *,
    expected: os.stat_result,
) -> int:
    parts = PurePosixPath(relative).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, _directory_open_flags(), dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stat_fingerprint(opened) != _stat_fingerprint(expected)
        ):
            os.close(descriptor)
            raise ArtifactValidationError("M5 tree member changed during validation")
        return descriptor
    finally:
        os.close(current)


def _read_tree_member(
    root_fd: int,
    relative: str,
    *,
    expected: os.stat_result,
) -> bytes:
    descriptor = _open_descendant_file(root_fd, relative, expected=expected)
    try:
        chunks: list[bytes] = []
        remaining = M5_RELEASE_MEMBER_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(M5_RELEASE_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) != expected.st_size or len(value) > M5_RELEASE_MEMBER_MAX_BYTES:
            raise ArtifactValidationError("M5 tree member changed during validation")
        return value
    finally:
        os.close(descriptor)


def _verify_tree_snapshot(
    root: Path,
    root_fd: int,
    snapshot: _SafeTreeSnapshot,
) -> None:
    current_root = os.fstat(root_fd)
    path_root = os.stat(root, follow_symlinks=False)
    if _stat_fingerprint(current_root) != _stat_fingerprint(
        snapshot.root_stat
    ) or _stat_fingerprint(path_root) != _stat_fingerprint(snapshot.root_stat):
        raise ArtifactValidationError("M5 tree root changed during validation")
    directories, files = _walk_tree_fd(root_fd)
    if set(directories) != set(snapshot.directories) or set(files) != set(snapshot.files):
        raise ArtifactValidationError("M5 tree allowlist changed during validation")
    for path, expected in snapshot.directories.items():
        current = directories[path]
        if _stat_fingerprint(current) != _stat_fingerprint(expected):
            raise ArtifactValidationError("M5 tree directory changed during validation")
    for path, expected in snapshot.files.items():
        current = files[path]
        if _stat_fingerprint(current) != _stat_fingerprint(expected):
            raise ArtifactValidationError("M5 tree member changed during validation")


def _load_exact_tree(
    root: Path,
    *,
    exact_paths: Sequence[str],
) -> Mapping[str, bytes]:
    root_fd, snapshot = _snapshot_tree(root, exact_paths=exact_paths)
    try:
        values = {
            path: _read_tree_member(
                root_fd,
                path,
                expected=snapshot.files[path],
            )
            for path in exact_paths
        }
        if sum(len(value) for value in values.values()) > M5_RELEASE_PACKAGE_MAX_BYTES:
            raise ArtifactValidationError("M5 tree exceeds the 50 MiB cap")
        _verify_tree_snapshot(root, root_fd, snapshot)
        return MappingProxyType(values)
    finally:
        os.close(root_fd)


def _mkdir_tree_at(root_fd: int, paths: Sequence[str]) -> None:
    opened: dict[str, int] = {"": os.dup(root_fd)}
    try:
        for relative in sorted(
            _expected_directories(paths),
            key=lambda value: (len(PurePosixPath(value).parts), value.encode("utf-8")),
        ):
            pure = PurePosixPath(relative)
            parent = pure.parent.as_posix()
            if parent == ".":
                parent = ""
            parent_fd = opened[parent]
            os.mkdir(pure.name, mode=0o700, dir_fd=parent_fd)
            descriptor = os.open(pure.name, _directory_open_flags(), dir_fd=parent_fd)
            opened[relative] = descriptor
    finally:
        for descriptor in opened.values():
            os.close(descriptor)


def _open_parent_at(root_fd: int, relative: str) -> tuple[int, str]:
    pure = PurePosixPath(relative)
    current = os.dup(root_fd)
    try:
        for part in pure.parts[:-1]:
            child = os.open(part, _directory_open_flags(), dir_fd=current)
            os.close(current)
            current = child
        return current, pure.name
    except BaseException:
        os.close(current)
        raise


def _open_directory_at(root_fd: int, relative: str) -> int:
    current = os.dup(root_fd)
    try:
        for part in PurePosixPath(relative).parts:
            child = os.open(part, _directory_open_flags(), dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _write_tree_member_at(root_fd: int, relative: str, value: bytes) -> None:
    parent_fd, name = _open_parent_at(root_fd, relative)
    try:
        write_exclusive_file_at(parent_fd, name, value)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _fsync_tree_at(root_fd: int, paths: Sequence[str]) -> None:
    for relative in sorted(
        _expected_directories(paths),
        key=lambda value: len(PurePosixPath(value).parts),
        reverse=True,
    ):
        directory_fd = _open_directory_at(root_fd, relative)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    os.fsync(root_fd)


def _cleanup_staging_tree_at(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    paths: Sequence[str],
) -> None:
    for relative in reversed(tuple(paths)):
        with suppress(OSError):
            parent, name = _open_parent_at(staging_fd, relative)
            try:
                os.unlink(name, dir_fd=parent)
            finally:
                os.close(parent)
    for relative in sorted(
        _expected_directories(paths),
        key=lambda value: len(PurePosixPath(value).parts),
        reverse=True,
    ):
        with suppress(OSError):
            parent, name = _open_parent_at(staging_fd, relative)
            try:
                os.rmdir(name, dir_fd=parent)
            finally:
                os.close(parent)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _publish_exact_tree[LoadedT](
    *,
    destination: Path,
    files: Mapping[str, bytes],
    exact_paths: Sequence[str],
    source_root: Path,
    loader: Callable[[Path], LoadedT],
) -> LoadedT:
    if tuple(files) != tuple(exact_paths):
        raise ArtifactValidationError("M5 publication bytes do not use exact canonical order")
    target = absolute_artifact_path(destination)
    reject_git_metadata_destination(target, discover_git_metadata_dirs(source_root))
    if os.path.lexists(target):
        raise FileExistsError("M5 publication destination already exists")
    parent_fd = open_or_create_real_directory(target.parent)
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            target.parent,
            label="M5 publication parent",
        )
        reject_directory_descriptor_in_git_metadata(
            parent_fd,
            discover_git_metadata_dirs(source_root),
        )
        if entry_exists_at(parent_fd, target.name):
            raise FileExistsError("M5 publication destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = target.parent / staging_name
        renamed = False
        try:
            _mkdir_tree_at(staging_fd, exact_paths)
            for path in exact_paths:
                _write_tree_member_at(staging_fd, path, files[path])
            _fsync_tree_at(staging_fd, exact_paths)
            loaded_staging = loader(staging)
            assert_directory_descriptor_matches_path(
                parent_fd,
                target.parent,
                label="M5 publication parent",
            )
            assert_directory_descriptor_matches_path(
                staging_fd,
                staging,
                label="M5 publication staging root",
            )
            if entry_exists_at(parent_fd, target.name):
                raise FileExistsError("M5 publication destination already exists")
            atomic_rename_directory_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            renamed = True
            try:
                os.fsync(parent_fd)
            except OSError as error:
                raise ArtifactValidationError(
                    "M5 publication durability is indeterminate after rename"
                ) from error
            assert_directory_descriptor_matches_path(
                parent_fd,
                target.parent,
                label="M5 publication parent",
            )
            assert_directory_descriptor_matches_path(
                staging_fd,
                target,
                label="M5 published tree",
            )
            loaded = loader(target)
            if loaded != loaded_staging:
                # Path is intentionally part of loaded values, so compare only if the
                # dataclass implementation elects to make equality path-independent.
                pass
            return loaded
        except BaseException:
            if not renamed:
                _cleanup_staging_tree_at(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=staging_name,
                    paths=exact_paths,
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def _validate_presentation_templates(members: Mapping[str, bytes]) -> None:
    templates = tuple(members[path] for path in M5_CANDIDATE_PRESENTATION_PATHS[:3])
    combined = b"\n".join(templates)
    for placeholder in M5_PRESENTATION_PLACEHOLDERS:
        if combined.count(placeholder.encode("ascii")) == 0:
            raise ArtifactValidationError("M5 presentation template omits an identity placeholder")
    unknown = set(re.findall(rb"@M5_[A-Z0-9_]+@", combined)) - {
        value.encode("ascii") for value in M5_PRESENTATION_PLACEHOLDERS
    }
    if unknown:
        raise ArtifactValidationError("M5 presentation template uses an unknown placeholder")


def load_review_candidate(path: Path) -> LoadedReviewCandidate:
    """Strictly reload the exact immutable 34-file candidate tree."""

    root = absolute_artifact_path(path)
    files = _load_exact_tree(root, exact_paths=M5_REVIEW_CANDIDATE_PATHS)
    index_bytes = files["candidate-index.json"]
    index = _load_canonical_model(
        index_bytes,
        label="M5 candidate index",
        adapter=TypeAdapter(ReviewCandidateIndexV1),
    )
    members = MappingProxyType({path: files[path] for path in M5_CANDIDATE_MEMBER_PATHS})
    entries = _candidate_entries(members)
    if entries != index.files:
        raise ArtifactValidationError("M5 candidate members disagree with their index")
    _validate_presentation_templates(members)
    return LoadedReviewCandidate(
        path=root,
        index=index,
        index_bytes=index_bytes,
        members=members,
        candidate_sha256=index.candidate_sha256,
        index_sha256=_sha256(index_bytes),
    )


def validate_review_candidate(path: Path) -> LoadedReviewCandidate:
    """Public alias emphasizing that successful load is full candidate validation."""

    return load_review_candidate(path)


def write_review_candidate(
    material: CandidateMaterial,
    destination: Path,
    *,
    source_root: Path,
) -> LoadedReviewCandidate:
    """Publish one immutable no-overwrite candidate and strictly reload it."""

    index = build_review_candidate_index(material)
    files = MappingProxyType(
        {
            "candidate-index.json": canonical_json_bytes(index),
            **{path: material.members[path] for path in M5_CANDIDATE_MEMBER_PATHS},
        }
    )
    return _publish_exact_tree(
        destination=destination,
        files=files,
        exact_paths=M5_REVIEW_CANDIDATE_PATHS,
        source_root=source_root,
        loader=load_review_candidate,
    )


_VALIDATION_INPUT_ADAPTER = TypeAdapter(ValidationInputsV1)
_RESULTS_REVIEW_ADAPTER = TypeAdapter(ResultsReviewAttestationV1)


def _validation_authorities(
    members: Mapping[str, bytes],
    *,
    results_review_attestation: bytes | None,
) -> Mapping[str, tuple[bytes, ...]]:
    def selected(*paths: str) -> tuple[bytes, ...]:
        try:
            return tuple(members[path] for path in paths)
        except KeyError as error:
            raise ArtifactValidationError("M5 validation authority is incomplete") from error

    claim_path = "presentation/public-claim-projections.json"
    software_path = "evidence/software-verification.json"
    implementation_path = "evidence/implementation-review-attestation.json"
    privacy_path = "evidence/privacy-license-attestation.json"
    authorities: dict[str, tuple[bytes, ...]] = {
        "intent-freeze": selected(
            "machine/intent.json",
            "evidence/release-pipeline-plan.md",
            "evidence/release-pipeline-plan-review.md",
            "evidence/resource-scope-amendment.md",
        ),
        "fixed-scene-population": selected(
            "machine/intent.json",
            "machine/replay-profile-summary.json",
            claim_path,
        ),
        "base-support": selected(
            "machine/replay-profile-summary.json",
            "machine/descriptor-aggregates.ndjson",
            "machine/source-member-commitments.ndjson",
        ),
        "health-schedules": selected(
            "machine/replay-profile-summary.json",
            "machine/health-panel-aggregates.ndjson",
            claim_path,
        ),
        "transform-and-timing-oracles": selected(
            software_path,
            implementation_path,
        ),
        "eligibility-and-fault-causality": selected(
            software_path,
            "machine/persistent-panel-aggregates.ndjson",
            claim_path,
        ),
        "health-feature-leakage": selected(
            software_path,
            "machine/health-panel-aggregates.ndjson",
        ),
        "persistent-panel-completeness": selected(
            "machine/persistent-panel-aggregates.ndjson",
            "machine/persistent-panel-crossovers.ndjson",
            claim_path,
        ),
        "health-panel-completeness": selected(
            "machine/health-panel-aggregates.ndjson",
            claim_path,
        ),
        "scene-bootstrap-and-cluster-sensitivity": selected(
            "machine/persistent-panel-aggregates.ndjson",
            "machine/health-panel-aggregates.ndjson",
            "machine/leave-one-cluster-sensitivity.ndjson",
        ),
        "repeat-scientific-members": selected(
            "machine/repeat-verification.json",
            "machine/source-member-commitments.ndjson",
        ),
        "cpu-and-memory-caps": selected(
            "machine/replay-profile-summary.json",
        ),
        "no-raw-payload-reads": selected(
            "machine/replay-profile-summary.json",
            privacy_path,
        ),
        "privacy-and-dataset-license": selected(
            privacy_path,
            "presentation/README.md",
            "presentation/claim-evidence.md",
            "presentation/release-summary.json",
        ),
        "implementation-review": selected(
            "evidence/implementation-review.md",
            implementation_path,
        ),
        "results-and-claims-review": (
            () if results_review_attestation is None else (results_review_attestation,)
        ),
        "software-verification": selected(software_path),
    }
    if tuple(authorities) != M5_RELEASE_VALIDATION_CHECK_IDS:
        raise RuntimeError("M5 validation authority map does not use frozen order")
    return MappingProxyType(authorities)


def build_validation_inputs(
    *,
    run_id: str,
    candidate_members_without_validation_inputs: Mapping[str, bytes],
) -> ValidationInputsV1:
    """Derive the first sixteen candidate authorities and mark review pending."""

    authorities = _validation_authorities(
        candidate_members_without_validation_inputs,
        results_review_attestation=None,
    )
    records = tuple(
        ValidationInputRecordV1(
            check_id=check_id,
            status=("pending" if check_id == "results-and-claims-review" else "passed"),
            authority_sha256=(
                None
                if check_id == "results-and-claims-review"
                else derive_validation_evidence_sha256(
                    check_id,
                    authorities[check_id],
                )
            ),
            evidence_scope=(
                "pending-independent-candidate-review"
                if check_id == "results-and-claims-review"
                else "content-derived-from-named-candidate-authorities"
            ),
        )
        for check_id in M5_RELEASE_VALIDATION_CHECK_IDS
    )
    return ValidationInputsV1(
        schema=M5_VALIDATION_INPUTS_SCHEMA,
        run_id=run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        records=records,
    )


def validate_validation_inputs(
    members: Mapping[str, bytes],
    *,
    expected_run_id: str,
) -> ValidationInputsV1:
    """Independently reconstruct all pre-review check digests."""

    value = members.get("evidence/validation-inputs.json")
    if value is None:
        raise ArtifactValidationError("M5 candidate validation inputs are missing")
    loaded = _load_canonical_model(
        value,
        label="M5 validation inputs",
        adapter=_VALIDATION_INPUT_ADAPTER,
    )
    if loaded.run_id != expected_run_id:
        raise ArtifactValidationError("M5 validation inputs have the wrong run binding")
    expected = build_validation_inputs(
        run_id=expected_run_id,
        candidate_members_without_validation_inputs=members,
    )
    if loaded != expected:
        raise ArtifactValidationError("M5 validation inputs disagree with their authorities")
    return loaded


def build_final_replay_validation(
    *,
    run_id: str,
    candidate_members: Mapping[str, bytes],
    results_review_attestation_bytes: bytes,
) -> ReplayValidationV1:
    """Derive all seventeen final validation checks from exact named bytes."""

    validate_validation_inputs(candidate_members, expected_run_id=run_id)
    authorities = _validation_authorities(
        candidate_members,
        results_review_attestation=results_review_attestation_bytes,
    )
    checks = tuple(
        ReplayValidationCheckV1(
            check_id=check_id,
            passed=True,
            evidence_sha256=derive_validation_evidence_sha256(
                check_id,
                authorities[check_id],
            ),
        )
        for check_id in M5_RELEASE_VALIDATION_CHECK_IDS
    )
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


def validate_final_replay_validation(
    validation: ReplayValidationV1,
    *,
    candidate_members: Mapping[str, bytes],
    results_review_attestation_bytes: bytes,
) -> None:
    """Require final machine validation to equal independent reconstruction."""

    expected = build_final_replay_validation(
        run_id=validation.run_id,
        candidate_members=candidate_members,
        results_review_attestation_bytes=results_review_attestation_bytes,
    )
    if validation != expected:
        raise ArtifactValidationError("M5 machine validation digests are not authoritative")


def _candidate_review_set_digests(
    candidate: LoadedReviewCandidate,
) -> Mapping[str, str]:
    members = candidate.members
    return MappingProxyType(
        {
            "scientific_member_set_sha256": compute_member_set_digest(
                label="scientific-members",
                paths=M5_MACHINE_CANDIDATE_PATHS,
                members=members,
            ),
            "claim_projection_set_sha256": compute_member_set_digest(
                label="claim-projections",
                paths=(
                    "presentation/release-summary.json",
                    "presentation/public-claim-projections.json",
                ),
                members=members,
            ),
            "figure_spec_set_sha256": compute_member_set_digest(
                label="figure-specs",
                paths=tuple(
                    path for path in M5_CANDIDATE_FIGURE_PATHS if path.endswith(".spec.json")
                ),
                members=members,
            ),
            "rendered_figure_set_sha256": compute_member_set_digest(
                label="rendered-figures",
                paths=tuple(path for path in M5_CANDIDATE_FIGURE_PATHS if path.endswith(".svg")),
                members=members,
            ),
            "presentation_template_set_sha256": compute_member_set_digest(
                label="presentation-templates",
                paths=M5_CANDIDATE_PRESENTATION_PATHS[:3],
                members=members,
            ),
        }
    )


def attest_results_review(
    *,
    candidate: Path,
    review_report: Path,
    decision: Path,
    output: Path,
) -> ResultsReviewAttestationV1:
    """Canonicalize a reviewer-authored disposition for one exact candidate."""

    loaded_candidate = load_review_candidate(candidate)
    report_bytes = _read_stable_public_file(review_report)
    scan_public_member(
        "evidence/results-review.md",
        report_bytes,
        role="independent-review-evidence",
    )
    decision_model = _load_canonical_model(
        _read_stable_public_file(decision),
        label="M5 results review decision",
        adapter=_RESULTS_DECISION_ADAPTER,
    )
    set_digests = _candidate_review_set_digests(loaded_candidate)
    attestation = ResultsReviewAttestationV1(
        schema=M5_RESULTS_REVIEW_SCHEMA,
        candidate_sha256=loaded_candidate.candidate_sha256,
        candidate_index_sha256=loaded_candidate.index_sha256,
        scientific_member_set_sha256=set_digests["scientific_member_set_sha256"],
        claim_projection_set_sha256=set_digests["claim_projection_set_sha256"],
        figure_spec_set_sha256=set_digests["figure_spec_set_sha256"],
        rendered_figure_set_sha256=set_digests["rendered_figure_set_sha256"],
        presentation_template_set_sha256=(set_digests["presentation_template_set_sha256"]),
        review_report_sha256=_sha256(report_bytes),
        reviewer=decision_model.reviewer,
        reviewer_identity_scope=("operator-recorded-not-cryptographically-authenticated"),
        p0_count=decision_model.p0_count,
        p1_count=decision_model.p1_count,
        p2_count=decision_model.p2_count,
        unresolved_finding_ids=decision_model.unresolved_finding_ids,
        negative_and_undefined_results_reviewed_and_retained=(
            decision_model.negative_and_undefined_results_reviewed_and_retained
        ),
        limitations_reviewed_and_retained=(decision_model.limitations_reviewed_and_retained),
        disposition=decision_model.disposition,
    )
    _exclusive_file(output, canonical_json_bytes(attestation))
    return attestation


def load_results_review_attestation(
    path: Path,
    *,
    candidate: LoadedReviewCandidate | None = None,
    review_report_bytes: bytes | None = None,
) -> ResultsReviewAttestationV1:
    """Strictly load and optionally bind a results review to exact reviewed bytes."""

    value = _read_stable_public_file(path)
    attestation = _load_canonical_model(
        value,
        label="M5 results review attestation",
        adapter=_RESULTS_REVIEW_ADAPTER,
    )
    if candidate is not None:
        digests = _candidate_review_set_digests(candidate)
        if (
            attestation.candidate_sha256 != candidate.candidate_sha256
            or attestation.candidate_index_sha256 != candidate.index_sha256
            or any(getattr(attestation, name) != digest for name, digest in digests.items())
        ):
            raise ArtifactValidationError("M5 results review does not bind the exact candidate")
    if review_report_bytes is not None and attestation.review_report_sha256 != _sha256(
        review_report_bytes
    ):
        raise ArtifactValidationError("M5 results review report binding is invalid")
    if attestation.disposition != "pass":
        raise ArtifactValidationError("M5 results review does not permit release")
    return attestation


_M5_RELEASE_PLAN_SOURCE = Path("docs/m5-release-pipeline-plan.md")
_M5_RELEASE_PLAN_REVIEW_SOURCE = Path("docs/reviews/m5-release-pipeline-plan-review.md")
_M5_RESOURCE_AMENDMENT_SOURCE = Path("docs/m5-resource-scope-amendment.md")
_M5_IMPLEMENTATION_REVIEW_SOURCE = Path("docs/reviews/m5-release-implementation-review.md")
_M5_IMPLEMENTATION_REVIEW_ATTESTATION_SOURCE = Path(
    "docs/reviews/m5-release-implementation-review-attestation.json"
)
_M5_REVIEW_CANDIDATE_ROOT = Path("reports/generated")
_M5_FINAL_RELEASE_RELATIVE = Path("reports/releases/m5-nuscenes-replay-v0.1.0")


def _typed_record_dump(value: object) -> dict[str, object]:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise ArtifactValidationError("M5 claim source is not a typed record")
    dumped = model_dump(mode="json", by_alias=True)
    if not isinstance(dumped, dict):
        raise ArtifactValidationError("M5 claim source is not a mapping")
    return cast("dict[str, object]", dumped)


def _build_public_claim_projections(
    *,
    evidence: ReplayCuratedAggregateEvidence,
    figure_files: Mapping[str, bytes],
    repeat_verification: object,
) -> bytes:
    from fusion_fault_bench.contracts.replay_artifact_v1 import (
        M5_HEALTH_HYPOTHESIS_COORDINATE_COUNT,
        M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256,
        M5_PERSISTENT_HYPOTHESIS_COORDINATE_COUNT,
        M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256,
        M5_TRACKED_AGGREGATE_TERMS,
    )

    persistent = cast("Sequence[object]", evidence.persistent_aggregates)
    health = cast("Sequence[object]", evidence.health_aggregates)
    crossovers = cast("Sequence[object]", evidence.persistent_crossovers)
    profile = evidence.profile_summary
    hypothesis_rows = tuple(
        _typed_record_dump(record)
        for record in (*persistent, *health)
        if getattr(record, "hypothesis_id", None) is not None
    )
    persistent_hypothesis_count = sum(
        str(row.get("hypothesis_id", "")).startswith("h5-a") for row in hypothesis_rows
    )
    health_hypothesis_count = sum(
        str(row.get("hypothesis_id", "")).startswith("h5-b") for row in hypothesis_rows
    )
    if (
        persistent_hypothesis_count != M5_PERSISTENT_HYPOTHESIS_COORDINATE_COUNT
        or health_hypothesis_count != M5_HEALTH_HYPOTHESIS_COORDINATE_COUNT
    ):
        raise ArtifactValidationError("M5 public hypothesis registry is incomplete")

    figure_projections: list[dict[str, object]] = []
    for stem in M5_FIGURE_STEMS:
        path = f"figures/{stem}.spec.json"
        spec = _strict_json_mapping(figure_files[path], label=path)
        marks = spec.get("marks")
        if not isinstance(marks, list):
            raise ArtifactValidationError("M5 figure projection registry is malformed")
        typed_marks = cast("list[object]", marks)
        figure_projections.append(
            {
                "figure_id": spec.get("figure_id"),
                "figure_kind": spec.get("figure_kind"),
                "source_member": path,
                "mark_count": len(typed_marks),
                "marks": typed_marks,
            }
        )
    resource_evidence = cast("Sequence[object]", profile.resource_evidence)
    registry = {
        "schema": M5_PUBLIC_CLAIMS_SCHEMA,
        "release_id": M5_REPLAY_RELEASE_ID,
        "run_id": profile.run_id,
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "hypothesis_registry": {
            "m5_a": {
                "coordinate_count": M5_PERSISTENT_HYPOTHESIS_COORDINATE_COUNT,
                "coordinate_set_sha256": (M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256),
            },
            "m5_b": {
                "coordinate_count": M5_HEALTH_HYPOTHESIS_COORDINATE_COUNT,
                "coordinate_set_sha256": M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256,
            },
            "records": list(hypothesis_rows),
        },
        "figure_projections": figure_projections,
        "crossovers": [_typed_record_dump(record) for record in crossovers],
        "dropout_nesting": {
            "derivation": (
                "fixed-predeclared-camera-dropout-selector-order;"
                "higher-dropout-membership-is-a-superset-under-one-shared-uniform-draw"
            ),
            "source_figure_id": "m5-persistent-panel-summary",
        },
        "resources": [_typed_record_dump(record) for record in resource_evidence],
        "repeat_facts": _typed_record_dump(repeat_verification),
        "population_facts": {
            "scene_count": profile.scene_count,
            "persistent_experiment_count": profile.persistent_experiment_count,
            "health_experiment_count": profile.health_experiment_count,
            "replay_experiment_count": profile.replay_experiment_count,
            "distinct_log_group_count": profile.distinct_log_group_count,
        },
        "finalization_metadata": {
            "machine_artifact_byte_length": None,
            "source": "artifact/release-index.json",
            "candidate_value_status": "pending-finalization",
        },
        "numeric_rendering": {
            "overview_and_figure_finite_float": ".6g",
            "elapsed_seconds": ".2f",
            "integers": "exact",
            "unsupported_states": ("undefined-not-applicable-censored-positive-infinity-literal"),
        },
        "claim_boundary": (
            "matched-center-estimator-output-proxy-only;"
            "no-raw-sensor-detector-fleet-planning-production-or-safety-claim"
        ),
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
        "attribution": "nuScenes / Motional; aggregate replay evidence only",
        "non_endorsement": True,
    }
    return canonical_json_bytes(registry)


def _format_public_scalar(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(0.0 if value == 0.0 else value, ".6g")
    return str(value)


def _presentation_templates(
    *,
    public_claims_bytes: bytes,
) -> Mapping[str, bytes]:
    registry = _strict_json_mapping(
        public_claims_bytes,
        label="public claim projections",
    )
    hypothesis = cast("dict[str, object]", registry["hypothesis_registry"])
    rows = cast("list[dict[str, object]]", hypothesis["records"])
    claim_lines = [
        "# M5 claim-evidence ledger",
        "",
        (
            "Every numeric token below is generated from "
            "`public-claim-projections.json`; failed, undefined, control, and "
            "non-persistent outcomes remain visible."
        ),
        "",
        (
            "| Hypothesis | Selector | Method | Metric | Status | Estimate | "
            "Lower | Upper | Persistence |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        claim_lines.append(
            "| "
            + " | ".join(
                (
                    str(row["hypothesis_id"]),
                    str(row["condition_selector"]),
                    str(row["method_id"]),
                    str(row["metric_id"]),
                    str(row["status"]),
                    _format_public_scalar(row.get("estimate")),
                    _format_public_scalar(row.get("interval_lower")),
                    _format_public_scalar(row.get("interval_upper")),
                    str(row["persistence_label"]),
                )
            )
            + " |"
        )
    identity_block = (
        "\n\nMachine artifact SHA-256: `@M5_RELEASE_ARTIFACT_SHA256@`  \n"
        "Run SHA-256: `@M5_RELEASE_RUN_SHA256@`  \n"
        "Results-review attestation SHA-256: "
        "`@M5_RESULTS_REVIEW_ATTESTATION_SHA256@`  \n"
        "Strict machine artifact bytes: `@M5_MACHINE_ARTIFACT_BYTES@`\n"
    )
    terms = (
        "\nEvidence terms: CC BY-NC-SA 4.0 plus Motional Dataset Terms; "
        "attribution required; no endorsement. The Apache-2.0 repository "
        "license does not relicense aggregate evidence.\n"
    )
    overview = (
        "# Fusion Fault Bench — M5 nuScenes-mini latent-scene replay\n\n"
        "This release evaluates matched-center estimator-output proxy loss over "
        "the fixed ten-scene mini population. It publishes all preregistered "
        "positive, negative, control, undefined, crossover, and finite-cluster "
        "sensitivity evidence. It does not evaluate raw-sensor detector "
        "performance, fleet generalization, planning, production readiness, or safety.\n\n"
        "The fixed M4 health rule is apply-only; no M5 outcome was used to refit "
        "or select the rule. See `claim-evidence.md` and the five deterministic figures."
        + identity_block
        + terms
    )
    verification = (
        "# M5 verification\n\n"
        "The 14-file machine artifact, 26 indexed sidecars, immutable review "
        "candidate, independent results review, seventeen check-specific "
        "validation digests, and both package identities are independently "
        "reconstructed by the offline validator. Dataset bytes are not "
        "authenticated and raw timing logs remain local." + identity_block + terms
    )
    claim_ledger = "\n".join(claim_lines) + identity_block + terms
    return MappingProxyType(
        {
            "presentation/README.md": overview.encode("utf-8"),
            "presentation/claim-evidence.md": claim_ledger.encode("utf-8"),
            "presentation/verification.md": verification.encode("utf-8"),
        }
    )


def _release_summary_bytes(
    *,
    public_claims_bytes: bytes,
) -> bytes:
    claims = _strict_json_mapping(
        public_claims_bytes,
        label="public claim projections",
    )
    hypothesis = cast("dict[str, object]", claims["hypothesis_registry"])
    figures = cast("list[dict[str, object]]", claims["figure_projections"])
    return canonical_json_bytes(
        {
            "schema": "ffb.m5-release-summary/v1",
            "release_id": M5_REPLAY_RELEASE_ID,
            "run_id": claims["run_id"],
            "hypothesis_record_count": len(cast("list[object]", hypothesis["records"])),
            "crossover_record_count": len(cast("list[object]", claims["crossovers"])),
            "figure_count": len(figures),
            "figure_mark_counts": [
                {
                    "figure_id": figure["figure_id"],
                    "mark_count": figure["mark_count"],
                }
                for figure in figures
            ],
            "claim_projection_sha256": _sha256(public_claims_bytes),
            "claim_boundary": claims["claim_boundary"],
            "tracked_aggregate_terms": claims["tracked_aggregate_terms"],
            "attribution": claims["attribution"],
            "non_endorsement": True,
        }
    )


def _privacy_attestation_bytes() -> bytes:
    return canonical_json_bytes(
        PrivacyLicenseAttestationV1(
            schema=M5_PRIVACY_LICENSE_SCHEMA,
            candidate_scanned_path_count=33,
            raw_sensor_payload_reads=0,
            dataset_root_serialized=False,
            local_sequence_rows_published=False,
            raw_resource_logs_published=False,
            private_or_credential_material_found=False,
            dataset_terms="CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms",
            attribution_required=True,
            non_endorsement_required=True,
            repository_license_does_not_relicense_evidence=True,
            all_checks_passed=True,
            evidence_scope=("deterministic-bounded-content-scan-and-operator-read-accounting"),
        )
    )


def _regenerate_candidate_material(
    inputs: M5ReleaseLocalInputs,
    *,
    source_root: Path,
) -> CandidateMaterial:
    """Freshly re-curate all reviewed bytes from exactly five local inputs."""

    from fusion_fault_bench.contracts.replay_v1 import M5_REPLAY_INTENT_PATH
    from fusion_fault_bench.provenance import (
        ProvenanceError,
        discover_clean_source,
        verify_locked_execution,
    )
    from fusion_fault_bench.replay_artifacts import canonical_replay_ndjson_bytes
    from fusion_fault_bench.replay_figures import build_replay_figure_bundle
    from fusion_fault_bench.replay_plan import load_replay_plan
    from fusion_fault_bench.replay_runner import (
        ReplayRunnerError,
        curate_replay_verified_repeat,
        verify_replay_repeat_artifacts,
    )

    root = source_root.resolve(strict=True)
    try:
        snapshot = discover_clean_source(root / M5_REPLAY_INTENT_PATH)
        verify_locked_execution(snapshot)
        implementation = compute_implementation_snapshot(root)
        software_bytes = _read_stable_public_file(inputs.software_verification)
        software = _load_canonical_model(
            software_bytes,
            label="M5 software verification",
            adapter=TypeAdapter(SoftwareVerificationV1),
        )
        if (
            not software.all_checks_passed
            or software.scientific_git_revision != snapshot.git_revision
            or software.implementation_snapshot_sha256 != implementation.sha256
            or software.lockfile_sha256 != snapshot.lockfile_sha256
            or software.package_version != snapshot.package_version
        ):
            raise ArtifactValidationError("M5 software verification has stale source authority")
        implementation_report = _strict_source_file(
            root,
            _M5_IMPLEMENTATION_REVIEW_SOURCE.as_posix(),
        )
        implementation_attestation_bytes = _strict_source_file(
            root,
            _M5_IMPLEMENTATION_REVIEW_ATTESTATION_SOURCE.as_posix(),
        )
        implementation_attestation = _load_canonical_model(
            implementation_attestation_bytes,
            label="M5 implementation review attestation",
            adapter=TypeAdapter(ImplementationReviewAttestationV1),
        )
        if (
            implementation_attestation.disposition != "pass"
            or implementation_attestation.implementation_snapshot_sha256 != implementation.sha256
            or implementation_attestation.implementation_snapshot_entry_count
            != len(implementation.entries)
            or implementation_attestation.review_report_sha256 != _sha256(implementation_report)
        ):
            raise ArtifactValidationError(
                "M5 implementation review has stale or blocking authority"
            )
        repeat = verify_replay_repeat_artifacts(
            primary_path=inputs.primary_artifact,
            repeat_path=inputs.repeat_artifact,
        )
        evidence = curate_replay_verified_repeat(
            repeat,
            primary_log_path=inputs.primary_time_l,
            repeat_log_path=inputs.repeat_time_l,
        )
        plan = load_replay_plan(source_root=root)
        figure_bundle = build_replay_figure_bundle(evidence, plan=plan)
        figure_files = figure_bundle.files()
        public_claims = _build_public_claim_projections(
            evidence=evidence,
            figure_files=figure_files,
            repeat_verification=repeat.repeat_verification,
        )
        templates = _presentation_templates(public_claims_bytes=public_claims)
        release_summary = _release_summary_bytes(public_claims_bytes=public_claims)
        partial: dict[str, bytes] = {
            "machine/intent.json": plan.intent.path.read_bytes(),
            "machine/replay-profile-summary.json": canonical_json_bytes(evidence.profile_summary),
            "machine/descriptor-aggregates.ndjson": canonical_replay_ndjson_bytes(
                evidence.descriptor_aggregates
            ),
            "machine/persistent-panel-aggregates.ndjson": (
                canonical_replay_ndjson_bytes(evidence.persistent_aggregates)
            ),
            "machine/persistent-panel-crossovers.ndjson": (
                canonical_replay_ndjson_bytes(evidence.persistent_crossovers)
            ),
            "machine/health-panel-aggregates.ndjson": canonical_replay_ndjson_bytes(
                evidence.health_aggregates
            ),
            "machine/leave-one-cluster-sensitivity.ndjson": (
                canonical_replay_ndjson_bytes(evidence.cluster_sensitivity)
            ),
            "machine/repeat-verification.json": canonical_json_bytes(repeat.repeat_verification),
            "machine/figure-records.ndjson": canonical_replay_ndjson_bytes(figure_bundle.bindings),
            "machine/source-member-commitments.ndjson": (
                canonical_replay_ndjson_bytes(repeat.source_commitments)
            ),
            "evidence/release-pipeline-plan.md": _strict_source_file(
                root, _M5_RELEASE_PLAN_SOURCE.as_posix()
            ),
            "evidence/release-pipeline-plan-review.md": _strict_source_file(
                root, _M5_RELEASE_PLAN_REVIEW_SOURCE.as_posix()
            ),
            "evidence/resource-scope-amendment.md": _strict_source_file(
                root, _M5_RESOURCE_AMENDMENT_SOURCE.as_posix()
            ),
            "evidence/implementation-review.md": implementation_report,
            "evidence/implementation-review-attestation.json": (implementation_attestation_bytes),
            "evidence/software-verification.json": software_bytes,
            "evidence/privacy-license-attestation.json": (_privacy_attestation_bytes()),
            **figure_files,
            **templates,
            "presentation/release-summary.json": release_summary,
            "presentation/public-claim-projections.json": public_claims,
        }
        validation_inputs = build_validation_inputs(
            run_id=evidence.run.run_id,
            candidate_members_without_validation_inputs=partial,
        )
        partial["evidence/validation-inputs.json"] = canonical_json_bytes(validation_inputs)
        members = MappingProxyType({path: partial[path] for path in M5_CANDIDATE_MEMBER_PATHS})
        _candidate_entries(members)
        validate_validation_inputs(members, expected_run_id=evidence.run.run_id)
        final_snapshot = discover_clean_source(root / M5_REPLAY_INTENT_PATH)
        verify_locked_execution(final_snapshot)
        if final_snapshot != snapshot or compute_implementation_snapshot(root) != implementation:
            raise ArtifactValidationError(
                "M5 source authority changed during candidate regeneration"
            )
    except (
        ArtifactValidationError,
        OSError,
        ProvenanceError,
        ReplayRunnerError,
        ValidationError,
        ValueError,
    ):
        raise ArtifactValidationError("M5 candidate regeneration failed closed") from None
    return CandidateMaterial(
        scientific_git_revision=snapshot.git_revision,
        implementation_snapshot_sha256=implementation.sha256,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version=snapshot.package_version,
        run_id=evidence.run.run_id,
        primary_local_artifact_sha256=repeat.primary.artifact_sha256,
        repeat_local_artifact_sha256=repeat.repeat.artifact_sha256,
        primary_run_sha256=repeat.primary.run_sha256,
        repeat_run_sha256=repeat.repeat.run_sha256,
        members=members,
    )


def prepare_review_candidate(
    inputs: M5ReleaseLocalInputs,
    *,
    output_dir: Path,
    source_root: Path,
) -> LoadedReviewCandidate:
    """Regenerate and atomically publish the exact ignored review candidate."""

    root = source_root.resolve(strict=True)
    if output_dir.is_absolute() or any(part in {"", ".", ".."} for part in output_dir.parts):
        raise ArtifactValidationError(
            "M5 review candidate output must be normalized repository-relative"
        )
    try:
        output_dir.relative_to(_M5_REVIEW_CANDIDATE_ROOT)
    except ValueError as error:
        raise ArtifactValidationError(
            "M5 review candidate output must remain under reports/generated"
        ) from error
    material = _regenerate_candidate_material(inputs, source_root=root)
    return write_review_candidate(
        material,
        root / output_dir,
        source_root=root,
    )


def _parse_ndjson_models[ModelT: ContractModel](
    value: bytes,
    *,
    label: str,
    adapter: TypeAdapter[ModelT],
) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    for line in value.splitlines(keepends=True):
        try:
            strict_json_object_body(line, label=label)
            record = adapter.validate_json(line)
        except (ArtifactValidationError, ValidationError, ValueError) as error:
            raise ArtifactValidationError("M5 machine NDJSON is invalid") from error
        if canonical_json_bytes(record) != line:
            raise ArtifactValidationError("M5 machine NDJSON is not canonical")
        records.append(record)
    if len(records) != _ndjson_record_count(value, label=label):
        raise ArtifactValidationError("M5 machine NDJSON record count changed")
    return tuple(records)


def _artifact_request_from_candidate(
    *,
    candidate: LoadedReviewCandidate,
    results_review_attestation_bytes: bytes,
    inputs: M5ReleaseLocalInputs,
) -> object:
    from fusion_fault_bench.contracts.replay_artifact_v1 import (
        ReplayClusterSensitivityV1,
        ReplayDescriptorAggregateV1,
        ReplayFigureSourceBindingV1,
        ReplayHealthAggregateV1,
        ReplayPersistentAggregateV1,
        ReplayPersistentCrossoverV1,
        ReplayProfileSummaryV1,
        ReplayRepeatVerificationV1,
        ReplaySourceMemberCommitmentV1,
    )
    from fusion_fault_bench.replay_artifacts import ReplayCuratedArtifactWriteRequest
    from fusion_fault_bench.replay_runner import verify_replay_repeat_artifacts

    members = candidate.members
    profile = _load_canonical_model(
        members["machine/replay-profile-summary.json"],
        label="replay profile summary",
        adapter=TypeAdapter(ReplayProfileSummaryV1),
    )
    repeat_record = _load_canonical_model(
        members["machine/repeat-verification.json"],
        label="repeat verification",
        adapter=TypeAdapter(ReplayRepeatVerificationV1),
    )
    validation = build_final_replay_validation(
        run_id=candidate.index.run_id,
        candidate_members=members,
        results_review_attestation_bytes=results_review_attestation_bytes,
    )
    repeat_sources = verify_replay_repeat_artifacts(
        primary_path=inputs.primary_artifact,
        repeat_path=inputs.repeat_artifact,
    )
    if (
        repeat_sources.repeat_verification != repeat_record
        or repeat_sources.primary.run.run_id != candidate.index.run_id
    ):
        raise ArtifactValidationError("M5 final replay source authority changed")
    return ReplayCuratedArtifactWriteRequest(
        profile_summary=profile,
        descriptor_aggregates=_parse_ndjson_models(
            members["machine/descriptor-aggregates.ndjson"],
            label="descriptor aggregates",
            adapter=TypeAdapter(ReplayDescriptorAggregateV1),
        ),
        persistent_aggregates=_parse_ndjson_models(
            members["machine/persistent-panel-aggregates.ndjson"],
            label="persistent aggregates",
            adapter=TypeAdapter(ReplayPersistentAggregateV1),
        ),
        persistent_crossovers=_parse_ndjson_models(
            members["machine/persistent-panel-crossovers.ndjson"],
            label="persistent crossovers",
            adapter=TypeAdapter(ReplayPersistentCrossoverV1),
        ),
        health_aggregates=_parse_ndjson_models(
            members["machine/health-panel-aggregates.ndjson"],
            label="health aggregates",
            adapter=TypeAdapter(ReplayHealthAggregateV1),
        ),
        cluster_sensitivity=_parse_ndjson_models(
            members["machine/leave-one-cluster-sensitivity.ndjson"],
            label="cluster sensitivity",
            adapter=TypeAdapter(ReplayClusterSensitivityV1),
        ),
        validation=validation,
        repeat_verification=repeat_record,
        figures=_parse_ndjson_models(
            members["machine/figure-records.ndjson"],
            label="figure source bindings",
            adapter=TypeAdapter(ReplayFigureSourceBindingV1),
        ),
        source_commitments=_parse_ndjson_models(
            members["machine/source-member-commitments.ndjson"],
            label="source commitments",
            adapter=TypeAdapter(ReplaySourceMemberCommitmentV1),
        ),
        run=repeat_sources.primary.run,
    )


def _build_machine_artifact_files(
    *,
    candidate: LoadedReviewCandidate,
    results_review_attestation_bytes: bytes,
    inputs: M5ReleaseLocalInputs,
    source_root: Path,
) -> Mapping[str, bytes]:
    from fusion_fault_bench.replay_artifacts import (
        ReplayCuratedArtifactWriteRequest,
        write_replay_curated_artifact,
    )

    request = cast(
        ReplayCuratedArtifactWriteRequest,
        _artifact_request_from_candidate(
            candidate=candidate,
            results_review_attestation_bytes=results_review_attestation_bytes,
            inputs=inputs,
        ),
    )
    temporary_root = Path(tempfile.mkdtemp(prefix="ffb-m5-machine-", dir="/private/tmp"))
    artifact_path = temporary_root / "artifact"
    try:
        write_replay_curated_artifact(
            request,
            artifact_path,
            source_root=source_root,
        )
        return _load_exact_tree(
            artifact_path,
            exact_paths=REPLAY_ARTIFACT_PATHS,
        )
    finally:
        shutil.rmtree(temporary_root)


def _finalize_template_bytes(
    value: bytes,
    *,
    artifact_sha256: str,
    run_sha256: str,
    results_review_attestation_sha256: str,
    machine_artifact_bytes: int,
) -> bytes:
    replacements = {
        "@M5_RELEASE_ARTIFACT_SHA256@": artifact_sha256,
        "@M5_RELEASE_RUN_SHA256@": run_sha256,
        "@M5_RESULTS_REVIEW_ATTESTATION_SHA256@": (results_review_attestation_sha256),
        "@M5_MACHINE_ARTIFACT_BYTES@": str(machine_artifact_bytes),
    }
    finalized = value
    for placeholder, replacement in replacements.items():
        encoded = placeholder.encode("ascii")
        if finalized.count(encoded) != 1:
            raise ArtifactValidationError(
                "M5 reviewed template has a noncanonical placeholder count"
            )
        finalized = finalized.replace(encoded, replacement.encode("ascii"))
    if re.search(rb"@M5_[A-Z0-9_]+@", finalized):
        raise ArtifactValidationError("M5 finalized presentation retains a placeholder")
    return finalized


def _recover_template_bytes(
    value: bytes,
    *,
    artifact_sha256: str,
    run_sha256: str,
    results_review_attestation_sha256: str,
    machine_artifact_bytes: int,
) -> bytes:
    replacements = (
        (artifact_sha256, "@M5_RELEASE_ARTIFACT_SHA256@"),
        (run_sha256, "@M5_RELEASE_RUN_SHA256@"),
        (
            results_review_attestation_sha256,
            "@M5_RESULTS_REVIEW_ATTESTATION_SHA256@",
        ),
        (str(machine_artifact_bytes), "@M5_MACHINE_ARTIFACT_BYTES@"),
    )
    recovered = value
    for realized, placeholder in replacements:
        encoded = realized.encode("ascii")
        if recovered.count(encoded) != 1:
            raise ArtifactValidationError(
                "M5 finalized presentation has an ambiguous identity substitution"
            )
        recovered = recovered.replace(encoded, placeholder.encode("ascii"))
    return recovered


def _sidecar_role(
    path: str,
) -> CandidateRole | Literal["results-review-evidence", "reviewed-candidate-index"]:
    candidate_translation = {
        "README.md": "presentation/README.md",
        "claim-evidence.md": "presentation/claim-evidence.md",
        "verification.md": "presentation/verification.md",
        "release-summary.json": "presentation/release-summary.json",
        "evidence/public-claim-projections.json": ("presentation/public-claim-projections.json"),
    }
    translated = candidate_translation.get(path, path)
    if translated in _CANDIDATE_ROLE_BY_PATH:
        return _CANDIDATE_ROLE_BY_PATH[translated]
    if path == "evidence/review-candidate-index.json":
        return "reviewed-candidate-index"
    if path in {
        "evidence/results-review.md",
        "evidence/results-review-attestation.json",
    }:
        return "results-review-evidence"
    raise ArtifactValidationError("M5 sidecar role is not declared")


def _build_sidecar_index(
    *,
    candidate: LoadedReviewCandidate,
    artifact_sha256: str,
    run_sha256: str,
    scientific_git_revision: str,
    sidecars: Mapping[str, bytes],
    results_review_attestation_sha256: str,
    machine_artifact_payload_bytes: int,
) -> ReleaseSidecarIndexV1:
    if tuple(sidecars) != M5_FINAL_SIDECAR_PATHS:
        raise ArtifactValidationError("M5 final sidecar set is incomplete")
    entries = tuple(
        ReleaseSidecarEntryV1(
            path=path,
            role=_sidecar_role(path),
            byte_length=len(sidecars[path]),
            sha256=_sha256(sidecars[path]),
            record_count=(
                _ndjson_record_count(sidecars[path], label=path)
                if path.endswith(".ndjson")
                else None
            ),
        )
        for path in M5_FINAL_SIDECAR_PATHS
    )
    provisional = ReleaseSidecarIndexV1.model_construct(
        schema_id=M5_RELEASE_SIDECAR_SCHEMA,
        release_id=M5_REPLAY_RELEASE_ID,
        reviewed_candidate_sha256=candidate.candidate_sha256,
        results_review_attestation_sha256=(results_review_attestation_sha256),
        machine_artifact_sha256=artifact_sha256,
        machine_run_sha256=run_sha256,
        scientific_git_revision=scientific_git_revision,
        files=entries,
        machine_artifact_payload_bytes=machine_artifact_payload_bytes,
        indexed_sidecar_payload_bytes=sum(len(value) for value in sidecars.values()),
        sidecar_set_sha256="0" * 64,
        release_package_sha256="0" * 64,
    )
    sidecar_sha256 = compute_sidecar_set_digest_from_index(provisional)
    package_sha256 = compute_release_package_digest(
        machine_artifact_sha256=artifact_sha256,
        sidecar_set_sha256=sidecar_sha256,
    )
    return ReleaseSidecarIndexV1(
        schema=M5_RELEASE_SIDECAR_SCHEMA,
        release_id=M5_REPLAY_RELEASE_ID,
        reviewed_candidate_sha256=candidate.candidate_sha256,
        results_review_attestation_sha256=(results_review_attestation_sha256),
        machine_artifact_sha256=artifact_sha256,
        machine_run_sha256=run_sha256,
        scientific_git_revision=scientific_git_revision,
        files=entries,
        machine_artifact_payload_bytes=machine_artifact_payload_bytes,
        indexed_sidecar_payload_bytes=sum(len(value) for value in sidecars.values()),
        sidecar_set_sha256=sidecar_sha256,
        release_package_sha256=package_sha256,
    )


def _build_final_sidecars(
    *,
    candidate: LoadedReviewCandidate,
    results_review_bytes: bytes,
    results_review_attestation_bytes: bytes,
    artifact_sha256: str,
    run_sha256: str,
    machine_artifact_bytes: int,
) -> Mapping[str, bytes]:
    attestation_sha256 = _sha256(results_review_attestation_bytes)
    candidate_members = candidate.members
    values: dict[str, bytes] = {
        "README.md": _finalize_template_bytes(
            candidate_members["presentation/README.md"],
            artifact_sha256=artifact_sha256,
            run_sha256=run_sha256,
            results_review_attestation_sha256=attestation_sha256,
            machine_artifact_bytes=machine_artifact_bytes,
        ),
        "claim-evidence.md": _finalize_template_bytes(
            candidate_members["presentation/claim-evidence.md"],
            artifact_sha256=artifact_sha256,
            run_sha256=run_sha256,
            results_review_attestation_sha256=attestation_sha256,
            machine_artifact_bytes=machine_artifact_bytes,
        ),
        "verification.md": _finalize_template_bytes(
            candidate_members["presentation/verification.md"],
            artifact_sha256=artifact_sha256,
            run_sha256=run_sha256,
            results_review_attestation_sha256=attestation_sha256,
            machine_artifact_bytes=machine_artifact_bytes,
        ),
        "release-summary.json": candidate_members["presentation/release-summary.json"],
        **{path: candidate_members[path] for path in M5_CANDIDATE_FIGURE_PATHS},
        "evidence/release-pipeline-plan.md": candidate_members["evidence/release-pipeline-plan.md"],
        "evidence/release-pipeline-plan-review.md": candidate_members[
            "evidence/release-pipeline-plan-review.md"
        ],
        "evidence/resource-scope-amendment.md": candidate_members[
            "evidence/resource-scope-amendment.md"
        ],
        "evidence/implementation-review.md": candidate_members["evidence/implementation-review.md"],
        "evidence/review-candidate-index.json": candidate.index_bytes,
        "evidence/validation-inputs.json": candidate_members["evidence/validation-inputs.json"],
        "evidence/implementation-review-attestation.json": candidate_members[
            "evidence/implementation-review-attestation.json"
        ],
        "evidence/software-verification.json": candidate_members[
            "evidence/software-verification.json"
        ],
        "evidence/privacy-license-attestation.json": candidate_members[
            "evidence/privacy-license-attestation.json"
        ],
        "evidence/public-claim-projections.json": candidate_members[
            "presentation/public-claim-projections.json"
        ],
        "evidence/results-review.md": results_review_bytes,
        "evidence/results-review-attestation.json": (results_review_attestation_bytes),
    }
    return MappingProxyType({path: values[path] for path in M5_FINAL_SIDECAR_PATHS})


def _candidate_from_release_files(
    *,
    root: Path,
    files: Mapping[str, bytes],
    sidecar_index: ReleaseSidecarIndexV1,
) -> LoadedReviewCandidate:
    artifact_sha256 = sidecar_index.machine_artifact_sha256
    run_sha256 = sidecar_index.machine_run_sha256
    review_sha256 = sidecar_index.results_review_attestation_sha256
    artifact_bytes = sidecar_index.machine_artifact_payload_bytes
    machine_translation = {
        "machine/intent.json": "artifact/intent.json",
        "machine/replay-profile-summary.json": ("artifact/replay-profile-summary.json"),
        "machine/descriptor-aggregates.ndjson": ("artifact/descriptor-aggregates.ndjson"),
        "machine/persistent-panel-aggregates.ndjson": (
            "artifact/persistent-panel-aggregates.ndjson"
        ),
        "machine/persistent-panel-crossovers.ndjson": (
            "artifact/persistent-panel-crossovers.ndjson"
        ),
        "machine/health-panel-aggregates.ndjson": ("artifact/health-panel-aggregates.ndjson"),
        "machine/leave-one-cluster-sensitivity.ndjson": (
            "artifact/leave-one-cluster-sensitivity.ndjson"
        ),
        "machine/repeat-verification.json": "artifact/repeat-verification.json",
        "machine/figure-records.ndjson": "artifact/figure-records.ndjson",
        "machine/source-member-commitments.ndjson": ("artifact/source-member-commitments.ndjson"),
    }
    candidate_members: dict[str, bytes] = {
        candidate_path: files[release_path]
        for candidate_path, release_path in machine_translation.items()
    }
    candidate_members.update(
        {
            "evidence/release-pipeline-plan.md": files["evidence/release-pipeline-plan.md"],
            "evidence/release-pipeline-plan-review.md": files[
                "evidence/release-pipeline-plan-review.md"
            ],
            "evidence/resource-scope-amendment.md": files["evidence/resource-scope-amendment.md"],
            "evidence/implementation-review.md": files["evidence/implementation-review.md"],
            "evidence/validation-inputs.json": files["evidence/validation-inputs.json"],
            "evidence/implementation-review-attestation.json": files[
                "evidence/implementation-review-attestation.json"
            ],
            "evidence/software-verification.json": files["evidence/software-verification.json"],
            "evidence/privacy-license-attestation.json": files[
                "evidence/privacy-license-attestation.json"
            ],
            **{path: files[path] for path in M5_CANDIDATE_FIGURE_PATHS},
            "presentation/README.md": _recover_template_bytes(
                files["README.md"],
                artifact_sha256=artifact_sha256,
                run_sha256=run_sha256,
                results_review_attestation_sha256=review_sha256,
                machine_artifact_bytes=artifact_bytes,
            ),
            "presentation/claim-evidence.md": _recover_template_bytes(
                files["claim-evidence.md"],
                artifact_sha256=artifact_sha256,
                run_sha256=run_sha256,
                results_review_attestation_sha256=review_sha256,
                machine_artifact_bytes=artifact_bytes,
            ),
            "presentation/verification.md": _recover_template_bytes(
                files["verification.md"],
                artifact_sha256=artifact_sha256,
                run_sha256=run_sha256,
                results_review_attestation_sha256=review_sha256,
                machine_artifact_bytes=artifact_bytes,
            ),
            "presentation/release-summary.json": files["release-summary.json"],
            "presentation/public-claim-projections.json": files[
                "evidence/public-claim-projections.json"
            ],
        }
    )
    ordered_members = MappingProxyType(
        {path: candidate_members[path] for path in M5_CANDIDATE_MEMBER_PATHS}
    )
    index_bytes = files["evidence/review-candidate-index.json"]
    index = _load_canonical_model(
        index_bytes,
        label="M5 reviewed candidate index",
        adapter=TypeAdapter(ReviewCandidateIndexV1),
    )
    if _candidate_entries(ordered_members) != index.files:
        raise ArtifactValidationError("M5 release bytes differ from the reviewed candidate")
    return LoadedReviewCandidate(
        path=root,
        index=index,
        index_bytes=index_bytes,
        members=ordered_members,
        candidate_sha256=index.candidate_sha256,
        index_sha256=_sha256(index_bytes),
    )


def _validate_figure_sidecars(
    *,
    candidate: LoadedReviewCandidate,
    machine_figures: Sequence[ReplayFigureSourceBindingV1],
) -> None:
    from fusion_fault_bench.replay_figures import render_replay_figure_svg

    bindings_by_figure: dict[str, list[ReplayFigureSourceBindingV1]] = {}
    for binding in machine_figures:
        figure_id = binding.figure_id
        bindings_by_figure.setdefault(figure_id, []).append(binding)
    for stem in M5_FIGURE_STEMS:
        spec_path = f"figures/{stem}.spec.json"
        svg_path = f"figures/{stem}.svg"
        spec_bytes = candidate.members[spec_path]
        svg_bytes = candidate.members[svg_path]
        spec = _strict_json_mapping(spec_bytes, label=spec_path)
        if render_replay_figure_svg(spec) != svg_bytes:
            raise ArtifactValidationError(
                "M5 rendered figure differs from deterministic regeneration"
            )
        marks_value = spec.get("marks")
        if not isinstance(marks_value, list):
            raise ArtifactValidationError("M5 figure marks are malformed")
        marks = cast("list[dict[str, object]]", marks_value)
        bindings = bindings_by_figure.get(stem, [])
        if len(bindings) != len(marks):
            raise ArtifactValidationError("M5 figure binding coverage is incomplete")
        for ordinal, (binding, mark) in enumerate(zip(bindings, marks, strict=True)):
            if (
                binding.mark_ordinal != ordinal
                or binding.source_kind != mark.get("source_kind")
                or binding.source_id != mark.get("source_id")
                or binding.source_record_sha256 != mark.get("source_record_sha256")
                or binding.replay_identity_sha256 != mark.get("replay_identity_sha256")
                or binding.figure_spec_sha256 != _sha256(spec_bytes)
                or binding.rendered_svg_path != svg_path
                or binding.rendered_svg_sha256 != _sha256(svg_bytes)
                or binding.rendered_svg_byte_length != len(svg_bytes)
            ):
                raise ArtifactValidationError("M5 figure mark differs from its machine binding")
    if tuple(bindings_by_figure) != M5_FIGURE_STEMS:
        raise ArtifactValidationError("M5 machine figure order is invalid")


def load_release(path: Path) -> LoadedM5Release:
    """Strictly validate the complete self-contained 41-file M5 package."""

    from fusion_fault_bench.replay_artifacts import load_replay_curated_artifact

    root = absolute_artifact_path(path)
    files = _load_exact_tree(root, exact_paths=M5_FINAL_RELEASE_PATHS)
    index_bytes = files["release-sidecar-index.json"]
    sidecar_index = _load_canonical_model(
        index_bytes,
        label="M5 release sidecar index",
        adapter=TypeAdapter(ReleaseSidecarIndexV1),
    )
    sidecar_entries = tuple(
        ReleaseSidecarEntryV1(
            path=path,
            role=_sidecar_role(path),
            byte_length=len(files[path]),
            sha256=_sha256(files[path]),
            record_count=(
                _ndjson_record_count(files[path], label=path) if path.endswith(".ndjson") else None
            ),
        )
        for path in M5_FINAL_SIDECAR_PATHS
    )
    if sidecar_entries != sidecar_index.files:
        raise ArtifactValidationError("M5 release sidecars disagree with their index")
    for entry in sidecar_entries:
        scan_public_member(entry.path, files[entry.path], role=entry.role)
    machine = load_replay_curated_artifact(root / "artifact")
    machine_bytes = sum(len(files[f"artifact/{path}"]) for path in REPLAY_ARTIFACT_PATHS)
    if (
        machine.artifact_sha256 != sidecar_index.machine_artifact_sha256
        or machine.run_sha256 != sidecar_index.machine_run_sha256
        or machine.run.git_revision != sidecar_index.scientific_git_revision
        or machine_bytes != sidecar_index.machine_artifact_payload_bytes
    ):
        raise ArtifactValidationError("M5 machine artifact disagrees with package identity")
    candidate = _candidate_from_release_files(
        root=root,
        files=files,
        sidecar_index=sidecar_index,
    )
    if candidate.candidate_sha256 != sidecar_index.reviewed_candidate_sha256:
        raise ArtifactValidationError("M5 package has the wrong reviewed candidate")
    review_report = files["evidence/results-review.md"]
    review_attestation_bytes = files["evidence/results-review-attestation.json"]
    review_attestation = _load_canonical_model(
        review_attestation_bytes,
        label="M5 results review attestation",
        adapter=TypeAdapter(ResultsReviewAttestationV1),
    )
    digests = _candidate_review_set_digests(candidate)
    if (
        review_attestation.disposition != "pass"
        or review_attestation.candidate_sha256 != candidate.candidate_sha256
        or review_attestation.candidate_index_sha256 != candidate.index_sha256
        or review_attestation.review_report_sha256 != _sha256(review_report)
        or _sha256(review_attestation_bytes) != sidecar_index.results_review_attestation_sha256
        or any(getattr(review_attestation, name) != value for name, value in digests.items())
    ):
        raise ArtifactValidationError("M5 results review binding is invalid")
    validate_final_replay_validation(
        machine.validation,
        candidate_members=candidate.members,
        results_review_attestation_bytes=review_attestation_bytes,
    )
    _validate_figure_sidecars(
        candidate=candidate,
        machine_figures=machine.figures,
    )
    if sum(len(value) for value in files.values()) > M5_RELEASE_PACKAGE_MAX_BYTES:
        raise ArtifactValidationError("M5 complete release exceeds the 50 MiB cap")
    return LoadedM5Release(
        path=root,
        sidecar_index=sidecar_index,
        sidecar_index_bytes=index_bytes,
        files=files,
        package_sha256=sidecar_index.release_package_sha256,
    )


def validate_release(path: Path) -> LoadedM5Release:
    """Package-only offline validation hook."""

    return load_release(path)


def _write_final_release(
    *,
    destination: Path,
    source_root: Path,
    candidate: LoadedReviewCandidate,
    artifact_files: Mapping[str, bytes],
    results_review_bytes: bytes,
    results_review_attestation_bytes: bytes,
) -> LoadedM5Release:
    from fusion_fault_bench.replay_artifacts import load_replay_curated_artifact

    temporary_root = Path(tempfile.mkdtemp(prefix="ffb-m5-artifact-check-", dir="/private/tmp"))
    artifact_check = temporary_root / "artifact"
    try:
        # Use the same exact transaction and strict loader for the machine bytes
        # before they enter the outer package transaction.
        _publish_exact_tree(
            destination=artifact_check,
            files=artifact_files,
            exact_paths=REPLAY_ARTIFACT_PATHS,
            source_root=source_root,
            loader=load_replay_curated_artifact,
        )
        machine = load_replay_curated_artifact(artifact_check)
    finally:
        shutil.rmtree(temporary_root)
    machine_bytes = sum(len(value) for value in artifact_files.values())
    sidecars = _build_final_sidecars(
        candidate=candidate,
        results_review_bytes=results_review_bytes,
        results_review_attestation_bytes=results_review_attestation_bytes,
        artifact_sha256=machine.artifact_sha256,
        run_sha256=machine.run_sha256,
        machine_artifact_bytes=machine_bytes,
    )
    sidecar_index = _build_sidecar_index(
        candidate=candidate,
        artifact_sha256=machine.artifact_sha256,
        run_sha256=machine.run_sha256,
        scientific_git_revision=candidate.index.scientific_git_revision,
        sidecars=sidecars,
        results_review_attestation_sha256=_sha256(results_review_attestation_bytes),
        machine_artifact_payload_bytes=machine_bytes,
    )
    files = MappingProxyType(
        {
            **{f"artifact/{path}": artifact_files[path] for path in REPLAY_ARTIFACT_PATHS},
            **sidecars,
            "release-sidecar-index.json": canonical_json_bytes(sidecar_index),
        }
    )
    return _publish_exact_tree(
        destination=destination,
        files=files,
        exact_paths=M5_FINAL_RELEASE_PATHS,
        source_root=source_root,
        loader=load_release,
    )


def build_release(
    *,
    candidate: Path,
    results_review: Path,
    results_review_attestation: Path,
    inputs: M5ReleaseLocalInputs,
    output_dir: Path,
    source_root: Path,
) -> LoadedM5Release:
    """Re-curate, compare with reviewed bytes, and atomically publish M5."""

    root = source_root.resolve(strict=True)
    if output_dir != _M5_FINAL_RELEASE_RELATIVE:
        raise ArtifactValidationError("M5 final release output path is not canonical")
    loaded_candidate = load_review_candidate(candidate)
    review_bytes = _read_stable_public_file(results_review)
    scan_public_member(
        "evidence/results-review.md",
        review_bytes,
        role="independent-review-evidence",
    )
    review_attestation_bytes = _read_stable_public_file(results_review_attestation)
    review_attestation = _load_canonical_model(
        review_attestation_bytes,
        label="M5 results review attestation",
        adapter=TypeAdapter(ResultsReviewAttestationV1),
    )
    digests = _candidate_review_set_digests(loaded_candidate)
    if (
        review_attestation.disposition != "pass"
        or review_attestation.candidate_sha256 != loaded_candidate.candidate_sha256
        or review_attestation.candidate_index_sha256 != loaded_candidate.index_sha256
        or review_attestation.review_report_sha256 != _sha256(review_bytes)
        or any(getattr(review_attestation, name) != value for name, value in digests.items())
    ):
        raise ArtifactValidationError("M5 release review does not bind the exact candidate")

    regenerated = _regenerate_candidate_material(inputs, source_root=root)
    regenerated_index = build_review_candidate_index(regenerated)
    if (
        regenerated.members != loaded_candidate.members
        or canonical_json_bytes(regenerated_index) != loaded_candidate.index_bytes
    ):
        raise ArtifactValidationError("M5 final regeneration differs from reviewed candidate bytes")
    artifact_files = _build_machine_artifact_files(
        candidate=loaded_candidate,
        results_review_attestation_bytes=review_attestation_bytes,
        inputs=inputs,
        source_root=root,
    )
    return _write_final_release(
        destination=root / output_dir,
        source_root=root,
        candidate=loaded_candidate,
        artifact_files=artifact_files,
        results_review_bytes=review_bytes,
        results_review_attestation_bytes=review_attestation_bytes,
    )


def sync_reviewed_evidence(
    *,
    release: Path,
    review_report_output: Path,
    review_attestation_output: Path,
) -> None:
    """Copy exact reviewed package evidence to two absent public destinations."""

    loaded = load_release(release)
    report = loaded.files["evidence/results-review.md"]
    attestation = loaded.files["evidence/results-review-attestation.json"]
    if (
        os.path.lexists(review_report_output)
        or os.path.lexists(review_attestation_output)
        or absolute_artifact_path(review_report_output)
        == absolute_artifact_path(review_attestation_output)
    ):
        raise FileExistsError("M5 reviewed evidence destination already exists")
    _exclusive_file(review_report_output, report)
    _exclusive_file(review_attestation_output, attestation)
    if (
        _read_stable_public_file(review_report_output) != report
        or _read_stable_public_file(review_attestation_output) != attestation
    ):
        raise ArtifactValidationError("M5 reviewed evidence synchronization failed")


_M5_PUBLICATION_DOC_PATHS = (
    "README.md",
    "docs/results.md",
    "docs/benchmark-card.md",
    "docs/limitations.md",
    "docs/reproducibility.md",
    "docs/project-plan.md",
    "docs/dataset-preparation.md",
    "docs/m5-technical-walkthrough.md",
)


def validate_publication(
    *,
    release: Path,
    source_root: Path,
) -> LoadedM5Release:
    """Validate package copies and release-specific public documentation."""

    loaded = load_release(release)
    root = source_root.resolve(strict=True)
    report = _strict_source_file(root, "docs/reviews/m5-results-review.md")
    attestation = _strict_source_file(
        root,
        "docs/reviews/m5-results-review-attestation.json",
    )
    if (
        report != loaded.files["evidence/results-review.md"]
        or attestation != loaded.files["evidence/results-review-attestation.json"]
    ):
        raise ArtifactValidationError("M5 public review copies differ from the release package")
    release_id = M5_REPLAY_RELEASE_ID.encode("ascii")
    package_sha256 = loaded.package_sha256.encode("ascii")
    package_identity_mentions = 0
    for relative in _M5_PUBLICATION_DOC_PATHS:
        value = _strict_source_file(root, relative)
        scan_public_member(
            relative,
            value,
            role="frozen-public-methodology",
        )
        if release_id not in value:
            raise ArtifactValidationError("M5 public documentation omits the release identity")
        package_identity_mentions += package_sha256 in value
    if package_identity_mentions < 2:
        raise ArtifactValidationError("M5 public documentation underbinds the package identity")
    return loaded
