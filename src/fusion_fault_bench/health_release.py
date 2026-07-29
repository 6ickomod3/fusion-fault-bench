"""Deterministic, privacy-bounded curation for the M4 health release.

The public release retains both complete fit artifacts, one exact copy of every
small evaluation scientific member, all aggregate rows, and the provenance
envelopes for two evaluation runs.  The three large sequence-level evaluation
members are intentionally omitted and committed by exact digest, byte length,
and record count.

Strict health artifact loaders remain the owners of scientific validation.
This module treats their aggregate, contrast, event, and loss records as opaque
after authentication; it does not duplicate evolving scientific rules.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    Field,
    FiniteFloat,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    absolute_artifact_path,
    assert_directory_descriptor_matches_path,
    atomic_rename_directory_no_replace_at,
    canonical_json_bytes,
    compute_run_record_digest,
    create_staging_directory_at,
    discover_git_metadata_dirs,
    entry_exists_at,
    open_or_create_real_directory,
    read_file_at,
    reject_directory_descriptor_in_git_metadata,
    reject_git_metadata_destination,
    write_exclusive_file_at,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_AGGREGATES_FILE,
    HEALTH_CANDIDATES_FILE,
    HEALTH_ECDF_FILE,
    HEALTH_EDGE_PROFILE_FILE,
    HEALTH_EVAL_ARTIFACT_CONTRACT,
    HEALTH_EVAL_INDEXED_PATHS,
    HEALTH_EVAL_VALIDATION_FILE,
    HEALTH_FIT_ARTIFACT_CONTRACT,
    HEALTH_FIT_INDEXED_PATHS,
    HEALTH_FIT_REFERENCE_FILE,
    HEALTH_FIT_SUMMARY_FILE,
    HEALTH_FIT_VALIDATION_FILE,
    HEALTH_INTENT_FILE,
    HEALTH_MAIN_PROFILE_FILE,
    HEALTH_PAYLOAD_INDEX_FILE,
    HEALTH_RUN_FILE,
    HEALTH_SEQUENCE_CONTRASTS_FILE,
    HEALTH_SEQUENCE_EVENTS_FILE,
    HEALTH_SEQUENCE_LOSSES_FILE,
    HEALTH_SUCCESS_FILE,
    HealthFitReferenceV1,
    HealthPayloadIndexV1,
)
from fusion_fault_bench.contracts.health_result_v1 import (
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
    HealthThresholdCandidateV1,
    HealthValidationV1,
)
from fusion_fault_bench.contracts.health_v1 import (
    HEALTH_BENCHMARK_INTENT_ADAPTER,
    HealthBenchmarkIntentV1,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    PROCEDURAL_PROFILE_ADAPTER,
    EdgeProceduralProfile,
    MainProceduralProfile,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    Digest,
    Identifier,
    RunRecordV1Alpha1,
)
from fusion_fault_bench.health import HealthCalibration
from fusion_fault_bench.health_artifacts import (
    HEALTH_EVAL_ARTIFACT_PATHS,
    HEALTH_FIT_ARTIFACT_PATHS,
    HealthEcdfArraysV1,
    LoadedHealthFitArtifact,
    build_health_fit_reference,
    canonical_health_ndjson_bytes,
    compute_health_artifact_digest,
    load_health_evaluation_artifact,
    load_health_fit_artifact,
    validate_health_aggregate_structure,
    validate_health_fit_bundle,
)

HEALTH_RELEASE_CONTRACT = "ffb.health-release-payload/v1"
HEALTH_RELEASE_SUMMARY_FILE = "release-summary.json"
HEALTH_RELEASE_COMMITMENTS_FILE = "source-member-commitments.ndjson"
HEALTH_RELEASE_REPEAT_FILE = "repeat-verification.json"
HEALTH_RELEASE_CLAIMS_FILE = "quantitative-claims.ndjson"
HEALTH_RELEASE_INDEX_FILE = "release-index.json"
HEALTH_RELEASE_SUCCESS_FILE = "_SUCCESS"

_RELEASE_DOMAIN = b"fusion-fault-bench/health-release-artifact/v1\x00"
_READ_CHUNK_BYTES = 1024 * 1024
_MAX_RECORD_BYTES = 1024 * 1024
_CURATED_RELEASE_BYTES_MAX = 52_428_800
_PEAK_RSS_BYTES_MAX = 1_073_741_824
_WALL_TIME_SECONDS_MAX = 1800.0
_EXPECTED_OMITTED_RECORD_COUNTS: Mapping[str, int] = {
    HEALTH_SEQUENCE_LOSSES_FILE: 264_600,
    HEALTH_SEQUENCE_CONTRASTS_FILE: 133_500,
    HEALTH_SEQUENCE_EVENTS_FILE: 35_600,
}
_OMITTED_EVALUATION_PATHS = tuple(_EXPECTED_OMITTED_RECORD_COUNTS)
_EVALUATION_RETAINED_PATHS = tuple(
    path for path in HEALTH_EVAL_INDEXED_PATHS if path not in _OMITTED_EVALUATION_PATHS
)
_PRIVATE_PATH_PATTERN = re.compile(
    rb"(?i)(?:file://|[A-Z]:\\\\|/(?:Users|home|private|tmp|var/folders)/)"
)
_SECRET_PATTERN = re.compile(
    rb"(?i)(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|"
    rb"AKIA[0-9A-Z]{16}|authorization\s*:)"
)
_TIME_L_REAL_PATTERN = re.compile(
    rb"^[ \t]*(?P<real>[0-9]+(?:\.[0-9]+)?)[ \t]+real"
    rb"[ \t]+[0-9]+(?:\.[0-9]+)?[ \t]+user"
    rb"[ \t]+[0-9]+(?:\.[0-9]+)?[ \t]+sys[ \t]*$"
)
_TIME_L_RSS_PATTERN = re.compile(rb"^[ \t]*(?P<rss>[0-9]+)[ \t]+maximum resident set size[ \t]*$")

type SourceArtifactKind = Literal["fit", "evaluation"]
type RunLabel = Literal[
    "primary-fit",
    "repeat-fit",
    "primary-evaluation",
    "repeat-evaluation",
]
type FitClaimField = Literal[
    "selected-candidate-index",
    "selected-self-threshold",
    "selected-cross-threshold",
]
type ResourceMetric = Literal["wall-time-seconds", "peak-rss-bytes"]


class HealthReleaseValidationError(ValueError):
    """Curated evidence disagrees with strict M4 source artifacts."""


def _prefixed_source_path(prefix: str, source_path: str) -> str:
    normalized = "success.json" if source_path == HEALTH_SUCCESS_FILE else source_path
    return f"{prefix}-{normalized}"


_PRIMARY_FIT_RELEASE_PATHS: Mapping[str, str] = {
    path: _prefixed_source_path("primary-fit", path) for path in HEALTH_FIT_ARTIFACT_PATHS
}
_REPEAT_FIT_RELEASE_PATHS: Mapping[str, str] = {
    path: _prefixed_source_path("repeat-fit", path) for path in HEALTH_FIT_ARTIFACT_PATHS
}
_EVALUATION_SCIENCE_RELEASE_PATHS: Mapping[str, str] = {
    HEALTH_INTENT_FILE: "evaluation-intent.json",
    HEALTH_MAIN_PROFILE_FILE: "evaluation-main-profile.json",
    HEALTH_EDGE_PROFILE_FILE: "evaluation-edge-profile.json",
    HEALTH_FIT_REFERENCE_FILE: "evaluation-fit-reference.json",
    HEALTH_AGGREGATES_FILE: "aggregate-metrics.ndjson",
    HEALTH_EVAL_VALIDATION_FILE: "evaluation-validation.json",
}
_PRIMARY_EVALUATION_ENVELOPE_PATHS: Mapping[str, str] = {
    HEALTH_PAYLOAD_INDEX_FILE: "primary-evaluation-payload-index.json",
    HEALTH_RUN_FILE: "primary-evaluation-run.json",
    HEALTH_SUCCESS_FILE: "primary-evaluation-success.json",
}
_REPEAT_EVALUATION_ENVELOPE_PATHS: Mapping[str, str] = {
    HEALTH_PAYLOAD_INDEX_FILE: "repeat-evaluation-payload-index.json",
    HEALTH_RUN_FILE: "repeat-evaluation-run.json",
    HEALTH_SUCCESS_FILE: "repeat-evaluation-success.json",
}
_RESOURCE_LOG_RELEASE_PATHS: Mapping[RunLabel, str] = {
    "primary-fit": "primary-fit-time-l.txt",
    "repeat-fit": "repeat-fit-time-l.txt",
    "primary-evaluation": "primary-evaluation-time-l.txt",
    "repeat-evaluation": "repeat-evaluation-time-l.txt",
}

HEALTH_RELEASE_INDEXED_PATHS = (
    *tuple(_PRIMARY_FIT_RELEASE_PATHS.values()),
    *tuple(_REPEAT_FIT_RELEASE_PATHS.values()),
    *tuple(_EVALUATION_SCIENCE_RELEASE_PATHS.values()),
    *tuple(_PRIMARY_EVALUATION_ENVELOPE_PATHS.values()),
    *tuple(_REPEAT_EVALUATION_ENVELOPE_PATHS.values()),
    *tuple(_RESOURCE_LOG_RELEASE_PATHS.values()),
    HEALTH_RELEASE_COMMITMENTS_FILE,
    HEALTH_RELEASE_REPEAT_FILE,
    HEALTH_RELEASE_CLAIMS_FILE,
    HEALTH_RELEASE_SUMMARY_FILE,
)
HEALTH_RELEASE_ARTIFACT_PATHS = (
    *HEALTH_RELEASE_INDEXED_PATHS,
    HEALTH_RELEASE_INDEX_FILE,
    HEALTH_RELEASE_SUCCESS_FILE,
)


@dataclass(frozen=True, slots=True)
class HealthRunResources:
    """One strict sidecar plus its exact Darwin ``time -l`` log bytes."""

    time_l_log: bytes
    measurement: HealthRunResourceEvidenceV1


class HealthSourceMemberCommitmentV1(ContractModel):
    """One primary/repeat scientific-member commitment and retention decision."""

    schema_id: Literal["ffb.health-source-member-commitment/v1"] = Field(alias="schema")
    artifact_kind: SourceArtifactKind
    path: str
    primary_artifact_sha256: Digest
    repeat_artifact_sha256: Digest
    primary_sha256: Digest
    repeat_sha256: Digest
    primary_byte_length: Annotated[int, Field(ge=1)]
    repeat_byte_length: Annotated[int, Field(ge=1)]
    primary_record_count: Annotated[int, Field(ge=1)] | None
    repeat_record_count: Annotated[int, Field(ge=1)] | None
    equal: bool
    primary_retained_release_path: str | None
    repeat_retained_release_path: str | None
    retention_scope: Literal[
        "primary-and-repeat-exact-source-bytes",
        "one-exact-copy-shared-by-identical-evaluations",
        "omitted-commitment-only-not-independently-recomputable",
    ]

    @model_validator(mode="after")
    def require_consistent_commitment(self) -> HealthSourceMemberCommitmentV1:
        expected_paths = (
            HEALTH_FIT_INDEXED_PATHS if self.artifact_kind == "fit" else HEALTH_EVAL_INDEXED_PATHS
        )
        if self.path not in expected_paths:
            raise ValueError("source commitment path is not indexed by its artifact")
        is_ndjson = self.path.endswith(".ndjson")
        if is_ndjson != (self.primary_record_count is not None):
            raise ValueError("primary record count must be present exactly for NDJSON")
        if is_ndjson != (self.repeat_record_count is not None):
            raise ValueError("repeat record count must be present exactly for NDJSON")
        expected_equal = (
            self.primary_sha256 == self.repeat_sha256
            and self.primary_byte_length == self.repeat_byte_length
            and self.primary_record_count == self.repeat_record_count
        )
        if self.equal != expected_equal:
            raise ValueError("source member equality contradicts its commitments")
        omitted = self.artifact_kind == "evaluation" and self.path in _OMITTED_EVALUATION_PATHS
        if omitted:
            expected_count = _EXPECTED_OMITTED_RECORD_COUNTS[self.path]
            if (
                self.primary_record_count != expected_count
                or self.repeat_record_count != expected_count
                or self.primary_retained_release_path is not None
                or self.repeat_retained_release_path is not None
                or self.retention_scope != "omitted-commitment-only-not-independently-recomputable"
            ):
                raise ValueError("omitted source member has an invalid retention record")
        else:
            if (
                self.primary_retained_release_path is None
                or self.repeat_retained_release_path is None
                or self.retention_scope == "omitted-commitment-only-not-independently-recomputable"
            ):
                raise ValueError("retained source member requires exact release paths")
            if self.artifact_kind == "fit":
                expected_primary = _PRIMARY_FIT_RELEASE_PATHS[self.path]
                expected_repeat = _REPEAT_FIT_RELEASE_PATHS[self.path]
                expected_scope = "primary-and-repeat-exact-source-bytes"
            else:
                expected_primary = _EVALUATION_SCIENCE_RELEASE_PATHS[self.path]
                expected_repeat = expected_primary
                expected_scope = "one-exact-copy-shared-by-identical-evaluations"
            if (
                self.primary_retained_release_path != expected_primary
                or self.repeat_retained_release_path != expected_repeat
                or self.retention_scope != expected_scope
            ):
                raise ValueError("retained source member mapping is not canonical")
        return self


class HealthRunResourceEvidenceV1(ContractModel):
    """One raw-log-backed complete fit or evaluation resource record."""

    run_label: RunLabel
    phase: Literal["fit", "evaluation"]
    artifact_contract: Literal["ffb.health-fit-payload/v1", "ffb.health-eval-payload/v1"]
    artifact_sha256: Digest
    run_sha256: Digest
    logical_command: Annotated[tuple[str, ...], Field(min_length=1)]
    cpu_model: Annotated[str, Field(min_length=1, max_length=256)]
    os_name: Annotated[str, Field(min_length=1, max_length=128)]
    os_release: Annotated[str, Field(min_length=1, max_length=128)]
    tool_path: Literal["/usr/bin/time"]
    tool_options: tuple[Literal["-l"]]
    parser_contract: Literal["ffb.darwin-time-l/v1"]
    raw_log_path: str
    raw_log_sha256: Digest
    raw_log_byte_length: Annotated[int, Field(ge=1, le=65_536)]
    wall_time_seconds: Annotated[FiniteFloat, Field(gt=0.0)]
    maximum_resident_set_size_raw: Annotated[int, Field(gt=0)]
    maximum_resident_set_size_unit: Literal["bytes"]
    peak_rss_bytes: Annotated[int, Field(gt=0)]
    exit_status: Literal[0]
    direct_child_count: Literal[1]
    accelerator_requested: Literal[False]
    wall_time_cap_seconds: FiniteFloat
    peak_rss_cap_bytes: Literal[1_073_741_824]
    wall_time_within_cap: bool
    peak_rss_within_cap: bool
    measurement_scope: Literal[
        "operator-recorded-darwin-time-l-for-one-direct-child;"
        "self-reported-not-cryptographic-execution-proof"
    ]

    @model_validator(mode="after")
    def require_recomputed_resource_gates(self) -> HealthRunResourceEvidenceV1:
        if self.wall_time_cap_seconds != _WALL_TIME_SECONDS_MAX:
            raise ValueError("wall-time cap is not the frozen M4 value")
        if self.phase != ("fit" if self.run_label.endswith("fit") else "evaluation"):
            raise ValueError("resource phase disagrees with run label")
        expected_contract = (
            HEALTH_FIT_ARTIFACT_CONTRACT if self.phase == "fit" else HEALTH_EVAL_ARTIFACT_CONTRACT
        )
        if self.artifact_contract != expected_contract:
            raise ValueError("resource artifact contract disagrees with phase")
        if self.raw_log_path != _RESOURCE_LOG_RELEASE_PATHS[self.run_label]:
            raise ValueError("resource raw-log path is not canonical")
        if (
            self.maximum_resident_set_size_unit != "bytes"
            or self.maximum_resident_set_size_raw != self.peak_rss_bytes
        ):
            raise ValueError("Darwin time -l RSS must be interpreted directly as bytes")
        if self.wall_time_within_cap != (self.wall_time_seconds < self.wall_time_cap_seconds):
            raise ValueError("wall-time resource gate is contradictory")
        if self.peak_rss_within_cap != (self.peak_rss_bytes < self.peak_rss_cap_bytes):
            raise ValueError("peak-RSS resource gate is contradictory")
        return self

    @property
    def all_checks_passed(self) -> bool:
        return self.wall_time_within_cap and self.peak_rss_within_cap


class HealthRepeatVerificationV1(ContractModel):
    """Two-fit plus two-evaluation deterministic execution evidence."""

    schema_id: Literal["ffb.health-repeat-verification/v1"] = Field(alias="schema")
    intent_sha256: Digest
    official_fit_artifact_sha256: Digest
    official_fit_run_sha256: Digest
    repeat_fit_artifact_sha256: Digest
    repeat_fit_run_sha256: Digest
    primary_evaluation_artifact_sha256: Digest
    primary_evaluation_run_sha256: Digest
    repeat_evaluation_artifact_sha256: Digest
    repeat_evaluation_run_sha256: Digest
    fit_reference_sha256: Digest
    source_member_commitments_sha256: Digest
    fit_member_comparison_count: Literal[7]
    evaluation_member_comparison_count: Literal[9]
    total_member_comparison_count: Literal[16]
    mismatch_count: Annotated[int, Field(ge=0, le=16)]
    fit_payload_index_equal: bool
    evaluation_payload_index_equal: bool
    scientific_members_all_equal: bool
    volatile_run_records_distinct: bool
    normalized_run_identity_equal_within_phase: bool
    same_exact_runtime_environment: bool
    source_paths_and_inodes_independent: bool
    resources: Annotated[
        tuple[HealthRunResourceEvidenceV1, ...],
        Field(min_length=4, max_length=4),
    ]
    resource_measurement_scope: Literal[
        "raw-darwin-time-l-logs-retained-and-reparsed;"
        "operator-recorded-sidecars-self-reported-not-cryptographic-execution-proof"
    ]
    execution_evidence_scope: Literal[
        "distinct-path-and-inode-consistency-not-cryptographic-proof-of-two-executions"
    ]
    inference_recomputation_scope: Literal[
        "aggregates-regenerate-claims-and-figures;"
        "omitted-sequence-rows-prevent-independent-bootstrap-recomputation"
    ]
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_recomputed_repeat_conjunction(self) -> HealthRepeatVerificationV1:
        expected_labels = (
            "primary-fit",
            "repeat-fit",
            "primary-evaluation",
            "repeat-evaluation",
        )
        if tuple(resource.run_label for resource in self.resources) != expected_labels:
            raise ValueError("resource evidence does not use canonical run order")
        expected_all_equal = self.mismatch_count == 0
        if self.scientific_members_all_equal != expected_all_equal:
            raise ValueError("scientific equality contradicts mismatch count")
        expected_pass = (
            self.scientific_members_all_equal
            and self.fit_payload_index_equal
            and self.evaluation_payload_index_equal
            and self.official_fit_artifact_sha256 == self.repeat_fit_artifact_sha256
            and self.primary_evaluation_artifact_sha256 == self.repeat_evaluation_artifact_sha256
            and self.volatile_run_records_distinct
            and self.normalized_run_identity_equal_within_phase
            and self.same_exact_runtime_environment
            and self.source_paths_and_inodes_independent
            and all(resource.all_checks_passed for resource in self.resources)
        )
        if self.all_checks_passed != expected_pass:
            raise ValueError("repeat all_checks_passed is not the exact conjunction")
        return self


class AggregateQuantitativeClaimV1(ContractModel):
    """A public outcome claim that must be an exact aggregate-row projection."""

    schema_id: Literal["ffb.health-quantitative-claim/v1"] = Field(alias="schema")
    source_kind: Literal["aggregate"]
    claim_id: Identifier
    presentation_id: Identifier
    aggregate: HealthAggregateMetricV1


class FitQuantitativeClaimV1(ContractModel):
    """A selected-fit claim sourced from the retained fit summary."""

    schema_id: Literal["ffb.health-quantitative-claim/v1"] = Field(alias="schema")
    source_kind: Literal["fit-summary"]
    claim_id: Identifier
    presentation_id: Identifier
    field: FitClaimField
    value: FiniteFloat
    unit: Literal["count", "fraction"]

    @model_validator(mode="after")
    def require_fit_claim_unit(self) -> FitQuantitativeClaimV1:
        expected = "count" if self.field == "selected-candidate-index" else "fraction"
        if self.unit != expected:
            raise ValueError("fit claim unit disagrees with its field")
        return self


class ResourceQuantitativeClaimV1(ContractModel):
    """A CPU resource claim sourced from one measured complete process."""

    schema_id: Literal["ffb.health-quantitative-claim/v1"] = Field(alias="schema")
    source_kind: Literal["resource"]
    claim_id: Identifier
    presentation_id: Identifier
    run_label: RunLabel
    metric: ResourceMetric
    value: FiniteFloat
    unit: Literal["s", "bytes"]
    cpu_model: Annotated[str, Field(min_length=1, max_length=256)]
    evidence_scope: Literal["operator-recorded-time-l-sidecar-not-independent-attestation"]

    @model_validator(mode="after")
    def require_resource_claim_unit(self) -> ResourceQuantitativeClaimV1:
        expected = "s" if self.metric == "wall-time-seconds" else "bytes"
        if self.unit != expected:
            raise ValueError("resource claim unit disagrees with its metric")
        return self


type HealthQuantitativeClaimV1 = Annotated[
    AggregateQuantitativeClaimV1 | FitQuantitativeClaimV1 | ResourceQuantitativeClaimV1,
    Field(discriminator="source_kind"),
]
_CLAIM_ADAPTER = TypeAdapter(HealthQuantitativeClaimV1)


class HealthReleaseSummaryV1(ContractModel):
    """Compact public identity, completeness, and scope summary."""

    schema_id: Literal["ffb.health-release-summary/v1"] = Field(alias="schema")
    release_id: Literal["m4-health-v0.1.0"]
    intent_sha256: Digest
    git_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    lockfile_sha256: Digest
    package_version: Annotated[str, Field(min_length=1, max_length=64)]
    official_fit_artifact_sha256: Digest
    official_fit_run_sha256: Digest
    evaluation_artifact_sha256: Digest
    selected_candidate_index: Annotated[int, Field(ge=0, le=35)]
    selected_self_threshold: Annotated[FiniteFloat, Field(ge=0.95, le=1.0)]
    selected_cross_threshold: Annotated[FiniteFloat, Field(ge=0.95, le=1.0)]
    condition_count: Literal[47]
    aggregate_record_count: Annotated[int, Field(ge=1)]
    aggregate_ok_count: Annotated[int, Field(ge=0)]
    aggregate_undefined_count: Annotated[int, Field(ge=0)]
    aggregate_not_applicable_count: Annotated[int, Field(ge=0)]
    quantitative_claim_count: Annotated[int, Field(ge=1)]
    source_member_commitment_count: Literal[16]
    omitted_sequence_member_count: Literal[3]
    omitted_sequence_record_count: Literal[433700]
    omitted_sequence_byte_count: Annotated[int, Field(ge=3)]
    curated_release_bytes_max: Literal[52_428_800]
    complete_aggregate_matrix_retained: Literal[True]
    aggregate_evidence_scope: Literal[
        "exact-source-aggregate-rows-regenerate-claims-tables-and-figures"
    ]
    omitted_inference_scope: Literal[
        "sequence-loss-contrast-event-rows-committed-not-retained;"
        "third-parties-cannot-independently-recompute-bootstrap-inference"
    ]
    source_artifacts_strictly_validated_before_curation: Literal[True]
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_status_partition(self) -> HealthReleaseSummaryV1:
        if (
            self.aggregate_ok_count
            + self.aggregate_undefined_count
            + self.aggregate_not_applicable_count
            != self.aggregate_record_count
        ):
            raise ValueError("aggregate status counts do not partition retained rows")
        return self


class HealthReleaseFileEntryV1(ContractModel):
    """One exact member in the curated release index."""

    path: str
    byte_length: Annotated[int, Field(ge=1, le=_CURATED_RELEASE_BYTES_MAX)]
    sha256: Digest
    record_count: Annotated[int, Field(ge=1)] | None

    @model_validator(mode="after")
    def require_ndjson_record_count(self) -> HealthReleaseFileEntryV1:
        if self.path.endswith(".ndjson") != (self.record_count is not None):
            raise ValueError("release record count must be present exactly for NDJSON")
        return self


class HealthReleaseIndexV1(ContractModel):
    """Ordered curated-member envelope with an explicit inference scope."""

    schema_id: Literal["ffb.health-release-index/v1"] = Field(alias="schema")
    artifact_contract: Literal["ffb.health-release-payload/v1"]
    release_id: Literal["m4-health-v0.1.0"]
    intent_sha256: Digest
    official_fit_artifact_sha256: Digest
    evaluation_artifact_sha256: Digest
    curated_release_bytes_max: Literal[52_428_800]
    omitted_sequence_rows_recomputable: Literal[False]
    files: tuple[HealthReleaseFileEntryV1, ...]

    @model_validator(mode="after")
    def require_exact_release_member_order(self) -> HealthReleaseIndexV1:
        if tuple(entry.path for entry in self.files) != HEALTH_RELEASE_INDEXED_PATHS:
            raise ValueError("release index does not use the exact member order")
        return self


class HealthReleaseSuccessV1(ContractModel):
    """Final marker binding the release index, summary, and repeat record."""

    schema_id: Literal["ffb.health-release-success/v1"] = Field(alias="schema")
    release_artifact_sha256: Digest
    release_summary_sha256: Digest
    repeat_verification_sha256: Digest


@dataclass(frozen=True, slots=True)
class HealthReleaseWriteRequest:
    """Four strict source artifacts, measurements, and declared public claims."""

    official_fit_path: Path
    repeat_fit_path: Path
    primary_evaluation_path: Path
    repeat_evaluation_path: Path
    primary_fit_resources: HealthRunResources
    repeat_fit_resources: HealthRunResources
    primary_evaluation_resources: HealthRunResources
    repeat_evaluation_resources: HealthRunResources
    quantitative_claims: Sequence[HealthQuantitativeClaimV1]


@dataclass(frozen=True, slots=True)
class LoadedHealthRelease:
    """One strictly loaded curated release."""

    path: Path
    summary: HealthReleaseSummaryV1
    commitments: tuple[HealthSourceMemberCommitmentV1, ...]
    repeat: HealthRepeatVerificationV1
    claims: tuple[HealthQuantitativeClaimV1, ...]
    aggregates: tuple[HealthAggregateMetricV1, ...]
    release_index: HealthReleaseIndexV1
    success: HealthReleaseSuccessV1
    release_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class _AuthenticatedSources:
    official_fit: LoadedHealthFitArtifact
    repeat_fit: LoadedHealthFitArtifact
    primary_evaluation: _LightweightEvaluation
    repeat_evaluation: _LightweightEvaluation


@dataclass(frozen=True, slots=True)
class _LightweightEvaluation:
    """Authenticated evaluation projection that never retains sequence rows."""

    path: Path
    fit_reference: HealthFitReferenceV1
    aggregates: tuple[HealthAggregateMetricV1, ...]
    validation: HealthValidationV1
    payload_index: HealthPayloadIndexV1
    run: RunRecordV1Alpha1
    success: SuccessMarkerV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _RetainedFitSemantics:
    intent: HealthBenchmarkIntentV1
    main_profile: MainProceduralProfile
    edge_profile: EdgeProceduralProfile
    summary: HealthFitSummaryV1


@dataclass(frozen=True, slots=True)
class _PreparedHealthRelease:
    files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class _ReleaseTreeSnapshot:
    root_stat: os.stat_result
    entries: Mapping[str, os.stat_result]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_records(records: Sequence[BaseModel]) -> bytes:
    if not records:
        raise HealthReleaseValidationError("release NDJSON records must be nonempty")
    output = bytearray()
    for record in records:
        line = canonical_json_bytes(record)
        if len(line) > _MAX_RECORD_BYTES:
            raise HealthReleaseValidationError("release NDJSON record exceeds its cap")
        output.extend(line)
        if len(output) > _CURATED_RELEASE_BYTES_MAX:
            raise HealthReleaseValidationError("release NDJSON exceeds the release cap")
    return bytes(output)


def compute_health_release_digest(index_file_bytes: bytes) -> str:
    """Hash exact canonical release-index bytes in a distinct domain."""

    return hashlib.sha256(
        b"".join(
            (
                _RELEASE_DOMAIN,
                len(index_file_bytes).to_bytes(8, "big"),
                index_file_bytes,
            )
        )
    ).hexdigest()


def _payload_entry(
    artifact: LoadedHealthFitArtifact | _LightweightEvaluation,
    path: str,
) -> Any:
    try:
        return next(entry for entry in artifact.payload_index.files if entry.path == path)
    except StopIteration as error:
        raise HealthReleaseValidationError("source payload index is incomplete") from error


def _authenticate_sources(request: HealthReleaseWriteRequest) -> _AuthenticatedSources:
    official_fit = load_health_fit_artifact(request.official_fit_path)
    repeat_fit = load_health_fit_artifact(request.repeat_fit_path)
    primary_loaded = load_health_evaluation_artifact(
        request.primary_evaluation_path,
        fit_artifact=official_fit,
    )
    primary_evaluation = _LightweightEvaluation(
        path=primary_loaded.path,
        fit_reference=primary_loaded.fit_reference,
        aggregates=primary_loaded.aggregates,
        validation=primary_loaded.validation,
        payload_index=primary_loaded.payload_index,
        run=primary_loaded.run,
        success=primary_loaded.success,
        artifact_sha256=primary_loaded.artifact_sha256,
        run_sha256=primary_loaded.run_sha256,
    )
    del primary_loaded
    gc.collect()

    repeat_loaded = load_health_evaluation_artifact(
        request.repeat_evaluation_path,
        fit_artifact=official_fit,
    )
    repeat_evaluation = _LightweightEvaluation(
        path=repeat_loaded.path,
        fit_reference=repeat_loaded.fit_reference,
        aggregates=(),
        validation=repeat_loaded.validation,
        payload_index=repeat_loaded.payload_index,
        run=repeat_loaded.run,
        success=repeat_loaded.success,
        artifact_sha256=repeat_loaded.artifact_sha256,
        run_sha256=repeat_loaded.run_sha256,
    )
    del repeat_loaded
    gc.collect()
    expected_reference = build_health_fit_reference(official_fit)
    if (
        primary_evaluation.fit_reference != expected_reference
        or repeat_evaluation.fit_reference != expected_reference
    ):
        raise HealthReleaseValidationError(
            "both evaluation reruns must bind the designated official fit"
        )
    return _AuthenticatedSources(
        official_fit=official_fit,
        repeat_fit=repeat_fit,
        primary_evaluation=primary_evaluation,
        repeat_evaluation=repeat_evaluation,
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _require_independent_source_trees(sources: _AuthenticatedSources) -> None:
    artifacts = (
        sources.official_fit,
        sources.repeat_fit,
        sources.primary_evaluation,
        sources.repeat_evaluation,
    )
    roots = tuple(artifact.path for artifact in artifacts)
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if _paths_overlap(first, second):
                raise HealthReleaseValidationError(
                    "source artifact roots must be distinct and non-overlapping"
                )
            try:
                if os.path.samestat(first.stat(), second.stat()):
                    raise HealthReleaseValidationError(
                        "source artifact roots must have independent inodes"
                    )
            except OSError as error:
                raise HealthReleaseValidationError(
                    "source artifact roots cannot be inspected"
                ) from error
    for primary, repeat, paths in (
        (sources.official_fit, sources.repeat_fit, HEALTH_FIT_ARTIFACT_PATHS),
        (
            sources.primary_evaluation,
            sources.repeat_evaluation,
            HEALTH_EVAL_ARTIFACT_PATHS,
        ),
    ):
        for path in paths:
            try:
                if os.path.samestat(
                    (primary.path / path).stat(),
                    (repeat.path / path).stat(),
                ):
                    raise HealthReleaseValidationError(
                        "repeat source members must have independent inodes"
                    )
            except OSError as error:
                raise HealthReleaseValidationError(
                    "repeat source members cannot be inspected"
                ) from error


def _normalized_command(run: RunRecordV1Alpha1) -> tuple[str, ...]:
    command = tuple(run.command)
    try:
        output_index = command.index("--output-dir")
    except ValueError as error:
        raise HealthReleaseValidationError(
            "health repeat command is missing --output-dir"
        ) from error
    if output_index != len(command) - 2:
        raise HealthReleaseValidationError(
            "health repeat command has a noncanonical output argument"
        )
    return (*command[: output_index + 1], "<artifact-output>")


def _normalized_run_identity(run: RunRecordV1Alpha1) -> Mapping[str, Any]:
    value = run.model_dump(mode="json", by_alias=True)
    value.pop("started_at")
    value.pop("ended_at")
    value["command"] = _normalized_command(run)
    return value


def _parse_darwin_time_l(value: bytes) -> tuple[float, int]:
    if type(value) is not bytes or not 0 < len(value) <= 65_536 or not value.endswith(b"\n"):
        raise HealthReleaseValidationError(
            "Darwin time -l log must be bounded newline-terminated bytes"
        )
    try:
        value.decode("ascii")
    except UnicodeDecodeError as error:
        raise HealthReleaseValidationError("Darwin time -l log must be ASCII") from error
    real_matches = tuple(
        match
        for line in value.splitlines()
        if (match := _TIME_L_REAL_PATTERN.fullmatch(line)) is not None
    )
    rss_matches = tuple(
        match
        for line in value.splitlines()
        if (match := _TIME_L_RSS_PATTERN.fullmatch(line)) is not None
    )
    if len(real_matches) != 1 or len(rss_matches) != 1:
        raise HealthReleaseValidationError(
            "Darwin time -l log must contain exactly one real-time and RSS record"
        )
    real_match = real_matches[0]
    rss_match = cast(re.Match[bytes], rss_matches[0])
    wall_time_seconds = float(real_match.group("real"))
    peak_rss_bytes = int(rss_match.group("rss"))
    if not math.isfinite(wall_time_seconds) or wall_time_seconds <= 0.0 or peak_rss_bytes <= 0:
        raise HealthReleaseValidationError("Darwin time -l resources must be positive")
    return wall_time_seconds, peak_rss_bytes


def _run_wall_time(run: RunRecordV1Alpha1) -> float:
    if run.ended_at is None:
        raise HealthReleaseValidationError("resource-bound run lacks timestamps")
    return (run.ended_at - run.started_at).total_seconds()


def build_health_resource_measurement(
    run_label: RunLabel,
    artifact: LoadedHealthFitArtifact | _LightweightEvaluation,
    time_l_log: bytes,
) -> HealthRunResourceEvidenceV1:
    """Build the strict self-reported sidecar for one exact retained run."""

    wall_time_seconds, peak_rss_bytes = _parse_darwin_time_l(time_l_log)
    internal_wall_time = _run_wall_time(artifact.run)
    if wall_time_seconds + 0.02 < internal_wall_time:
        raise HealthReleaseValidationError(
            "external resource wall time does not cover the retained run interval"
        )
    phase = "fit" if run_label.endswith("fit") else "evaluation"
    return HealthRunResourceEvidenceV1(
        run_label=run_label,
        phase=phase,
        artifact_contract=(
            HEALTH_FIT_ARTIFACT_CONTRACT if phase == "fit" else HEALTH_EVAL_ARTIFACT_CONTRACT
        ),
        artifact_sha256=artifact.artifact_sha256,
        run_sha256=artifact.run_sha256,
        logical_command=tuple(artifact.run.command),
        cpu_model=artifact.run.environment.cpu_model,
        os_name=artifact.run.environment.os_name,
        os_release=artifact.run.environment.os_release,
        tool_path="/usr/bin/time",
        tool_options=("-l",),
        parser_contract="ffb.darwin-time-l/v1",
        raw_log_path=_RESOURCE_LOG_RELEASE_PATHS[run_label],
        raw_log_sha256=_sha256_bytes(time_l_log),
        raw_log_byte_length=len(time_l_log),
        wall_time_seconds=wall_time_seconds,
        maximum_resident_set_size_raw=peak_rss_bytes,
        maximum_resident_set_size_unit="bytes",
        peak_rss_bytes=peak_rss_bytes,
        exit_status=0,
        direct_child_count=1,
        accelerator_requested=False,
        wall_time_cap_seconds=1800.0,
        peak_rss_cap_bytes=_PEAK_RSS_BYTES_MAX,
        wall_time_within_cap=wall_time_seconds < _WALL_TIME_SECONDS_MAX,
        peak_rss_within_cap=peak_rss_bytes < _PEAK_RSS_BYTES_MAX,
        measurement_scope=(
            "operator-recorded-darwin-time-l-for-one-direct-child;"
            "self-reported-not-cryptographic-execution-proof"
        ),
    )


def _validated_resources(
    run_label: RunLabel,
    artifact: LoadedHealthFitArtifact | _LightweightEvaluation,
    resources: HealthRunResources,
) -> HealthRunResourceEvidenceV1:
    expected = build_health_resource_measurement(
        run_label,
        artifact,
        resources.time_l_log,
    )
    if resources.measurement != expected:
        raise HealthReleaseValidationError(
            "resource sidecar is not bound to the exact retained run and raw log"
        )
    return expected


def _commitment(
    *,
    artifact_kind: SourceArtifactKind,
    path: str,
    primary: LoadedHealthFitArtifact | _LightweightEvaluation,
    repeat: LoadedHealthFitArtifact | _LightweightEvaluation,
) -> HealthSourceMemberCommitmentV1:
    first = _payload_entry(primary, path)
    second = _payload_entry(repeat, path)
    if artifact_kind == "fit":
        primary_retained_path = _PRIMARY_FIT_RELEASE_PATHS[path]
        repeat_retained_path = _REPEAT_FIT_RELEASE_PATHS[path]
        scope = "primary-and-repeat-exact-source-bytes"
    elif path in _OMITTED_EVALUATION_PATHS:
        primary_retained_path = None
        repeat_retained_path = None
        scope = "omitted-commitment-only-not-independently-recomputable"
    else:
        primary_retained_path = _EVALUATION_SCIENCE_RELEASE_PATHS[path]
        repeat_retained_path = primary_retained_path
        scope = "one-exact-copy-shared-by-identical-evaluations"
    return HealthSourceMemberCommitmentV1(
        schema="ffb.health-source-member-commitment/v1",
        artifact_kind=artifact_kind,
        path=path,
        primary_artifact_sha256=primary.artifact_sha256,
        repeat_artifact_sha256=repeat.artifact_sha256,
        primary_sha256=first.sha256,
        repeat_sha256=second.sha256,
        primary_byte_length=first.byte_length,
        repeat_byte_length=second.byte_length,
        primary_record_count=first.record_count,
        repeat_record_count=second.record_count,
        equal=(
            first.sha256 == second.sha256
            and first.byte_length == second.byte_length
            and first.record_count == second.record_count
        ),
        primary_retained_release_path=primary_retained_path,
        repeat_retained_release_path=repeat_retained_path,
        retention_scope=cast(Any, scope),
    )


def _payload_index_bytes_equal(
    primary: LoadedHealthFitArtifact | _LightweightEvaluation,
    repeat: LoadedHealthFitArtifact | _LightweightEvaluation,
) -> bool:
    return (primary.path / HEALTH_PAYLOAD_INDEX_FILE).read_bytes() == (
        repeat.path / HEALTH_PAYLOAD_INDEX_FILE
    ).read_bytes()


def build_health_repeat_evidence(
    sources: _AuthenticatedSources,
    *,
    primary_fit_resources: HealthRunResources,
    repeat_fit_resources: HealthRunResources,
    primary_evaluation_resources: HealthRunResources,
    repeat_evaluation_resources: HealthRunResources,
) -> tuple[
    tuple[HealthSourceMemberCommitmentV1, ...],
    HealthRepeatVerificationV1,
]:
    """Build exact two-fit/two-evaluation repeat and omission evidence."""

    _require_independent_source_trees(sources)
    commitments = tuple(
        [
            *(
                _commitment(
                    artifact_kind="fit",
                    path=path,
                    primary=sources.official_fit,
                    repeat=sources.repeat_fit,
                )
                for path in HEALTH_FIT_INDEXED_PATHS
            ),
            *(
                _commitment(
                    artifact_kind="evaluation",
                    path=path,
                    primary=sources.primary_evaluation,
                    repeat=sources.repeat_evaluation,
                )
                for path in HEALTH_EVAL_INDEXED_PATHS
            ),
        ]
    )
    commitment_bytes = _canonical_records(commitments)
    runs = (
        sources.official_fit,
        sources.repeat_fit,
        sources.primary_evaluation,
        sources.repeat_evaluation,
    )
    environments = tuple(artifact.run.environment for artifact in runs)
    resources = (
        _validated_resources(
            "primary-fit",
            sources.official_fit,
            primary_fit_resources,
        ),
        _validated_resources(
            "repeat-fit",
            sources.repeat_fit,
            repeat_fit_resources,
        ),
        _validated_resources(
            "primary-evaluation",
            sources.primary_evaluation,
            primary_evaluation_resources,
        ),
        _validated_resources(
            "repeat-evaluation",
            sources.repeat_evaluation,
            repeat_evaluation_resources,
        ),
    )
    mismatch_count = sum(not item.equal for item in commitments)
    distinct_run_records = (
        sources.official_fit.run_sha256 != sources.repeat_fit.run_sha256
        and sources.primary_evaluation.run_sha256 != sources.repeat_evaluation.run_sha256
    )
    normalized_equal = _normalized_run_identity(
        sources.official_fit.run
    ) == _normalized_run_identity(sources.repeat_fit.run) and _normalized_run_identity(
        sources.primary_evaluation.run
    ) == _normalized_run_identity(sources.repeat_evaluation.run)
    same_environment = all(environment == environments[0] for environment in environments[1:])
    fit_index_equal = _payload_index_bytes_equal(
        sources.official_fit,
        sources.repeat_fit,
    )
    evaluation_index_equal = _payload_index_bytes_equal(
        sources.primary_evaluation,
        sources.repeat_evaluation,
    )
    all_checks = (
        mismatch_count == 0
        and fit_index_equal
        and evaluation_index_equal
        and sources.official_fit.artifact_sha256 == sources.repeat_fit.artifact_sha256
        and sources.primary_evaluation.artifact_sha256 == sources.repeat_evaluation.artifact_sha256
        and distinct_run_records
        and normalized_equal
        and same_environment
        and all(resource.all_checks_passed for resource in resources)
    )
    repeat = HealthRepeatVerificationV1(
        schema="ffb.health-repeat-verification/v1",
        intent_sha256=sha256_digest(sources.official_fit.intent),
        official_fit_artifact_sha256=sources.official_fit.artifact_sha256,
        official_fit_run_sha256=sources.official_fit.run_sha256,
        repeat_fit_artifact_sha256=sources.repeat_fit.artifact_sha256,
        repeat_fit_run_sha256=sources.repeat_fit.run_sha256,
        primary_evaluation_artifact_sha256=(sources.primary_evaluation.artifact_sha256),
        primary_evaluation_run_sha256=sources.primary_evaluation.run_sha256,
        repeat_evaluation_artifact_sha256=(sources.repeat_evaluation.artifact_sha256),
        repeat_evaluation_run_sha256=sources.repeat_evaluation.run_sha256,
        fit_reference_sha256=sha256_digest(sources.primary_evaluation.fit_reference),
        source_member_commitments_sha256=_sha256_bytes(commitment_bytes),
        fit_member_comparison_count=7,
        evaluation_member_comparison_count=9,
        total_member_comparison_count=16,
        mismatch_count=mismatch_count,
        fit_payload_index_equal=fit_index_equal,
        evaluation_payload_index_equal=evaluation_index_equal,
        scientific_members_all_equal=mismatch_count == 0,
        volatile_run_records_distinct=distinct_run_records,
        normalized_run_identity_equal_within_phase=normalized_equal,
        same_exact_runtime_environment=same_environment,
        source_paths_and_inodes_independent=True,
        resources=resources,
        resource_measurement_scope=(
            "raw-darwin-time-l-logs-retained-and-reparsed;"
            "operator-recorded-sidecars-self-reported-not-cryptographic-execution-proof"
        ),
        execution_evidence_scope=(
            "distinct-path-and-inode-consistency-not-cryptographic-proof-of-two-executions"
        ),
        inference_recomputation_scope=(
            "aggregates-regenerate-claims-and-figures;"
            "omitted-sequence-rows-prevent-independent-bootstrap-recomputation"
        ),
        all_checks_passed=all_checks,
    )
    return commitments, repeat


def _aggregate_key(
    record: HealthAggregateMetricV1,
) -> tuple[str, str, str, str]:
    return (
        record.condition_id,
        "" if record.method is None else record.method,
        record.metric_name,
        "" if record.window is None else record.window,
    )


def validate_health_quantitative_claims(
    claims: Sequence[HealthQuantitativeClaimV1],
    *,
    aggregates: Sequence[HealthAggregateMetricV1],
    fit_summary: HealthFitSummaryV1,
    repeat: HealthRepeatVerificationV1,
) -> tuple[HealthQuantitativeClaimV1, ...]:
    """Require every numeric claim to equal one retained typed source value."""

    ordered = tuple(sorted(claims, key=lambda claim: claim.claim_id))
    if not ordered:
        raise HealthReleaseValidationError(
            "health release requires at least one declared quantitative claim"
        )
    if len({claim.claim_id for claim in ordered}) != len(ordered):
        raise HealthReleaseValidationError("quantitative claim IDs must be unique")
    aggregate_by_key = {_aggregate_key(record): record for record in aggregates}
    if len(aggregate_by_key) != len(aggregates):
        raise HealthReleaseValidationError("retained aggregates contain duplicate keys")
    resources = {resource.run_label: resource for resource in repeat.resources}
    fit_values: Mapping[FitClaimField, float] = {
        "selected-candidate-index": float(fit_summary.selected_candidate_index),
        "selected-self-threshold": float(fit_summary.selected_self_threshold),
        "selected-cross-threshold": float(fit_summary.selected_cross_threshold),
    }
    for claim in ordered:
        if isinstance(claim, AggregateQuantitativeClaimV1):
            retained = aggregate_by_key.get(_aggregate_key(claim.aggregate))
            if retained != claim.aggregate:
                raise HealthReleaseValidationError(
                    "aggregate claim is not an exact retained-row projection"
                )
        elif isinstance(claim, FitQuantitativeClaimV1):
            if float(claim.value) != fit_values[claim.field]:
                raise HealthReleaseValidationError("fit claim disagrees with retained fit summary")
        else:
            resource = resources[claim.run_label]
            expected_value = (
                float(resource.wall_time_seconds)
                if claim.metric == "wall-time-seconds"
                else float(resource.peak_rss_bytes)
            )
            if float(claim.value) != expected_value or claim.cpu_model != resource.cpu_model:
                raise HealthReleaseValidationError(
                    "resource claim disagrees with measured run evidence"
                )
    return ordered


def build_health_release_summary(
    *,
    sources: _AuthenticatedSources,
    commitments: Sequence[HealthSourceMemberCommitmentV1],
    repeat: HealthRepeatVerificationV1,
    claims: Sequence[HealthQuantitativeClaimV1],
) -> HealthReleaseSummaryV1:
    """Recompute the compact release summary from retained typed evidence."""

    aggregates = sources.primary_evaluation.aggregates
    status_counts = {
        status: sum(record.status == status for record in aggregates)
        for status in ("ok", "undefined", "not-applicable")
    }
    conditions = {record.condition_id for record in aggregates}
    omitted = tuple(
        item
        for item in commitments
        if item.retention_scope == "omitted-commitment-only-not-independently-recomputable"
    )
    return HealthReleaseSummaryV1(
        schema="ffb.health-release-summary/v1",
        release_id=sources.official_fit.intent.release_id,
        intent_sha256=sha256_digest(sources.official_fit.intent),
        git_revision=sources.official_fit.run.git_revision,
        lockfile_sha256=sources.official_fit.run.lockfile_sha256,
        package_version=sources.official_fit.run.package_version,
        official_fit_artifact_sha256=sources.official_fit.artifact_sha256,
        official_fit_run_sha256=sources.official_fit.run_sha256,
        evaluation_artifact_sha256=(sources.primary_evaluation.artifact_sha256),
        selected_candidate_index=(sources.official_fit.summary.selected_candidate_index),
        selected_self_threshold=(sources.official_fit.summary.selected_self_threshold),
        selected_cross_threshold=(sources.official_fit.summary.selected_cross_threshold),
        condition_count=cast(Any, len(conditions)),
        aggregate_record_count=len(aggregates),
        aggregate_ok_count=status_counts["ok"],
        aggregate_undefined_count=status_counts["undefined"],
        aggregate_not_applicable_count=status_counts["not-applicable"],
        quantitative_claim_count=len(claims),
        source_member_commitment_count=16,
        omitted_sequence_member_count=cast(Any, len(omitted)),
        omitted_sequence_record_count=cast(
            Any,
            sum(cast(int, item.primary_record_count) for item in omitted),
        ),
        omitted_sequence_byte_count=sum(item.primary_byte_length for item in omitted),
        curated_release_bytes_max=_CURATED_RELEASE_BYTES_MAX,
        complete_aggregate_matrix_retained=True,
        aggregate_evidence_scope=(
            "exact-source-aggregate-rows-regenerate-claims-tables-and-figures"
        ),
        omitted_inference_scope=(
            "sequence-loss-contrast-event-rows-committed-not-retained;"
            "third-parties-cannot-independently-recompute-bootstrap-inference"
        ),
        source_artifacts_strictly_validated_before_curation=True,
        all_checks_passed=(
            repeat.all_checks_passed
            and sources.official_fit.validation.all_checks_passed
            and sources.repeat_fit.validation.all_checks_passed
            and sources.primary_evaluation.validation.all_checks_passed
            and sources.repeat_evaluation.validation.all_checks_passed
            and len(commitments) == 16
            and len(conditions) == 47
            and len(omitted) == 3
        ),
    )


def build_health_release_index(
    files: Mapping[str, bytes],
    *,
    summary: HealthReleaseSummaryV1,
    record_counts: Mapping[str, int],
) -> HealthReleaseIndexV1:
    """Build the exact ordered index for already prepared curated members."""

    if set(files) != set(HEALTH_RELEASE_INDEXED_PATHS):
        raise HealthReleaseValidationError("prepared release member allowlist is invalid")
    entries = tuple(
        HealthReleaseFileEntryV1(
            path=path,
            byte_length=len(files[path]),
            sha256=_sha256_bytes(files[path]),
            record_count=record_counts.get(path),
        )
        for path in HEALTH_RELEASE_INDEXED_PATHS
    )
    return HealthReleaseIndexV1(
        schema="ffb.health-release-index/v1",
        artifact_contract=HEALTH_RELEASE_CONTRACT,
        release_id=summary.release_id,
        intent_sha256=summary.intent_sha256,
        official_fit_artifact_sha256=summary.official_fit_artifact_sha256,
        evaluation_artifact_sha256=summary.evaluation_artifact_sha256,
        curated_release_bytes_max=_CURATED_RELEASE_BYTES_MAX,
        omitted_sequence_rows_recomputable=False,
        files=entries,
    )


def _read_source_member(
    root: Path,
    path: str,
    *,
    byte_cap: int = _CURATED_RELEASE_BYTES_MAX,
) -> bytes:
    member = root / path
    try:
        before = os.lstat(member)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > byte_cap
        ):
            raise HealthReleaseValidationError("source member is not a bounded regular file")
        descriptor = os.open(
            member,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size
            ):
                raise HealthReleaseValidationError("source member changed before curation")
            chunks: list[bytes] = []
            remaining = byte_cap + 1
            while remaining:
                chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            value = b"".join(chunks)
            after = os.fstat(descriptor)
            if (
                len(value) != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns
            ):
                raise HealthReleaseValidationError("source member changed during curation")
            return value
        finally:
            os.close(descriptor)
    except HealthReleaseValidationError:
        raise
    except OSError as error:
        raise HealthReleaseValidationError("source member cannot be read") from error


def _copy_source_members(
    files: dict[str, bytes],
    artifact: LoadedHealthFitArtifact | _LightweightEvaluation,
    mapping: Mapping[str, str],
) -> None:
    for source_path, release_path in mapping.items():
        value = _read_source_member(artifact.path, source_path)
        if source_path in {entry.path for entry in artifact.payload_index.files}:
            entry = _payload_entry(artifact, source_path)
            if len(value) != entry.byte_length or _sha256_bytes(value) != entry.sha256:
                raise HealthReleaseValidationError(
                    "source bytes disagree with strict payload commitments"
                )
        files[release_path] = value


def validate_health_release_candidate_bytes(files: Mapping[str, bytes]) -> None:
    """Reject absolute local paths, private notes, and credential-like bytes."""

    for path, value in files.items():
        if (
            _PRIVATE_PATH_PATTERN.search(value)
            or _SECRET_PATTERN.search(value)
            or b"interview/" in value
        ):
            raise HealthReleaseValidationError(f"curated release privacy scan failed for {path}")


def _privacy_scan(files: Mapping[str, bytes], sources: _AuthenticatedSources) -> None:
    validate_health_release_candidate_bytes(files)
    forbidden_exact = tuple(
        os.fsencode(artifact.path)
        for artifact in (
            sources.official_fit,
            sources.repeat_fit,
            sources.primary_evaluation,
            sources.repeat_evaluation,
        )
    )
    for path, value in files.items():
        if any(secret in value for secret in forbidden_exact):
            raise HealthReleaseValidationError(f"curated release privacy scan failed for {path}")


def _prepare_health_release(
    request: HealthReleaseWriteRequest,
) -> _PreparedHealthRelease:
    sources = _authenticate_sources(request)
    commitments, repeat = build_health_repeat_evidence(
        sources,
        primary_fit_resources=request.primary_fit_resources,
        repeat_fit_resources=request.repeat_fit_resources,
        primary_evaluation_resources=request.primary_evaluation_resources,
        repeat_evaluation_resources=request.repeat_evaluation_resources,
    )
    claims = validate_health_quantitative_claims(
        request.quantitative_claims,
        aggregates=sources.primary_evaluation.aggregates,
        fit_summary=sources.official_fit.summary,
        repeat=repeat,
    )
    summary = build_health_release_summary(
        sources=sources,
        commitments=commitments,
        repeat=repeat,
        claims=claims,
    )
    if not summary.all_checks_passed:
        raise HealthReleaseValidationError("M4 release gates did not all pass")

    files: dict[str, bytes] = {}
    _copy_source_members(
        files,
        sources.official_fit,
        _PRIMARY_FIT_RELEASE_PATHS,
    )
    _copy_source_members(
        files,
        sources.repeat_fit,
        _REPEAT_FIT_RELEASE_PATHS,
    )
    _copy_source_members(
        files,
        sources.primary_evaluation,
        _EVALUATION_SCIENCE_RELEASE_PATHS,
    )
    _copy_source_members(
        files,
        sources.primary_evaluation,
        _PRIMARY_EVALUATION_ENVELOPE_PATHS,
    )
    _copy_source_members(
        files,
        sources.repeat_evaluation,
        _REPEAT_EVALUATION_ENVELOPE_PATHS,
    )
    for run_label, resources in (
        ("primary-fit", request.primary_fit_resources),
        ("repeat-fit", request.repeat_fit_resources),
        ("primary-evaluation", request.primary_evaluation_resources),
        ("repeat-evaluation", request.repeat_evaluation_resources),
    ):
        files[_RESOURCE_LOG_RELEASE_PATHS[cast(RunLabel, run_label)]] = resources.time_l_log
    aggregate_bytes = files[_EVALUATION_SCIENCE_RELEASE_PATHS[HEALTH_AGGREGATES_FILE]]
    if aggregate_bytes != canonical_health_ndjson_bytes(sources.primary_evaluation.aggregates):
        raise HealthReleaseValidationError(
            "retained aggregate bytes are not the exact canonical source rows"
        )
    files[HEALTH_RELEASE_COMMITMENTS_FILE] = _canonical_records(commitments)
    files[HEALTH_RELEASE_REPEAT_FILE] = canonical_json_bytes(repeat)
    files[HEALTH_RELEASE_CLAIMS_FILE] = _canonical_records(cast(Sequence[BaseModel], claims))
    files[HEALTH_RELEASE_SUMMARY_FILE] = canonical_json_bytes(summary)
    record_counts = {
        _PRIMARY_FIT_RELEASE_PATHS[HEALTH_CANDIDATES_FILE]: 36,
        _EVALUATION_SCIENCE_RELEASE_PATHS[HEALTH_AGGREGATES_FILE]: len(
            sources.primary_evaluation.aggregates
        ),
        _REPEAT_FIT_RELEASE_PATHS[HEALTH_CANDIDATES_FILE]: 36,
        HEALTH_RELEASE_COMMITMENTS_FILE: len(commitments),
        HEALTH_RELEASE_CLAIMS_FILE: len(claims),
    }
    index = build_health_release_index(
        files,
        summary=summary,
        record_counts=record_counts,
    )
    index_bytes = canonical_json_bytes(index)
    release_digest = compute_health_release_digest(index_bytes)
    success = HealthReleaseSuccessV1(
        schema="ffb.health-release-success/v1",
        release_artifact_sha256=release_digest,
        release_summary_sha256=_sha256_bytes(files[HEALTH_RELEASE_SUMMARY_FILE]),
        repeat_verification_sha256=_sha256_bytes(files[HEALTH_RELEASE_REPEAT_FILE]),
    )
    files[HEALTH_RELEASE_INDEX_FILE] = index_bytes
    files[HEALTH_RELEASE_SUCCESS_FILE] = canonical_json_bytes(success)
    _privacy_scan(files, sources)
    if sum(len(value) for value in files.values()) >= _CURATED_RELEASE_BYTES_MAX:
        raise HealthReleaseValidationError("curated release exceeds its frozen byte cap")
    return _PreparedHealthRelease(files=files)


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


def _reject_symlink_components(root: Path) -> None:
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise HealthReleaseValidationError("release path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise HealthReleaseValidationError("release path contains a symlink")


def _require_safe_release_tree(root: Path) -> _ReleaseTreeSnapshot:
    absolute = absolute_artifact_path(root)
    _reject_symlink_components(absolute)
    try:
        root_stat = os.lstat(absolute)
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise HealthReleaseValidationError("release root must be a real directory")
        entries: dict[str, os.stat_result] = {}
        with os.scandir(absolute) as iterator:
            for entry in iterator:
                metadata = entry.stat(follow_symlinks=False)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise HealthReleaseValidationError(
                        "release members must be independent regular files"
                    )
                entries[entry.name] = metadata
    except HealthReleaseValidationError:
        raise
    except OSError as error:
        raise HealthReleaseValidationError("release tree cannot be inspected") from error
    if set(entries) != set(HEALTH_RELEASE_ARTIFACT_PATHS):
        raise HealthReleaseValidationError("release file allowlist mismatch")
    if sum(entry.st_size for entry in entries.values()) >= _CURATED_RELEASE_BYTES_MAX:
        raise HealthReleaseValidationError("release tree exceeds its byte cap")
    return _ReleaseTreeSnapshot(root_stat=root_stat, entries=entries)


def _read_release_member(
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
) -> bytes:
    descriptor = os.open(
        root / name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stat_fingerprint(opened) != _stat_fingerprint(expected_stat)
        ):
            raise HealthReleaseValidationError("release member changed during validation")
        chunks: list[bytes] = []
        remaining = expected_stat.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise HealthReleaseValidationError("release member changed during reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise HealthReleaseValidationError("release member grew during reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_canonical_model[ModelT: BaseModel](
    value: bytes,
    *,
    validate: Callable[[bytes], ModelT],
    label: str,
) -> ModelT:
    try:
        model = validate(value)
    except (ValueError, ValidationError) as error:
        raise HealthReleaseValidationError(
            f"{label} does not satisfy its strict contract"
        ) from error
    if canonical_json_bytes(model) != value:
        raise HealthReleaseValidationError(f"{label} is not canonical JSON")
    return model


def _load_canonical_ndjson[ModelT: BaseModel](
    value: bytes,
    *,
    validate: Callable[[bytes], ModelT],
    label: str,
) -> tuple[ModelT, ...]:
    lines = value.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise HealthReleaseValidationError(f"{label} is not canonical NDJSON")
    result = tuple(_load_canonical_model(line, validate=validate, label=label) for line in lines)
    return result


def _load_claim(value: bytes) -> HealthQuantitativeClaimV1:
    return _CLAIM_ADAPTER.validate_json(value, strict=True)


def _source_graph(
    files: Mapping[str, bytes],
    *,
    prefix_mapping: Mapping[str, str],
    expected_artifact_sha256: str,
    expected_run_sha256: str,
) -> tuple[HealthPayloadIndexV1, RunRecordV1Alpha1]:
    index_bytes = files[prefix_mapping[HEALTH_PAYLOAD_INDEX_FILE]]
    run_bytes = files[prefix_mapping[HEALTH_RUN_FILE]]
    success_bytes = files[prefix_mapping[HEALTH_SUCCESS_FILE]]
    index = _load_canonical_model(
        index_bytes,
        validate=HealthPayloadIndexV1.model_validate_json,
        label=prefix_mapping[HEALTH_PAYLOAD_INDEX_FILE],
    )
    run = _load_canonical_model(
        run_bytes,
        validate=RunRecordV1Alpha1.model_validate_json,
        label=prefix_mapping[HEALTH_RUN_FILE],
    )
    success = _load_canonical_model(
        success_bytes,
        validate=SuccessMarkerV1Alpha1.model_validate_json,
        label=prefix_mapping[HEALTH_SUCCESS_FILE],
    )
    artifact_sha256 = compute_health_artifact_digest(
        index_bytes,
        artifact_contract=index.artifact_contract,
    )
    run_sha256 = compute_run_record_digest(run_bytes)
    if (
        artifact_sha256 != expected_artifact_sha256
        or run_sha256 != expected_run_sha256
        or run.artifact_sha256 != artifact_sha256
        or index.run_id != run.run_id
        or success.artifact_sha256 != artifact_sha256
        or success.run_sha256 != run_sha256
    ):
        raise HealthReleaseValidationError("retained source artifact identity graph is invalid")
    return index, run


def _validate_fit_copies(
    files: Mapping[str, bytes],
    *,
    mapping: Mapping[str, str],
    index: HealthPayloadIndexV1,
) -> None:
    for entry in index.files:
        value = files[mapping[entry.path]]
        if len(value) != entry.byte_length or _sha256_bytes(value) != entry.sha256:
            raise HealthReleaseValidationError(
                "retained fit source bytes disagree with its payload index"
            )


def _parse_retained_intent(value: bytes) -> HealthBenchmarkIntentV1:
    return HEALTH_BENCHMARK_INTENT_ADAPTER.validate_json(value, strict=True)


def _load_retained_intent(value: bytes, *, label: str) -> HealthBenchmarkIntentV1:
    return _load_canonical_model(
        value,
        validate=_parse_retained_intent,
        label=label,
    )


def _load_retained_profile(
    value: bytes,
    *,
    label: str,
) -> MainProceduralProfile | EdgeProceduralProfile:
    profile = _load_canonical_model(
        value,
        validate=_parse_retained_profile,
        label=label,
    )
    return profile


def _parse_retained_profile(
    value: bytes,
) -> MainProceduralProfile | EdgeProceduralProfile:
    profile = PROCEDURAL_PROFILE_ADAPTER.validate_json(value, strict=True)
    if not isinstance(profile, (MainProceduralProfile, EdgeProceduralProfile)):
        raise ValueError("retained profile is not main or edge")
    return profile


def _calibration_from_envelope(envelope: HealthEcdfArraysV1) -> HealthCalibration:
    channels = {channel.channel: channel.values for channel in envelope.channels}
    return HealthCalibration(
        camera_self_mean=cast(Any, channels["camera_self_mean"]),
        camera_self_maximum=cast(Any, channels["camera_self_maximum"]),
        lidar_self_mean=cast(Any, channels["lidar_self_mean"]),
        lidar_self_maximum=cast(Any, channels["lidar_self_maximum"]),
        camera_from_lidar_cross_mean=cast(
            Any,
            channels["camera_from_lidar_cross_mean"],
        ),
        camera_from_lidar_cross_maximum=cast(
            Any,
            channels["camera_from_lidar_cross_maximum"],
        ),
        lidar_from_camera_cross_mean=cast(
            Any,
            channels["lidar_from_camera_cross_mean"],
        ),
        lidar_from_camera_cross_maximum=cast(
            Any,
            channels["lidar_from_camera_cross_maximum"],
        ),
    )


def _validate_retained_fit_semantics(
    files: Mapping[str, bytes],
    *,
    mapping: Mapping[str, str],
    run: RunRecordV1Alpha1,
) -> _RetainedFitSemantics:
    intent = _load_retained_intent(
        files[mapping[HEALTH_INTENT_FILE]],
        label=mapping[HEALTH_INTENT_FILE],
    )
    main_profile = _load_retained_profile(
        files[mapping[HEALTH_MAIN_PROFILE_FILE]],
        label=mapping[HEALTH_MAIN_PROFILE_FILE],
    )
    edge_profile = _load_retained_profile(
        files[mapping[HEALTH_EDGE_PROFILE_FILE]],
        label=mapping[HEALTH_EDGE_PROFILE_FILE],
    )
    if not isinstance(main_profile, MainProceduralProfile) or not isinstance(
        edge_profile,
        EdgeProceduralProfile,
    ):
        raise HealthReleaseValidationError(
            "retained fit profiles do not use the main/edge populations"
        )
    ecdf = _load_canonical_model(
        files[mapping[HEALTH_ECDF_FILE]],
        validate=HealthEcdfArraysV1.model_validate_json,
        label=mapping[HEALTH_ECDF_FILE],
    )
    candidates = _load_canonical_ndjson(
        files[mapping[HEALTH_CANDIDATES_FILE]],
        validate=HealthThresholdCandidateV1.model_validate_json,
        label=mapping[HEALTH_CANDIDATES_FILE],
    )
    summary = _load_canonical_model(
        files[mapping[HEALTH_FIT_SUMMARY_FILE]],
        validate=HealthFitSummaryV1.model_validate_json,
        label=mapping[HEALTH_FIT_SUMMARY_FILE],
    )
    validation = _load_canonical_model(
        files[mapping[HEALTH_FIT_VALIDATION_FILE]],
        validate=HealthValidationV1.model_validate_json,
        label=mapping[HEALTH_FIT_VALIDATION_FILE],
    )
    ordered = validate_health_fit_bundle(
        intent,
        main_profile,
        edge_profile,
        _calibration_from_envelope(ecdf),
        candidates,
        summary,
        validation,
        run,
    )
    if candidates != ordered:
        raise HealthReleaseValidationError("retained fit candidates are not in canonical order")
    return _RetainedFitSemantics(
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        summary=summary,
    )


def _validate_commitment_envelope(
    commitments: Sequence[HealthSourceMemberCommitmentV1],
    *,
    repeat: HealthRepeatVerificationV1,
    primary_fit_index: HealthPayloadIndexV1,
    repeat_fit_index: HealthPayloadIndexV1,
    primary_evaluation_index: HealthPayloadIndexV1,
    repeat_evaluation_index: HealthPayloadIndexV1,
) -> None:
    expected_keys = (
        *(("fit", path) for path in HEALTH_FIT_INDEXED_PATHS),
        *(("evaluation", path) for path in HEALTH_EVAL_INDEXED_PATHS),
    )
    if tuple((item.artifact_kind, item.path) for item in commitments) != expected_keys:
        raise HealthReleaseValidationError(
            "source commitments do not use the exact canonical member order"
        )
    indexes = (
        (
            "fit",
            primary_fit_index,
            repeat_fit_index,
            repeat.official_fit_artifact_sha256,
            repeat.repeat_fit_artifact_sha256,
        ),
        (
            "evaluation",
            primary_evaluation_index,
            repeat_evaluation_index,
            repeat.primary_evaluation_artifact_sha256,
            repeat.repeat_evaluation_artifact_sha256,
        ),
    )
    offset = 0
    for artifact_kind, primary_index, repeat_index, primary_artifact, repeat_artifact in indexes:
        for primary_entry, repeat_entry in zip(
            primary_index.files,
            repeat_index.files,
            strict=True,
        ):
            commitment = commitments[offset]
            offset += 1
            if (
                commitment.artifact_kind != artifact_kind
                or commitment.path != primary_entry.path
                or primary_entry.path != repeat_entry.path
                or commitment.primary_artifact_sha256 != primary_artifact
                or commitment.repeat_artifact_sha256 != repeat_artifact
                or commitment.primary_sha256 != primary_entry.sha256
                or commitment.repeat_sha256 != repeat_entry.sha256
                or commitment.primary_byte_length != primary_entry.byte_length
                or commitment.repeat_byte_length != repeat_entry.byte_length
                or commitment.primary_record_count != primary_entry.record_count
                or commitment.repeat_record_count != repeat_entry.record_count
            ):
                raise HealthReleaseValidationError(
                    "source payload envelopes disagree with member commitments"
                )
    if sum(not item.equal for item in commitments) != repeat.mismatch_count:
        raise HealthReleaseValidationError(
            "source commitment mismatches disagree with repeat evidence"
        )


def _validate_evaluation_commitments(
    files: Mapping[str, bytes],
    *,
    commitments: Sequence[HealthSourceMemberCommitmentV1],
    primary_index: HealthPayloadIndexV1,
    repeat_index: HealthPayloadIndexV1,
) -> None:
    by_path = {item.path: item for item in commitments if item.artifact_kind == "evaluation"}
    if len(by_path) != len(HEALTH_EVAL_INDEXED_PATHS):
        raise HealthReleaseValidationError("evaluation commitments are incomplete")
    for primary, repeat in zip(
        primary_index.files,
        repeat_index.files,
        strict=True,
    ):
        commitment = by_path[primary.path]
        if (
            primary.path != repeat.path
            or primary.sha256 != commitment.primary_sha256
            or repeat.sha256 != commitment.repeat_sha256
            or primary.byte_length != commitment.primary_byte_length
            or repeat.byte_length != commitment.repeat_byte_length
            or primary.record_count != commitment.primary_record_count
            or repeat.record_count != commitment.repeat_record_count
        ):
            raise HealthReleaseValidationError(
                "evaluation source envelope disagrees with commitments"
            )
        if commitment.primary_retained_release_path is not None:
            primary_value = files[commitment.primary_retained_release_path]
            repeat_value = files[cast(str, commitment.repeat_retained_release_path)]
            if (
                len(primary_value) != primary.byte_length
                or _sha256_bytes(primary_value) != primary.sha256
                or len(repeat_value) != repeat.byte_length
                or _sha256_bytes(repeat_value) != repeat.sha256
            ):
                raise HealthReleaseValidationError("retained evaluation source bytes are invalid")


def _validate_release_record_counts(
    index: HealthReleaseIndexV1,
    *,
    primary_fit_index: HealthPayloadIndexV1,
    repeat_fit_index: HealthPayloadIndexV1,
    aggregates: Sequence[HealthAggregateMetricV1],
    commitments: Sequence[HealthSourceMemberCommitmentV1],
    claims: Sequence[HealthQuantitativeClaimV1],
) -> None:
    primary_candidate_count = cast(
        int,
        next(
            entry.record_count
            for entry in primary_fit_index.files
            if entry.path == HEALTH_CANDIDATES_FILE
        ),
    )
    repeat_candidate_count = cast(
        int,
        next(
            entry.record_count
            for entry in repeat_fit_index.files
            if entry.path == HEALTH_CANDIDATES_FILE
        ),
    )
    expected_counts: Mapping[str, int] = {
        _PRIMARY_FIT_RELEASE_PATHS[HEALTH_CANDIDATES_FILE]: primary_candidate_count,
        _REPEAT_FIT_RELEASE_PATHS[HEALTH_CANDIDATES_FILE]: repeat_candidate_count,
        _EVALUATION_SCIENCE_RELEASE_PATHS[HEALTH_AGGREGATES_FILE]: len(aggregates),
        HEALTH_RELEASE_COMMITMENTS_FILE: len(commitments),
        HEALTH_RELEASE_CLAIMS_FILE: len(claims),
    }
    if primary_candidate_count != 36 or repeat_candidate_count != 36:
        raise HealthReleaseValidationError(
            "retained fit candidate counts disagree with the frozen grid"
        )
    for entry in index.files:
        if entry.record_count != expected_counts.get(entry.path):
            raise HealthReleaseValidationError(
                "release index record count does not recompute from retained evidence"
            )


def _validate_retained_source_evidence(
    files: Mapping[str, bytes],
    *,
    repeat: HealthRepeatVerificationV1,
    primary_fit_index: HealthPayloadIndexV1,
    repeat_fit_index: HealthPayloadIndexV1,
    primary_evaluation_index: HealthPayloadIndexV1,
    repeat_evaluation_index: HealthPayloadIndexV1,
    primary_fit_run: RunRecordV1Alpha1,
    repeat_fit_run: RunRecordV1Alpha1,
    primary_evaluation_run: RunRecordV1Alpha1,
    repeat_evaluation_run: RunRecordV1Alpha1,
) -> None:
    fit_index_equal = (
        files[_PRIMARY_FIT_RELEASE_PATHS[HEALTH_PAYLOAD_INDEX_FILE]]
        == files[_REPEAT_FIT_RELEASE_PATHS[HEALTH_PAYLOAD_INDEX_FILE]]
    )
    evaluation_index_equal = (
        files[_PRIMARY_EVALUATION_ENVELOPE_PATHS[HEALTH_PAYLOAD_INDEX_FILE]]
        == files[_REPEAT_EVALUATION_ENVELOPE_PATHS[HEALTH_PAYLOAD_INDEX_FILE]]
    )
    run_pairs = (
        (primary_fit_run, repeat_fit_run),
        (primary_evaluation_run, repeat_evaluation_run),
    )
    normalized_equal = all(
        _normalized_run_identity(primary) == _normalized_run_identity(repeated)
        for primary, repeated in run_pairs
    )
    all_runs = (
        primary_fit_run,
        repeat_fit_run,
        primary_evaluation_run,
        repeat_evaluation_run,
    )
    same_environment = all(run.environment == all_runs[0].environment for run in all_runs[1:])
    volatile_distinct = (
        repeat.official_fit_run_sha256 != repeat.repeat_fit_run_sha256
        and repeat.primary_evaluation_run_sha256 != repeat.repeat_evaluation_run_sha256
    )
    indexes = (
        primary_fit_index,
        repeat_fit_index,
        primary_evaluation_index,
        repeat_evaluation_index,
    )
    if (
        primary_fit_index.artifact_contract != HEALTH_FIT_ARTIFACT_CONTRACT
        or repeat_fit_index.artifact_contract != HEALTH_FIT_ARTIFACT_CONTRACT
        or primary_evaluation_index.artifact_contract != HEALTH_EVAL_ARTIFACT_CONTRACT
        or repeat_evaluation_index.artifact_contract != HEALTH_EVAL_ARTIFACT_CONTRACT
        or any(index.intent_sha256 != repeat.intent_sha256 for index in indexes)
        or len({index.main_profile_sha256 for index in indexes}) != 1
        or len({index.edge_profile_sha256 for index in indexes}) != 1
        or repeat.fit_payload_index_equal != fit_index_equal
        or repeat.evaluation_payload_index_equal != evaluation_index_equal
        or repeat.normalized_run_identity_equal_within_phase != normalized_equal
        or repeat.same_exact_runtime_environment != same_environment
        or repeat.volatile_run_records_distinct != volatile_distinct
        or not repeat.source_paths_and_inodes_independent
        or not repeat.all_checks_passed
    ):
        raise HealthReleaseValidationError(
            "repeat evidence does not recompute from retained source envelopes"
        )
    expected_resources = {
        "primary-fit": (
            repeat.official_fit_artifact_sha256,
            repeat.official_fit_run_sha256,
            primary_fit_run,
        ),
        "repeat-fit": (
            repeat.repeat_fit_artifact_sha256,
            repeat.repeat_fit_run_sha256,
            repeat_fit_run,
        ),
        "primary-evaluation": (
            repeat.primary_evaluation_artifact_sha256,
            repeat.primary_evaluation_run_sha256,
            primary_evaluation_run,
        ),
        "repeat-evaluation": (
            repeat.repeat_evaluation_artifact_sha256,
            repeat.repeat_evaluation_run_sha256,
            repeat_evaluation_run,
        ),
    }
    for resource in repeat.resources:
        artifact_sha256, run_sha256, run = expected_resources[resource.run_label]
        raw_log = files[resource.raw_log_path]
        wall_time_seconds, peak_rss_bytes = _parse_darwin_time_l(raw_log)
        internal_wall_time = _run_wall_time(run)
        if (
            resource.artifact_sha256 != artifact_sha256
            or resource.run_sha256 != run_sha256
            or resource.logical_command != tuple(run.command)
            or resource.cpu_model != run.environment.cpu_model
            or resource.os_name != run.environment.os_name
            or resource.os_release != run.environment.os_release
            or resource.raw_log_sha256 != _sha256_bytes(raw_log)
            or resource.raw_log_byte_length != len(raw_log)
            or float(resource.wall_time_seconds) != wall_time_seconds
            or resource.maximum_resident_set_size_raw != peak_rss_bytes
            or resource.peak_rss_bytes != peak_rss_bytes
            or wall_time_seconds + 0.02 < internal_wall_time
        ):
            raise HealthReleaseValidationError("resource evidence is not bound to its retained run")


def _validate_retained_validations(
    files: Mapping[str, bytes],
    *,
    intent_sha256: str,
) -> None:
    validation_paths = (
        _PRIMARY_FIT_RELEASE_PATHS[HEALTH_FIT_VALIDATION_FILE],
        _REPEAT_FIT_RELEASE_PATHS[HEALTH_FIT_VALIDATION_FILE],
        _EVALUATION_SCIENCE_RELEASE_PATHS[HEALTH_EVAL_VALIDATION_FILE],
    )
    validations = tuple(
        _load_canonical_model(
            files[path],
            validate=HealthValidationV1.model_validate_json,
            label=path,
        )
        for path in validation_paths
    )
    if any(
        validation.intent_sha256 != intent_sha256 or not validation.all_checks_passed
        for validation in validations
    ):
        raise HealthReleaseValidationError(
            "retained source validations did not all pass for this intent"
        )


def _validate_loaded_summary(
    summary: HealthReleaseSummaryV1,
    *,
    aggregates: Sequence[HealthAggregateMetricV1],
    claims: Sequence[HealthQuantitativeClaimV1],
    commitments: Sequence[HealthSourceMemberCommitmentV1],
    repeat: HealthRepeatVerificationV1,
    fit_summary: HealthFitSummaryV1,
    official_fit_run: RunRecordV1Alpha1,
) -> None:
    status_counts = {
        status: sum(record.status == status for record in aggregates)
        for status in ("ok", "undefined", "not-applicable")
    }
    omitted = tuple(
        item
        for item in commitments
        if item.retention_scope == "omitted-commitment-only-not-independently-recomputable"
    )
    expected = summary.model_copy(
        update={
            "intent_sha256": repeat.intent_sha256,
            "git_revision": official_fit_run.git_revision,
            "lockfile_sha256": official_fit_run.lockfile_sha256,
            "package_version": official_fit_run.package_version,
            "official_fit_artifact_sha256": (repeat.official_fit_artifact_sha256),
            "official_fit_run_sha256": repeat.official_fit_run_sha256,
            "evaluation_artifact_sha256": (repeat.primary_evaluation_artifact_sha256),
            "selected_candidate_index": fit_summary.selected_candidate_index,
            "selected_self_threshold": fit_summary.selected_self_threshold,
            "selected_cross_threshold": fit_summary.selected_cross_threshold,
            "condition_count": len({row.condition_id for row in aggregates}),
            "aggregate_record_count": len(aggregates),
            "aggregate_ok_count": status_counts["ok"],
            "aggregate_undefined_count": status_counts["undefined"],
            "aggregate_not_applicable_count": status_counts["not-applicable"],
            "quantitative_claim_count": len(claims),
            "source_member_commitment_count": len(commitments),
            "omitted_sequence_member_count": len(omitted),
            "omitted_sequence_record_count": sum(
                cast(int, item.primary_record_count) for item in omitted
            ),
            "omitted_sequence_byte_count": sum(item.primary_byte_length for item in omitted),
            "all_checks_passed": repeat.all_checks_passed,
        }
    )
    if expected != summary:
        raise HealthReleaseValidationError(
            "release summary does not recompute from retained evidence"
        )


def _load_health_release(path: Path) -> LoadedHealthRelease:
    root = absolute_artifact_path(path)
    snapshot = _require_safe_release_tree(root)
    files = {
        name: _read_release_member(
            root,
            name,
            expected_stat=snapshot.entries[name],
        )
        for name in HEALTH_RELEASE_ARTIFACT_PATHS
    }
    validate_health_release_candidate_bytes(files)
    index = _load_canonical_model(
        files[HEALTH_RELEASE_INDEX_FILE],
        validate=HealthReleaseIndexV1.model_validate_json,
        label=HEALTH_RELEASE_INDEX_FILE,
    )
    for expected_path, entry in zip(
        HEALTH_RELEASE_INDEXED_PATHS,
        index.files,
        strict=True,
    ):
        value = files[expected_path]
        if (
            entry.path != expected_path
            or entry.byte_length != len(value)
            or entry.sha256 != _sha256_bytes(value)
        ):
            raise HealthReleaseValidationError("release member disagrees with release index")
    release_sha256 = compute_health_release_digest(files[HEALTH_RELEASE_INDEX_FILE])
    success = _load_canonical_model(
        files[HEALTH_RELEASE_SUCCESS_FILE],
        validate=HealthReleaseSuccessV1.model_validate_json,
        label=HEALTH_RELEASE_SUCCESS_FILE,
    )
    if success.release_artifact_sha256 != release_sha256:
        raise HealthReleaseValidationError("release success digest is invalid")

    summary = _load_canonical_model(
        files[HEALTH_RELEASE_SUMMARY_FILE],
        validate=HealthReleaseSummaryV1.model_validate_json,
        label=HEALTH_RELEASE_SUMMARY_FILE,
    )
    repeat = _load_canonical_model(
        files[HEALTH_RELEASE_REPEAT_FILE],
        validate=HealthRepeatVerificationV1.model_validate_json,
        label=HEALTH_RELEASE_REPEAT_FILE,
    )
    commitments = _load_canonical_ndjson(
        files[HEALTH_RELEASE_COMMITMENTS_FILE],
        validate=HealthSourceMemberCommitmentV1.model_validate_json,
        label=HEALTH_RELEASE_COMMITMENTS_FILE,
    )
    claims = _load_canonical_ndjson(
        files[HEALTH_RELEASE_CLAIMS_FILE],
        validate=_load_claim,
        label=HEALTH_RELEASE_CLAIMS_FILE,
    )
    aggregates = _load_canonical_ndjson(
        files[_EVALUATION_SCIENCE_RELEASE_PATHS[HEALTH_AGGREGATES_FILE]],
        validate=HealthAggregateMetricV1.model_validate_json,
        label=HEALTH_AGGREGATES_FILE,
    )
    if (
        success.release_summary_sha256 != _sha256_bytes(files[HEALTH_RELEASE_SUMMARY_FILE])
        or success.repeat_verification_sha256 != _sha256_bytes(files[HEALTH_RELEASE_REPEAT_FILE])
        or repeat.source_member_commitments_sha256
        != _sha256_bytes(files[HEALTH_RELEASE_COMMITMENTS_FILE])
    ):
        raise HealthReleaseValidationError("release success or repeat binding is invalid")
    fit_summary = _load_canonical_model(
        files[_PRIMARY_FIT_RELEASE_PATHS[HEALTH_FIT_SUMMARY_FILE]],
        validate=HealthFitSummaryV1.model_validate_json,
        label=HEALTH_FIT_SUMMARY_FILE,
    )
    primary_fit_index, official_fit_run = _source_graph(
        files,
        prefix_mapping=_PRIMARY_FIT_RELEASE_PATHS,
        expected_artifact_sha256=repeat.official_fit_artifact_sha256,
        expected_run_sha256=repeat.official_fit_run_sha256,
    )
    repeat_fit_index, repeat_fit_run = _source_graph(
        files,
        prefix_mapping=_REPEAT_FIT_RELEASE_PATHS,
        expected_artifact_sha256=repeat.repeat_fit_artifact_sha256,
        expected_run_sha256=repeat.repeat_fit_run_sha256,
    )
    primary_evaluation_index, primary_evaluation_run = _source_graph(
        files,
        prefix_mapping=_PRIMARY_EVALUATION_ENVELOPE_PATHS,
        expected_artifact_sha256=repeat.primary_evaluation_artifact_sha256,
        expected_run_sha256=repeat.primary_evaluation_run_sha256,
    )
    repeat_evaluation_index, repeat_evaluation_run = _source_graph(
        files,
        prefix_mapping=_REPEAT_EVALUATION_ENVELOPE_PATHS,
        expected_artifact_sha256=repeat.repeat_evaluation_artifact_sha256,
        expected_run_sha256=repeat.repeat_evaluation_run_sha256,
    )
    _validate_fit_copies(
        files,
        mapping=_PRIMARY_FIT_RELEASE_PATHS,
        index=primary_fit_index,
    )
    _validate_fit_copies(
        files,
        mapping=_REPEAT_FIT_RELEASE_PATHS,
        index=repeat_fit_index,
    )
    primary_fit_semantics = _validate_retained_fit_semantics(
        files,
        mapping=_PRIMARY_FIT_RELEASE_PATHS,
        run=official_fit_run,
    )
    repeat_fit_semantics = _validate_retained_fit_semantics(
        files,
        mapping=_REPEAT_FIT_RELEASE_PATHS,
        run=repeat_fit_run,
    )
    if (
        primary_fit_semantics != repeat_fit_semantics
        or primary_fit_semantics.summary != fit_summary
    ):
        raise HealthReleaseValidationError(
            "retained fit copies do not have identical valid semantics"
        )
    _validate_commitment_envelope(
        commitments,
        repeat=repeat,
        primary_fit_index=primary_fit_index,
        repeat_fit_index=repeat_fit_index,
        primary_evaluation_index=primary_evaluation_index,
        repeat_evaluation_index=repeat_evaluation_index,
    )
    _validate_evaluation_commitments(
        files,
        commitments=commitments,
        primary_index=primary_evaluation_index,
        repeat_index=repeat_evaluation_index,
    )
    _validate_release_record_counts(
        index,
        primary_fit_index=primary_fit_index,
        repeat_fit_index=repeat_fit_index,
        aggregates=aggregates,
        commitments=commitments,
        claims=claims,
    )
    _validate_retained_source_evidence(
        files,
        repeat=repeat,
        primary_fit_index=primary_fit_index,
        repeat_fit_index=repeat_fit_index,
        primary_evaluation_index=primary_evaluation_index,
        repeat_evaluation_index=repeat_evaluation_index,
        primary_fit_run=official_fit_run,
        repeat_fit_run=repeat_fit_run,
        primary_evaluation_run=primary_evaluation_run,
        repeat_evaluation_run=repeat_evaluation_run,
    )
    _validate_retained_validations(
        files,
        intent_sha256=repeat.intent_sha256,
    )
    for source_path in (
        HEALTH_INTENT_FILE,
        HEALTH_MAIN_PROFILE_FILE,
        HEALTH_EDGE_PROFILE_FILE,
    ):
        if (
            files[_EVALUATION_SCIENCE_RELEASE_PATHS[source_path]]
            != files[_PRIMARY_FIT_RELEASE_PATHS[source_path]]
        ):
            raise HealthReleaseValidationError(
                "retained evaluation intent or profiles disagree with the official fit"
            )
    ordered_aggregates = validate_health_aggregate_structure(
        primary_fit_semantics.intent,
        primary_fit_semantics.main_profile,
        primary_fit_semantics.edge_profile,
        aggregates,
    )
    if aggregates != ordered_aggregates:
        raise HealthReleaseValidationError("retained aggregate matrix is not in canonical order")
    fit_reference = _load_canonical_model(
        files[_EVALUATION_SCIENCE_RELEASE_PATHS[HEALTH_FIT_REFERENCE_FILE]],
        validate=HealthFitReferenceV1.model_validate_json,
        label=HEALTH_FIT_REFERENCE_FILE,
    )
    if (
        fit_reference.fit_artifact_sha256 != repeat.official_fit_artifact_sha256
        or fit_reference.fit_run_sha256 != repeat.official_fit_run_sha256
        or fit_reference.intent_sha256 != repeat.intent_sha256
        or fit_reference.selected_candidate_index != fit_summary.selected_candidate_index
        or fit_reference.selected_self_threshold != fit_summary.selected_self_threshold
        or fit_reference.selected_cross_threshold != fit_summary.selected_cross_threshold
        or repeat.fit_reference_sha256 != sha256_digest(fit_reference)
    ):
        raise HealthReleaseValidationError("retained evaluation fit reference is invalid")
    validate_health_quantitative_claims(
        claims,
        aggregates=aggregates,
        fit_summary=fit_summary,
        repeat=repeat,
    )
    _validate_loaded_summary(
        summary,
        aggregates=aggregates,
        claims=claims,
        commitments=commitments,
        repeat=repeat,
        fit_summary=fit_summary,
        official_fit_run=official_fit_run,
    )
    if (
        index.intent_sha256 != summary.intent_sha256
        or index.official_fit_artifact_sha256 != summary.official_fit_artifact_sha256
        or index.evaluation_artifact_sha256 != summary.evaluation_artifact_sha256
    ):
        raise HealthReleaseValidationError("release index identity disagrees with summary")
    return LoadedHealthRelease(
        path=root,
        summary=summary,
        commitments=commitments,
        repeat=repeat,
        claims=claims,
        aggregates=aggregates,
        release_index=index,
        success=success,
        release_artifact_sha256=release_sha256,
    )


def load_health_release(path: Path) -> LoadedHealthRelease:
    """Strictly load one complete offline-verifiable curated M4 release."""

    try:
        return _load_health_release(path)
    except HealthReleaseValidationError:
        raise
    except (OSError, KeyError, StopIteration, ValueError, ValidationError) as error:
        raise HealthReleaseValidationError("invalid M4 health release") from error


def _cleanup_staging(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
) -> None:
    for name in HEALTH_RELEASE_ARTIFACT_PATHS:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=staging_fd)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _publish_health_release(
    prepared: _PreparedHealthRelease,
    destination: Path,
    *,
    source_root: Path | None,
    git_metadata_dirs: Sequence[Path] | None,
) -> LoadedHealthRelease:
    target = absolute_artifact_path(destination)
    metadata_dirs = (
        discover_git_metadata_dirs(source_root)
        if git_metadata_dirs is None
        else tuple(git_metadata_dirs)
    )
    reject_git_metadata_destination(target, metadata_dirs)
    if os.path.lexists(target):
        raise FileExistsError("health release destination already exists")
    parent = target.parent
    parent_fd = open_or_create_real_directory(parent)
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            parent,
            label="release destination parent",
        )
        reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
        if entry_exists_at(parent_fd, target.name):
            raise FileExistsError("health release destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        published = False
        try:
            for name in HEALTH_RELEASE_ARTIFACT_PATHS[:-1]:
                write_exclusive_file_at(staging_fd, name, prepared.files[name])
            for name in HEALTH_RELEASE_ARTIFACT_PATHS[:-1]:
                if (
                    read_file_at(
                        staging_fd,
                        name,
                        byte_cap=len(prepared.files[name]),
                    )
                    != prepared.files[name]
                ):
                    raise HealthReleaseValidationError("release staging verification failed")
            write_exclusive_file_at(
                staging_fd,
                HEALTH_RELEASE_SUCCESS_FILE,
                prepared.files[HEALTH_RELEASE_SUCCESS_FILE],
            )
            os.fsync(staging_fd)
            load_health_release(staging)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="release destination parent",
            )
            reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
            if entry_exists_at(parent_fd, target.name):
                raise FileExistsError("health release destination already exists")
            atomic_rename_directory_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            published = True
            os.fsync(parent_fd)
            return load_health_release(target)
        except BaseException:
            if not published:
                _cleanup_staging(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=staging_name,
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def write_health_release(
    request: HealthReleaseWriteRequest,
    destination: Path,
    *,
    source_root: Path | None = None,
    git_metadata_dirs: Sequence[Path] | None = None,
) -> LoadedHealthRelease:
    """Authenticate, curate, stage, and atomically publish one M4 release."""

    try:
        target = absolute_artifact_path(destination)
        if os.path.lexists(target):
            raise FileExistsError("health release destination already exists")
        for artifact_path in (
            request.official_fit_path,
            request.repeat_fit_path,
            request.primary_evaluation_path,
            request.repeat_evaluation_path,
        ):
            if _paths_overlap(target, absolute_artifact_path(artifact_path)):
                raise HealthReleaseValidationError(
                    "release destination must be disjoint from source artifacts"
                )
        prepared = _prepare_health_release(request)
        return _publish_health_release(
            prepared,
            target,
            source_root=source_root,
            git_metadata_dirs=git_metadata_dirs,
        )
    except FileExistsError:
        raise
    except HealthReleaseValidationError:
        raise
    except (ArtifactValidationError, OSError, ValueError, ValidationError) as error:
        raise HealthReleaseValidationError("M4 health release publication failed") from error
