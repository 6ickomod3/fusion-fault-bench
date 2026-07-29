"""Strict construction, publication, and loading of M4 health artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, FiniteFloat, ValidationError, model_validator

from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    absolute_artifact_path,
    assert_directory_descriptor_matches_path,
    atomic_rename_directory_no_replace_at,
    canonical_json_bytes,
    compute_run_record_digest,
    create_staging_directory_at,
    derive_run_id,
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
    HEALTH_MAX_ARTIFACT_BYTES,
    HEALTH_MAX_MEMBER_BYTES,
    HEALTH_MAX_RECORD_BYTES,
    HEALTH_PAYLOAD_INDEX_FILE,
    HEALTH_RUN_FILE,
    HEALTH_SEQUENCE_CONTRASTS_FILE,
    HEALTH_SEQUENCE_EVENTS_FILE,
    HEALTH_SEQUENCE_LOSSES_FILE,
    HEALTH_SUCCESS_FILE,
    HealthArtifactContract,
    HealthFitReferenceV1,
    HealthPayloadFileEntryV1,
    HealthPayloadIndexV1,
)
from fusion_fault_bench.contracts.health_result_v1 import (
    CROSS_THRESHOLDS,
    SELF_THRESHOLDS,
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
    HealthSequenceContrastV1,
    HealthSequenceEventV1,
    HealthSequenceLossV1,
    HealthThresholdCandidateV1,
    HealthValidationV1,
)
from fusion_fault_bench.contracts.health_v1 import (
    HEALTH_BENCHMARK_INTENT_ADAPTER,
    M4_HEALTH_INTENT_SHA256,
    HealthBenchmarkIntentV1,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EDGE_PROFILE_SHA256,
    MAIN_PROFILE_SHA256,
    PROCEDURAL_PROFILE_ADAPTER,
    EdgeProceduralProfile,
    MainProceduralProfile,
    ProceduralProfileV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RunRecordV1Alpha1
from fusion_fault_bench.health import HealthCalibration

_HEALTH_FIT_ARTIFACT_DOMAIN = b"fusion-fault-bench/health-fit-artifact/v1\x00"
_HEALTH_EVAL_ARTIFACT_DOMAIN = b"fusion-fault-bench/health-eval-artifact/v1\x00"
_READ_CHUNK_BYTES = 1024 * 1024
_ECDF_VALUES_PER_CHANNEL = 9200
_CANDIDATE_COUNT = len(SELF_THRESHOLDS) * len(CROSS_THRESHOLDS)
_MAX_EVALUATION_LOSS_RECORDS = 264_600
_MAX_EVALUATION_CONTRAST_RECORDS = 133_500
_MAX_EVALUATION_EVENT_RECORDS = 35_600
_MAX_EVALUATION_AGGREGATE_RECORDS = 20_000
_EXPECTED_EVALUATION_CONDITION_COUNT = 47

HEALTH_FIT_ARTIFACT_PATHS = (
    *HEALTH_FIT_INDEXED_PATHS,
    HEALTH_PAYLOAD_INDEX_FILE,
    HEALTH_RUN_FILE,
    HEALTH_SUCCESS_FILE,
)
HEALTH_EVAL_ARTIFACT_PATHS = (
    *HEALTH_EVAL_INDEXED_PATHS,
    HEALTH_PAYLOAD_INDEX_FILE,
    HEALTH_RUN_FILE,
    HEALTH_SUCCESS_FILE,
)

type HealthEcdfChannel = Literal[
    "camera_self_mean",
    "camera_self_maximum",
    "lidar_self_mean",
    "lidar_self_maximum",
    "camera_from_lidar_cross_mean",
    "camera_from_lidar_cross_maximum",
    "lidar_from_camera_cross_mean",
    "lidar_from_camera_cross_maximum",
]

HEALTH_ECDF_CHANNEL_ORDER: tuple[HealthEcdfChannel, ...] = (
    "camera_self_mean",
    "camera_self_maximum",
    "lidar_self_mean",
    "lidar_self_maximum",
    "camera_from_lidar_cross_mean",
    "camera_from_lidar_cross_maximum",
    "lidar_from_camera_cross_mean",
    "lidar_from_camera_cross_maximum",
)


class HealthEcdfChannelV1(ContractModel):
    """One exact sorted clean ECDF array."""

    channel: HealthEcdfChannel
    values: Annotated[
        tuple[FiniteFloat, ...],
        Field(min_length=_ECDF_VALUES_PER_CHANNEL, max_length=_ECDF_VALUES_PER_CHANNEL),
    ]

    @model_validator(mode="after")
    def require_sorted_values(self) -> HealthEcdfChannelV1:
        if any(
            self.values[index + 1] < self.values[index] for index in range(len(self.values) - 1)
        ):
            raise ValueError("health ECDF values must be sorted in nondecreasing order")
        return self


class HealthEcdfArraysV1(ContractModel):
    """The eight ordered clean ECDF arrays committed by a fit artifact."""

    schema_id: Literal["ffb.health-ecdf-arrays/v1"] = Field(alias="schema")
    channels: Annotated[
        tuple[HealthEcdfChannelV1, ...],
        Field(min_length=8, max_length=8),
    ]

    @model_validator(mode="after")
    def require_exact_channel_order(self) -> HealthEcdfArraysV1:
        if tuple(channel.channel for channel in self.channels) != HEALTH_ECDF_CHANNEL_ORDER:
            raise ValueError("health ECDF arrays do not use the exact channel order")
        return self


@dataclass(frozen=True, slots=True)
class HealthFitArtifactWriteRequest:
    """Already-built M4 fit evidence and completed run provenance."""

    intent: HealthBenchmarkIntentV1
    main_profile: MainProceduralProfile
    edge_profile: EdgeProceduralProfile
    calibration: HealthCalibration
    candidates: Sequence[HealthThresholdCandidateV1]
    summary: HealthFitSummaryV1
    validation: HealthValidationV1
    run: RunRecordV1Alpha1


@dataclass(frozen=True, slots=True)
class HealthEvaluationArtifactWriteRequest:
    """Already-built M4 evaluation evidence and completed run provenance."""

    intent: HealthBenchmarkIntentV1
    main_profile: MainProceduralProfile
    edge_profile: EdgeProceduralProfile
    fit_reference: HealthFitReferenceV1
    sequence_losses: Sequence[HealthSequenceLossV1]
    sequence_contrasts: Sequence[HealthSequenceContrastV1]
    sequence_events: Sequence[HealthSequenceEventV1]
    aggregates: Sequence[HealthAggregateMetricV1]
    validation: HealthValidationV1
    run: RunRecordV1Alpha1


@dataclass(frozen=True, slots=True, eq=False)
class LoadedHealthFitArtifact:
    """One strictly reloaded and cross-validated M4 fit artifact."""

    path: Path
    intent: HealthBenchmarkIntentV1
    main_profile: MainProceduralProfile
    edge_profile: EdgeProceduralProfile
    calibration: HealthCalibration
    candidates: tuple[HealthThresholdCandidateV1, ...]
    summary: HealthFitSummaryV1
    validation: HealthValidationV1
    payload_index: HealthPayloadIndexV1
    run: RunRecordV1Alpha1
    success: SuccessMarkerV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedHealthEvaluationArtifact:
    """One strictly reloaded and fit-bound M4 evaluation artifact."""

    path: Path
    intent: HealthBenchmarkIntentV1
    main_profile: MainProceduralProfile
    edge_profile: EdgeProceduralProfile
    fit_reference: HealthFitReferenceV1
    sequence_rows_materialized: bool
    sequence_losses: tuple[HealthSequenceLossV1, ...]
    sequence_contrasts: tuple[HealthSequenceContrastV1, ...]
    sequence_events: tuple[HealthSequenceEventV1, ...]
    aggregates: tuple[HealthAggregateMetricV1, ...]
    validation: HealthValidationV1
    payload_index: HealthPayloadIndexV1
    run: RunRecordV1Alpha1
    success: SuccessMarkerV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedHealthArtifact:
    files: Mapping[str, bytes]
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CanonicalNdjsonMember:
    records: Sequence[BaseModel]
    byte_length: int
    sha256: str
    record_count: int


@dataclass(frozen=True, slots=True)
class _StreamMemberMetadata:
    byte_length: int
    sha256: str
    record_count: int


class _OpenCanonicalNdjsonStream:
    """One exclusive staged NDJSON member with incremental commitments."""

    def __init__(self, directory_fd: int, name: str) -> None:
        self.name = name
        self._descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        self._digest = hashlib.sha256()
        self._byte_length = 0
        self._record_count = 0
        self._previous_key: tuple[str, ...] | None = None

    def append[ModelT: BaseModel](
        self,
        records: Sequence[ModelT],
        *,
        key: Callable[[ModelT], tuple[str, ...]],
    ) -> None:
        for record in records:
            logical_key = key(record)
            if self._previous_key is not None and logical_key <= self._previous_key:
                raise ArtifactValidationError("health evaluation stream is not in canonical order")
            line = canonical_json_bytes(record)
            if len(line) > HEALTH_MAX_RECORD_BYTES:
                raise ArtifactValidationError(f"{self.name} line exceeds the 1 MiB cap")
            self._byte_length += len(line)
            if self._byte_length > HEALTH_MAX_MEMBER_BYTES:
                raise ArtifactValidationError(f"{self.name} exceeds the health member cap")
            self._digest.update(line)
            remaining = memoryview(line)
            while remaining:
                written = os.write(self._descriptor, remaining)
                if written == 0:
                    raise OSError("short write while streaming health evaluation")
                remaining = remaining[written:]
            self._record_count += 1
            self._previous_key = logical_key

    def finish(self) -> _StreamMemberMetadata:
        if self._descriptor < 0:
            raise ArtifactValidationError("health evaluation stream is already closed")
        if self._record_count == 0:
            raise ArtifactValidationError(f"{self.name} must contain records")
        os.fsync(self._descriptor)
        file_stat = os.fstat(self._descriptor)
        os.close(self._descriptor)
        self._descriptor = -1
        if file_stat.st_size != self._byte_length:
            raise ArtifactValidationError("streamed health member has the wrong size")
        return _StreamMemberMetadata(
            byte_length=self._byte_length,
            sha256=self._digest.hexdigest(),
            record_count=self._record_count,
        )

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1


@dataclass(frozen=True, slots=True)
class _PreparedStreamingHealthEvaluation:
    small_files: Mapping[str, bytes]
    ndjson_members: Mapping[str, _CanonicalNdjsonMember]
    paths: tuple[str, ...]
    sequence_losses: tuple[HealthSequenceLossV1, ...]
    sequence_contrasts: tuple[HealthSequenceContrastV1, ...]
    sequence_events: tuple[HealthSequenceEventV1, ...]
    aggregates: tuple[HealthAggregateMetricV1, ...]


@dataclass(frozen=True, slots=True)
class _TreeSnapshot:
    root_stat: os.stat_result
    entries: Mapping[str, os.stat_result]


def compute_health_artifact_digest(
    payload_index_file_bytes: bytes,
    *,
    artifact_contract: HealthArtifactContract,
) -> str:
    """Hash exact canonical payload-index bytes in the fit or evaluation domain."""

    if artifact_contract == HEALTH_FIT_ARTIFACT_CONTRACT:
        domain = _HEALTH_FIT_ARTIFACT_DOMAIN
    elif artifact_contract == HEALTH_EVAL_ARTIFACT_CONTRACT:
        domain = _HEALTH_EVAL_ARTIFACT_DOMAIN
    else:
        raise ArtifactValidationError("unknown health artifact digest domain")
    preimage = b"".join(
        (
            domain,
            len(payload_index_file_bytes).to_bytes(8, "big"),
            payload_index_file_bytes,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def canonical_health_ndjson_bytes(records: Sequence[BaseModel]) -> bytes:
    """Serialize one nonempty sequence of canonical, bounded health records."""

    output = bytearray()
    for index, record in enumerate(records):
        line = canonical_json_bytes(record)
        if len(line) > HEALTH_MAX_RECORD_BYTES:
            raise ArtifactValidationError(
                f"health NDJSON record {index} exceeds the 1 MiB line cap"
            )
        if len(output) + len(line) > HEALTH_MAX_MEMBER_BYTES:
            raise ArtifactValidationError("health NDJSON exceeds the member cap")
        output.extend(line)
    if not output:
        raise ArtifactValidationError("health NDJSON must contain at least one record")
    return bytes(output)


def _canonical_health_ndjson_member(
    records: Sequence[BaseModel],
) -> _CanonicalNdjsonMember:
    """Measure canonical NDJSON without retaining a member-sized byte buffer."""

    digest = hashlib.sha256()
    byte_length = 0
    for index, record in enumerate(records):
        line = canonical_json_bytes(record)
        if len(line) > HEALTH_MAX_RECORD_BYTES:
            raise ArtifactValidationError(
                f"health NDJSON record {index} exceeds the 1 MiB line cap"
            )
        byte_length += len(line)
        if byte_length > HEALTH_MAX_MEMBER_BYTES:
            raise ArtifactValidationError("health NDJSON exceeds the member cap")
        digest.update(line)
    if not records:
        raise ArtifactValidationError("health NDJSON must contain at least one record")
    return _CanonicalNdjsonMember(
        records=records,
        byte_length=byte_length,
        sha256=digest.hexdigest(),
        record_count=len(records),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_profiles(
    intent: HealthBenchmarkIntentV1,
    main_profile: ProceduralProfileV1,
    edge_profile: ProceduralProfileV1,
) -> tuple[str, str, str]:
    intent_sha256 = sha256_digest(intent)
    main_sha256 = sha256_digest(main_profile)
    edge_sha256 = sha256_digest(edge_profile)
    if intent_sha256 != M4_HEALTH_INTENT_SHA256:
        raise ArtifactValidationError("health intent digest is not preregistered")
    if not isinstance(main_profile, MainProceduralProfile):
        raise ArtifactValidationError("health main profile has the wrong profile type")
    if not isinstance(edge_profile, EdgeProceduralProfile):
        raise ArtifactValidationError("health edge profile has the wrong profile type")
    source = intent.source_population
    if (
        main_profile.profile_id != source.profile_id
        or main_sha256 != source.profile_sha256
        or main_sha256 != MAIN_PROFILE_SHA256
    ):
        raise ArtifactValidationError("health main profile identity is invalid")
    if (
        edge_profile.profile_id != source.edge_profile_id
        or edge_sha256 != source.edge_profile_sha256
        or edge_sha256 != EDGE_PROFILE_SHA256
    ):
        raise ArtifactValidationError("health edge profile identity is invalid")
    return intent_sha256, main_sha256, edge_sha256


def _expected_run_id(
    intent_sha256: str,
    run: RunRecordV1Alpha1,
    *,
    artifact_contract: HealthArtifactContract,
) -> str:
    return derive_run_id(
        manifest_sha256=intent_sha256,
        git_revision=run.git_revision,
        lockfile_sha256=run.lockfile_sha256,
        package_version=run.package_version,
        artifact_contract=artifact_contract,
    )


def _validate_run(
    intent_sha256: str,
    run: RunRecordV1Alpha1,
    *,
    artifact_contract: HealthArtifactContract,
    artifact_sha256: str | None = None,
) -> None:
    if run.manifest_sha256 != intent_sha256:
        raise ArtifactValidationError("health run intent identity is invalid")
    if run.run_id != _expected_run_id(
        intent_sha256,
        run,
        artifact_contract=artifact_contract,
    ):
        raise ArtifactValidationError("health run_id is invalid")
    if run.source_dirty or run.status != "succeeded":
        raise ArtifactValidationError("health artifact requires a clean successful run")
    if artifact_sha256 is not None and run.artifact_sha256 != artifact_sha256:
        raise ArtifactValidationError("health run artifact identity is invalid")


def _calibration_envelope(calibration: HealthCalibration) -> HealthEcdfArraysV1:
    channels: list[HealthEcdfChannelV1] = []
    for name in HEALTH_ECDF_CHANNEL_ORDER:
        values = tuple(float(value) for value in getattr(calibration, name))
        channels.append(HealthEcdfChannelV1(channel=name, values=values))
    return HealthEcdfArraysV1(
        schema="ffb.health-ecdf-arrays/v1",
        channels=tuple(channels),
    )


def _health_calibration(envelope: HealthEcdfArraysV1) -> HealthCalibration:
    channels = envelope.channels
    return HealthCalibration(
        camera_self_mean=np.asarray(channels[0].values, dtype=np.float64),
        camera_self_maximum=np.asarray(channels[1].values, dtype=np.float64),
        lidar_self_mean=np.asarray(channels[2].values, dtype=np.float64),
        lidar_self_maximum=np.asarray(channels[3].values, dtype=np.float64),
        camera_from_lidar_cross_mean=np.asarray(channels[4].values, dtype=np.float64),
        camera_from_lidar_cross_maximum=np.asarray(
            channels[5].values,
            dtype=np.float64,
        ),
        lidar_from_camera_cross_mean=np.asarray(channels[6].values, dtype=np.float64),
        lidar_from_camera_cross_maximum=np.asarray(
            channels[7].values,
            dtype=np.float64,
        ),
    )


def _ordered_candidates(
    candidates: Sequence[HealthThresholdCandidateV1],
) -> tuple[HealthThresholdCandidateV1, ...]:
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.candidate_index))
    if len(ordered) != _CANDIDATE_COUNT or tuple(
        candidate.candidate_index for candidate in ordered
    ) != tuple(range(_CANDIDATE_COUNT)):
        raise ArtifactValidationError("health fit requires the exact 36 candidate records")
    return ordered


def _selected_candidate(
    intent: HealthBenchmarkIntentV1,
    candidates: Sequence[HealthThresholdCandidateV1],
) -> HealthThresholdCandidateV1:
    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    if not feasible:
        raise ArtifactValidationError("health fit has no feasible threshold candidate")
    minimum_regret = min(candidate.validation_regret_m2 for candidate in feasible)
    tied = tuple(
        candidate
        for candidate in feasible
        if candidate.validation_regret_m2
        <= minimum_regret + intent.threshold_selection.tie_tolerance_m2
    )
    return min(
        tied,
        key=lambda candidate: (
            candidate.false_alert_episode_starts_per_sequence,
            candidate.mean_clean_regression_m2,
            -candidate.self_threshold,
            -candidate.cross_threshold,
        ),
    )


def _validate_validation(
    validation: HealthValidationV1,
    *,
    intent_sha256: str,
    label: str,
) -> None:
    if validation.intent_sha256 != intent_sha256:
        raise ArtifactValidationError(f"{label} validation intent identity is invalid")
    if not validation.all_checks_passed:
        raise ArtifactValidationError(f"{label} validation did not pass every release gate")


def _require_exact_validation_checks(
    validation: HealthValidationV1,
    *,
    expected: tuple[tuple[str, object, object], ...],
    label: str,
) -> None:
    observed = tuple(
        (check.check_id, check.observed, check.expected) for check in validation.checks
    )
    if observed != expected or not all(check.passed for check in validation.checks):
        raise ArtifactValidationError(f"{label} validation evidence is not the frozen conjunction")


def _require_exact_fit_validation(
    validation: HealthValidationV1,
    *,
    intent: HealthBenchmarkIntentV1,
    intent_sha256: str,
    main_profile_sha256: str,
    edge_profile_sha256: str,
) -> None:
    expected = (
        ("intent-digest", intent_sha256, M4_HEALTH_INTENT_SHA256),
        (
            "main-profile-digest",
            main_profile_sha256,
            intent.source_population.profile_sha256,
        ),
        (
            "edge-profile-digest",
            edge_profile_sha256,
            intent.source_population.edge_profile_sha256,
        ),
        ("train-sequence-count", 200, 200),
        ("validation-sequence-count", 200, 200),
        ("train-validation-sequence-id-overlap", 0, 0),
        ("validation-value-case-count", 33, 33),
        ("test-value-case-count", 47, 47),
        ("value-case-id-uniqueness", 80, 80),
        ("selection-value-case-count", 20, 20),
        ("identity-outside-active-event", 4_000, 4_000),
        ("ecdf-channel-count", 8, 8),
        ("ecdf-values-per-channel", 9_200, 9_200),
        ("threshold-candidate-count", 36, 36),
        ("threshold-candidate-order", True, True),
        ("metadata-leakage-boundary", True, True),
        ("future-prefix-causality", True, True),
        ("current-preupdate-causality", True, True),
        ("independent-modality-histories", True, True),
        ("frame-oracle-comparison-count", 144_000, 144_000),
        ("frame-oracle-dominance", 0, 0),
        (
            "candidate-frame-evaluation-cap",
            7_257_600,
            intent.resource_caps.candidate_frame_evaluations_max,
        ),
        (
            "bootstrap-cell-cap",
            400_000,
            intent.resource_caps.bootstrap_cells_max,
        ),
        ("scientific-feature-trace-count", 4_400, 4_400),
        ("selected-candidate-feasible", True, True),
    )
    _require_exact_validation_checks(validation, expected=expected, label="health fit")


def _require_exact_evaluation_validation(
    validation: HealthValidationV1,
) -> None:
    expected = (
        ("intent-identity", M4_HEALTH_INTENT_SHA256, M4_HEALTH_INTENT_SHA256),
        ("condition-and-population-counts", 8_900, 8_900),
        ("condition-value-count", 47, 47),
        ("sequence-loss-row-count", 264_600, 264_600),
        ("sequence-event-row-count", 35_600, 35_600),
        ("sequence-contrast-row-count", 133_500, 133_500),
        ("oracle-frame-action-dominance", True, True),
        ("common-mode-hindsight-boundary", True, True),
        ("cold-start-schedule-windows", True, True),
        ("held-out-yaw-present", True, True),
        ("required-negative-controls-present", True, True),
        ("feature-computed-once-per-sequence-condition", True, True),
        ("test-fit-apply-only", True, True),
    )
    _require_exact_validation_checks(
        validation,
        expected=expected,
        label="health evaluation",
    )


_HEALTH_WINDOWS = ("score", "event", "recovery")
_HEALTH_POLICIES = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
)
_HEALTH_POLICY_METHODS = (*_HEALTH_POLICIES, "combined-health-gate-abstain")


def _expected_case_sequence_ids(
    *,
    population: str,
    main_profile: MainProceduralProfile,
    edge_profile: EdgeProceduralProfile,
) -> tuple[str, ...]:
    if population == "main-test":
        profile_id = main_profile.profile_id
        count = 200
    elif population == "edge-test":
        profile_id = edge_profile.profile_id
        count = 100
    else:
        raise ArtifactValidationError("health evaluation case has an invalid population")
    return tuple(f"procedural:{profile_id}:test:{index:06d}" for index in range(count))


def _expected_aggregate_keys(
    *,
    condition_id: str,
    fault_family: str,
    fault_target: str,
    methods: tuple[str, ...],
) -> tuple[
    set[tuple[str, str | None, str, str]],
    set[tuple[str, str | None, str, str]],
]:
    required: set[tuple[str, str | None, str, str]] = set()
    optional: set[tuple[str, str | None, str, str]] = set()
    for method in methods:
        for window in _HEALTH_WINDOWS:
            for metric in ("matched-center-mse", "coverage", "undefined-output-rate"):
                required.add((condition_id, method, metric, window))

    common_mode = fault_family == "common-mode-position-bias"
    for policy in _HEALTH_POLICY_METHODS:
        for window in _HEALTH_WINDOWS:
            required.add((condition_id, policy, "policy-gain-vs-fixed", window))
            if not common_mode:
                required.add((condition_id, policy, "gap-vs-fault-target-drop", window))
                required.add((condition_id, policy, "gap-vs-frame-oracle", window))
                optional.add(
                    (
                        condition_id,
                        policy,
                        "frame-oracle-recoverable-loss-fraction",
                        window,
                    )
                )

    targetless = fault_target in {"none", "both"}
    for policy in _HEALTH_POLICIES:
        required.add((condition_id, policy, "detection-fraction", "event"))
        outcomes = (
            ("ambiguous", "missed")
            if targetless
            else ("correct", "ambiguous", "wrong-sensor", "missed")
        )
        for outcome in outcomes:
            required.add(
                (
                    condition_id,
                    policy,
                    f"event-outcome-{outcome}-fraction",
                    "event",
                )
            )
        if common_mode:
            for label in ("camera-fault", "lidar-fault", "ambiguous"):
                required.add(
                    (
                        condition_id,
                        policy,
                        f"first-latch-label-{label}-fraction",
                        "event",
                    )
                )
        if not targetless:
            required.add((condition_id, policy, "attribution-fraction", "event"))
            required.add((condition_id, policy, "attribution-latency", "event"))
        required.update(
            {
                (condition_id, policy, "early-clear-fraction", "event"),
                (condition_id, policy, "recovery-denominator-fraction", "event"),
                (condition_id, policy, "recovery-fraction", "recovery"),
                (condition_id, policy, "detection-latency", "event"),
                (condition_id, policy, "recovery-latency", "recovery"),
                (condition_id, policy, "false-alert-episode-starts", "score"),
                (condition_id, policy, "latch-episode-starts", "event"),
            }
        )
        for state in ("healthy", "camera-fault", "lidar-fault", "ambiguous"):
            required.add(
                (
                    condition_id,
                    policy,
                    f"final-active-state-{state}-fraction",
                    "event",
                )
            )
        for metric in (
            "state-healthy-occupancy",
            "state-camera-fault-occupancy",
            "state-lidar-fault-occupancy",
            "state-ambiguous-occupancy",
            "action-camera-occupancy",
            "action-lidar-occupancy",
            "action-fixed-occupancy",
            "action-undefined-occupancy",
        ):
            required.add((condition_id, policy, metric, "event"))
        if fault_family == "dropout":
            required.add(
                (
                    condition_id,
                    policy,
                    "detection-among-realized-dropout-fraction",
                    "event",
                )
            )
            required.add(
                (
                    condition_id,
                    policy,
                    "detection-minus-first-missing",
                    "event",
                )
            )
    if fault_family == "dropout":
        required.add((condition_id, None, "realized-dropout-fraction", "event"))
        required.add(
            (
                condition_id,
                None,
                "first-missing-frame-minus-event-start",
                "event",
            )
        )
    return required, optional


def _require_health_aggregate_structure_for_cases(
    *,
    intent: HealthBenchmarkIntentV1,
    main_profile: MainProceduralProfile,
    edge_profile: EdgeProceduralProfile,
    aggregates: Sequence[HealthAggregateMetricV1],
    cases: Sequence[Any],
) -> None:
    case_by_id = {case.condition_id: case for case in cases}
    sequence_counts = {
        case.condition_id: len(
            _expected_case_sequence_ids(
                population=case.population,
                main_profile=main_profile,
                edge_profile=edge_profile,
            )
        )
        for case in cases
    }
    methods_by_condition = {
        case.condition_id: tuple(
            method
            for method in intent.methods
            if not (
                case.fault.family == "common-mode-position-bias"
                and method
                in {
                    "fault-target-drop-policy",
                    "frame-action-performance-oracle",
                }
            )
        )
        for case in cases
    }
    required_aggregate_keys: set[tuple[str, str | None, str, str]] = set()
    optional_aggregate_keys: set[tuple[str, str | None, str, str]] = set()
    for case in cases:
        required, optional = _expected_aggregate_keys(
            condition_id=case.condition_id,
            fault_family=case.fault.family,
            fault_target=case.fault.target,
            methods=methods_by_condition[case.condition_id],
        )
        required_aggregate_keys.update(required)
        optional_aggregate_keys.update(optional)
    observed_aggregate_keys: set[tuple[str, str | None, str, str]] = set()
    for row in aggregates:
        if row.window is None:
            raise ArtifactValidationError("health aggregate lies outside the frozen matrix")
        key = (row.condition_id, row.method, row.metric_name, row.window)
        if row.condition_id not in case_by_id or row.sequence_count != sequence_counts.get(
            row.condition_id
        ):
            raise ArtifactValidationError(
                "health aggregate lies outside the frozen matrix or uses the wrong denominator"
            )
        if key in observed_aggregate_keys:
            raise ArtifactValidationError("health aggregate rows contain duplicate keys")
        observed_aggregate_keys.add(key)
    if not required_aggregate_keys.issubset(observed_aggregate_keys) or not (
        observed_aggregate_keys <= required_aggregate_keys | optional_aggregate_keys
    ):
        raise ArtifactValidationError("health aggregate rows do not match the frozen metric matrix")


def validate_health_aggregate_structure(
    intent: HealthBenchmarkIntentV1,
    main_profile: MainProceduralProfile,
    edge_profile: EdgeProceduralProfile,
    aggregates: Sequence[HealthAggregateMetricV1],
) -> tuple[HealthAggregateMetricV1, ...]:
    """Validate the complete aggregate matrix without omitted sequence values."""

    _validate_profiles(intent, main_profile, edge_profile)
    from fusion_fault_bench.health_benchmark import expand_test_cases

    cases = tuple(expand_test_cases(intent))
    if len({case.condition_id for case in cases}) != _EXPECTED_EVALUATION_CONDITION_COUNT:
        raise ArtifactValidationError("health evaluation matrix identity is invalid")
    ordered = _ordered_unique(
        aggregates,
        key=_aggregate_key,
        label="health aggregate metrics",
    )
    _require_health_aggregate_structure_for_cases(
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        aggregates=ordered,
        cases=cases,
    )
    return ordered


def _require_exact_evaluation_rows(
    *,
    intent: HealthBenchmarkIntentV1,
    main_profile: MainProceduralProfile,
    edge_profile: EdgeProceduralProfile,
    losses: tuple[HealthSequenceLossV1, ...],
    contrasts: tuple[HealthSequenceContrastV1, ...],
    events: tuple[HealthSequenceEventV1, ...],
    aggregates: tuple[HealthAggregateMetricV1, ...],
    cases_override: Sequence[Any] | None = None,
) -> None:
    # Keep one canonical value-level expansion without placing publication in
    # the fitting module's import graph.
    from fusion_fault_bench.health_benchmark import expand_test_cases

    cases = expand_test_cases(intent) if cases_override is None else tuple(cases_override)
    case_by_id = {case.condition_id: case for case in cases}
    if cases_override is None and len(case_by_id) != 47:
        raise ArtifactValidationError("health evaluation matrix identity is invalid")
    sequence_ids_by_condition = {
        case.condition_id: frozenset(
            _expected_case_sequence_ids(
                population=case.population,
                main_profile=main_profile,
                edge_profile=edge_profile,
            )
        )
        for case in cases
    }
    methods_by_condition = {
        case.condition_id: tuple(
            method
            for method in intent.methods
            if not (
                case.fault.family == "common-mode-position-bias"
                and method
                in {
                    "fault-target-drop-policy",
                    "frame-action-performance-oracle",
                }
            )
        )
        for case in cases
    }

    loss_counts: Counter[tuple[str, str, str]] = Counter()
    loss_by_key: dict[tuple[str, str, str, str], HealthSequenceLossV1] = {}
    for row in losses:
        case = case_by_id.get(row.condition_id)
        if (
            case is None
            or row.sequence_id not in sequence_ids_by_condition[row.condition_id]
            or row.method not in methods_by_condition[row.condition_id]
            or row.window not in _HEALTH_WINDOWS
        ):
            raise ArtifactValidationError("health sequence loss lies outside the frozen matrix")
        if case.population == "edge-test":
            expected_eligible = {"score": 184, "event": 96, "recovery": 48}[row.window]
        elif case.fault.schedule == "cold_start":
            expected_eligible = {"score": 288, "event": 144, "recovery": 144}[row.window]
        else:
            expected_eligible = {"score": 276, "event": 144, "recovery": 72}[row.window]
        if row.eligible_object_frame_count != expected_eligible:
            raise ArtifactValidationError("health sequence loss uses a nonfrozen denominator")
        key = (row.condition_id, row.sequence_id, row.method, row.window)
        if key in loss_by_key:
            raise ArtifactValidationError("health sequence losses contain duplicate keys")
        loss_by_key[key] = row
        loss_counts[(row.condition_id, row.method, row.window)] += 1
    for case in cases:
        expected_count = len(sequence_ids_by_condition[case.condition_id])
        for method in methods_by_condition[case.condition_id]:
            for window in _HEALTH_WINDOWS:
                if loss_counts[(case.condition_id, method, window)] != expected_count:
                    raise ArtifactValidationError(
                        "health sequence loss rows do not cover the frozen matrix"
                    )
    for case in cases:
        methods = methods_by_condition[case.condition_id]
        for sequence_id in sequence_ids_by_condition[case.condition_id]:
            for window in _HEALTH_WINDOWS:
                rows = {
                    method: loss_by_key[(case.condition_id, sequence_id, method, window)]
                    for method in methods
                }
                eligible = rows["fixed-fusion"].eligible_object_frame_count
                camera_count = rows["camera-only"].valid_object_frame_count
                lidar_count = rows["lidar-only"].valid_object_frame_count
                fixed_count = rows["fixed-fusion"].valid_object_frame_count
                if any(
                    rows[policy].valid_object_frame_count != eligible for policy in _HEALTH_POLICIES
                ):
                    raise ArtifactValidationError(
                        "nonabstaining policy support must cover exact eligibility"
                    )
                if case.fault.family == "dropout":
                    target_method = f"{case.fault.target}-only"
                    healthy_method = (
                        "lidar-only" if case.fault.target == "camera" else "camera-only"
                    )
                    if (
                        rows[healthy_method].valid_object_frame_count != eligible
                        or rows[target_method].valid_object_frame_count != fixed_count
                        or rows["fault-target-drop-policy"].valid_object_frame_count != eligible
                        or rows["frame-action-performance-oracle"].valid_object_frame_count
                        != eligible
                    ):
                        raise ArtifactValidationError(
                            "dropout method support violates the frozen availability table"
                        )
                else:
                    full_support_methods = {"camera-only", "lidar-only", "fixed-fusion"}
                    if any(
                        rows[method].valid_object_frame_count != eligible
                        for method in full_support_methods
                    ):
                        raise ArtifactValidationError(
                            "non-dropout method support must cover exact eligibility"
                        )
                    if case.fault.family != "common-mode-position-bias" and any(
                        rows[method].valid_object_frame_count != eligible
                        for method in {
                            "fault-target-drop-policy",
                            "frame-action-performance-oracle",
                        }
                    ):
                        raise ArtifactValidationError(
                            "applicable hindsight support must cover exact eligibility"
                        )
                if case.fault.family == "dropout" and case.fault.value == 1.0:
                    expected_full_dropout_support = {
                        "score": 132,
                        "event": 0,
                        "recovery": 72,
                    }[window]
                    target_method = f"{case.fault.target}-only"
                    if (
                        fixed_count != expected_full_dropout_support
                        or rows[target_method].valid_object_frame_count
                        != expected_full_dropout_support
                    ):
                        raise ArtifactValidationError(
                            "full-dropout fixed and target support violates the frozen schedule"
                        )
                if fixed_count > min(camera_count, lidar_count):
                    raise ArtifactValidationError("fixed support exceeds unimodal availability")
                if "frame-action-performance-oracle" in rows:
                    oracle_count = rows["frame-action-performance-oracle"].valid_object_frame_count
                    if oracle_count != camera_count + lidar_count - fixed_count:
                        raise ArtifactValidationError(
                            "oracle support disagrees with modality availability"
                        )
                    for policy in _HEALTH_POLICIES:
                        policy_count = rows[policy].valid_object_frame_count
                        if not fixed_count <= policy_count <= oracle_count:
                            raise ArtifactValidationError(
                                "deployable policy support violates availability bounds"
                            )
                    if (
                        rows["combined-health-gate-abstain"].valid_object_frame_count
                        > rows["combined-health-gate"].valid_object_frame_count
                    ):
                        raise ArtifactValidationError(
                            "abstaining support exceeds its nonabstaining policy"
                        )
                if case.fault.target == "none":
                    target_drop = rows["fault-target-drop-policy"]
                    fixed = rows["fixed-fusion"]
                    if (
                        target_drop.valid_object_frame_count != fixed.valid_object_frame_count
                        or not _loss_sum_close(target_drop.loss_sum_m2, fixed.loss_sum_m2)
                    ):
                        raise ArtifactValidationError(
                            "targetless hindsight policy must equal fixed fusion"
                        )

    contrast_counts: Counter[tuple[str, str, str]] = Counter()
    fixed_supports: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    oracle_supports: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in contrasts:
        case = case_by_id.get(row.condition_id)
        if (
            case is None
            or row.sequence_id not in sequence_ids_by_condition[row.condition_id]
            or row.policy not in _HEALTH_POLICY_METHODS
            or row.window not in _HEALTH_WINDOWS
        ):
            raise ArtifactValidationError("health sequence contrast lies outside the frozen matrix")
        common_mode = case.fault.family == "common-mode-position-bias"
        if common_mode and (row.target_drop_applicable or row.frame_oracle_applicable):
            raise ArtifactValidationError("common-mode contrast carries hindsight statistics")
        if not common_mode and (not row.target_drop_applicable or not row.frame_oracle_applicable):
            raise ArtifactValidationError("health contrast omits applicable hindsight statistics")

        fixed_loss = loss_by_key[(row.condition_id, row.sequence_id, "fixed-fusion", row.window)]
        policy_loss = loss_by_key[(row.condition_id, row.sequence_id, row.policy, row.window)]
        if row.fixed_policy_common_count > min(
            fixed_loss.valid_object_frame_count,
            policy_loss.valid_object_frame_count,
        ):
            raise ArtifactValidationError("fixed-policy contrast support is impossible")
        if (
            row.fixed_on_common_loss_sum_m2 > fixed_loss.loss_sum_m2 + 1e-12
            or row.policy_on_fixed_common_loss_sum_m2 > policy_loss.loss_sum_m2 + 1e-12
        ):
            raise ArtifactValidationError("fixed-policy common loss exceeds full support")
        fixed_supports[(row.condition_id, row.sequence_id, row.window)].add(
            row.fixed_support_sha256
        )
        known_nested_fixed_policy = not (
            case.fault.family == "dropout" and row.policy == "combined-health-gate-abstain"
        )
        if known_nested_fixed_policy:
            expected_common_count = min(
                fixed_loss.valid_object_frame_count,
                policy_loss.valid_object_frame_count,
            )
            if row.fixed_policy_common_count != expected_common_count:
                raise ArtifactValidationError(
                    "fixed-policy common support violates known support nesting"
                )
            if (
                fixed_loss.valid_object_frame_count <= policy_loss.valid_object_frame_count
                and not _loss_sum_close(
                    row.fixed_on_common_loss_sum_m2,
                    fixed_loss.loss_sum_m2,
                )
            ) or (
                policy_loss.valid_object_frame_count <= fixed_loss.valid_object_frame_count
                and not _loss_sum_close(
                    row.policy_on_fixed_common_loss_sum_m2,
                    policy_loss.loss_sum_m2,
                )
            ):
                raise ArtifactValidationError(
                    "fixed-policy common loss violates known support nesting"
                )
        fixed_policy_full_common = (
            row.fixed_policy_common_count == fixed_loss.valid_object_frame_count
            and row.fixed_policy_common_count == policy_loss.valid_object_frame_count
        )
        if row.fixed_support_sha256 == row.policy_support_sha256 and (
            not fixed_policy_full_common
            or not _loss_sum_close(
                row.fixed_on_common_loss_sum_m2,
                fixed_loss.loss_sum_m2,
            )
            or not _loss_sum_close(
                row.policy_on_fixed_common_loss_sum_m2,
                policy_loss.loss_sum_m2,
            )
        ):
            raise ArtifactValidationError(
                "equal fixed-policy support commitments require full retained support"
            )
        if fixed_policy_full_common and (
            row.fixed_support_sha256 != row.policy_support_sha256
            or not _loss_sum_close(
                row.fixed_on_common_loss_sum_m2,
                fixed_loss.loss_sum_m2,
            )
            or not _loss_sum_close(
                row.policy_on_fixed_common_loss_sum_m2,
                policy_loss.loss_sum_m2,
            )
        ):
            raise ArtifactValidationError(
                "full fixed-policy support disagrees with its commitments or losses"
            )

        if row.target_drop_applicable:
            assert row.policy_target_drop_common_count is not None
            assert row.policy_on_target_common_loss_sum_m2 is not None
            assert row.target_drop_on_common_loss_sum_m2 is not None
            target_loss = loss_by_key[
                (
                    row.condition_id,
                    row.sequence_id,
                    "fault-target-drop-policy",
                    row.window,
                )
            ]
            if row.policy_target_drop_common_count > min(
                policy_loss.valid_object_frame_count,
                target_loss.valid_object_frame_count,
            ):
                raise ArtifactValidationError("policy-target-drop support is impossible")
            if (
                row.policy_on_target_common_loss_sum_m2 > policy_loss.loss_sum_m2 + 1e-12
                or row.target_drop_on_common_loss_sum_m2 > target_loss.loss_sum_m2 + 1e-12
            ):
                raise ArtifactValidationError("policy-target-drop common loss exceeds full support")
            if (
                row.policy_target_drop_common_count != policy_loss.valid_object_frame_count
                or not _loss_sum_close(
                    row.policy_on_target_common_loss_sum_m2,
                    policy_loss.loss_sum_m2,
                )
            ):
                raise ArtifactValidationError(
                    "policy-target-drop contrast violates known support nesting"
                )
            if (
                row.policy_target_drop_common_count
                == policy_loss.valid_object_frame_count
                == target_loss.valid_object_frame_count
                and (
                    not _loss_sum_close(
                        row.policy_on_target_common_loss_sum_m2,
                        policy_loss.loss_sum_m2,
                    )
                    or not _loss_sum_close(
                        row.target_drop_on_common_loss_sum_m2,
                        target_loss.loss_sum_m2,
                    )
                )
            ):
                raise ArtifactValidationError(
                    "full policy-target-drop support disagrees with retained losses"
                )

        if row.frame_oracle_applicable:
            assert row.policy_frame_oracle_common_count is not None
            assert row.policy_on_oracle_common_loss_sum_m2 is not None
            assert row.frame_oracle_on_common_loss_sum_m2 is not None
            assert row.frame_oracle_support_sha256 is not None
            oracle_loss = loss_by_key[
                (
                    row.condition_id,
                    row.sequence_id,
                    "frame-action-performance-oracle",
                    row.window,
                )
            ]
            if row.policy_frame_oracle_common_count > min(
                policy_loss.valid_object_frame_count,
                oracle_loss.valid_object_frame_count,
            ):
                raise ArtifactValidationError("policy-frame-oracle support is impossible")
            if (
                row.policy_on_oracle_common_loss_sum_m2 > policy_loss.loss_sum_m2 + 1e-12
                or row.frame_oracle_on_common_loss_sum_m2 > oracle_loss.loss_sum_m2 + 1e-12
            ):
                raise ArtifactValidationError(
                    "policy-frame-oracle common loss exceeds full support"
                )
            if (
                row.policy_frame_oracle_common_count != policy_loss.valid_object_frame_count
                or not _loss_sum_close(
                    row.policy_on_oracle_common_loss_sum_m2,
                    policy_loss.loss_sum_m2,
                )
            ):
                raise ArtifactValidationError(
                    "policy-frame-oracle contrast violates known support nesting"
                )
            oracle_supports[(row.condition_id, row.sequence_id, row.window)].add(
                row.frame_oracle_support_sha256
            )
            if (
                row.fixed_support_sha256 == row.frame_oracle_support_sha256
                and fixed_loss.valid_object_frame_count != oracle_loss.valid_object_frame_count
            ):
                raise ArtifactValidationError(
                    "equal fixed-oracle support commitments require equal full counts"
                )
            oracle_full_common = (
                row.policy_frame_oracle_common_count
                == policy_loss.valid_object_frame_count
                == oracle_loss.valid_object_frame_count
            )
            if row.policy_support_sha256 == row.frame_oracle_support_sha256 and (
                not oracle_full_common
                or not _loss_sum_close(
                    row.policy_on_oracle_common_loss_sum_m2,
                    policy_loss.loss_sum_m2,
                )
                or not _loss_sum_close(
                    row.frame_oracle_on_common_loss_sum_m2,
                    oracle_loss.loss_sum_m2,
                )
            ):
                raise ArtifactValidationError(
                    "equal policy-oracle support commitments require full retained support"
                )
            if oracle_full_common and (
                row.policy_support_sha256 != row.frame_oracle_support_sha256
                or not _loss_sum_close(
                    row.policy_on_oracle_common_loss_sum_m2,
                    policy_loss.loss_sum_m2,
                )
                or not _loss_sum_close(
                    row.frame_oracle_on_common_loss_sum_m2,
                    oracle_loss.loss_sum_m2,
                )
            ):
                raise ArtifactValidationError(
                    "full policy-frame-oracle support disagrees with commitments or losses"
                )
            if row.identical_support_recovery_applicable and (
                row.fixed_policy_common_count != oracle_loss.valid_object_frame_count
                or not _loss_sum_close(
                    row.frame_oracle_on_common_loss_sum_m2,
                    oracle_loss.loss_sum_m2,
                )
            ):
                raise ArtifactValidationError(
                    "recovery support disagrees with retained oracle loss"
                )

        contrast_counts[(row.condition_id, row.policy, row.window)] += 1
    if any(len(digests) != 1 for digests in (*fixed_supports.values(), *oracle_supports.values())):
        raise ArtifactValidationError("shared contrast method has inconsistent support commitments")
    for case in cases:
        expected_count = len(sequence_ids_by_condition[case.condition_id])
        for policy in _HEALTH_POLICY_METHODS:
            for window in _HEALTH_WINDOWS:
                if contrast_counts[(case.condition_id, policy, window)] != expected_count:
                    raise ArtifactValidationError(
                        "health sequence contrast rows do not cover the frozen matrix"
                    )

    event_counts: Counter[tuple[str, str]] = Counter()
    for row in events:
        case = case_by_id.get(row.condition_id)
        if (
            case is None
            or row.sequence_id not in sequence_ids_by_condition[row.condition_id]
            or row.policy not in _HEALTH_POLICIES
        ):
            raise ArtifactValidationError("health event row lies outside the frozen matrix")
        is_dropout = case.fault.family == "dropout"
        targetless = case.fault.target in {"none", "both"}
        if is_dropout != (row.realized_dropout is not None):
            raise ArtifactValidationError("health event dropout fields disagree with its family")
        if targetless and (row.outcome not in {"ambiguous", "missed"} or row.correctly_attributed):
            raise ArtifactValidationError("health targetless event defines target attribution")
        if case.fault.family == "identity" and (row.detected or row.outcome != "missed"):
            raise ArtifactValidationError("health identity event cannot be a detected fault")
        if (
            is_dropout
            and row.realized_dropout is False
            and (row.detected or row.outcome != "missed")
        ):
            raise ArtifactValidationError(
                "health unrealized dropout must remain a missed regime event"
            )
        force_undetected = case.fault.family == "identity" or (
            is_dropout and row.realized_dropout is False
        )
        if force_undetected:
            if row.detected:
                raise ArtifactValidationError("health control event cannot be detected")
        elif row.detected != (row.latch_episode_count > 0):
            raise ArtifactValidationError(
                "health event detection disagrees with its latch episodes"
            )
        if row.detected:
            if targetless or row.first_latch_label == "ambiguous":
                expected_outcome = "ambiguous"
            elif row.first_latch_label == f"{case.fault.target}-fault":
                expected_outcome = "correct"
            else:
                expected_outcome = "wrong-sensor"
            if row.outcome != expected_outcome:
                raise ArtifactValidationError(
                    "health event outcome disagrees with its frozen target"
                )
        if row.latch_episode_count > 12:
            raise ArtifactValidationError("health latch episodes exceed the active-window cap")
        false_alert_cap = 24 if case.fault.schedule == "cold_start" else 23
        if row.false_alert_episode_count > false_alert_cap:
            raise ArtifactValidationError("health false alerts exceed the score-window cap")
        expected_early_clear = row.detected and (
            row.final_active_state == "healthy" or row.latch_episode_count > 1
        )
        if row.early_clear != expected_early_clear:
            raise ArtifactValidationError(
                "health early-clear flag disagrees with active-window transitions"
            )
        if case.fault.family in {"identity", "clean-predictor-mismatch"}:
            false_alert_valid = row.false_alert_episode_count >= row.latch_episode_count
        elif case.fault.family == "dropout" and row.realized_dropout is False:
            false_alert_valid = row.false_alert_episode_count == row.latch_episode_count
        else:
            false_alert_valid = row.false_alert_episode_count == 0
        if not false_alert_valid:
            raise ArtifactValidationError("health event false-alert accounting is invalid")
        if is_dropout and case.fault.value == 1.0:
            expected_healthy_action = (
                row.active_lidar_action_frames
                if case.fault.target == "camera"
                else row.active_camera_action_frames
            )
            unexpected_actions = (
                row.active_camera_action_frames
                + row.active_fixed_action_frames
                + row.active_undefined_action_frames
                if case.fault.target == "camera"
                else row.active_lidar_action_frames
                + row.active_fixed_action_frames
                + row.active_undefined_action_frames
            )
            if (
                row.realized_dropout is not True
                or row.first_missing_frame_minus_event_start != 0
                or expected_healthy_action != row.active_frame_count
                or unexpected_actions != 0
            ):
                raise ArtifactValidationError(
                    "full-dropout event violates deterministic availability semantics"
                )
        if case.fault.family != "dropout" or row.realized_dropout is False:
            if (
                row.active_camera_action_frames != row.active_lidar_fault_frames
                or row.active_lidar_action_frames != row.active_camera_fault_frames
                or row.active_fixed_action_frames
                != row.active_healthy_frames + row.active_ambiguous_frames
                or row.active_undefined_action_frames != 0
            ):
                raise ArtifactValidationError(
                    "health fully-available actions disagree with the frozen state table"
                )
        elif case.fault.target == "camera":
            if (
                row.active_camera_action_frames > row.active_lidar_fault_frames
                or row.active_lidar_action_frames < row.active_camera_fault_frames
                or row.active_fixed_action_frames
                > row.active_healthy_frames + row.active_ambiguous_frames
                or row.active_undefined_action_frames != 0
            ):
                raise ArtifactValidationError(
                    "camera-dropout actions violate availability precedence"
                )
        elif (
            row.active_lidar_action_frames > row.active_camera_fault_frames
            or row.active_camera_action_frames < row.active_lidar_fault_frames
            or row.active_fixed_action_frames
            > row.active_healthy_frames + row.active_ambiguous_frames
            or row.active_undefined_action_frames != 0
        ):
            raise ArtifactValidationError("lidar-dropout actions violate availability precedence")
        recovery_cap = 23 if case.fault.schedule == "cold_start" else 11
        if row.recovery_latency_frames is not None and row.recovery_latency_frames > recovery_cap:
            raise ArtifactValidationError("health recovery latency exceeds its schedule window")
        event_counts[(row.condition_id, row.policy)] += 1
    for case in cases:
        expected_count = len(sequence_ids_by_condition[case.condition_id])
        for policy in _HEALTH_POLICIES:
            if event_counts[(case.condition_id, policy)] != expected_count:
                raise ArtifactValidationError("health event rows do not cover the frozen matrix")

    _require_health_aggregate_structure_for_cases(
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        aggregates=aggregates,
        cases=cases,
    )

    from fusion_fault_bench.health_aggregation import (
        recompute_row_derived_health_aggregates,
    )

    aggregate_by_key = {_aggregate_key(row): row for row in aggregates}
    for case in cases:
        recomputed = recompute_row_derived_health_aggregates(
            condition_id=case.condition_id,
            fault=case.fault,
            sequence_ids=tuple(sorted(sequence_ids_by_condition[case.condition_id])),
            sequence_losses=tuple(row for row in losses if row.condition_id == case.condition_id),
            sequence_contrasts=tuple(
                row for row in contrasts if row.condition_id == case.condition_id
            ),
            sequence_events=tuple(row for row in events if row.condition_id == case.condition_id),
        )
        recomputed_by_key = {_aggregate_key(row): row for row in recomputed}
        observed_by_key = {
            key: row
            for key, row in aggregate_by_key.items()
            if row.condition_id == case.condition_id
        }
        if recomputed_by_key != observed_by_key:
            raise ArtifactValidationError(
                "health aggregate values disagree with retained sequence rows"
            )


def _require_exact_condition_rows(
    *,
    intent: HealthBenchmarkIntentV1,
    main_profile: MainProceduralProfile,
    edge_profile: EdgeProceduralProfile,
    case: Any,
    losses: tuple[HealthSequenceLossV1, ...],
    contrasts: tuple[HealthSequenceContrastV1, ...],
    events: tuple[HealthSequenceEventV1, ...],
    aggregates: tuple[HealthAggregateMetricV1, ...],
) -> None:
    """Apply every exact scientific row check to one bounded condition group."""

    _require_exact_evaluation_rows(
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        losses=losses,
        contrasts=contrasts,
        events=events,
        aggregates=aggregates,
        cases_override=(case,),
    )


def validate_health_fit_bundle(
    intent: HealthBenchmarkIntentV1,
    main_profile: MainProceduralProfile,
    edge_profile: EdgeProceduralProfile,
    calibration: HealthCalibration,
    candidates: Sequence[HealthThresholdCandidateV1],
    summary: HealthFitSummaryV1,
    validation: HealthValidationV1,
    run: RunRecordV1Alpha1,
) -> tuple[HealthThresholdCandidateV1, ...]:
    """Cross-validate one fit bundle before it can be committed."""

    intent_sha256, main_sha256, edge_sha256 = _validate_profiles(
        intent,
        main_profile,
        edge_profile,
    )
    _validate_run(
        intent_sha256,
        run,
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
    )
    _calibration_envelope(calibration)
    ordered = _ordered_candidates(candidates)
    selected = _selected_candidate(intent, ordered)
    if (
        summary.intent_sha256 != intent_sha256
        or summary.main_profile_sha256 != main_sha256
        or summary.edge_profile_sha256 != edge_sha256
    ):
        raise ArtifactValidationError("health fit summary identity is invalid")
    if (
        summary.selected_candidate_index != selected.candidate_index
        or summary.selected_self_threshold != selected.self_threshold
        or summary.selected_cross_threshold != selected.cross_threshold
    ):
        raise ArtifactValidationError("health fit summary disagrees with frozen selection")
    _validate_validation(
        validation,
        intent_sha256=intent_sha256,
        label="health fit",
    )
    _require_exact_fit_validation(
        validation,
        intent=intent,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
    )
    return ordered


def build_health_fit_reference(fit_artifact: LoadedHealthFitArtifact) -> HealthFitReferenceV1:
    """Build the exact evaluation reference for an authenticated fit artifact."""

    return HealthFitReferenceV1(
        schema="ffb.health-fit-reference/v1",
        fit_artifact_sha256=fit_artifact.artifact_sha256,
        fit_run_sha256=fit_artifact.run_sha256,
        intent_sha256=sha256_digest(fit_artifact.intent),
        selected_candidate_index=fit_artifact.summary.selected_candidate_index,
        selected_self_threshold=fit_artifact.summary.selected_self_threshold,
        selected_cross_threshold=fit_artifact.summary.selected_cross_threshold,
    )


def _loss_key(record: HealthSequenceLossV1) -> tuple[str, str, str, str]:
    return record.condition_id, record.sequence_id, record.method, record.window


def _contrast_key(record: HealthSequenceContrastV1) -> tuple[str, str, str, str]:
    return record.condition_id, record.sequence_id, record.policy, record.window


def _event_key(record: HealthSequenceEventV1) -> tuple[str, str, str]:
    return record.condition_id, record.sequence_id, record.policy


def _aggregate_key(record: HealthAggregateMetricV1) -> tuple[str, str, str, str]:
    return (
        record.condition_id,
        "" if record.method is None else record.method,
        record.metric_name,
        "" if record.window is None else record.window,
    )


def _loss_sum_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _ordered_unique[RecordT](
    records: Sequence[RecordT],
    *,
    key: Callable[[RecordT], tuple[str, ...]],
    label: str,
) -> tuple[RecordT, ...]:
    if not records:
        raise ArtifactValidationError(f"{label} must contain at least one record")
    ordered = tuple(sorted(records, key=key))
    keys = tuple(key(record) for record in ordered)
    if len(set(keys)) != len(keys):
        raise ArtifactValidationError(f"{label} contains duplicate logical records")
    return ordered


def _authenticate_fit_handle(
    fit_artifact: LoadedHealthFitArtifact,
) -> LoadedHealthFitArtifact:
    authenticated = load_health_fit_artifact(fit_artifact.path)
    if (
        authenticated.artifact_sha256 != fit_artifact.artifact_sha256
        or authenticated.run_sha256 != fit_artifact.run_sha256
    ):
        raise ArtifactValidationError("provided health fit handle is not authentic")
    return authenticated


def validate_health_evaluation_bundle(
    intent: HealthBenchmarkIntentV1,
    main_profile: MainProceduralProfile,
    edge_profile: EdgeProceduralProfile,
    fit_reference: HealthFitReferenceV1,
    sequence_losses: Sequence[HealthSequenceLossV1],
    sequence_contrasts: Sequence[HealthSequenceContrastV1],
    sequence_events: Sequence[HealthSequenceEventV1],
    aggregates: Sequence[HealthAggregateMetricV1],
    validation: HealthValidationV1,
    run: RunRecordV1Alpha1,
    *,
    fit_artifact: LoadedHealthFitArtifact,
) -> tuple[
    tuple[HealthSequenceLossV1, ...],
    tuple[HealthSequenceContrastV1, ...],
    tuple[HealthSequenceEventV1, ...],
    tuple[HealthAggregateMetricV1, ...],
]:
    """Cross-validate one evaluation bundle against its exact published fit."""

    intent_sha256, main_sha256, edge_sha256 = _validate_profiles(
        intent,
        main_profile,
        edge_profile,
    )
    _validate_run(
        intent_sha256,
        run,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
    )
    if fit_reference != build_health_fit_reference(fit_artifact):
        raise ArtifactValidationError("health evaluation fit reference is invalid")
    if (
        sha256_digest(fit_artifact.intent) != intent_sha256
        or sha256_digest(fit_artifact.main_profile) != main_sha256
        or sha256_digest(fit_artifact.edge_profile) != edge_sha256
    ):
        raise ArtifactValidationError("health evaluation profiles disagree with its fit")
    _validate_validation(
        validation,
        intent_sha256=intent_sha256,
        label="health evaluation",
    )
    _require_exact_evaluation_validation(validation)
    ordered_losses = _ordered_unique(
        sequence_losses,
        key=_loss_key,
        label="health sequence losses",
    )
    ordered_contrasts = _ordered_unique(
        sequence_contrasts,
        key=_contrast_key,
        label="health sequence contrasts",
    )
    ordered_events = _ordered_unique(
        sequence_events,
        key=_event_key,
        label="health sequence events",
    )
    ordered_aggregates = _ordered_unique(
        aggregates,
        key=_aggregate_key,
        label="health aggregate metrics",
    )
    _require_exact_evaluation_rows(
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        losses=ordered_losses,
        contrasts=ordered_contrasts,
        events=ordered_events,
        aggregates=ordered_aggregates,
    )
    return ordered_losses, ordered_contrasts, ordered_events, ordered_aggregates


def _finalize_run(run: RunRecordV1Alpha1, artifact_sha256: str) -> RunRecordV1Alpha1:
    value = run.model_dump(mode="python", by_alias=True)
    value["artifact_sha256"] = artifact_sha256
    return RunRecordV1Alpha1.model_validate(value)


def _payload_index(
    *,
    artifact_contract: HealthArtifactContract,
    run_id: str,
    intent_sha256: str,
    main_profile_sha256: str,
    edge_profile_sha256: str,
    indexed_files: Mapping[str, bytes],
    indexed_paths: tuple[str, ...],
    record_counts: Mapping[str, int],
) -> HealthPayloadIndexV1:
    return HealthPayloadIndexV1(
        schema="ffb.health-payload-index/v1",
        artifact_contract=artifact_contract,
        run_id=run_id,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_profile_sha256,
        edge_profile_sha256=edge_profile_sha256,
        files=tuple(
            HealthPayloadFileEntryV1(
                path=cast(Any, path),
                byte_length=len(indexed_files[path]),
                sha256=_sha256_bytes(indexed_files[path]),
                record_count=record_counts.get(path),
            )
            for path in indexed_paths
        ),
    )


def _complete_files(
    *,
    indexed_files: Mapping[str, bytes],
    payload_index: HealthPayloadIndexV1,
    run: RunRecordV1Alpha1,
) -> Mapping[str, bytes]:
    payload_index_bytes = canonical_json_bytes(payload_index)
    artifact_sha256 = compute_health_artifact_digest(
        payload_index_bytes,
        artifact_contract=payload_index.artifact_contract,
    )
    finalized_run = _finalize_run(run, artifact_sha256)
    _validate_run(
        payload_index.intent_sha256,
        finalized_run,
        artifact_contract=payload_index.artifact_contract,
        artifact_sha256=artifact_sha256,
    )
    run_bytes = canonical_json_bytes(finalized_run)
    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=compute_run_record_digest(run_bytes),
    )
    files = {
        **indexed_files,
        HEALTH_PAYLOAD_INDEX_FILE: payload_index_bytes,
        HEALTH_RUN_FILE: run_bytes,
        HEALTH_SUCCESS_FILE: canonical_json_bytes(success),
    }
    if any(len(value) > HEALTH_MAX_MEMBER_BYTES for value in files.values()):
        raise ArtifactValidationError("health artifact member exceeds its byte cap")
    bounded_json_members = tuple(
        name for name in files if name != HEALTH_ECDF_FILE and not name.endswith(".ndjson")
    )
    if any(len(files[name]) > HEALTH_MAX_RECORD_BYTES for name in bounded_json_members):
        raise ArtifactValidationError("health canonical JSON member exceeds 1 MiB")
    if sum(map(len, files.values())) > HEALTH_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("health artifact exceeds the 1 GiB cap")
    return files


def _prepare_health_fit_artifact(
    request: HealthFitArtifactWriteRequest,
) -> _PreparedHealthArtifact:
    candidates = validate_health_fit_bundle(
        request.intent,
        request.main_profile,
        request.edge_profile,
        request.calibration,
        request.candidates,
        request.summary,
        request.validation,
        request.run,
    )
    intent_sha256, main_sha256, edge_sha256 = _validate_profiles(
        request.intent,
        request.main_profile,
        request.edge_profile,
    )
    indexed_files: dict[str, bytes] = {
        HEALTH_INTENT_FILE: canonical_json_bytes(request.intent),
        HEALTH_MAIN_PROFILE_FILE: canonical_json_bytes(request.main_profile),
        HEALTH_EDGE_PROFILE_FILE: canonical_json_bytes(request.edge_profile),
        HEALTH_ECDF_FILE: canonical_json_bytes(_calibration_envelope(request.calibration)),
        HEALTH_CANDIDATES_FILE: canonical_health_ndjson_bytes(candidates),
        HEALTH_FIT_SUMMARY_FILE: canonical_json_bytes(request.summary),
        HEALTH_FIT_VALIDATION_FILE: canonical_json_bytes(request.validation),
    }
    payload_index = _payload_index(
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
        run_id=request.run.run_id,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
        indexed_files=indexed_files,
        indexed_paths=HEALTH_FIT_INDEXED_PATHS,
        record_counts={HEALTH_CANDIDATES_FILE: len(candidates)},
    )
    return _PreparedHealthArtifact(
        files=_complete_files(
            indexed_files=indexed_files,
            payload_index=payload_index,
            run=request.run,
        ),
        paths=HEALTH_FIT_ARTIFACT_PATHS,
    )


def _prepare_health_evaluation_artifact(
    request: HealthEvaluationArtifactWriteRequest,
    *,
    fit_artifact: LoadedHealthFitArtifact,
) -> _PreparedStreamingHealthEvaluation:
    losses, contrasts, events, aggregates = validate_health_evaluation_bundle(
        request.intent,
        request.main_profile,
        request.edge_profile,
        request.fit_reference,
        request.sequence_losses,
        request.sequence_contrasts,
        request.sequence_events,
        request.aggregates,
        request.validation,
        request.run,
        fit_artifact=fit_artifact,
    )
    intent_sha256, main_sha256, edge_sha256 = _validate_profiles(
        request.intent,
        request.main_profile,
        request.edge_profile,
    )
    indexed_small_files: dict[str, bytes] = {
        HEALTH_INTENT_FILE: canonical_json_bytes(request.intent),
        HEALTH_MAIN_PROFILE_FILE: canonical_json_bytes(request.main_profile),
        HEALTH_EDGE_PROFILE_FILE: canonical_json_bytes(request.edge_profile),
        HEALTH_FIT_REFERENCE_FILE: canonical_json_bytes(request.fit_reference),
        HEALTH_EVAL_VALIDATION_FILE: canonical_json_bytes(request.validation),
    }
    ndjson_members = {
        HEALTH_SEQUENCE_LOSSES_FILE: _canonical_health_ndjson_member(losses),
        HEALTH_SEQUENCE_CONTRASTS_FILE: _canonical_health_ndjson_member(contrasts),
        HEALTH_SEQUENCE_EVENTS_FILE: _canonical_health_ndjson_member(events),
        HEALTH_AGGREGATES_FILE: _canonical_health_ndjson_member(aggregates),
    }
    indexed_metadata = {
        name: (len(value), _sha256_bytes(value), None)
        for name, value in indexed_small_files.items()
    } | {
        name: (member.byte_length, member.sha256, member.record_count)
        for name, member in ndjson_members.items()
    }
    payload_index = HealthPayloadIndexV1(
        schema="ffb.health-payload-index/v1",
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        run_id=request.run.run_id,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
        files=tuple(
            HealthPayloadFileEntryV1(
                path=cast(Any, path),
                byte_length=indexed_metadata[path][0],
                sha256=indexed_metadata[path][1],
                record_count=indexed_metadata[path][2],
            )
            for path in HEALTH_EVAL_INDEXED_PATHS
        ),
    )
    payload_index_bytes = canonical_json_bytes(payload_index)
    artifact_sha256 = compute_health_artifact_digest(
        payload_index_bytes,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
    )
    finalized_run = _finalize_run(request.run, artifact_sha256)
    _validate_run(
        intent_sha256,
        finalized_run,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        artifact_sha256=artifact_sha256,
    )
    run_bytes = canonical_json_bytes(finalized_run)
    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=compute_run_record_digest(run_bytes),
    )
    small_files = {
        **indexed_small_files,
        HEALTH_PAYLOAD_INDEX_FILE: payload_index_bytes,
        HEALTH_RUN_FILE: run_bytes,
        HEALTH_SUCCESS_FILE: canonical_json_bytes(success),
    }
    if any(len(value) > HEALTH_MAX_RECORD_BYTES for value in small_files.values()):
        raise ArtifactValidationError("health canonical JSON member exceeds 1 MiB")
    total_bytes = sum(map(len, small_files.values())) + sum(
        member.byte_length for member in ndjson_members.values()
    )
    if total_bytes > HEALTH_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("health artifact exceeds the 1 GiB cap")
    return _PreparedStreamingHealthEvaluation(
        small_files=small_files,
        ndjson_members=ndjson_members,
        paths=HEALTH_EVAL_ARTIFACT_PATHS,
        sequence_losses=losses,
        sequence_contrasts=contrasts,
        sequence_events=events,
        aggregates=aggregates,
    )


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


def _reject_root_symlink_components(root: Path) -> None:
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            component_stat = os.lstat(current)
        except OSError as error:
            raise ArtifactValidationError("health artifact path cannot be inspected") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ArtifactValidationError("health artifact path contains a symlink")


def _require_safe_tree(root: Path, *, expected_paths: tuple[str, ...]) -> _TreeSnapshot:
    absolute = absolute_artifact_path(root)
    _reject_root_symlink_components(absolute)
    try:
        root_stat = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError("health artifact directory is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ArtifactValidationError("health artifact root must be a real directory")

    entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(absolute) as iterator:
            for entry in iterator:
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ArtifactValidationError("health artifact contains a symlink member")
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ArtifactValidationError("health artifact members must be regular files")
                if entry_stat.st_nlink != 1:
                    raise ArtifactValidationError("health artifact contains a hard-linked member")
                entries[entry.name] = entry_stat
    except ArtifactValidationError:
        raise
    except OSError as error:
        raise ArtifactValidationError("health artifact members cannot be inspected") from error

    if set(entries) != set(expected_paths):
        raise ArtifactValidationError("health artifact file allowlist mismatch")
    if any(entry.st_size > HEALTH_MAX_MEMBER_BYTES for entry in entries.values()):
        raise ArtifactValidationError("health artifact member exceeds its cap")
    if sum(entry.st_size for entry in entries.values()) > HEALTH_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("health artifact exceeds its tree cap")
    return _TreeSnapshot(root_stat=root_stat, entries=entries)


def _open_regular_member(
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
) -> int:
    descriptor = os.open(
        root / name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactValidationError("health artifact member is not a regular file")
        if file_stat.st_nlink != 1:
            raise ArtifactValidationError("health artifact member is hard linked")
        if _stat_fingerprint(file_stat) != _stat_fingerprint(expected_stat):
            raise ArtifactValidationError("health artifact member changed during validation")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_member(
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
    byte_cap: int,
) -> bytes:
    descriptor = _open_regular_member(root, name, expected_stat=expected_stat)
    try:
        file_stat = os.fstat(descriptor)
        if file_stat.st_size > byte_cap:
            raise ArtifactValidationError(f"health artifact member exceeds its cap: {name}")
        chunks: list[bytes] = []
        remaining = byte_cap + 1
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) > byte_cap:
            raise ArtifactValidationError(f"health artifact member exceeds its cap: {name}")
        if len(value) != file_stat.st_size:
            raise ArtifactValidationError("health artifact member changed during reading")
        return value
    finally:
        os.close(descriptor)


def _sha256_member(
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
) -> str:
    digest = hashlib.sha256()
    descriptor = _open_regular_member(root, name, expected_stat=expected_stat)
    try:
        remaining = expected_stat.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise ArtifactValidationError("health artifact member changed during hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ArtifactValidationError("health artifact member changed during hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number must be finite")
    return parsed


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _strict_object(
    data: bytes,
    *,
    label: str,
    byte_cap: int,
) -> dict[str, Any]:
    if len(data) > byte_cap:
        raise ArtifactValidationError(f"{label} exceeds its byte cap")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ArtifactValidationError(f"{label} contains a UTF-8 BOM")
    if b"\r" in data:
        raise ArtifactValidationError(f"{label} contains a forbidden CR byte")
    if not data.endswith(b"\n"):
        raise ArtifactValidationError(f"{label} is missing its terminal LF")
    body = data[:-1]
    if not body or b"\n" in body:
        raise ArtifactValidationError(f"{label} has noncanonical LF bytes")
    try:
        raw = body.decode("utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must contain a JSON object")
    return value


def _load_model[ModelT: BaseModel](
    data: bytes,
    *,
    label: str,
    validate: Callable[[bytes], ModelT],
    byte_cap: int = HEALTH_MAX_RECORD_BYTES,
) -> ModelT:
    _strict_object(data, label=label, byte_cap=byte_cap)
    try:
        model = validate(data)
    except (ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"{label} violates its fixed schema") from error
    if canonical_json_bytes(model) != data:
        raise ArtifactValidationError(f"{label} is not canonical JSON")
    return model


def _load_ndjson[ModelT: BaseModel](
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
    validate: Callable[[bytes], ModelT],
    record_cap: int,
) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    descriptor = _open_regular_member(root, name, expected_stat=expected_stat)
    with os.fdopen(descriptor, "rb") as stream:
        while True:
            line = stream.readline(HEALTH_MAX_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > HEALTH_MAX_RECORD_BYTES:
                raise ArtifactValidationError(f"{name} line exceeds the 1 MiB cap")
            if len(records) >= record_cap:
                raise ArtifactValidationError(f"{name} exceeds its record-count cap")
            records.append(
                _load_model(
                    line,
                    label=f"{name} line {len(records) + 1}",
                    validate=validate,
                )
            )
    if not records:
        raise ArtifactValidationError(f"{name} must contain at least one record")
    return tuple(records)


def _iter_ordered_ndjson[ModelT: BaseModel](
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
    validate: Callable[[bytes], ModelT],
    expected_count: int,
    record_cap: int,
    key: Callable[[ModelT], tuple[str, ...]],
) -> Iterator[ModelT]:
    """Yield canonical rows in strict logical order with bounded memory."""

    descriptor = _open_regular_member(root, name, expected_stat=expected_stat)
    count = 0
    previous_key: tuple[str, ...] | None = None
    with os.fdopen(descriptor, "rb") as stream:
        while True:
            line = stream.readline(HEALTH_MAX_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > HEALTH_MAX_RECORD_BYTES:
                raise ArtifactValidationError(f"{name} line exceeds the 1 MiB cap")
            count += 1
            if count > record_cap or count > expected_count:
                raise ArtifactValidationError(f"{name} exceeds its record-count cap")
            record = _load_model(
                line,
                label=f"{name} line {count}",
                validate=validate,
            )
            logical_key = key(record)
            if previous_key is not None and logical_key <= previous_key:
                raise ArtifactValidationError(
                    "health evaluation records are not in canonical order"
                )
            previous_key = logical_key
            yield record
    if count != expected_count:
        raise ArtifactValidationError("health evaluation record count is invalid")


def _validate_profile_json(value: bytes) -> ProceduralProfileV1:
    return PROCEDURAL_PROFILE_ADAPTER.validate_json(value)


def _verify_tree_snapshot(root: Path, snapshot: _TreeSnapshot) -> None:
    _reject_root_symlink_components(root)
    try:
        current_root_stat = os.lstat(root)
    except OSError as error:
        raise ArtifactValidationError(
            "health artifact root disappeared during validation"
        ) from error
    if _stat_fingerprint(current_root_stat) != _stat_fingerprint(snapshot.root_stat):
        raise ArtifactValidationError("health artifact root changed during validation")
    current_entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                current_entries[entry.name] = entry.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactValidationError("health artifact cannot be rechecked") from error
    if set(current_entries) != set(snapshot.entries):
        raise ArtifactValidationError("health artifact allowlist changed during validation")
    for name, expected_stat in snapshot.entries.items():
        current_stat = current_entries.get(name)
        if current_stat is None or _stat_fingerprint(current_stat) != _stat_fingerprint(
            expected_stat
        ):
            raise ArtifactValidationError("health artifact member changed during validation")


def _validate_index(
    root: Path,
    snapshot: _TreeSnapshot,
    payload_index: HealthPayloadIndexV1,
    *,
    expected_paths: tuple[str, ...],
) -> None:
    for expected_path, entry in zip(expected_paths, payload_index.files, strict=True):
        metadata = snapshot.entries[expected_path]
        if entry.path != expected_path:
            raise ArtifactValidationError("health payload index order is invalid")
        if entry.byte_length != metadata.st_size:
            raise ArtifactValidationError("health payload byte length is invalid")
        if entry.sha256 != _sha256_member(
            root,
            expected_path,
            expected_stat=metadata,
        ):
            raise ArtifactValidationError("health payload digest is invalid")


def _load_common(
    root: Path,
    snapshot: _TreeSnapshot,
    *,
    artifact_contract: HealthArtifactContract,
    indexed_paths: tuple[str, ...],
) -> tuple[
    HealthBenchmarkIntentV1,
    MainProceduralProfile,
    EdgeProceduralProfile,
    HealthPayloadIndexV1,
    RunRecordV1Alpha1,
    SuccessMarkerV1Alpha1,
    str,
    str,
]:
    small_bytes = {
        name: _read_member(
            root,
            name,
            expected_stat=snapshot.entries[name],
            byte_cap=HEALTH_MAX_RECORD_BYTES,
        )
        for name in (
            HEALTH_INTENT_FILE,
            HEALTH_MAIN_PROFILE_FILE,
            HEALTH_EDGE_PROFILE_FILE,
            HEALTH_PAYLOAD_INDEX_FILE,
            HEALTH_RUN_FILE,
            HEALTH_SUCCESS_FILE,
        )
    }
    intent = _load_model(
        small_bytes[HEALTH_INTENT_FILE],
        label=HEALTH_INTENT_FILE,
        validate=HEALTH_BENCHMARK_INTENT_ADAPTER.validate_json,
    )
    main_union = _load_model(
        small_bytes[HEALTH_MAIN_PROFILE_FILE],
        label=HEALTH_MAIN_PROFILE_FILE,
        validate=_validate_profile_json,
    )
    edge_union = _load_model(
        small_bytes[HEALTH_EDGE_PROFILE_FILE],
        label=HEALTH_EDGE_PROFILE_FILE,
        validate=_validate_profile_json,
    )
    if not isinstance(main_union, MainProceduralProfile):
        raise ArtifactValidationError("health main profile has the wrong type")
    if not isinstance(edge_union, EdgeProceduralProfile):
        raise ArtifactValidationError("health edge profile has the wrong type")
    payload_index = _load_model(
        small_bytes[HEALTH_PAYLOAD_INDEX_FILE],
        label=HEALTH_PAYLOAD_INDEX_FILE,
        validate=HealthPayloadIndexV1.model_validate_json,
    )
    run = _load_model(
        small_bytes[HEALTH_RUN_FILE],
        label=HEALTH_RUN_FILE,
        validate=RunRecordV1Alpha1.model_validate_json,
    )
    success = _load_model(
        small_bytes[HEALTH_SUCCESS_FILE],
        label=HEALTH_SUCCESS_FILE,
        validate=SuccessMarkerV1Alpha1.model_validate_json,
    )
    if payload_index.artifact_contract != artifact_contract:
        raise ArtifactValidationError("health payload has the wrong artifact contract")
    _validate_index(
        root,
        snapshot,
        payload_index,
        expected_paths=indexed_paths,
    )
    intent_sha256, main_sha256, edge_sha256 = _validate_profiles(
        intent,
        main_union,
        edge_union,
    )
    if (
        payload_index.run_id != run.run_id
        or payload_index.intent_sha256 != intent_sha256
        or payload_index.main_profile_sha256 != main_sha256
        or payload_index.edge_profile_sha256 != edge_sha256
    ):
        raise ArtifactValidationError("health payload identity graph is invalid")
    index_bytes = small_bytes[HEALTH_PAYLOAD_INDEX_FILE]
    artifact_sha256 = compute_health_artifact_digest(
        index_bytes,
        artifact_contract=artifact_contract,
    )
    _validate_run(
        intent_sha256,
        run,
        artifact_contract=artifact_contract,
        artifact_sha256=artifact_sha256,
    )
    run_sha256 = compute_run_record_digest(small_bytes[HEALTH_RUN_FILE])
    if success.artifact_sha256 != artifact_sha256 or success.run_sha256 != run_sha256:
        raise ArtifactValidationError("health success marker identity is invalid")
    return (
        intent,
        main_union,
        edge_union,
        payload_index,
        run,
        success,
        artifact_sha256,
        run_sha256,
    )


def _record_count(payload_index: HealthPayloadIndexV1, path: str) -> int:
    entry = next(entry for entry in payload_index.files if entry.path == path)
    if entry.record_count is None:
        raise ArtifactValidationError("health NDJSON index is missing its record count")
    return entry.record_count


def _load_health_fit_artifact(path: Path) -> LoadedHealthFitArtifact:
    root = absolute_artifact_path(path)
    snapshot = _require_safe_tree(root, expected_paths=HEALTH_FIT_ARTIFACT_PATHS)
    (
        intent,
        main_profile,
        edge_profile,
        payload_index,
        run,
        success,
        artifact_sha256,
        run_sha256,
    ) = _load_common(
        root,
        snapshot,
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
        indexed_paths=HEALTH_FIT_INDEXED_PATHS,
    )
    ecdf_bytes = _read_member(
        root,
        HEALTH_ECDF_FILE,
        expected_stat=snapshot.entries[HEALTH_ECDF_FILE],
        byte_cap=HEALTH_MAX_MEMBER_BYTES,
    )
    ecdf = _load_model(
        ecdf_bytes,
        label=HEALTH_ECDF_FILE,
        validate=HealthEcdfArraysV1.model_validate_json,
        byte_cap=HEALTH_MAX_MEMBER_BYTES,
    )
    calibration = _health_calibration(ecdf)
    candidates = _load_ndjson(
        root,
        HEALTH_CANDIDATES_FILE,
        expected_stat=snapshot.entries[HEALTH_CANDIDATES_FILE],
        validate=HealthThresholdCandidateV1.model_validate_json,
        record_cap=_CANDIDATE_COUNT,
    )
    summary = _load_model(
        _read_member(
            root,
            HEALTH_FIT_SUMMARY_FILE,
            expected_stat=snapshot.entries[HEALTH_FIT_SUMMARY_FILE],
            byte_cap=HEALTH_MAX_RECORD_BYTES,
        ),
        label=HEALTH_FIT_SUMMARY_FILE,
        validate=HealthFitSummaryV1.model_validate_json,
    )
    validation = _load_model(
        _read_member(
            root,
            HEALTH_FIT_VALIDATION_FILE,
            expected_stat=snapshot.entries[HEALTH_FIT_VALIDATION_FILE],
            byte_cap=HEALTH_MAX_RECORD_BYTES,
        ),
        label=HEALTH_FIT_VALIDATION_FILE,
        validate=HealthValidationV1.model_validate_json,
    )
    if len(candidates) != _record_count(payload_index, HEALTH_CANDIDATES_FILE):
        raise ArtifactValidationError("health candidate record count is invalid")
    ordered = validate_health_fit_bundle(
        intent,
        main_profile,
        edge_profile,
        calibration,
        candidates,
        summary,
        validation,
        run,
    )
    if candidates != ordered:
        raise ArtifactValidationError("health candidates are not in canonical order")
    _verify_tree_snapshot(root, snapshot)
    return LoadedHealthFitArtifact(
        path=root,
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        calibration=calibration,
        candidates=candidates,
        summary=summary,
        validation=validation,
        payload_index=payload_index,
        run=run,
        success=success,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )


def load_health_fit_artifact(path: Path) -> LoadedHealthFitArtifact:
    """Strictly load one complete, immutable M4 fit artifact."""

    try:
        return _load_health_fit_artifact(path)
    except ArtifactValidationError:
        raise
    except (OSError, StopIteration, ValueError, ValidationError) as error:
        raise ArtifactValidationError("invalid M4 health fit artifact") from error


def _load_health_evaluation_artifact(
    path: Path,
    *,
    fit_artifact: LoadedHealthFitArtifact,
) -> LoadedHealthEvaluationArtifact:
    root = absolute_artifact_path(path)
    snapshot = _require_safe_tree(root, expected_paths=HEALTH_EVAL_ARTIFACT_PATHS)
    (
        intent,
        main_profile,
        edge_profile,
        payload_index,
        run,
        success,
        artifact_sha256,
        run_sha256,
    ) = _load_common(
        root,
        snapshot,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        indexed_paths=HEALTH_EVAL_INDEXED_PATHS,
    )
    fit_reference = _load_model(
        _read_member(
            root,
            HEALTH_FIT_REFERENCE_FILE,
            expected_stat=snapshot.entries[HEALTH_FIT_REFERENCE_FILE],
            byte_cap=HEALTH_MAX_RECORD_BYTES,
        ),
        label=HEALTH_FIT_REFERENCE_FILE,
        validate=HealthFitReferenceV1.model_validate_json,
    )
    loss_count = _record_count(payload_index, HEALTH_SEQUENCE_LOSSES_FILE)
    contrast_count = _record_count(payload_index, HEALTH_SEQUENCE_CONTRASTS_FILE)
    event_count = _record_count(payload_index, HEALTH_SEQUENCE_EVENTS_FILE)
    aggregate_count = _record_count(payload_index, HEALTH_AGGREGATES_FILE)
    validation = _load_model(
        _read_member(
            root,
            HEALTH_EVAL_VALIDATION_FILE,
            expected_stat=snapshot.entries[HEALTH_EVAL_VALIDATION_FILE],
            byte_cap=HEALTH_MAX_RECORD_BYTES,
        ),
        label=HEALTH_EVAL_VALIDATION_FILE,
        validate=HealthValidationV1.model_validate_json,
    )
    if (
        loss_count != _MAX_EVALUATION_LOSS_RECORDS
        or contrast_count != _MAX_EVALUATION_CONTRAST_RECORDS
        or event_count != _MAX_EVALUATION_EVENT_RECORDS
    ):
        raise ArtifactValidationError(
            "health evaluation sequence record counts do not match the frozen matrix"
        )
    if not 0 < aggregate_count <= _MAX_EVALUATION_AGGREGATE_RECORDS:
        raise ArtifactValidationError("health evaluation record count is invalid")
    if fit_reference != build_health_fit_reference(fit_artifact):
        raise ArtifactValidationError("health evaluation fit reference is invalid")
    intent_sha256 = sha256_digest(intent)
    if (
        sha256_digest(fit_artifact.intent) != intent_sha256
        or sha256_digest(fit_artifact.main_profile) != sha256_digest(main_profile)
        or sha256_digest(fit_artifact.edge_profile) != sha256_digest(edge_profile)
    ):
        raise ArtifactValidationError("health evaluation profiles disagree with its fit")
    _validate_validation(
        validation,
        intent_sha256=intent_sha256,
        label="health evaluation",
    )
    _require_exact_evaluation_validation(validation)

    loss_rows = _iter_ordered_ndjson(
        root,
        HEALTH_SEQUENCE_LOSSES_FILE,
        expected_stat=snapshot.entries[HEALTH_SEQUENCE_LOSSES_FILE],
        validate=HealthSequenceLossV1.model_validate_json,
        expected_count=loss_count,
        record_cap=_MAX_EVALUATION_LOSS_RECORDS,
        key=_loss_key,
    )
    contrast_rows = _iter_ordered_ndjson(
        root,
        HEALTH_SEQUENCE_CONTRASTS_FILE,
        expected_stat=snapshot.entries[HEALTH_SEQUENCE_CONTRASTS_FILE],
        validate=HealthSequenceContrastV1.model_validate_json,
        expected_count=contrast_count,
        record_cap=_MAX_EVALUATION_CONTRAST_RECORDS,
        key=_contrast_key,
    )
    event_rows = _iter_ordered_ndjson(
        root,
        HEALTH_SEQUENCE_EVENTS_FILE,
        expected_stat=snapshot.entries[HEALTH_SEQUENCE_EVENTS_FILE],
        validate=HealthSequenceEventV1.model_validate_json,
        expected_count=event_count,
        record_cap=_MAX_EVALUATION_EVENT_RECORDS,
        key=_event_key,
    )
    aggregate_rows = _iter_ordered_ndjson(
        root,
        HEALTH_AGGREGATES_FILE,
        expected_stat=snapshot.entries[HEALTH_AGGREGATES_FILE],
        validate=HealthAggregateMetricV1.model_validate_json,
        expected_count=aggregate_count,
        record_cap=_MAX_EVALUATION_AGGREGATE_RECORDS,
        key=_aggregate_key,
    )
    grouped = (
        iter(groupby(loss_rows, key=lambda row: row.condition_id)),
        iter(groupby(contrast_rows, key=lambda row: row.condition_id)),
        iter(groupby(event_rows, key=lambda row: row.condition_id)),
        iter(groupby(aggregate_rows, key=lambda row: row.condition_id)),
    )
    from fusion_fault_bench.health_benchmark import expand_test_cases

    cases = tuple(sorted(expand_test_cases(intent), key=lambda case: case.condition_id))
    if len(cases) != _EXPECTED_EVALUATION_CONDITION_COUNT:
        raise ArtifactValidationError("health evaluation matrix identity is invalid")
    retained_aggregates: list[HealthAggregateMetricV1] = []
    for case in cases:
        condition_groups: list[tuple[BaseModel, ...]] = []
        for groups in grouped:
            try:
                condition_id, rows = next(groups)
            except StopIteration as error:
                raise ArtifactValidationError(
                    "health evaluation condition groups are incomplete"
                ) from error
            if condition_id != case.condition_id:
                raise ArtifactValidationError(
                    "health evaluation condition groups are not canonical"
                )
            condition_groups.append(tuple(rows))
        condition_losses = cast(tuple[HealthSequenceLossV1, ...], condition_groups[0])
        condition_contrasts = cast(
            tuple[HealthSequenceContrastV1, ...],
            condition_groups[1],
        )
        condition_events = cast(tuple[HealthSequenceEventV1, ...], condition_groups[2])
        condition_aggregates = cast(
            tuple[HealthAggregateMetricV1, ...],
            condition_groups[3],
        )
        _require_exact_condition_rows(
            intent=intent,
            main_profile=main_profile,
            edge_profile=edge_profile,
            case=case,
            losses=condition_losses,
            contrasts=condition_contrasts,
            events=condition_events,
            aggregates=condition_aggregates,
        )
        retained_aggregates.extend(condition_aggregates)
    for groups in grouped:
        with suppress(StopIteration):
            next(groups)
            raise ArtifactValidationError("health evaluation contains extra condition groups")
    _verify_tree_snapshot(root, snapshot)
    return LoadedHealthEvaluationArtifact(
        path=root,
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        fit_reference=fit_reference,
        sequence_rows_materialized=False,
        sequence_losses=(),
        sequence_contrasts=(),
        sequence_events=(),
        aggregates=tuple(retained_aggregates),
        validation=validation,
        payload_index=payload_index,
        run=run,
        success=success,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )


def load_health_evaluation_artifact(
    path: Path,
    *,
    fit_artifact: LoadedHealthFitArtifact,
) -> LoadedHealthEvaluationArtifact:
    """Strictly load an M4 evaluation artifact against its exact fit."""

    try:
        authenticated_fit = _authenticate_fit_handle(fit_artifact)
        return _load_health_evaluation_artifact(
            path,
            fit_artifact=authenticated_fit,
        )
    except ArtifactValidationError:
        raise
    except (OSError, StopIteration, ValueError, ValidationError) as error:
        raise ArtifactValidationError("invalid M4 health evaluation artifact") from error


def _safe_cleanup_staging_at(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    paths: tuple[str, ...],
) -> None:
    for name in paths:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=staging_fd)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _publish_health_artifact[LoadedT](
    prepared: _PreparedHealthArtifact,
    destination: Path,
    *,
    load_staging: Callable[[Path], LoadedT],
    load_published: Callable[[Path], LoadedT],
    source_root: Path | None,
    git_metadata_dirs: Sequence[Path] | None,
) -> LoadedT:
    target = absolute_artifact_path(destination)
    metadata_dirs = (
        discover_git_metadata_dirs(source_root)
        if git_metadata_dirs is None
        else tuple(git_metadata_dirs)
    )
    reject_git_metadata_destination(target, metadata_dirs)
    if os.path.lexists(target):
        raise FileExistsError("health artifact destination already exists")
    parent = target.parent
    parent_fd = open_or_create_real_directory(parent)
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            parent,
            label="destination parent",
        )
        reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
        if entry_exists_at(parent_fd, target.name):
            raise FileExistsError("health artifact destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        published = False
        try:
            for name in prepared.paths[:-1]:
                write_exclusive_file_at(staging_fd, name, prepared.files[name])
            for name in prepared.paths[:-1]:
                if (
                    read_file_at(
                        staging_fd,
                        name,
                        byte_cap=len(prepared.files[name]),
                    )
                    != prepared.files[name]
                ):
                    raise ArtifactValidationError("health staging verification failed")
            write_exclusive_file_at(
                staging_fd,
                HEALTH_SUCCESS_FILE,
                prepared.files[HEALTH_SUCCESS_FILE],
            )
            os.fsync(staging_fd)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            load_staging(staging)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
            if entry_exists_at(parent_fd, target.name):
                raise FileExistsError("health artifact destination already exists")
            atomic_rename_directory_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            published = True
            os.fsync(parent_fd)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            loaded = load_published(target)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            return loaded
        except BaseException:
            if not published:
                _safe_cleanup_staging_at(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=staging_name,
                    paths=prepared.paths,
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def _write_canonical_ndjson_at(
    directory_fd: int,
    name: str,
    member: _CanonicalNdjsonMember,
) -> None:
    """Write one canonical NDJSON member with constant additional memory."""

    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    digest = hashlib.sha256()
    byte_length = 0
    try:
        for index, record in enumerate(member.records):
            line = canonical_json_bytes(record)
            if len(line) > HEALTH_MAX_RECORD_BYTES:
                raise ArtifactValidationError(
                    f"health NDJSON record {index} exceeds the 1 MiB line cap"
                )
            digest.update(line)
            byte_length += len(line)
            remaining = memoryview(line)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("short write while staging health NDJSON")
                remaining = remaining[written:]
        os.fsync(descriptor)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ArtifactValidationError("staged health NDJSON is not one private regular file")
        if (
            byte_length != member.byte_length
            or file_stat.st_size != member.byte_length
            or digest.hexdigest() != member.sha256
        ):
            raise ArtifactValidationError("staged health NDJSON disagrees with its preparation")
    finally:
        os.close(descriptor)


def _load_streamed_health_evaluation_handle(
    path: Path,
    *,
    prepared: _PreparedStreamingHealthEvaluation,
    fit_artifact: LoadedHealthFitArtifact,
) -> LoadedHealthEvaluationArtifact:
    """Authenticate streamed bytes while reusing already-validated row objects."""

    root = absolute_artifact_path(path)
    snapshot = _require_safe_tree(root, expected_paths=HEALTH_EVAL_ARTIFACT_PATHS)
    (
        intent,
        main_profile,
        edge_profile,
        payload_index,
        run,
        success,
        artifact_sha256,
        run_sha256,
    ) = _load_common(
        root,
        snapshot,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        indexed_paths=HEALTH_EVAL_INDEXED_PATHS,
    )
    fit_reference = _load_model(
        _read_member(
            root,
            HEALTH_FIT_REFERENCE_FILE,
            expected_stat=snapshot.entries[HEALTH_FIT_REFERENCE_FILE],
            byte_cap=HEALTH_MAX_RECORD_BYTES,
        ),
        label=HEALTH_FIT_REFERENCE_FILE,
        validate=HealthFitReferenceV1.model_validate_json,
    )
    validation = _load_model(
        _read_member(
            root,
            HEALTH_EVAL_VALIDATION_FILE,
            expected_stat=snapshot.entries[HEALTH_EVAL_VALIDATION_FILE],
            byte_cap=HEALTH_MAX_RECORD_BYTES,
        ),
        label=HEALTH_EVAL_VALIDATION_FILE,
        validate=HealthValidationV1.model_validate_json,
    )
    intent_sha256 = sha256_digest(intent)
    if fit_reference != build_health_fit_reference(fit_artifact):
        raise ArtifactValidationError("health evaluation fit reference is invalid")
    if (
        sha256_digest(fit_artifact.intent) != intent_sha256
        or sha256_digest(fit_artifact.main_profile) != sha256_digest(main_profile)
        or sha256_digest(fit_artifact.edge_profile) != sha256_digest(edge_profile)
    ):
        raise ArtifactValidationError("health evaluation profiles disagree with its fit")
    _validate_validation(
        validation,
        intent_sha256=intent_sha256,
        label="health evaluation",
    )
    _require_exact_evaluation_validation(validation)
    for name, member in prepared.ndjson_members.items():
        if _record_count(payload_index, name) != member.record_count:
            raise ArtifactValidationError("health evaluation record count is invalid")
    _verify_tree_snapshot(root, snapshot)
    return LoadedHealthEvaluationArtifact(
        path=root,
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        fit_reference=fit_reference,
        sequence_rows_materialized=True,
        sequence_losses=prepared.sequence_losses,
        sequence_contrasts=prepared.sequence_contrasts,
        sequence_events=prepared.sequence_events,
        aggregates=prepared.aggregates,
        validation=validation,
        payload_index=payload_index,
        run=run,
        success=success,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )


def _publish_streaming_health_evaluation(
    prepared: _PreparedStreamingHealthEvaluation,
    destination: Path,
    *,
    fit_artifact: LoadedHealthFitArtifact,
    source_root: Path | None,
    git_metadata_dirs: Sequence[Path] | None,
) -> LoadedHealthEvaluationArtifact:
    """Atomically publish evaluation rows without member-sized byte buffers."""

    target = absolute_artifact_path(destination)
    metadata_dirs = (
        discover_git_metadata_dirs(source_root)
        if git_metadata_dirs is None
        else tuple(git_metadata_dirs)
    )
    reject_git_metadata_destination(target, metadata_dirs)
    if os.path.lexists(target):
        raise FileExistsError("health artifact destination already exists")
    parent = target.parent
    parent_fd = open_or_create_real_directory(parent)
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            parent,
            label="destination parent",
        )
        reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
        if entry_exists_at(parent_fd, target.name):
            raise FileExistsError("health artifact destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        published = False
        try:
            for name in prepared.paths[:-1]:
                member = prepared.ndjson_members.get(name)
                if member is None:
                    write_exclusive_file_at(staging_fd, name, prepared.small_files[name])
                else:
                    _write_canonical_ndjson_at(staging_fd, name, member)
            write_exclusive_file_at(
                staging_fd,
                HEALTH_SUCCESS_FILE,
                prepared.small_files[HEALTH_SUCCESS_FILE],
            )
            os.fsync(staging_fd)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            _load_streamed_health_evaluation_handle(
                staging,
                prepared=prepared,
                fit_artifact=fit_artifact,
            )
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
            if entry_exists_at(parent_fd, target.name):
                raise FileExistsError("health artifact destination already exists")
            atomic_rename_directory_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            published = True
            os.fsync(parent_fd)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            loaded = _load_streamed_health_evaluation_handle(
                target,
                prepared=prepared,
                fit_artifact=fit_artifact,
            )
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            return loaded
        except BaseException:
            if not published:
                _safe_cleanup_staging_at(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=staging_name,
                    paths=prepared.paths,
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


class HealthEvaluationArtifactTransaction:
    """Bounded-memory, fail-closed publisher for the exact M4 evaluation."""

    def __init__(
        self,
        destination: Path,
        *,
        fit_artifact: LoadedHealthFitArtifact,
        source_root: Path | None = None,
        git_metadata_dirs: Sequence[Path] | None = None,
    ) -> None:
        self.fit_artifact = _authenticate_fit_handle(fit_artifact)
        try:
            self._fit_root_stat = os.lstat(self.fit_artifact.path)
        except OSError as error:
            raise ArtifactValidationError("health fit source cannot be snapshotted") from error
        self.intent = self.fit_artifact.intent
        self.main_profile = self.fit_artifact.main_profile
        self.edge_profile = self.fit_artifact.edge_profile
        self.fit_reference = build_health_fit_reference(self.fit_artifact)
        self.target = absolute_artifact_path(destination)
        metadata_dirs = (
            discover_git_metadata_dirs(source_root)
            if git_metadata_dirs is None
            else tuple(git_metadata_dirs)
        )
        reject_git_metadata_destination(self.target, metadata_dirs)
        if os.path.lexists(self.target):
            raise FileExistsError("health artifact destination already exists")
        self._metadata_dirs = metadata_dirs
        self._parent = self.target.parent
        self._parent_fd = open_or_create_real_directory(self._parent)
        self._staging_fd = -1
        self._staging_name = ""
        self._published = False
        self._closed = False
        self._streams: dict[str, _OpenCanonicalNdjsonStream] = {}
        from fusion_fault_bench.health_benchmark import expand_test_cases

        self._cases = tuple(
            sorted(expand_test_cases(self.intent), key=lambda case: case.condition_id)
        )
        self._next_condition = 0
        self._indexed_small_files = {
            HEALTH_INTENT_FILE: canonical_json_bytes(self.intent),
            HEALTH_MAIN_PROFILE_FILE: canonical_json_bytes(self.main_profile),
            HEALTH_EDGE_PROFILE_FILE: canonical_json_bytes(self.edge_profile),
            HEALTH_FIT_REFERENCE_FILE: canonical_json_bytes(self.fit_reference),
        }
        try:
            assert_directory_descriptor_matches_path(
                self._parent_fd,
                self._parent,
                label="destination parent",
            )
            reject_directory_descriptor_in_git_metadata(self._parent_fd, metadata_dirs)
            if entry_exists_at(self._parent_fd, self.target.name):
                raise FileExistsError("health artifact destination already exists")
            self._staging_name, self._staging_fd = create_staging_directory_at(self._parent_fd)
            for name, value in self._indexed_small_files.items():
                write_exclusive_file_at(self._staging_fd, name, value)
            for name in (
                HEALTH_SEQUENCE_LOSSES_FILE,
                HEALTH_SEQUENCE_CONTRASTS_FILE,
                HEALTH_SEQUENCE_EVENTS_FILE,
                HEALTH_AGGREGATES_FILE,
            ):
                self._streams[name] = _OpenCanonicalNdjsonStream(self._staging_fd, name)
        except BaseException:
            self.abort()
            raise

    @property
    def staging_path(self) -> Path:
        return self._parent / self._staging_name

    def _reauthenticate_fit_source(self) -> LoadedHealthFitArtifact:
        try:
            current_stat = os.lstat(self.fit_artifact.path)
        except OSError as error:
            raise ArtifactValidationError(
                "health evaluation fit source changed during publication"
            ) from error
        if _stat_fingerprint(current_stat) != _stat_fingerprint(self._fit_root_stat):
            raise ArtifactValidationError("health evaluation fit source changed during publication")
        authenticated = _authenticate_fit_handle(self.fit_artifact)
        if (
            authenticated.artifact_sha256 != self.fit_artifact.artifact_sha256
            or authenticated.run_sha256 != self.fit_artifact.run_sha256
        ):
            raise ArtifactValidationError("health evaluation fit source changed during publication")
        return authenticated

    def append_condition(self, batch: Any) -> None:
        """Validate and append exactly one canonical condition batch."""

        if self._closed or self._next_condition >= len(self._cases):
            raise ArtifactValidationError("health evaluation transaction is not appendable")
        case = self._cases[self._next_condition]
        if batch.condition_id != case.condition_id:
            raise ArtifactValidationError("health evaluation conditions are not in canonical order")
        losses = tuple(batch.sequence_losses)
        contrasts = tuple(batch.sequence_contrasts)
        events = tuple(batch.sequence_events)
        aggregates = tuple(batch.aggregates)
        _require_exact_condition_rows(
            intent=self.intent,
            main_profile=self.main_profile,
            edge_profile=self.edge_profile,
            case=case,
            losses=losses,
            contrasts=contrasts,
            events=events,
            aggregates=aggregates,
        )
        self._streams[HEALTH_SEQUENCE_LOSSES_FILE].append(losses, key=_loss_key)
        self._streams[HEALTH_SEQUENCE_CONTRASTS_FILE].append(
            contrasts,
            key=_contrast_key,
        )
        self._streams[HEALTH_SEQUENCE_EVENTS_FILE].append(events, key=_event_key)
        self._streams[HEALTH_AGGREGATES_FILE].append(aggregates, key=_aggregate_key)
        self._next_condition += 1

    def finalize(
        self,
        *,
        validation: HealthValidationV1,
        run: RunRecordV1Alpha1,
    ) -> LoadedHealthEvaluationArtifact:
        """Commit the completed stream, authenticate it twice, and publish."""

        if self._closed or self._next_condition != len(self._cases):
            raise ArtifactValidationError("health evaluation condition stream is incomplete")
        intent_sha256, main_sha256, edge_sha256 = _validate_profiles(
            self.intent,
            self.main_profile,
            self.edge_profile,
        )
        _validate_validation(
            validation,
            intent_sha256=intent_sha256,
            label="health evaluation",
        )
        _require_exact_evaluation_validation(validation)
        _validate_run(
            intent_sha256,
            run,
            artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        )
        stream_metadata = {name: stream.finish() for name, stream in self._streams.items()}
        if (
            stream_metadata[HEALTH_SEQUENCE_LOSSES_FILE].record_count
            != _MAX_EVALUATION_LOSS_RECORDS
            or stream_metadata[HEALTH_SEQUENCE_CONTRASTS_FILE].record_count
            != _MAX_EVALUATION_CONTRAST_RECORDS
            or stream_metadata[HEALTH_SEQUENCE_EVENTS_FILE].record_count
            != _MAX_EVALUATION_EVENT_RECORDS
        ):
            raise ArtifactValidationError("health evaluation stream has wrong global counts")
        validation_bytes = canonical_json_bytes(validation)
        self._indexed_small_files[HEALTH_EVAL_VALIDATION_FILE] = validation_bytes
        write_exclusive_file_at(
            self._staging_fd,
            HEALTH_EVAL_VALIDATION_FILE,
            validation_bytes,
        )
        indexed_metadata = {
            name: (len(value), _sha256_bytes(value), None)
            for name, value in self._indexed_small_files.items()
        } | {
            name: (metadata.byte_length, metadata.sha256, metadata.record_count)
            for name, metadata in stream_metadata.items()
        }
        payload_index = HealthPayloadIndexV1(
            schema="ffb.health-payload-index/v1",
            artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
            run_id=run.run_id,
            intent_sha256=intent_sha256,
            main_profile_sha256=main_sha256,
            edge_profile_sha256=edge_sha256,
            files=tuple(
                HealthPayloadFileEntryV1(
                    path=cast(Any, path),
                    byte_length=indexed_metadata[path][0],
                    sha256=indexed_metadata[path][1],
                    record_count=indexed_metadata[path][2],
                )
                for path in HEALTH_EVAL_INDEXED_PATHS
            ),
        )
        payload_index_bytes = canonical_json_bytes(payload_index)
        artifact_sha256 = compute_health_artifact_digest(
            payload_index_bytes,
            artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        )
        finalized_run = _finalize_run(run, artifact_sha256)
        run_bytes = canonical_json_bytes(finalized_run)
        expected_run_sha256 = compute_run_record_digest(run_bytes)
        success_bytes = canonical_json_bytes(
            SuccessMarkerV1Alpha1(
                schema="ffb.success/v1alpha1",
                artifact_sha256=artifact_sha256,
                run_sha256=expected_run_sha256,
            )
        )
        total_bytes = (
            sum(metadata.byte_length for metadata in stream_metadata.values())
            + sum(len(value) for value in self._indexed_small_files.values())
            + len(payload_index_bytes)
            + len(run_bytes)
            + len(success_bytes)
        )
        if total_bytes > HEALTH_MAX_ARTIFACT_BYTES:
            raise ArtifactValidationError("health artifact exceeds the 1 GiB cap")
        write_exclusive_file_at(
            self._staging_fd,
            HEALTH_PAYLOAD_INDEX_FILE,
            payload_index_bytes,
        )
        write_exclusive_file_at(self._staging_fd, HEALTH_RUN_FILE, run_bytes)
        write_exclusive_file_at(self._staging_fd, HEALTH_SUCCESS_FILE, success_bytes)
        os.fsync(self._staging_fd)
        authenticated_fit = self._reauthenticate_fit_source()
        staged = _load_health_evaluation_artifact(
            self.staging_path,
            fit_artifact=authenticated_fit,
        )
        if staged.artifact_sha256 != artifact_sha256 or staged.run_sha256 != expected_run_sha256:
            raise ArtifactValidationError(
                "staged health evaluation identity changed during authentication"
            )
        assert_directory_descriptor_matches_path(
            self._staging_fd,
            self.staging_path,
            label="staging artifact",
        )
        assert_directory_descriptor_matches_path(
            self._parent_fd,
            self._parent,
            label="destination parent",
        )
        reject_directory_descriptor_in_git_metadata(
            self._parent_fd,
            self._metadata_dirs,
        )
        if entry_exists_at(self._parent_fd, self.target.name):
            raise FileExistsError("health artifact destination already exists")
        self._reauthenticate_fit_source()
        atomic_rename_directory_no_replace_at(
            self._parent_fd,
            self._staging_name,
            self._parent_fd,
            self.target.name,
        )
        renamed = True
        try:
            os.fsync(self._parent_fd)
            assert_directory_descriptor_matches_path(
                self._parent_fd,
                self._parent,
                label="destination parent",
            )
            assert_directory_descriptor_matches_path(
                self._staging_fd,
                self.target,
                label="published artifact",
            )
            authenticated_fit = self._reauthenticate_fit_source()
            loaded = _load_health_evaluation_artifact(
                self.target,
                fit_artifact=authenticated_fit,
            )
            if (
                loaded.artifact_sha256 != artifact_sha256
                or loaded.run_sha256 != expected_run_sha256
            ):
                raise ArtifactValidationError(
                    "published health evaluation identity changed during authentication"
                )
            assert_directory_descriptor_matches_path(
                self._parent_fd,
                self._parent,
                label="destination parent",
            )
            assert_directory_descriptor_matches_path(
                self._staging_fd,
                self.target,
                label="published artifact",
            )
            self._reauthenticate_fit_source()
            assert_directory_descriptor_matches_path(
                self._parent_fd,
                self._parent,
                label="destination parent",
            )
            assert_directory_descriptor_matches_path(
                self._staging_fd,
                self.target,
                label="published artifact",
            )
        except BaseException:
            if renamed:
                with suppress(OSError, ArtifactValidationError):
                    assert_directory_descriptor_matches_path(
                        self._staging_fd,
                        self.target,
                        label="published artifact",
                    )
                    _safe_cleanup_staging_at(
                        parent_fd=self._parent_fd,
                        staging_fd=self._staging_fd,
                        staging_name=self.target.name,
                        paths=HEALTH_EVAL_ARTIFACT_PATHS,
                    )
                    os.fsync(self._parent_fd)
            raise
        self._published = True
        self._close_descriptors()
        return loaded

    def _close_descriptors(self) -> None:
        for stream in self._streams.values():
            with suppress(OSError):
                stream.close()
        if self._staging_fd >= 0:
            with suppress(OSError):
                os.close(self._staging_fd)
            self._staging_fd = -1
        if self._parent_fd >= 0:
            with suppress(OSError):
                os.close(self._parent_fd)
            self._parent_fd = -1
        self._closed = True

    def abort(self) -> None:
        """Close streams and remove an unpublished staging directory."""

        if self._closed:
            return
        cleanup_error: BaseException | None = None
        try:
            for stream in self._streams.values():
                stream.close()
            if (
                not self._published
                and self._parent_fd >= 0
                and self._staging_fd >= 0
                and self._staging_name
            ):
                _safe_cleanup_staging_at(
                    parent_fd=self._parent_fd,
                    staging_fd=self._staging_fd,
                    staging_name=self._staging_name,
                    paths=HEALTH_EVAL_ARTIFACT_PATHS,
                )
        except BaseException as error:
            cleanup_error = error
        finally:
            self._close_descriptors()
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self) -> HealthEvaluationArtifactTransaction:
        return self

    def __exit__(self, *_: object) -> None:
        if not self._closed:
            self.abort()


def write_health_fit_artifact(
    request: HealthFitArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path | None = None,
    git_metadata_dirs: Sequence[Path] | None = None,
) -> LoadedHealthFitArtifact:
    """Validate, stage, and atomically publish one no-overwrite M4 fit."""

    try:
        prepared = _prepare_health_fit_artifact(request)
        return _publish_health_artifact(
            prepared,
            destination,
            load_staging=_load_health_fit_artifact,
            load_published=load_health_fit_artifact,
            source_root=source_root,
            git_metadata_dirs=git_metadata_dirs,
        )
    except FileExistsError:
        raise
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactValidationError("M4 health fit publication failed") from error


def write_health_evaluation_artifact(
    request: HealthEvaluationArtifactWriteRequest,
    destination: Path,
    *,
    fit_artifact: LoadedHealthFitArtifact,
    source_root: Path | None = None,
    git_metadata_dirs: Sequence[Path] | None = None,
) -> LoadedHealthEvaluationArtifact:
    """Validate, stage, and atomically publish one fit-bound M4 evaluation."""

    try:
        authenticated_fit = _authenticate_fit_handle(fit_artifact)
        prepared = _prepare_health_evaluation_artifact(
            request,
            fit_artifact=authenticated_fit,
        )
        return _publish_streaming_health_evaluation(
            prepared,
            destination,
            fit_artifact=authenticated_fit,
            source_root=source_root,
            git_metadata_dirs=git_metadata_dirs,
        )
    except FileExistsError:
        raise
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactValidationError("M4 health evaluation publication failed") from error
