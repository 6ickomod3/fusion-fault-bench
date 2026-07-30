"""Deterministic, aggregate-only publication for the frozen M5 replay release."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

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
    strict_json_object_body,
    write_exclusive_file_at,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_AGGREGATE_COORDINATE_COUNT,
    M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256,
    M5_HEALTH_CONDITION_SELECTOR_COUNT,
    M5_HEALTH_CONDITION_SELECTOR_SET_SHA256,
    M5_HEALTH_FIT_RUN_SHA256,
    M5_HEALTH_HYPOTHESIS_COORDINATE_COUNT,
    M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256,
    M5_HEALTH_HYPOTHESIS_COORDINATES,
    M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT,
    M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256,
    M5_PERSISTENT_CONDITION_SELECTOR_COUNT,
    M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256,
    M5_PERSISTENT_HYPOTHESIS_COORDINATE_COUNT,
    M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256,
    M5_PERSISTENT_HYPOTHESIS_COORDINATES,
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    M5_SCIENTIFIC_SOURCE_ROLES,
    M5_TRACKED_AGGREGATE_TERMS,
    REPLAY_ARTIFACT_PATHS,
    REPLAY_CLUSTER_SENSITIVITY_FILE,
    REPLAY_CURATED_ARTIFACT_CONTRACT,
    REPLAY_DESCRIPTOR_AGGREGATES_FILE,
    REPLAY_FIGURE_RECORDS_FILE,
    REPLAY_HEALTH_AGGREGATES_FILE,
    REPLAY_INDEXED_PATHS,
    REPLAY_INTENT_FILE,
    REPLAY_MAX_ARTIFACT_BYTES,
    REPLAY_MAX_MEMBER_BYTES,
    REPLAY_MAX_NDJSON_RECORDS,
    REPLAY_MAX_RECORD_BYTES,
    REPLAY_PERSISTENT_AGGREGATES_FILE,
    REPLAY_PERSISTENT_CROSSOVERS_FILE,
    REPLAY_PROFILE_SUMMARY_FILE,
    REPLAY_RELEASE_INDEX_FILE,
    REPLAY_REPEAT_VERIFICATION_FILE,
    REPLAY_RUN_FILE,
    REPLAY_SOURCE_COMMITMENTS_FILE,
    REPLAY_SUCCESS_FILE,
    REPLAY_VALIDATION_FILE,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayExecutionResourceEvidenceV1,
    ReplayFigureRecordV1,
    ReplayHealthAggregateV1,
    ReplayPayloadFileEntryV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    ReplayProfileSummaryV1,
    ReplayReleaseIndexV1,
    ReplayRepeatVerificationV1,
    ReplaySourceMemberCommitmentV1,
    ReplaySuccessV1,
    ReplayValidationV1,
    replay_resource_evidence_sha256,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_HEALTH_PANEL_ID,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_PERSISTENT_PANEL_ID,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
    ReplayExperimentIdentityV1,
    expected_replay_identities,
    load_replay_intent,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RunRecordV1Alpha1
from fusion_fault_bench.replay_resources import (
    replay_environment_sha256,
    replay_logical_command_sha256,
)

_ARTIFACT_DOMAIN = b"fusion-fault-bench/replay-curated-artifact/v1\x00"
_READ_CHUNK_BYTES = 1024 * 1024
_ZERO_DIGEST = "0" * 64
_PRIVATE_PATH_PATTERN = re.compile(
    rb"(?:file:(?://)?|/(?:Users|home|private|tmp|Volumes)/|[A-Za-z]:[\\/])",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    rb"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    rb"(?:[\"'\s]*[:=][\"'\s]*|_)[A-Za-z0-9+/=_-]{8,}",
    re.IGNORECASE,
)
_SCENE_ID_PATTERN = re.compile(rb"(?:nuscenes:)?scene-[0-9]{4}", re.IGNORECASE)
_RAW_PAYLOAD_PATTERN = re.compile(
    rb"[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|pcd|las|laz|bin|tar|tgz|zip)(?:[\"'\s]|$)",
    re.IGNORECASE,
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "sample_token",
        "annotation_token",
        "instance_token",
        "calibrated_sensor_token",
        "ego_pose_token",
        "log_token",
        "filename",
        "file_name",
        "filepath",
        "file_path",
        "dataset_path",
        "dataset_root",
        "image_path",
        "pointcloud_path",
        "point_cloud_path",
        "timestamp",
        "timestamp_us",
        "pose",
        "translation",
        "rotation",
        "coordinates",
    }
)

_EXPECTED_IDENTITIES = expected_replay_identities()
_IDENTITY_RANK = {
    replay_experiment_identity_sha256(identity): index
    for index, identity in enumerate(_EXPECTED_IDENTITIES)
}
_PERSISTENT_IDENTITIES = tuple(
    identity for identity in _EXPECTED_IDENTITIES if identity.panel_id == M5_PERSISTENT_PANEL_ID
)
_HEALTH_IDENTITIES = tuple(
    identity for identity in _EXPECTED_IDENTITIES if identity.panel_id == M5_HEALTH_PANEL_ID
)


@dataclass(frozen=True, slots=True)
class ReplayCuratedArtifactWriteRequest:
    """Already-aggregated, privacy-safe M5 records and curation provenance."""

    profile_summary: ReplayProfileSummaryV1
    descriptor_aggregates: Sequence[ReplayDescriptorAggregateV1]
    persistent_aggregates: Sequence[ReplayPersistentAggregateV1]
    persistent_crossovers: Sequence[ReplayPersistentCrossoverV1]
    health_aggregates: Sequence[ReplayHealthAggregateV1]
    cluster_sensitivity: Sequence[ReplayClusterSensitivityV1]
    validation: ReplayValidationV1
    repeat_verification: ReplayRepeatVerificationV1
    figures: Sequence[ReplayFigureRecordV1]
    source_commitments: Sequence[ReplaySourceMemberCommitmentV1]
    run: RunRecordV1Alpha1


@dataclass(frozen=True, slots=True)
class LoadedReplayCuratedArtifact:
    """One strictly reloaded and cross-validated aggregate-only M5 artifact."""

    path: Path
    intent_bytes: bytes
    profile_summary: ReplayProfileSummaryV1
    descriptor_aggregates: tuple[ReplayDescriptorAggregateV1, ...]
    persistent_aggregates: tuple[ReplayPersistentAggregateV1, ...]
    persistent_crossovers: tuple[ReplayPersistentCrossoverV1, ...]
    health_aggregates: tuple[ReplayHealthAggregateV1, ...]
    cluster_sensitivity: tuple[ReplayClusterSensitivityV1, ...]
    validation: ReplayValidationV1
    repeat_verification: ReplayRepeatVerificationV1
    figures: tuple[ReplayFigureRecordV1, ...]
    source_commitments: tuple[ReplaySourceMemberCommitmentV1, ...]
    release_index: ReplayReleaseIndexV1
    run: RunRecordV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedReplayArtifact:
    files: Mapping[str, bytes]
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _TreeSnapshot:
    root_stat: os.stat_result
    entries: Mapping[str, os.stat_result]


def canonical_replay_ndjson_bytes(records: Sequence[BaseModel]) -> bytes:
    """Serialize a nonempty sequence as canonical, bounded NDJSON."""

    output = bytearray()
    if len(records) > REPLAY_MAX_NDJSON_RECORDS:
        raise ArtifactValidationError("replay NDJSON exceeds its record-count cap")
    for index, record in enumerate(records):
        line = canonical_json_bytes(record)
        if len(line) > REPLAY_MAX_RECORD_BYTES:
            raise ArtifactValidationError(
                f"replay NDJSON record {index} exceeds the 1 MiB record cap"
            )
        if len(output) + len(line) > REPLAY_MAX_MEMBER_BYTES:
            raise ArtifactValidationError("replay NDJSON exceeds its member cap")
        output.extend(line)
    if not output:
        raise ArtifactValidationError("replay NDJSON members must not be empty")
    return bytes(output)


def compute_replay_curated_artifact_digest(release_index_file_bytes: bytes) -> str:
    """Hash exact canonical release-index bytes using the frozen M5 domain."""

    preimage = b"".join(
        (
            _ARTIFACT_DOMAIN,
            len(release_index_file_bytes).to_bytes(8, "big"),
            release_index_file_bytes,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_digest(record: Any) -> str:
    identity = getattr(record, "identity", None)
    if not isinstance(identity, ReplayExperimentIdentityV1):
        raise ArtifactValidationError("replay record is missing its experiment identity")
    return replay_experiment_identity_sha256(identity)


def _first_identity_order(records: Sequence[Any]) -> tuple[ReplayExperimentIdentityV1, ...]:
    seen: set[str] = set()
    identities: list[ReplayExperimentIdentityV1] = []
    for record in records:
        digest = _identity_digest(record)
        if digest not in seen:
            seen.add(digest)
            identities.append(record.identity)
    return tuple(identities)


def _ordered_records(
    request: ReplayCuratedArtifactWriteRequest,
) -> tuple[
    tuple[ReplayDescriptorAggregateV1, ...],
    tuple[ReplayPersistentAggregateV1, ...],
    tuple[ReplayPersistentCrossoverV1, ...],
    tuple[ReplayHealthAggregateV1, ...],
    tuple[ReplayClusterSensitivityV1, ...],
    tuple[ReplayFigureRecordV1, ...],
    tuple[ReplaySourceMemberCommitmentV1, ...],
]:
    try:
        descriptors = tuple(
            sorted(
                request.descriptor_aggregates,
                key=lambda record: (
                    record.population,
                    record.descriptor_id,
                    record.statistic,
                    record.category_label or "",
                ),
            )
        )
        persistent = tuple(
            sorted(
                request.persistent_aggregates,
                key=lambda record: (
                    _IDENTITY_RANK[_identity_digest(record)],
                    record.condition_id,
                    record.condition_selector,
                    record.method_id,
                    record.metric_id,
                    record.window,
                    record.result_id,
                ),
            )
        )
        crossovers = tuple(
            sorted(
                request.persistent_crossovers,
                key=lambda record: (
                    _IDENTITY_RANK[_identity_digest(record)],
                    record.direction,
                    record.crossover_id,
                ),
            )
        )
        health = tuple(
            sorted(
                request.health_aggregates,
                key=lambda record: (
                    _IDENTITY_RANK[_identity_digest(record)],
                    record.condition_id,
                    record.condition_selector,
                    record.method_id,
                    record.metric_id,
                    record.window,
                    record.result_id,
                ),
            )
        )
        sensitivity = tuple(
            sorted(
                request.cluster_sensitivity,
                key=lambda record: (
                    _IDENTITY_RANK[_identity_digest(record)],
                    record.source_result_id,
                    record.cluster_kind,
                    record.cluster_id,
                    record.sensitivity_id,
                ),
            )
        )
        figures = tuple(
            sorted(
                request.figures,
                key=lambda record: (
                    _IDENTITY_RANK[_identity_digest(record)],
                    record.source_result_id,
                    record.figure_kind,
                    record.figure_id,
                ),
            )
        )
        commitments = tuple(request.source_commitments)
    except KeyError as error:
        raise ArtifactValidationError(
            "replay record cannot be ordered from the frozen identity set"
        ) from error
    return (
        descriptors,
        persistent,
        crossovers,
        health,
        sensitivity,
        figures,
        commitments,
    )


def _expected_run_id(run: RunRecordV1Alpha1) -> str:
    return derive_run_id(
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        git_revision=run.git_revision,
        lockfile_sha256=run.lockfile_sha256,
        package_version=run.package_version,
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
    )


def _validate_run(
    run: RunRecordV1Alpha1,
    *,
    artifact_sha256: str | None = None,
) -> None:
    if run.manifest_sha256 != M5_REPLAY_INTENT_SHA256:
        raise ArtifactValidationError("replay run does not bind the frozen intent")
    if run.run_id != _expected_run_id(run):
        raise ArtifactValidationError("replay run_id is invalid")
    if run.source_dirty or run.status != "succeeded":
        raise ArtifactValidationError("replay release requires a clean successful run")
    try:
        replay_logical_command_sha256(tuple(run.command))
    except ValueError as error:
        raise ArtifactValidationError("replay run command is not the exact M5 authority") from error
    if artifact_sha256 is None:
        if run.artifact_sha256 not in {_ZERO_DIGEST, None}:
            raise ArtifactValidationError(
                "unfinalized replay run has an unexpected artifact digest"
            )
    elif run.artifact_sha256 != artifact_sha256:
        raise ArtifactValidationError("replay run artifact identity is invalid")


def _validate_frozen_intent_bytes(intent_bytes: bytes) -> None:
    if (
        len(intent_bytes) > REPLAY_MAX_MEMBER_BYTES
        or _sha256_bytes(intent_bytes) != M5_REPLAY_INTENT_BYTE_SHA256
    ):
        raise ArtifactValidationError("replay artifact does not contain the byte-frozen intent")
    try:
        value = cast(object, json.loads(intent_bytes))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("frozen replay intent is invalid JSON") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError("replay intent canonical identity is invalid")
    intent = cast("dict[str, Any]", value)
    if sha256_digest(intent) != M5_REPLAY_INTENT_SHA256:
        raise ArtifactValidationError("replay intent canonical identity is invalid")


def _require_unique(values: Sequence[object], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ArtifactValidationError(f"replay {label} contains duplicate logical keys")


def _condition_selector_set_sha256(
    panel_id: str,
    records: Sequence[ReplayPersistentAggregateV1 | ReplayHealthAggregateV1],
) -> str:
    return sha256_digest(
        {
            "panel_id": panel_id,
            "condition_selectors": sorted({record.condition_selector for record in records}),
        }
    )


def _persistent_aggregate_coordinate_set_sha256(
    records: Sequence[ReplayPersistentAggregateV1],
) -> str:
    coordinates = {
        (
            record.condition_selector,
            record.method_id,
            record.metric_id,
            record.window,
            record.unit,
            record.aggregation,
        )
        for record in records
    }
    return sha256_digest(
        {
            "panel_id": M5_PERSISTENT_PANEL_ID,
            "aggregate_coordinates": sorted(coordinates),
        }
    )


def _health_aggregate_coordinate_set_sha256(
    records: Sequence[ReplayHealthAggregateV1],
) -> str:
    coordinates = {
        (
            record.condition_selector,
            record.method_id,
            record.metric_id,
            record.window,
            record.unit,
            record.aggregation,
        )
        for record in records
    }
    return sha256_digest(
        {
            "panel_id": M5_HEALTH_PANEL_ID,
            "aggregate_coordinates": sorted(coordinates),
        }
    )


def _expected_crossover_keys() -> tuple[tuple[str, str, str, float], ...]:
    specifications = (
        (0, "negative", "m", 4.0),
        (0, "positive", "m", 4.0),
        (1, "increase", "std-scale", 4.0),
        (2, "increase", "std-scale", 4.0),
        (3, "negative", "m", 4.0),
        (3, "positive", "m", 4.0),
        (4, "negative", "rad", 0.08),
        (4, "positive", "rad", 0.08),
        (5, "negative", "s", 0.8),
        (5, "positive", "s", 0.8),
    )
    return tuple(
        (
            replay_experiment_identity_sha256(_PERSISTENT_IDENTITIES[index]),
            direction,
            unit,
            tested_maximum,
        )
        for index, direction, unit, tested_maximum in specifications
    )


def _stable_coordinate_id(prefix: str, coordinate: dict[str, object]) -> str:
    return f"{prefix}-{sha256_digest(coordinate)}"


def _validate_deterministic_ids(
    *,
    persistent: Sequence[ReplayPersistentAggregateV1],
    crossovers: Sequence[ReplayPersistentCrossoverV1],
    health: Sequence[ReplayHealthAggregateV1],
    sensitivity: Sequence[ReplayClusterSensitivityV1],
) -> None:
    for record in (*persistent, *health):
        expected = _stable_coordinate_id(
            "replay-result",
            {
                "schema": "ffb.replay-result-coordinate/v1",
                "panel_id": record.identity.panel_id,
                "replay_experiment_identity_sha256": record.replay_identity_sha256,
                "condition_selector": record.condition_selector,
                "method_id": record.method_id,
                "metric_id": record.metric_id,
                "window": record.window,
            },
        )
        if record.result_id != expected:
            raise ArtifactValidationError("panel result identifier is not coordinate-derived")
    for record in crossovers:
        expected = _stable_coordinate_id(
            "replay-crossover",
            {
                "schema": "ffb.replay-crossover-coordinate/v1",
                "replay_experiment_identity_sha256": record.replay_identity_sha256,
                "direction": record.direction,
                "severity_unit": record.severity_unit,
            },
        )
        if record.crossover_id != expected:
            raise ArtifactValidationError("crossover identifier is not coordinate-derived")
    for record in sensitivity:
        expected = _stable_coordinate_id(
            "replay-sensitivity",
            {
                "schema": "ffb.replay-sensitivity-coordinate/v1",
                "source_result_id": record.source_result_id,
                "cluster_kind": record.cluster_kind,
                "cluster_id": record.cluster_id,
            },
        )
        if record.sensitivity_id != expected:
            raise ArtifactValidationError("sensitivity identifier is not coordinate-derived")


def _validate_health_hypothesis_coordinates(
    persistent: Sequence[ReplayPersistentAggregateV1],
    health: Sequence[ReplayHealthAggregateV1],
) -> None:
    if any(
        record.hypothesis_id is not None and record.hypothesis_id.startswith("h5-b")
        for record in persistent
    ):
        raise ArtifactValidationError("M5-B hypotheses cannot bind persistent-panel rows")
    if any(
        record.hypothesis_id is not None and record.hypothesis_id.startswith("h5-a")
        for record in health
    ):
        raise ArtifactValidationError("M5-A hypotheses cannot bind health-panel rows")
    actual_persistent = tuple(
        (
            record.hypothesis_id,
            record.condition_selector,
            record.method_id,
            record.metric_id,
            record.window,
            record.unit,
            record.aggregation,
            record.inference_role,
            record.expected_direction,
        )
        for record in persistent
        if record.hypothesis_id is not None
    )
    if len(actual_persistent) != M5_PERSISTENT_HYPOTHESIS_COORDINATE_COUNT or set(
        actual_persistent
    ) != set(M5_PERSISTENT_HYPOTHESIS_COORDINATES):
        raise ArtifactValidationError("persistent hypothesis coordinate map is incomplete")
    actual = tuple(
        (
            record.hypothesis_id,
            record.condition_selector,
            record.method_id,
            record.metric_id,
            record.window,
            record.unit,
            record.inference_role,
            record.expected_direction,
        )
        for record in health
        if record.hypothesis_id is not None
    )
    if len(actual) != M5_HEALTH_HYPOTHESIS_COORDINATE_COUNT or set(actual) != set(
        M5_HEALTH_HYPOTHESIS_COORDINATES
    ):
        raise ArtifactValidationError("health hypothesis coordinate map is incomplete")
    if any(
        record.hypothesis_id is not None and record.aggregation != "equal-scene-mean"
        for record in health
    ):
        raise ArtifactValidationError("health hypothesis rows require equal-scene aggregation")


def _validate_health_not_applicable_semantics(
    records: Sequence[ReplayHealthAggregateV1],
) -> None:
    structural_metrics = {
        "gap-vs-fault-target-drop",
        "gap-vs-frame-oracle",
        "frame-oracle-recoverable-loss-fraction",
    }
    for record in records:
        structural_common_mode = (
            record.condition_id == "replay-common-mode-x" and record.metric_id in structural_metrics
        )
        recovery = record.metric_id == "frame-oracle-recoverable-loss-fraction"
        if structural_common_mode:
            valid = (
                record.status == "not-applicable"
                and record.applicability_basis == "structural-unavailable"
                and record.recovery_support_compatible_scene_count is None
            )
        elif recovery:
            count = record.recovery_support_compatible_scene_count
            valid = count is not None and (
                (
                    count == 10
                    and record.status != "not-applicable"
                    and record.applicability_basis == "applicable"
                )
                or (
                    count < 10
                    and record.status == "not-applicable"
                    and record.applicability_basis == "support-incompatible"
                )
            )
        else:
            valid = (
                record.status != "not-applicable"
                and record.applicability_basis == "applicable"
                and record.recovery_support_compatible_scene_count is None
            )
        if record.status == "not-applicable":
            valid = valid and (
                record.hypothesis_id is None
                and record.inference_role == "descriptive"
                and record.positive_scene_count is None
                and record.zero_scene_count is None
                and record.negative_scene_count is None
                and record.undefined_scene_count is None
            )
        if not valid:
            raise ArtifactValidationError(
                "health applicability status disagrees with released support evidence"
            )


def _classify_directional_aggregate(
    record: ReplayPersistentAggregateV1 | ReplayHealthAggregateV1,
    sensitivity: Sequence[ReplayClusterSensitivityV1],
    *,
    distinct_log_group_count: int,
) -> str:
    if record.status != "ok":
        return "undefined"
    assert record.estimate is not None
    assert record.interval_lower is not None
    assert record.interval_upper is not None
    expected_positive = record.expected_direction == "positive"
    point_matches = record.estimate > 0.0 if expected_positive else record.estimate < 0.0
    if not point_matches:
        return "non-persistent"
    interval_matches = (
        record.interval_lower > 0.0 if expected_positive else record.interval_upper < 0.0
    )
    matching_scene_count = (
        record.positive_scene_count if expected_positive else record.negative_scene_count
    )
    assert matching_scene_count is not None
    sensitivity_matches = all(
        row.status == "ok"
        and row.estimate is not None
        and (row.estimate > 0.0 if expected_positive else row.estimate < 0.0)
        for row in sensitivity
    )
    robust = (
        interval_matches
        and matching_scene_count >= 8
        and sensitivity_matches
        and distinct_log_group_count >= 2
    )
    return "robustly-persistent" if robust else "directionally-consistent"


def _validate_sensitivity_coverage(
    *,
    profile_summary: ReplayProfileSummaryV1,
    aggregates: Sequence[ReplayPersistentAggregateV1 | ReplayHealthAggregateV1],
    sensitivity: Sequence[ReplayClusterSensitivityV1],
) -> None:
    by_source: dict[str, list[ReplayClusterSensitivityV1]] = {}
    for record in sensitivity:
        by_source.setdefault(record.source_result_id, []).append(record)
    expected_coordinates = {
        *(("leave-one-scene-out", f"scene-ordinal:{index:02d}") for index in range(10)),
        *(
            ("leave-one-log-group-out", f"log-group:{index:02d}")
            for index in range(profile_summary.distinct_log_group_count)
        ),
    }
    aggregate_by_id = {record.result_id: record for record in aggregates}
    if set(by_source) - set(aggregate_by_id):
        raise ArtifactValidationError("cluster sensitivity names an unknown aggregate")
    for source_result_id, rows in by_source.items():
        coordinates = {(row.cluster_kind, row.cluster_id) for row in rows}
        if coordinates != expected_coordinates or len(rows) != len(expected_coordinates):
            raise ArtifactValidationError(
                f"cluster sensitivity coverage is incomplete for {source_result_id}"
            )
    for record in aggregates:
        rows = tuple(by_source.get(record.result_id, ()))
        requires_sensitivity = record.inference_role in {
            "primary-directional",
            "nonpositive-control",
        }
        if rows and not requires_sensitivity:
            raise ArtifactValidationError(
                f"descriptive aggregate carries cluster sensitivity: {record.result_id}"
            )
        if requires_sensitivity and not rows:
            raise ArtifactValidationError(
                f"primary aggregate lacks cluster sensitivity: {record.result_id}"
            )
        if any(row.unit != record.unit for row in rows):
            raise ArtifactValidationError("cluster sensitivity unit disagrees with its aggregate")
        if record.inference_role == "primary-directional":
            expected_label = _classify_directional_aggregate(
                record,
                rows,
                distinct_log_group_count=profile_summary.distinct_log_group_count,
            )
            if record.persistence_label != expected_label:
                raise ArtifactValidationError(
                    f"persistence label is not derived from released evidence: {record.result_id}"
                )


def _validate_record_links(
    *,
    run_id: str,
    runtime_environment_sha256: str,
    logical_command_sha256: str,
    profile_summary: ReplayProfileSummaryV1,
    descriptors: Sequence[ReplayDescriptorAggregateV1],
    persistent: Sequence[ReplayPersistentAggregateV1],
    crossovers: Sequence[ReplayPersistentCrossoverV1],
    health: Sequence[ReplayHealthAggregateV1],
    sensitivity: Sequence[ReplayClusterSensitivityV1],
    validation: ReplayValidationV1,
    repeat: ReplayRepeatVerificationV1,
    figures: Sequence[ReplayFigureRecordV1],
    commitments: Sequence[ReplaySourceMemberCommitmentV1],
) -> None:
    globally_bound: tuple[Any, ...] = (
        profile_summary,
        *descriptors,
        *persistent,
        *crossovers,
        *health,
        *sensitivity,
        validation,
        repeat,
        *figures,
        *commitments,
    )
    if any(record.run_id != run_id for record in globally_bound):
        raise ArtifactValidationError("replay record run_id binding is inconsistent")

    if _first_identity_order(persistent) != _PERSISTENT_IDENTITIES:
        raise ArtifactValidationError("persistent panel identity coverage or order is incomplete")
    if _first_identity_order(health) != _HEALTH_IDENTITIES:
        raise ArtifactValidationError("health panel identity coverage or order is incomplete")
    actual_crossover_keys = tuple(
        (
            record.replay_identity_sha256,
            record.direction,
            record.severity_unit,
            record.tested_maximum,
        )
        for record in crossovers
    )
    if actual_crossover_keys != _expected_crossover_keys():
        raise ArtifactValidationError("persistent crossover coordinate coverage is incomplete")
    if _first_identity_order(figures) != _EXPECTED_IDENTITIES:
        raise ArtifactValidationError("figure-record identity coverage is incomplete")
    persistent_selectors = {record.condition_selector for record in persistent}
    health_selectors = {record.condition_selector for record in health}
    if (
        len(persistent_selectors) != M5_PERSISTENT_CONDITION_SELECTOR_COUNT
        or _condition_selector_set_sha256(M5_PERSISTENT_PANEL_ID, persistent)
        != M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256
    ):
        raise ArtifactValidationError("persistent condition grid is incomplete")
    if (
        len(persistent) != M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT
        or _persistent_aggregate_coordinate_set_sha256(persistent)
        != M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256
    ):
        raise ArtifactValidationError("persistent aggregate coordinate matrix is incomplete")
    if (
        len(health_selectors) != M5_HEALTH_CONDITION_SELECTOR_COUNT
        or _condition_selector_set_sha256(M5_HEALTH_PANEL_ID, health)
        != M5_HEALTH_CONDITION_SELECTOR_SET_SHA256
    ):
        raise ArtifactValidationError("health condition grid is incomplete")
    if (
        len(health) != M5_HEALTH_AGGREGATE_COORDINATE_COUNT
        or _health_aggregate_coordinate_set_sha256(health)
        != M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256
    ):
        raise ArtifactValidationError("health aggregate coordinate matrix is incomplete")
    _validate_health_hypothesis_coordinates(persistent, health)
    _validate_health_not_applicable_semantics(health)
    _validate_deterministic_ids(
        persistent=persistent,
        crossovers=crossovers,
        health=health,
        sensitivity=sensitivity,
    )

    _require_unique(
        [
            (
                record.population,
                record.descriptor_id,
                record.statistic,
                record.category_label,
            )
            for record in descriptors
        ],
        label="descriptor aggregates",
    )
    _require_unique(
        [
            (
                record.replay_identity_sha256,
                record.condition_id,
                record.condition_selector,
                record.method_id,
                record.metric_id,
                record.window,
            )
            for record in (*persistent, *health)
        ],
        label="panel aggregates",
    )
    _require_unique(
        [record.result_id for record in (*persistent, *health)],
        label="panel result identifiers",
    )
    _require_unique(
        [record.crossover_id for record in crossovers],
        label="crossover identifiers",
    )
    _require_unique(
        [
            (record.source_result_id, record.cluster_kind, record.cluster_id)
            for record in sensitivity
        ],
        label="cluster-sensitivity coordinates",
    )
    _require_unique(
        [record.sensitivity_id for record in sensitivity],
        label="cluster-sensitivity identifiers",
    )
    _require_unique(
        [record.figure_id for record in figures],
        label="figure identifiers",
    )
    commitment_roles = tuple(record.relative_role for record in commitments)
    if commitment_roles != M5_SCIENTIFIC_SOURCE_ROLES:
        raise ArtifactValidationError(
            "replay source commitment roles do not match the frozen canonical set"
        )

    sources: dict[str, list[tuple[ReplayExperimentIdentityV1 | None, str]]] = {}
    for record in descriptors:
        sources.setdefault(record.descriptor_id, []).append((None, sha256_digest(record)))
    for record in (*persistent, *health):
        if record.result_id in sources:
            raise ArtifactValidationError("replay source-result identifiers are not global")
        sources[record.result_id] = [(record.identity, sha256_digest(record))]
    for record in crossovers:
        if record.crossover_id in sources:
            raise ArtifactValidationError("replay source-result identifiers are not global")
        sources[record.crossover_id] = [(record.identity, sha256_digest(record))]

    for record in sensitivity:
        candidates = sources.get(record.source_result_id, [])
        if (
            len(candidates) != 1
            or candidates[0][0] is None
            or candidates[0][0] != record.identity
            or candidates[0][1] != record.source_record_sha256
        ):
            raise ArtifactValidationError("cluster sensitivity has an invalid source binding")
        if record.sensitivity_id in sources:
            raise ArtifactValidationError("replay source-result identifiers are not global")
        sources[record.sensitivity_id] = [(record.identity, sha256_digest(record))]

    for record in figures:
        candidates = sources.get(record.source_result_id, [])
        matches = tuple(
            source
            for source in candidates
            if source[1] == record.source_record_sha256
            and (source[0] is None or source[0] == record.identity)
        )
        if len(matches) != 1:
            raise ArtifactValidationError("figure record has an invalid aggregate binding")
        source = matches[0]
        if source[0] is not None and source[0] != record.identity:
            raise ArtifactValidationError("figure identity disagrees with its aggregate")
        if source[0] is None and record.figure_kind != "descriptor-comparison":
            raise ArtifactValidationError(
                "only descriptor-comparison figures may bind descriptor aggregates"
            )

    _validate_sensitivity_coverage(
        profile_summary=profile_summary,
        aggregates=(*persistent, *health),
        sensitivity=sensitivity,
    )
    if not descriptors:
        raise ArtifactValidationError("replay descriptor aggregates must not be empty")
    if not commitments or any(not record.equal for record in commitments):
        raise ArtifactValidationError("replay release requires equal source commitments")
    commitment_bytes = canonical_replay_ndjson_bytes(commitments)
    if (
        repeat.source_member_commitments_sha256 != _sha256_bytes(commitment_bytes)
        or repeat.scientific_member_count != len(M5_SCIENTIFIC_SOURCE_ROLES)
        or repeat.mismatch_count != 0
        or not repeat.all_checks_passed
    ):
        raise ArtifactValidationError("repeat verification disagrees with source commitments")
    if not validation.all_checks_passed:
        raise ArtifactValidationError("replay validation did not pass every release gate")
    if tuple(check.check_id for check in validation.checks) != M5_RELEASE_VALIDATION_CHECK_IDS:
        raise ArtifactValidationError("replay validation check order is invalid")
    try:
        resources = tuple(
            ReplayExecutionResourceEvidenceV1.model_validate(
                record.model_dump(mode="python", by_alias=True)
            )
            for record in profile_summary.resource_evidence
        )
    except ValidationError as error:
        raise ArtifactValidationError(
            "external resource evidence violates its strict contract"
        ) from error
    if (
        len(resources) != 2
        or tuple(record.run_label for record in resources) != ("primary", "repeat")
        or any(
            record.run_id != run_id
            or record.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or record.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            for record in resources
        )
        or profile_summary.elapsed_seconds != max(record.elapsed_seconds for record in resources)
        or profile_summary.peak_rss_bytes != max(record.peak_rss_bytes for record in resources)
        or tuple(record.local_run_sha256 for record in resources)
        != (repeat.primary_run_sha256, repeat.repeat_run_sha256)
        or tuple(record.local_artifact_sha256 for record in resources)
        != (
            repeat.primary_local_artifact_sha256,
            repeat.repeat_local_artifact_sha256,
        )
        or resources[0].local_run_sha256 == resources[1].local_run_sha256
        or any(record.environment_sha256 != runtime_environment_sha256 for record in resources)
        or any(record.logical_command_sha256 != logical_command_sha256 for record in resources)
    ):
        raise ArtifactValidationError(
            "external resource evidence is not bound to repeat and runtime provenance"
        )
    resource_check = next(
        check for check in validation.checks if check.check_id == "cpu-and-memory-caps"
    )
    if resource_check.evidence_sha256 != replay_resource_evidence_sha256(resources):
        raise ArtifactValidationError(
            "CPU-and-memory validation does not bind canonical resource evidence"
        )


def validate_replay_curated_bundle(
    *,
    intent_bytes: bytes,
    profile_summary: ReplayProfileSummaryV1,
    descriptor_aggregates: Sequence[ReplayDescriptorAggregateV1],
    persistent_aggregates: Sequence[ReplayPersistentAggregateV1],
    persistent_crossovers: Sequence[ReplayPersistentCrossoverV1],
    health_aggregates: Sequence[ReplayHealthAggregateV1],
    cluster_sensitivity: Sequence[ReplayClusterSensitivityV1],
    validation: ReplayValidationV1,
    repeat_verification: ReplayRepeatVerificationV1,
    figures: Sequence[ReplayFigureRecordV1],
    source_commitments: Sequence[ReplaySourceMemberCommitmentV1],
    run: RunRecordV1Alpha1,
    artifact_sha256: str | None = None,
) -> None:
    """Cross-validate aggregate records, references, repeat evidence, and provenance."""

    _validate_frozen_intent_bytes(intent_bytes)
    _validate_run(run, artifact_sha256=artifact_sha256)
    _validate_record_links(
        run_id=run.run_id,
        runtime_environment_sha256=replay_environment_sha256(run.environment),
        logical_command_sha256=replay_logical_command_sha256(tuple(run.command)),
        profile_summary=profile_summary,
        descriptors=descriptor_aggregates,
        persistent=persistent_aggregates,
        crossovers=persistent_crossovers,
        health=health_aggregates,
        sensitivity=cluster_sensitivity,
        validation=validation,
        repeat=repeat_verification,
        figures=figures,
        commitments=source_commitments,
    )


def _scan_public_value(value: object) -> None:
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        for key, child in mapping.items():
            if not isinstance(key, str):
                raise ArtifactValidationError("replay candidate contains a non-string JSON key")
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_PUBLIC_KEYS:
                raise ArtifactValidationError("replay candidate contains private source fields")
            _scan_public_value(child)
    elif isinstance(value, list):
        for child in cast("list[object]", value):
            _scan_public_value(child)
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        if (
            _PRIVATE_PATH_PATTERN.search(encoded)
            or _SECRET_PATTERN.search(encoded)
            or _SCENE_ID_PATTERN.search(encoded)
            or _RAW_PAYLOAD_PATTERN.search(encoded)
            or b"interview/" in encoded.lower()
        ):
            raise ArtifactValidationError("replay candidate contains private source material")


def _json_values_for_candidate(path: str, value: bytes) -> tuple[object, ...]:
    lines = value.splitlines(keepends=True) if path.endswith(".ndjson") else [value]
    if path.endswith(".ndjson") and not lines:
        raise ArtifactValidationError("replay candidate NDJSON must not be empty")
    parsed: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        if len(line) > REPLAY_MAX_RECORD_BYTES:
            raise ArtifactValidationError("replay candidate record exceeds its cap")
        if not line.endswith(b"\n") or b"\r" in line:
            raise ArtifactValidationError("replay candidate is not LF-delimited JSON")
        try:
            item = cast(object, json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactValidationError(
                f"replay candidate contains invalid JSON at line {line_number}"
            ) from error
        if not isinstance(item, dict):
            raise ArtifactValidationError("replay candidate records must be JSON objects")
        parsed.append(cast("dict[str, Any]", item))
    return tuple(parsed)


def validate_replay_candidate_bytes(files: Mapping[str, bytes]) -> None:
    """Reject non-allowlisted, oversized, raw, private, or credential-like bytes."""

    if set(files) != set(REPLAY_ARTIFACT_PATHS):
        raise ArtifactValidationError("replay candidate file allowlist mismatch")
    if any(len(value) > REPLAY_MAX_MEMBER_BYTES for value in files.values()):
        raise ArtifactValidationError("replay candidate member exceeds its byte cap")
    if sum(len(value) for value in files.values()) > REPLAY_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("replay candidate exceeds the 50 MiB cap")
    for path, value in files.items():
        if path == REPLAY_INTENT_FILE:
            _validate_frozen_intent_bytes(value)
            continue
        if (
            _PRIVATE_PATH_PATTERN.search(value)
            or _SECRET_PATTERN.search(value)
            or _SCENE_ID_PATTERN.search(value)
            or _RAW_PAYLOAD_PATTERN.search(value)
            or b"interview/" in value.lower()
        ):
            raise ArtifactValidationError(f"replay candidate privacy scan failed for {path}")
        for parsed in _json_values_for_candidate(path, value):
            _scan_public_value(parsed)


def _finalize_run(run: RunRecordV1Alpha1, artifact_sha256: str) -> RunRecordV1Alpha1:
    value = run.model_dump(mode="python", by_alias=True)
    value["artifact_sha256"] = artifact_sha256
    return RunRecordV1Alpha1.model_validate(value)


def _prepare_replay_artifact(
    request: ReplayCuratedArtifactWriteRequest,
    *,
    source_root: Path,
) -> _PreparedReplayArtifact:
    loaded_intent = load_replay_intent(source_root=source_root)
    intent_bytes = loaded_intent.path.read_bytes()
    (
        descriptors,
        persistent,
        crossovers,
        health,
        sensitivity,
        figures,
        commitments,
    ) = _ordered_records(request)
    validate_replay_curated_bundle(
        intent_bytes=intent_bytes,
        profile_summary=request.profile_summary,
        descriptor_aggregates=descriptors,
        persistent_aggregates=persistent,
        persistent_crossovers=crossovers,
        health_aggregates=health,
        cluster_sensitivity=sensitivity,
        validation=request.validation,
        repeat_verification=request.repeat_verification,
        figures=figures,
        source_commitments=commitments,
        run=request.run,
    )
    indexed_files: dict[str, bytes] = {
        REPLAY_INTENT_FILE: intent_bytes,
        REPLAY_PROFILE_SUMMARY_FILE: canonical_json_bytes(request.profile_summary),
        REPLAY_DESCRIPTOR_AGGREGATES_FILE: canonical_replay_ndjson_bytes(descriptors),
        REPLAY_PERSISTENT_AGGREGATES_FILE: canonical_replay_ndjson_bytes(persistent),
        REPLAY_PERSISTENT_CROSSOVERS_FILE: canonical_replay_ndjson_bytes(crossovers),
        REPLAY_HEALTH_AGGREGATES_FILE: canonical_replay_ndjson_bytes(health),
        REPLAY_CLUSTER_SENSITIVITY_FILE: canonical_replay_ndjson_bytes(sensitivity),
        REPLAY_VALIDATION_FILE: canonical_json_bytes(request.validation),
        REPLAY_REPEAT_VERIFICATION_FILE: canonical_json_bytes(request.repeat_verification),
        REPLAY_FIGURE_RECORDS_FILE: canonical_replay_ndjson_bytes(figures),
        REPLAY_SOURCE_COMMITMENTS_FILE: canonical_replay_ndjson_bytes(commitments),
    }
    record_counts = {
        REPLAY_DESCRIPTOR_AGGREGATES_FILE: len(descriptors),
        REPLAY_PERSISTENT_AGGREGATES_FILE: len(persistent),
        REPLAY_PERSISTENT_CROSSOVERS_FILE: len(crossovers),
        REPLAY_HEALTH_AGGREGATES_FILE: len(health),
        REPLAY_CLUSTER_SENSITIVITY_FILE: len(sensitivity),
        REPLAY_FIGURE_RECORDS_FILE: len(figures),
        REPLAY_SOURCE_COMMITMENTS_FILE: len(commitments),
    }
    release_index = ReplayReleaseIndexV1(
        schema="ffb.replay-release-index/v1",
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
        release_id=M5_REPLAY_RELEASE_ID,
        run_id=request.run.run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        identities=_EXPECTED_IDENTITIES,
        persistent_source_matrix_sha256=M5_PERSISTENT_MATRIX_SHA256,
        persistent_condition_selector_count=M5_PERSISTENT_CONDITION_SELECTOR_COUNT,
        persistent_condition_selector_set_sha256=(M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256),
        persistent_aggregate_coordinate_count=M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT,
        persistent_aggregate_coordinate_set_sha256=(M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256),
        persistent_hypothesis_coordinate_count=(M5_PERSISTENT_HYPOTHESIS_COORDINATE_COUNT),
        persistent_hypothesis_coordinate_set_sha256=(
            M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256
        ),
        health_fit_artifact_sha256=M5_HEALTH_FIT_SHA256,
        health_fit_run_sha256=M5_HEALTH_FIT_RUN_SHA256,
        health_condition_selector_count=M5_HEALTH_CONDITION_SELECTOR_COUNT,
        health_condition_selector_set_sha256=M5_HEALTH_CONDITION_SELECTOR_SET_SHA256,
        health_aggregate_coordinate_count=M5_HEALTH_AGGREGATE_COORDINATE_COUNT,
        health_aggregate_coordinate_set_sha256=M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256,
        health_hypothesis_coordinate_count=M5_HEALTH_HYPOTHESIS_COORDINATE_COUNT,
        health_hypothesis_coordinate_set_sha256=(M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256),
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        files=tuple(
            ReplayPayloadFileEntryV1(
                path=path,  # type: ignore[arg-type]
                byte_length=len(indexed_files[path]),
                sha256=_sha256_bytes(indexed_files[path]),
                record_count=record_counts.get(path),
                replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            )
            for path in REPLAY_INDEXED_PATHS
        ),
    )
    index_bytes = canonical_json_bytes(release_index)
    artifact_sha256 = compute_replay_curated_artifact_digest(index_bytes)
    run = _finalize_run(request.run, artifact_sha256)
    validate_replay_curated_bundle(
        intent_bytes=intent_bytes,
        profile_summary=request.profile_summary,
        descriptor_aggregates=descriptors,
        persistent_aggregates=persistent,
        persistent_crossovers=crossovers,
        health_aggregates=health,
        cluster_sensitivity=sensitivity,
        validation=request.validation,
        repeat_verification=request.repeat_verification,
        figures=figures,
        source_commitments=commitments,
        run=run,
        artifact_sha256=artifact_sha256,
    )
    run_bytes = canonical_json_bytes(run)
    run_sha256 = compute_run_record_digest(run_bytes)
    success = ReplaySuccessV1(
        schema="ffb.replay-success/v1",
        release_artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
    )
    files = {
        **indexed_files,
        REPLAY_RELEASE_INDEX_FILE: index_bytes,
        REPLAY_RUN_FILE: run_bytes,
        REPLAY_SUCCESS_FILE: canonical_json_bytes(success),
    }
    validate_replay_candidate_bytes(files)
    return _PreparedReplayArtifact(
        files=files,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
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
            metadata = os.lstat(current)
        except OSError as error:
            raise ArtifactValidationError("replay artifact path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactValidationError("replay artifact path contains a symlink")


def _require_safe_tree(root: Path) -> _TreeSnapshot:
    absolute = absolute_artifact_path(root)
    _reject_root_symlink_components(absolute)
    try:
        root_stat = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError("replay artifact directory is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ArtifactValidationError("replay artifact root must be a real directory")
    entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(absolute) as iterator:
            for entry in iterator:
                metadata = entry.stat(follow_symlinks=False)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise ArtifactValidationError(
                        "replay artifact members must be private regular files"
                    )
                entries[entry.name] = metadata
    except ArtifactValidationError:
        raise
    except OSError as error:
        raise ArtifactValidationError("replay artifact members cannot be inspected") from error
    if set(entries) != set(REPLAY_ARTIFACT_PATHS):
        raise ArtifactValidationError("replay artifact file allowlist mismatch")
    if any(entry.st_size > REPLAY_MAX_MEMBER_BYTES for entry in entries.values()):
        raise ArtifactValidationError("replay artifact member exceeds its cap")
    if sum(entry.st_size for entry in entries.values()) > REPLAY_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("replay artifact exceeds the 50 MiB tree cap")
    return _TreeSnapshot(root_stat=root_stat, entries=entries)


def _open_regular_member(
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
) -> int:
    descriptor = os.open(
        root / name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or _stat_fingerprint(metadata) != _stat_fingerprint(expected_stat)
        ):
            raise ArtifactValidationError("replay artifact member changed during validation")
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
        chunks: list[bytes] = []
        remaining = byte_cap + 1
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) > byte_cap or len(value) != expected_stat.st_size:
            raise ArtifactValidationError(f"replay artifact member exceeds its cap: {name}")
        return value
    finally:
        os.close(descriptor)


def _load_model[ModelT: BaseModel](
    data: bytes,
    *,
    label: str,
    validate: Callable[[bytes], ModelT],
) -> ModelT:
    try:
        body = strict_json_object_body(data, label=label)
        model = validate(body)
    except (ArtifactValidationError, ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"{label} violates strict canonical JSON") from error
    if canonical_json_bytes(model) != data:
        raise ArtifactValidationError(f"{label} is not canonical JSON")
    return model


def _load_ndjson[ModelT: BaseModel](
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
    validate: Callable[[bytes], ModelT],
) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    descriptor = _open_regular_member(root, name, expected_stat=expected_stat)
    with os.fdopen(descriptor, "rb") as stream:
        while True:
            line = stream.readline(REPLAY_MAX_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > REPLAY_MAX_RECORD_BYTES:
                raise ArtifactValidationError(f"{name} line exceeds the record cap")
            if len(records) >= REPLAY_MAX_NDJSON_RECORDS:
                raise ArtifactValidationError(f"{name} exceeds the record-count cap")
            records.append(
                _load_model(
                    line,
                    label=f"{name} line {len(records) + 1}",
                    validate=validate,
                )
            )
    if not records:
        raise ArtifactValidationError(f"{name} must not be empty")
    return tuple(records)


def _verify_tree_snapshot(root: Path, snapshot: _TreeSnapshot) -> None:
    _reject_root_symlink_components(root)
    try:
        current_root = os.lstat(root)
    except OSError as error:
        raise ArtifactValidationError("replay artifact root disappeared") from error
    if _stat_fingerprint(current_root) != _stat_fingerprint(snapshot.root_stat):
        raise ArtifactValidationError("replay artifact root changed during validation")
    current_entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                current_entries[entry.name] = entry.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactValidationError("replay artifact cannot be rechecked") from error
    if set(current_entries) != set(snapshot.entries):
        raise ArtifactValidationError("replay artifact allowlist changed during validation")
    for name, expected in snapshot.entries.items():
        current = current_entries.get(name)
        if current is None or _stat_fingerprint(current) != _stat_fingerprint(expected):
            raise ArtifactValidationError("replay artifact member changed during validation")


def _load_replay_curated_artifact(path: Path) -> LoadedReplayCuratedArtifact:
    root = absolute_artifact_path(path)
    snapshot = _require_safe_tree(root)
    all_bytes = {
        name: _read_member(
            root,
            name,
            expected_stat=snapshot.entries[name],
            byte_cap=REPLAY_MAX_MEMBER_BYTES,
        )
        for name in REPLAY_ARTIFACT_PATHS
    }
    validate_replay_candidate_bytes(all_bytes)
    intent_bytes = all_bytes[REPLAY_INTENT_FILE]
    profile = _load_model(
        all_bytes[REPLAY_PROFILE_SUMMARY_FILE],
        label=REPLAY_PROFILE_SUMMARY_FILE,
        validate=ReplayProfileSummaryV1.model_validate_json,
    )
    descriptors = _load_ndjson(
        root,
        REPLAY_DESCRIPTOR_AGGREGATES_FILE,
        expected_stat=snapshot.entries[REPLAY_DESCRIPTOR_AGGREGATES_FILE],
        validate=ReplayDescriptorAggregateV1.model_validate_json,
    )
    persistent = _load_ndjson(
        root,
        REPLAY_PERSISTENT_AGGREGATES_FILE,
        expected_stat=snapshot.entries[REPLAY_PERSISTENT_AGGREGATES_FILE],
        validate=ReplayPersistentAggregateV1.model_validate_json,
    )
    crossovers = _load_ndjson(
        root,
        REPLAY_PERSISTENT_CROSSOVERS_FILE,
        expected_stat=snapshot.entries[REPLAY_PERSISTENT_CROSSOVERS_FILE],
        validate=ReplayPersistentCrossoverV1.model_validate_json,
    )
    health = _load_ndjson(
        root,
        REPLAY_HEALTH_AGGREGATES_FILE,
        expected_stat=snapshot.entries[REPLAY_HEALTH_AGGREGATES_FILE],
        validate=ReplayHealthAggregateV1.model_validate_json,
    )
    sensitivity = _load_ndjson(
        root,
        REPLAY_CLUSTER_SENSITIVITY_FILE,
        expected_stat=snapshot.entries[REPLAY_CLUSTER_SENSITIVITY_FILE],
        validate=ReplayClusterSensitivityV1.model_validate_json,
    )
    validation = _load_model(
        all_bytes[REPLAY_VALIDATION_FILE],
        label=REPLAY_VALIDATION_FILE,
        validate=ReplayValidationV1.model_validate_json,
    )
    repeat = _load_model(
        all_bytes[REPLAY_REPEAT_VERIFICATION_FILE],
        label=REPLAY_REPEAT_VERIFICATION_FILE,
        validate=ReplayRepeatVerificationV1.model_validate_json,
    )
    figures = _load_ndjson(
        root,
        REPLAY_FIGURE_RECORDS_FILE,
        expected_stat=snapshot.entries[REPLAY_FIGURE_RECORDS_FILE],
        validate=ReplayFigureRecordV1.model_validate_json,
    )
    commitments = _load_ndjson(
        root,
        REPLAY_SOURCE_COMMITMENTS_FILE,
        expected_stat=snapshot.entries[REPLAY_SOURCE_COMMITMENTS_FILE],
        validate=ReplaySourceMemberCommitmentV1.model_validate_json,
    )
    release_index = _load_model(
        all_bytes[REPLAY_RELEASE_INDEX_FILE],
        label=REPLAY_RELEASE_INDEX_FILE,
        validate=ReplayReleaseIndexV1.model_validate_json,
    )
    run = _load_model(
        all_bytes[REPLAY_RUN_FILE],
        label=REPLAY_RUN_FILE,
        validate=RunRecordV1Alpha1.model_validate_json,
    )
    success = _load_model(
        all_bytes[REPLAY_SUCCESS_FILE],
        label=REPLAY_SUCCESS_FILE,
        validate=ReplaySuccessV1.model_validate_json,
    )

    request = ReplayCuratedArtifactWriteRequest(
        profile_summary=profile,
        descriptor_aggregates=descriptors,
        persistent_aggregates=persistent,
        persistent_crossovers=crossovers,
        health_aggregates=health,
        cluster_sensitivity=sensitivity,
        validation=validation,
        repeat_verification=repeat,
        figures=figures,
        source_commitments=commitments,
        run=run,
    )
    expected_order = _ordered_records(request)
    if (
        descriptors,
        persistent,
        crossovers,
        health,
        sensitivity,
        figures,
        commitments,
    ) != expected_order:
        raise ArtifactValidationError("replay records are not in canonical order")

    record_counts = {
        REPLAY_DESCRIPTOR_AGGREGATES_FILE: len(descriptors),
        REPLAY_PERSISTENT_AGGREGATES_FILE: len(persistent),
        REPLAY_PERSISTENT_CROSSOVERS_FILE: len(crossovers),
        REPLAY_HEALTH_AGGREGATES_FILE: len(health),
        REPLAY_CLUSTER_SENSITIVITY_FILE: len(sensitivity),
        REPLAY_FIGURE_RECORDS_FILE: len(figures),
        REPLAY_SOURCE_COMMITMENTS_FILE: len(commitments),
    }
    for expected_path, entry in zip(REPLAY_INDEXED_PATHS, release_index.files, strict=True):
        member = all_bytes[expected_path]
        expected_record_count = record_counts.get(expected_path)
        if (
            entry.path != expected_path
            or entry.byte_length != len(member)
            or entry.sha256 != _sha256_bytes(member)
            or entry.record_count != expected_record_count
        ):
            raise ArtifactValidationError("replay member disagrees with the release index")
    if release_index.run_id != run.run_id:
        raise ArtifactValidationError("replay release index run binding is invalid")

    artifact_sha256 = compute_replay_curated_artifact_digest(all_bytes[REPLAY_RELEASE_INDEX_FILE])
    validate_replay_curated_bundle(
        intent_bytes=intent_bytes,
        profile_summary=profile,
        descriptor_aggregates=descriptors,
        persistent_aggregates=persistent,
        persistent_crossovers=crossovers,
        health_aggregates=health,
        cluster_sensitivity=sensitivity,
        validation=validation,
        repeat_verification=repeat,
        figures=figures,
        source_commitments=commitments,
        run=run,
        artifact_sha256=artifact_sha256,
    )
    run_sha256 = compute_run_record_digest(all_bytes[REPLAY_RUN_FILE])
    if success.release_artifact_sha256 != artifact_sha256 or success.run_sha256 != run_sha256:
        raise ArtifactValidationError("replay success marker binding is invalid")
    _verify_tree_snapshot(root, snapshot)
    return LoadedReplayCuratedArtifact(
        path=root,
        intent_bytes=intent_bytes,
        profile_summary=profile,
        descriptor_aggregates=descriptors,
        persistent_aggregates=persistent,
        persistent_crossovers=crossovers,
        health_aggregates=health,
        cluster_sensitivity=sensitivity,
        validation=validation,
        repeat_verification=repeat,
        figures=figures,
        source_commitments=commitments,
        release_index=release_index,
        run=run,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )


def load_replay_curated_artifact(path: Path) -> LoadedReplayCuratedArtifact:
    """Strictly load one complete, aggregate-only M5 curated artifact."""

    try:
        return _load_replay_curated_artifact(path)
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactValidationError("invalid M5 replay curated artifact") from error


def _safe_cleanup_at(
    *,
    parent_fd: int,
    artifact_fd: int,
    artifact_name: str,
) -> None:
    for name in REPLAY_ARTIFACT_PATHS:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=artifact_fd)
    with suppress(OSError):
        os.rmdir(artifact_name, dir_fd=parent_fd)


def _write_replay_curated_artifact(
    request: ReplayCuratedArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path,
    git_metadata_dirs: Sequence[Path] | None,
) -> LoadedReplayCuratedArtifact:
    prepared = _prepare_replay_artifact(request, source_root=source_root)
    target = absolute_artifact_path(destination)
    metadata_dirs = (
        discover_git_metadata_dirs(source_root)
        if git_metadata_dirs is None
        else tuple(git_metadata_dirs)
    )
    reject_git_metadata_destination(target, metadata_dirs)
    if os.path.lexists(target):
        raise FileExistsError("replay artifact destination already exists")
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
            raise FileExistsError("replay artifact destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        renamed = False
        try:
            for name in REPLAY_ARTIFACT_PATHS[:-1]:
                write_exclusive_file_at(staging_fd, name, prepared.files[name])
            for name in REPLAY_ARTIFACT_PATHS[:-1]:
                if (
                    read_file_at(staging_fd, name, byte_cap=len(prepared.files[name]))
                    != prepared.files[name]
                ):
                    raise ArtifactValidationError("replay staging verification failed")
            write_exclusive_file_at(
                staging_fd,
                REPLAY_SUCCESS_FILE,
                prepared.files[REPLAY_SUCCESS_FILE],
            )
            os.fsync(staging_fd)
            assert_directory_descriptor_matches_path(
                staging_fd,
                staging,
                label="staging artifact",
            )
            staged = _load_replay_curated_artifact(staging)
            if (
                staged.artifact_sha256 != prepared.artifact_sha256
                or staged.run_sha256 != prepared.run_sha256
            ):
                raise ArtifactValidationError("staged replay artifact identity is invalid")
            assert_directory_descriptor_matches_path(
                staging_fd,
                staging,
                label="staging artifact",
            )
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
            if entry_exists_at(parent_fd, target.name):
                raise FileExistsError("replay artifact destination already exists")
            atomic_rename_directory_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            renamed = True
            try:
                os.fsync(parent_fd)
                assert_directory_descriptor_matches_path(
                    parent_fd,
                    parent,
                    label="destination parent",
                )
                assert_directory_descriptor_matches_path(
                    staging_fd,
                    target,
                    label="published replay artifact",
                )
                loaded = _load_replay_curated_artifact(target)
                if (
                    loaded.artifact_sha256 != prepared.artifact_sha256
                    or loaded.run_sha256 != prepared.run_sha256
                ):
                    raise ArtifactValidationError("published replay artifact identity is invalid")
                assert_directory_descriptor_matches_path(
                    parent_fd,
                    parent,
                    label="destination parent",
                )
                assert_directory_descriptor_matches_path(
                    staging_fd,
                    target,
                    label="published replay artifact",
                )
            except BaseException:
                with suppress(OSError, ArtifactValidationError):
                    assert_directory_descriptor_matches_path(
                        staging_fd,
                        target,
                        label="published replay artifact",
                    )
                    _safe_cleanup_at(
                        parent_fd=parent_fd,
                        artifact_fd=staging_fd,
                        artifact_name=target.name,
                    )
                    os.fsync(parent_fd)
                raise
            return loaded
        except BaseException:
            if not renamed:
                _safe_cleanup_at(
                    parent_fd=parent_fd,
                    artifact_fd=staging_fd,
                    artifact_name=staging_name,
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def write_replay_curated_artifact(
    request: ReplayCuratedArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path,
    git_metadata_dirs: Sequence[Path] | None = None,
) -> LoadedReplayCuratedArtifact:
    """Validate, stage, and atomically publish one no-overwrite M5 artifact."""

    try:
        return _write_replay_curated_artifact(
            request,
            destination,
            source_root=source_root,
            git_metadata_dirs=git_metadata_dirs,
        )
    except FileExistsError:
        raise
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactValidationError("M5 replay artifact publication failed") from error
