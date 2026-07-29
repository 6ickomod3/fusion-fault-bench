"""Strict construction, loading, and publication of M3 procedural artifacts."""

from __future__ import annotations

import hashlib
import os
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
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    expected_conditions,
    expected_sequence_ids,
    validate_result_bundle,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    EXPERIMENT_MANIFEST_ADAPTER,
    AvailabilityControlManifest,
    CommonModeControlManifest,
    CorrectlyReportedNoiseFault,
    GeometryCrossoverManifest,
    ProceduralSource,
    UnderreportedNoiseFault,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_AGGREGATE_METRICS_FILE,
    PROCEDURAL_ARTIFACT_CONTRACT,
    PROCEDURAL_ARTIFACT_PATHS,
    PROCEDURAL_CROSSOVERS_FILE,
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
    PROCEDURAL_MANIFEST_FILE,
    PROCEDURAL_MAX_ARTIFACT_BYTES,
    PROCEDURAL_MAX_BOOTSTRAP_CELLS,
    PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
    PROCEDURAL_MAX_MEMBER_BYTES,
    PROCEDURAL_MAX_RECORD_BYTES,
    PROCEDURAL_MAX_SEQUENCE_COUNT,
    PROCEDURAL_MAX_SEQUENCE_ROWS,
    PROCEDURAL_PAYLOAD_INDEX_FILE,
    PROCEDURAL_PROFILE_FILE,
    PROCEDURAL_RUN_FILE,
    PROCEDURAL_SEQUENCE_METRICS_FILE,
    PROCEDURAL_SUCCESS_FILE,
    PROCEDURAL_VALIDATION_FILE,
    ProceduralPayloadFileEntryV1Alpha2,
    ProceduralPayloadIndexV1Alpha2,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EXPECTED_PROFILE_DIGESTS,
    PROCEDURAL_PROFILE_ADAPTER,
    ProceduralProfileV1,
    profile_sequence_count,
)
from fusion_fault_bench.contracts.procedural_validation_v1 import ProceduralValidationV1
from fusion_fault_bench.contracts.result_v1alpha1 import (
    METRIC_RECORD_ADAPTER,
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    MetricRecordV1Alpha1,
    RunRecordV1Alpha1,
)
from fusion_fault_bench.procedural_validation import build_procedural_validation
from fusion_fault_bench.scenarios.procedural import generate_procedural_sequences

_PROCEDURAL_ARTIFACT_DOMAIN = b"fusion-fault-bench/procedural-artifact/v1\x00"
_READ_CHUNK_BYTES = 1024 * 1024

type ProceduralManifest = (
    GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest
)


@dataclass(frozen=True, slots=True)
class ProceduralArtifactWriteRequest:
    """Already-built M3 scientific records and completed run provenance."""

    manifest: ProceduralManifest
    profile: ProceduralProfileV1
    metrics: Sequence[MetricRecordV1Alpha1]
    aggregates: Sequence[AggregateMetricRecordV1Alpha1]
    crossovers: Sequence[CrossoverRecordV1Alpha1]
    validation: ProceduralValidationV1
    run: RunRecordV1Alpha1


@dataclass(frozen=True, slots=True)
class LoadedProceduralArtifact:
    """One strictly reloaded and cross-validated M3 artifact."""

    path: Path
    manifest: ProceduralManifest
    profile: ProceduralProfileV1
    metrics: tuple[MetricRecordV1Alpha1, ...]
    aggregates: tuple[AggregateMetricRecordV1Alpha1, ...]
    crossovers: tuple[CrossoverRecordV1Alpha1, ...]
    validation: ProceduralValidationV1
    payload_index: ProceduralPayloadIndexV1Alpha2
    run: RunRecordV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedProceduralArtifact:
    files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class _TreeSnapshot:
    root_stat: os.stat_result
    entries: Mapping[str, os.stat_result]


def compute_procedural_artifact_digest(payload_index_file_bytes: bytes) -> str:
    """Hash exact canonical v1alpha2 index bytes using the M3 domain."""

    preimage = b"".join(
        (
            _PROCEDURAL_ARTIFACT_DOMAIN,
            len(payload_index_file_bytes).to_bytes(8, "big"),
            payload_index_file_bytes,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def canonical_procedural_ndjson_bytes(
    records: Sequence[BaseModel],
    *,
    allow_empty: bool = False,
) -> bytes:
    """Serialize canonical bounded NDJSON, optionally permitting zero bytes."""

    output = bytearray()
    for index, record in enumerate(records):
        line = canonical_json_bytes(record)
        if len(line) > PROCEDURAL_MAX_RECORD_BYTES:
            raise ArtifactValidationError(
                f"procedural NDJSON record {index} exceeds the 1 MiB line cap"
            )
        if len(output) + len(line) > PROCEDURAL_MAX_MEMBER_BYTES:
            raise ArtifactValidationError("procedural NDJSON exceeds the member cap")
        output.extend(line)
    if not output and not allow_empty:
        raise ArtifactValidationError("procedural NDJSON must contain at least one record")
    return bytes(output)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_run_id(manifest_sha256: str, run: RunRecordV1Alpha1) -> str:
    return derive_run_id(
        manifest_sha256=manifest_sha256,
        git_revision=run.git_revision,
        lockfile_sha256=run.lockfile_sha256,
        package_version=run.package_version,
        artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
    )


def _validate_manifest(manifest: ProceduralManifest) -> ProceduralSource:
    source = manifest.source
    if not isinstance(source, ProceduralSource):
        raise ArtifactValidationError("M3 artifacts require a procedural source")
    return source


def _validate_run(
    manifest: ProceduralManifest,
    run: RunRecordV1Alpha1,
    *,
    artifact_sha256: str | None = None,
) -> str:
    manifest_sha256 = sha256_digest(manifest)
    if run.manifest_sha256 != manifest_sha256:
        raise ArtifactValidationError("procedural run manifest identity is invalid")
    if run.run_id != _expected_run_id(manifest_sha256, run):
        raise ArtifactValidationError("procedural run_id is invalid")
    if run.source_dirty or run.status != "succeeded":
        raise ArtifactValidationError("procedural artifact requires a clean successful run")
    if artifact_sha256 is not None and run.artifact_sha256 != artifact_sha256:
        raise ArtifactValidationError("procedural run artifact identity is invalid")
    return manifest_sha256


def _profile_shape(profile: ProceduralProfileV1) -> tuple[int, int]:
    return profile.source.frame_count, profile.source.object_count


def _implied_counts(manifest: ProceduralManifest) -> tuple[int, int, int, int]:
    source = _validate_manifest(manifest)
    sequence_count = source.sequence_count
    bootstrap_replicates = manifest.evaluation.bootstrap.replicates
    condition_count = len(expected_conditions(manifest))
    if isinstance(manifest, AvailabilityControlManifest):
        sequence_pairs = len(manifest.methods) * len(manifest.evaluation.metrics)
        aggregate_pairs = sequence_pairs
        crossover_count = 0
    elif isinstance(manifest, CommonModeControlManifest):
        sequence_pairs = len(manifest.methods)
        aggregate_pairs = len(manifest.methods)
        crossover_count = 0
    else:
        sequence_pairs = len(manifest.methods)
        aggregate_pairs = len(manifest.methods) + 1
        if isinstance(
            manifest.fault_sweep,
            (CorrectlyReportedNoiseFault, UnderreportedNoiseFault),
        ):
            crossover_count = 1
        else:
            crossover_count = 2
    return (
        sequence_count * condition_count * sequence_pairs,
        condition_count * aggregate_pairs,
        crossover_count,
        sequence_count * bootstrap_replicates,
    )


def _validate_execution_caps(
    manifest: ProceduralManifest,
    *,
    sequence_rows: int | None = None,
) -> tuple[int, int, int, int]:
    source = _validate_manifest(manifest)
    implied = _implied_counts(manifest)
    implied_rows, _, _, bootstrap_cells = implied
    bootstrap_replicates = manifest.evaluation.bootstrap.replicates
    if source.sequence_count > PROCEDURAL_MAX_SEQUENCE_COUNT:
        raise ArtifactValidationError("procedural sequence_count exceeds its cap")
    if bootstrap_replicates > PROCEDURAL_MAX_BOOTSTRAP_REPLICATES:
        raise ArtifactValidationError("procedural bootstrap replicates exceed their cap")
    if bootstrap_cells > PROCEDURAL_MAX_BOOTSTRAP_CELLS:
        raise ArtifactValidationError("procedural bootstrap matrix exceeds its cap")
    if implied_rows > PROCEDURAL_MAX_SEQUENCE_ROWS:
        raise ArtifactValidationError("manifest-implied procedural rows exceed their cap")
    if sequence_rows is not None and sequence_rows > PROCEDURAL_MAX_SEQUENCE_ROWS:
        raise ArtifactValidationError("procedural sequence rows exceed their cap")
    return implied


def _validate_profile_link(
    manifest: ProceduralManifest,
    profile: ProceduralProfileV1,
) -> str:
    source = _validate_manifest(manifest)
    profile_sha256 = sha256_digest(profile)
    if profile_sha256 != EXPECTED_PROFILE_DIGESTS[profile.profile_id]:
        raise ArtifactValidationError("procedural profile digest is not preregistered")
    if profile.profile_id != source.profile_id:
        raise ArtifactValidationError("procedural profile ID disagrees with the manifest")
    if profile_sha256 != source.profile_sha256:
        raise ArtifactValidationError("procedural profile digest disagrees with the manifest")
    if profile_sequence_count(profile, source.split) != source.sequence_count:
        raise ArtifactValidationError("procedural profile split count disagrees with the manifest")
    eligibility = profile.eligibility
    roi = manifest.roi
    if (
        eligibility.frame != roi.frame
        or eligibility.x_min_m != roi.x_min_m
        or eligibility.x_max_m != roi.x_max_m
        or eligibility.abs_y_max_m != roi.abs_y_max_m
        or eligibility.camera_half_fov_rad != roi.camera_half_fov_rad
    ):
        raise ArtifactValidationError("procedural profile ROI disagrees with the manifest")
    return profile_sha256


def _validate_validation_links(
    manifest: ProceduralManifest,
    profile: ProceduralProfileV1,
    validation: ProceduralValidationV1,
    run: RunRecordV1Alpha1,
) -> None:
    source = _validate_manifest(manifest)
    manifest_sha256 = sha256_digest(manifest)
    profile_sha256 = sha256_digest(profile)
    frame_count, object_count = _profile_shape(profile)
    implied_rows, _, _, bootstrap_cells = _implied_counts(manifest)
    if validation.run_id != run.run_id:
        raise ArtifactValidationError("procedural validation run_id is invalid")
    if validation.manifest_sha256 != manifest_sha256:
        raise ArtifactValidationError("procedural validation manifest identity is invalid")
    if validation.profile_id != profile.profile_id or validation.profile_sha256 != profile_sha256:
        raise ArtifactValidationError("procedural validation profile identity is invalid")
    if (
        validation.split != source.split
        or validation.sequence_count != source.sequence_count
        or validation.frame_count != frame_count
        or validation.object_count != object_count
    ):
        raise ArtifactValidationError("procedural validation population shape is invalid")
    if validation.resources.implied_sequence_row_count != implied_rows:
        raise ArtifactValidationError("procedural validation row count is invalid")
    if validation.resources.implied_bootstrap_cell_count != bootstrap_cells:
        raise ArtifactValidationError("procedural validation bootstrap cell count is invalid")
    if validation.resources.bootstrap_replicates != manifest.evaluation.bootstrap.replicates:
        raise ArtifactValidationError("procedural validation replicate count is invalid")
    if isinstance(manifest, AvailabilityControlManifest):
        if validation.dropout_validation.status != "applicable":
            raise ArtifactValidationError("availability validation requires dropout evidence")
        if validation.expected_loss_checks:
            raise ArtifactValidationError("availability validation cannot contain affine loss rows")
    elif validation.dropout_validation.status != "not-applicable":
        raise ArtifactValidationError("non-availability validation cannot contain dropout evidence")
    if isinstance(manifest, CommonModeControlManifest):
        if validation.common_mode_validation.status != "applicable":
            raise ArtifactValidationError("common-mode manifest requires disagreement evidence")
        if validation.identity_comparison.status != "not-applicable":
            raise ArtifactValidationError("edge common-mode identity comparison must be excluded")
    else:
        if validation.common_mode_validation.status != "not-applicable":
            raise ArtifactValidationError(
                "non-common-mode validation cannot contain common evidence"
            )
        if validation.identity_comparison.status != "deferred-to-matrix":
            raise ArtifactValidationError(
                "main-profile cross-manifest identity must be deferred to matrix evidence"
            )
    if (
        not isinstance(manifest, AvailabilityControlManifest)
        and not validation.expected_loss_checks
    ):
        raise ArtifactValidationError("non-availability validation requires expected-loss rows")
    if not validation.all_checks_passed:
        raise ArtifactValidationError("procedural validation did not pass every release gate")


def validate_procedural_bundle(
    manifest: ProceduralManifest,
    profile: ProceduralProfileV1,
    metrics: Sequence[MetricRecordV1Alpha1],
    aggregates: Sequence[AggregateMetricRecordV1Alpha1],
    crossovers: Sequence[CrossoverRecordV1Alpha1],
    validation: ProceduralValidationV1,
    run: RunRecordV1Alpha1,
) -> None:
    """Cross-validate the immutable profile, records, evidence, and provenance."""

    _validate_execution_caps(manifest, sequence_rows=len(metrics))
    _validate_run(manifest, run)
    _validate_profile_link(manifest, profile)
    _validate_validation_links(manifest, profile, validation, run)
    validate_result_bundle(manifest, run, metrics, aggregates, crossovers)
    source = _validate_manifest(manifest)
    sequences = generate_procedural_sequences(
        profile,
        split=source.split,
        sequence_count=source.sequence_count,
        data_master_seed=manifest.rng.data_master_seed,
    )
    independently_rebuilt = build_procedural_validation(
        manifest,
        profile=profile,
        run_id=run.run_id,
        sequences=sequences,
        metrics=metrics,
    )
    if canonical_json_bytes(independently_rebuilt) != canonical_json_bytes(validation):
        raise ArtifactValidationError(
            "procedural validation evidence disagrees with independent recomputation"
        )


def _condition_key(record: Any) -> tuple[str, str, int, float, str, str]:
    return (
        record.fault_family,
        record.fault_axis,
        record.severity.index,
        record.severity.magnitude,
        record.severity.direction,
        record.severity.unit,
    )


def _ordered_records(
    manifest: ProceduralManifest,
    metrics: Sequence[MetricRecordV1Alpha1],
    aggregates: Sequence[AggregateMetricRecordV1Alpha1],
    crossovers: Sequence[CrossoverRecordV1Alpha1],
) -> tuple[
    tuple[MetricRecordV1Alpha1, ...],
    tuple[AggregateMetricRecordV1Alpha1, ...],
    tuple[CrossoverRecordV1Alpha1, ...],
]:
    sequence_rank = {
        sequence_id: index for index, sequence_id in enumerate(expected_sequence_ids(manifest))
    }
    condition_rank = {
        (
            condition.fault_family,
            condition.fault_axis,
            condition.severity_index,
            condition.magnitude,
            condition.direction,
            condition.unit,
        ): index
        for index, condition in enumerate(expected_conditions(manifest))
    }
    method_rank = {method: index for index, method in enumerate(manifest.methods)}
    if isinstance(manifest, AvailabilityControlManifest):
        metric_rank = {metric: index for index, metric in enumerate(manifest.evaluation.metrics)}
    else:
        metric_rank = {"matched-center-mse": 0, "fused-minus-healthy": 1}
    directions = tuple(
        dict.fromkeys(
            condition.direction
            for condition in expected_conditions(manifest)
            if condition.direction != "identity"
        )
    )
    direction_rank = {direction: index for index, direction in enumerate(directions)}

    try:
        ordered_metrics = tuple(
            sorted(
                metrics,
                key=lambda record: (
                    sequence_rank[record.sequence_id],
                    condition_rank[_condition_key(record)],
                    method_rank[record.method_id],
                    metric_rank[record.metric_name],
                ),
            )
        )
        ordered_aggregates = tuple(
            sorted(
                aggregates,
                key=lambda record: (
                    condition_rank[_condition_key(record)],
                    method_rank[record.method_id],
                    metric_rank[record.metric_name],
                ),
            )
        )
        ordered_crossovers = tuple(
            sorted(crossovers, key=lambda record: direction_rank[record.direction])
        )
    except KeyError as error:
        raise ArtifactValidationError(
            "procedural records cannot be ordered from the manifest"
        ) from error
    return ordered_metrics, ordered_aggregates, ordered_crossovers


def _finalize_run(run: RunRecordV1Alpha1, artifact_sha256: str) -> RunRecordV1Alpha1:
    value = run.model_dump(mode="python", by_alias=True)
    value["artifact_sha256"] = artifact_sha256
    return RunRecordV1Alpha1.model_validate(value)


def _prepare_procedural_artifact(
    request: ProceduralArtifactWriteRequest,
) -> _PreparedProceduralArtifact:
    validate_procedural_bundle(
        request.manifest,
        request.profile,
        request.metrics,
        request.aggregates,
        request.crossovers,
        request.validation,
        request.run,
    )
    metrics, aggregates, crossovers = _ordered_records(
        request.manifest,
        request.metrics,
        request.aggregates,
        request.crossovers,
    )
    is_control = isinstance(
        request.manifest,
        (AvailabilityControlManifest, CommonModeControlManifest),
    )
    manifest_sha256 = sha256_digest(request.manifest)
    profile_sha256 = sha256_digest(request.profile)
    indexed_files: dict[str, bytes] = {
        PROCEDURAL_MANIFEST_FILE: canonical_json_bytes(request.manifest),
        PROCEDURAL_PROFILE_FILE: canonical_json_bytes(request.profile),
        PROCEDURAL_SEQUENCE_METRICS_FILE: canonical_procedural_ndjson_bytes(metrics),
        PROCEDURAL_AGGREGATE_METRICS_FILE: canonical_procedural_ndjson_bytes(aggregates),
        PROCEDURAL_CROSSOVERS_FILE: canonical_procedural_ndjson_bytes(
            crossovers,
            allow_empty=is_control,
        ),
        PROCEDURAL_VALIDATION_FILE: canonical_json_bytes(request.validation),
    }
    if any(len(value) > PROCEDURAL_MAX_MEMBER_BYTES for value in indexed_files.values()):
        raise ArtifactValidationError("procedural indexed member exceeds its byte cap")
    if is_control != (len(indexed_files[PROCEDURAL_CROSSOVERS_FILE]) == 0):
        raise ArtifactValidationError("procedural crossover emptiness disagrees with manifest kind")

    payload_index = ProceduralPayloadIndexV1Alpha2(
        schema="ffb.payload-index/v1alpha2",
        artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
        run_id=request.run.run_id,
        manifest_sha256=manifest_sha256,
        profile_sha256=profile_sha256,
        files=tuple(
            ProceduralPayloadFileEntryV1Alpha2(
                path=path,  # type: ignore[arg-type]
                byte_length=len(indexed_files[path]),
                sha256=_sha256_bytes(indexed_files[path]),
            )
            for path in PROCEDURAL_INDEXED_PAYLOAD_PATHS
        ),
    )
    payload_index_bytes = canonical_json_bytes(payload_index)
    artifact_sha256 = compute_procedural_artifact_digest(payload_index_bytes)
    run = _finalize_run(request.run, artifact_sha256)
    _validate_run(request.manifest, run, artifact_sha256=artifact_sha256)
    run_bytes = canonical_json_bytes(run)
    run_sha256 = compute_run_record_digest(run_bytes)
    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )
    files = {
        **indexed_files,
        PROCEDURAL_PAYLOAD_INDEX_FILE: payload_index_bytes,
        PROCEDURAL_RUN_FILE: run_bytes,
        PROCEDURAL_SUCCESS_FILE: canonical_json_bytes(success),
    }
    if any(
        len(files[path]) > PROCEDURAL_MAX_RECORD_BYTES
        for path in (
            PROCEDURAL_MANIFEST_FILE,
            PROCEDURAL_PROFILE_FILE,
            PROCEDURAL_VALIDATION_FILE,
            PROCEDURAL_PAYLOAD_INDEX_FILE,
            PROCEDURAL_RUN_FILE,
            PROCEDURAL_SUCCESS_FILE,
        )
    ):
        raise ArtifactValidationError("procedural canonical JSON member exceeds 1 MiB")
    if sum(map(len, files.values())) > PROCEDURAL_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("procedural artifact exceeds the 1 GiB cap")
    return _PreparedProceduralArtifact(files=files)


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
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
            raise ArtifactValidationError("procedural artifact path cannot be inspected") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ArtifactValidationError("procedural artifact path contains a symlink")


def _require_safe_tree(root: Path) -> _TreeSnapshot:
    absolute = absolute_artifact_path(root)
    _reject_root_symlink_components(absolute)
    try:
        root_stat = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError("procedural artifact directory is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ArtifactValidationError("procedural artifact root must be a real directory")

    entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(absolute) as iterator:
            for entry in iterator:
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ArtifactValidationError("procedural artifact contains a symlink member")
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ArtifactValidationError(
                        "procedural artifact members must be regular files"
                    )
                entries[entry.name] = entry_stat
    except ArtifactValidationError:
        raise
    except OSError as error:
        raise ArtifactValidationError("procedural artifact members cannot be inspected") from error

    if set(entries) != set(PROCEDURAL_ARTIFACT_PATHS):
        raise ArtifactValidationError("procedural artifact file allowlist mismatch")
    if any(entry.st_size > PROCEDURAL_MAX_MEMBER_BYTES for entry in entries.values()):
        raise ArtifactValidationError("procedural artifact member exceeds its cap")
    if sum(entry.st_size for entry in entries.values()) > PROCEDURAL_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("procedural artifact exceeds its tree cap")
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
            raise ArtifactValidationError("procedural artifact member is not a regular file")
        if _stat_fingerprint(file_stat) != _stat_fingerprint(expected_stat):
            raise ArtifactValidationError("procedural artifact member changed during validation")
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
            raise ArtifactValidationError(f"procedural artifact member exceeds its cap: {name}")
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
            raise ArtifactValidationError(f"procedural artifact member exceeds its cap: {name}")
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
                raise ArtifactValidationError("procedural artifact member changed during hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ArtifactValidationError("procedural artifact member changed during hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _load_model[ModelT: BaseModel](
    data: bytes,
    *,
    label: str,
    validate: Callable[[bytes], ModelT],
) -> ModelT:
    try:
        body = strict_json_object_body(data, label=label)
    except ArtifactValidationError as error:
        raise ArtifactValidationError(f"{label} is not strict canonical JSON") from error
    try:
        model = validate(body)
    except (ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"{label} violates its fixed schema") from error
    if canonical_json_bytes(model) != data:
        raise ArtifactValidationError(f"{label} is not canonical JSON")
    return model


def _load_metric(data: bytes) -> MetricRecordV1Alpha1:
    return cast(MetricRecordV1Alpha1, METRIC_RECORD_ADAPTER.validate_json(data))


def _load_ndjson[ModelT: BaseModel](
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
    validate: Callable[[bytes], ModelT],
    record_cap: int,
    allow_empty: bool = False,
) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    descriptor = _open_regular_member(root, name, expected_stat=expected_stat)
    with os.fdopen(descriptor, "rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(PROCEDURAL_MAX_RECORD_BYTES + 1)
            if not line:
                break
            line_number += 1
            if len(line) > PROCEDURAL_MAX_RECORD_BYTES:
                raise ArtifactValidationError(f"{name} line exceeds the 1 MiB cap")
            if line_number > record_cap:
                raise ArtifactValidationError(f"{name} exceeds its record-count cap")
            records.append(
                _load_model(
                    line,
                    label=f"{name} line {line_number}",
                    validate=validate,
                )
            )
    if not records and not allow_empty:
        raise ArtifactValidationError(f"{name} must contain at least one record")
    return tuple(records)


def _validate_manifest_json(data: bytes) -> ProceduralManifest:
    manifest = EXPERIMENT_MANIFEST_ADAPTER.validate_json(data)
    if not isinstance(
        manifest,
        (GeometryCrossoverManifest, AvailabilityControlManifest, CommonModeControlManifest),
    ) or not isinstance(manifest.source, ProceduralSource):
        raise ValueError("manifest is not an M3 procedural manifest")
    return manifest


def _validate_profile_json(data: bytes) -> ProceduralProfileV1:
    return PROCEDURAL_PROFILE_ADAPTER.validate_json(data)


def _verify_tree_snapshot(root: Path, snapshot: _TreeSnapshot) -> None:
    _reject_root_symlink_components(root)
    try:
        current_root_stat = os.lstat(root)
    except OSError as error:
        raise ArtifactValidationError(
            "procedural artifact root disappeared during validation"
        ) from error
    if _stat_fingerprint(current_root_stat) != _stat_fingerprint(snapshot.root_stat):
        raise ArtifactValidationError("procedural artifact root changed during validation")
    current_entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                current_entries[entry.name] = entry.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactValidationError("procedural artifact cannot be rechecked") from error
    if set(current_entries) != set(snapshot.entries):
        raise ArtifactValidationError("procedural artifact allowlist changed during validation")
    for name, expected_stat in snapshot.entries.items():
        current_stat = current_entries.get(name)
        if current_stat is None or _stat_fingerprint(current_stat) != _stat_fingerprint(
            expected_stat
        ):
            raise ArtifactValidationError("procedural artifact member changed during validation")


def _validate_index(
    root: Path,
    snapshot: _TreeSnapshot,
    payload_index: ProceduralPayloadIndexV1Alpha2,
) -> None:
    for expected_path, entry in zip(
        PROCEDURAL_INDEXED_PAYLOAD_PATHS,
        payload_index.files,
        strict=True,
    ):
        metadata = snapshot.entries[expected_path]
        if entry.path != expected_path:
            raise ArtifactValidationError("procedural payload index order is invalid")
        if entry.byte_length != metadata.st_size:
            raise ArtifactValidationError("procedural payload byte length is invalid")
        if entry.sha256 != _sha256_member(
            root,
            expected_path,
            expected_stat=metadata,
        ):
            raise ArtifactValidationError("procedural payload digest is invalid")


def _load_procedural_artifact(path: Path) -> LoadedProceduralArtifact:
    root = absolute_artifact_path(path)
    snapshot = _require_safe_tree(root)
    small_members = {
        name: _read_member(
            root,
            name,
            expected_stat=snapshot.entries[name],
            byte_cap=PROCEDURAL_MAX_RECORD_BYTES,
        )
        for name in (
            PROCEDURAL_MANIFEST_FILE,
            PROCEDURAL_PROFILE_FILE,
            PROCEDURAL_VALIDATION_FILE,
            PROCEDURAL_PAYLOAD_INDEX_FILE,
            PROCEDURAL_RUN_FILE,
            PROCEDURAL_SUCCESS_FILE,
        )
    }
    manifest = _load_model(
        small_members[PROCEDURAL_MANIFEST_FILE],
        label=PROCEDURAL_MANIFEST_FILE,
        validate=_validate_manifest_json,
    )
    profile = _load_model(
        small_members[PROCEDURAL_PROFILE_FILE],
        label=PROCEDURAL_PROFILE_FILE,
        validate=_validate_profile_json,
    )
    implied_rows, implied_aggregates, implied_crossovers, _ = _validate_execution_caps(manifest)
    is_control = isinstance(
        manifest,
        (AvailabilityControlManifest, CommonModeControlManifest),
    )
    metrics = _load_ndjson(
        root,
        PROCEDURAL_SEQUENCE_METRICS_FILE,
        expected_stat=snapshot.entries[PROCEDURAL_SEQUENCE_METRICS_FILE],
        validate=_load_metric,
        record_cap=implied_rows,
    )
    aggregates = _load_ndjson(
        root,
        PROCEDURAL_AGGREGATE_METRICS_FILE,
        expected_stat=snapshot.entries[PROCEDURAL_AGGREGATE_METRICS_FILE],
        validate=AggregateMetricRecordV1Alpha1.model_validate_json,
        record_cap=implied_aggregates,
    )
    crossovers = _load_ndjson(
        root,
        PROCEDURAL_CROSSOVERS_FILE,
        expected_stat=snapshot.entries[PROCEDURAL_CROSSOVERS_FILE],
        validate=CrossoverRecordV1Alpha1.model_validate_json,
        record_cap=max(1, implied_crossovers),
        allow_empty=is_control,
    )
    validation = _load_model(
        small_members[PROCEDURAL_VALIDATION_FILE],
        label=PROCEDURAL_VALIDATION_FILE,
        validate=ProceduralValidationV1.model_validate_json,
    )
    payload_index = _load_model(
        small_members[PROCEDURAL_PAYLOAD_INDEX_FILE],
        label=PROCEDURAL_PAYLOAD_INDEX_FILE,
        validate=ProceduralPayloadIndexV1Alpha2.model_validate_json,
    )
    run = _load_model(
        small_members[PROCEDURAL_RUN_FILE],
        label=PROCEDURAL_RUN_FILE,
        validate=RunRecordV1Alpha1.model_validate_json,
    )
    success = _load_model(
        small_members[PROCEDURAL_SUCCESS_FILE],
        label=PROCEDURAL_SUCCESS_FILE,
        validate=SuccessMarkerV1Alpha1.model_validate_json,
    )

    expected_order = _ordered_records(manifest, metrics, aggregates, crossovers)
    if (metrics, aggregates, crossovers) != expected_order:
        raise ArtifactValidationError("procedural records are not in canonical order")
    if len(metrics) != implied_rows:
        raise ArtifactValidationError("procedural sequence record count is invalid")
    if len(aggregates) != implied_aggregates:
        raise ArtifactValidationError("procedural aggregate record count is invalid")
    if len(crossovers) != implied_crossovers:
        raise ArtifactValidationError("procedural crossover record count is invalid")
    validate_procedural_bundle(
        manifest,
        profile,
        metrics,
        aggregates,
        crossovers,
        validation,
        run,
    )
    manifest_sha256 = sha256_digest(manifest)
    profile_sha256 = sha256_digest(profile)
    if payload_index.run_id != run.run_id:
        raise ArtifactValidationError("procedural payload run_id is invalid")
    if payload_index.manifest_sha256 != manifest_sha256:
        raise ArtifactValidationError("procedural payload manifest identity is invalid")
    if payload_index.profile_sha256 != profile_sha256:
        raise ArtifactValidationError("procedural payload profile identity is invalid")
    _validate_index(root, snapshot, payload_index)

    index_bytes = small_members[PROCEDURAL_PAYLOAD_INDEX_FILE]
    artifact_sha256 = compute_procedural_artifact_digest(index_bytes)
    _validate_run(manifest, run, artifact_sha256=artifact_sha256)
    run_sha256 = compute_run_record_digest(small_members[PROCEDURAL_RUN_FILE])
    if success.artifact_sha256 != artifact_sha256:
        raise ArtifactValidationError("procedural completion artifact identity is invalid")
    if success.run_sha256 != run_sha256:
        raise ArtifactValidationError("procedural completion run identity is invalid")
    _verify_tree_snapshot(root, snapshot)
    return LoadedProceduralArtifact(
        path=root,
        manifest=manifest,
        profile=profile,
        metrics=metrics,
        aggregates=aggregates,
        crossovers=crossovers,
        validation=validation,
        payload_index=payload_index,
        run=run,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )


def load_procedural_artifact(path: Path) -> LoadedProceduralArtifact:
    """Strictly load and independently validate one complete M3 artifact."""

    try:
        return _load_procedural_artifact(path)
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactValidationError("invalid M3 procedural artifact") from error


def _safe_cleanup_staging_at(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
) -> None:
    for name in PROCEDURAL_ARTIFACT_PATHS:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=staging_fd)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _write_procedural_artifact(
    request: ProceduralArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path | None,
    git_metadata_dirs: Sequence[Path] | None,
) -> LoadedProceduralArtifact:
    prepared = _prepare_procedural_artifact(request)
    target = absolute_artifact_path(destination)
    metadata_dirs = (
        discover_git_metadata_dirs(source_root)
        if git_metadata_dirs is None
        else tuple(git_metadata_dirs)
    )
    reject_git_metadata_destination(target, metadata_dirs)
    if os.path.lexists(target):
        raise FileExistsError("procedural artifact destination already exists")

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
            raise FileExistsError("procedural artifact destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        published = False
        try:
            for name in PROCEDURAL_ARTIFACT_PATHS[:-1]:
                write_exclusive_file_at(staging_fd, name, prepared.files[name])
            for name in PROCEDURAL_ARTIFACT_PATHS[:-1]:
                if (
                    read_file_at(
                        staging_fd,
                        name,
                        byte_cap=len(prepared.files[name]),
                    )
                    != prepared.files[name]
                ):
                    raise ArtifactValidationError("procedural staging verification failed")
            write_exclusive_file_at(
                staging_fd,
                PROCEDURAL_SUCCESS_FILE,
                prepared.files[PROCEDURAL_SUCCESS_FILE],
            )
            os.fsync(staging_fd)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            _load_procedural_artifact(staging)
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="destination parent",
            )
            reject_directory_descriptor_in_git_metadata(parent_fd, metadata_dirs)
            if entry_exists_at(parent_fd, target.name):
                raise FileExistsError("procedural artifact destination already exists")
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
            loaded = load_procedural_artifact(target)
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
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def write_procedural_artifact(
    request: ProceduralArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path | None = None,
    git_metadata_dirs: Sequence[Path] | None = None,
) -> LoadedProceduralArtifact:
    """Validate, stage, and atomically publish one no-overwrite M3 artifact."""

    try:
        return _write_procedural_artifact(
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
        raise ArtifactValidationError("M3 procedural artifact publication failed") from error
