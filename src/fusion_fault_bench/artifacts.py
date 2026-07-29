"""Deterministic construction and strict loading of M1 scientific artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from pydantic import BaseModel, ValidationError

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import (
    AGGREGATE_METRICS_FILE,
    ANALYTIC_VALIDATION_FILE,
    ARTIFACT_PATHS,
    CROSSOVERS_FILE,
    INDEXED_PAYLOAD_PATHS,
    MANIFEST_FILE,
    PAYLOAD_INDEX_FILE,
    RUN_FILE,
    SEQUENCE_METRICS_FILE,
    SUCCESS_FILE,
    AnalyticValidationV1Alpha1,
    PayloadFileEntryV1Alpha1,
    PayloadIndexV1Alpha1,
    SuccessMarkerV1Alpha1,
)
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    expected_conditions,
    expected_sequence_ids,
    validate_result_bundle,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    EXPERIMENT_MANIFEST_ADAPTER,
    AnalyticCrossoverManifest,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    METRIC_RECORD_ADAPTER,
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    MetricRecordV1Alpha1,
    RunRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.validation import build_analytic_validation

MAX_SEQUENCE_COUNT = 10_000
MAX_BOOTSTRAP_REPLICATES = 20_000
MAX_BOOTSTRAP_CELLS = 20_000_000
MAX_SEQUENCE_ROWS = 2_000_000
MAX_LINE_BYTES = 1024 * 1024
MAX_SCIENTIFIC_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024

_STAGING_PREFIX = ".ffb-staging-"
_ARTIFACT_DOMAIN = b"fusion-fault-bench/artifact/v1\x00"
_RUN_RECORD_DOMAIN = b"fusion-fault-bench/run-record/v1\x00"
_RUN_ID_DOMAIN = b"fusion-fault-bench/run-id/v1\x00"
_ARTIFACT_CONTRACT = b"ffb.scientific-payload/v1"
_READ_CHUNK_BYTES = 1024 * 1024
_SAFE_REPOSITORY_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactValidationError(ValueError):
    """The artifact is malformed, unsafe, noncanonical, or contradictory."""


@dataclass(frozen=True, slots=True)
class ArtifactWriteRequest:
    """Already-built scientific records and completed run provenance."""

    manifest: AnalyticCrossoverManifest
    run: RunRecordV1Alpha1
    metrics: Sequence[MetricRecordV1Alpha1]
    aggregates: Sequence[AggregateMetricRecordV1Alpha1]
    crossovers: Sequence[CrossoverRecordV1Alpha1]
    analytic_validation: AnalyticValidationV1Alpha1


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """One fully reloaded and cross-validated M1 artifact."""

    path: Path
    manifest: AnalyticCrossoverManifest
    run: RunRecordV1Alpha1
    metrics: tuple[MetricRecordV1Alpha1, ...]
    aggregates: tuple[AggregateMetricRecordV1Alpha1, ...]
    crossovers: tuple[CrossoverRecordV1Alpha1, ...]
    analytic_validation: AnalyticValidationV1Alpha1
    payload_index: PayloadIndexV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedArtifact:
    files: Mapping[str, bytes]
    manifest: AnalyticCrossoverManifest
    run: RunRecordV1Alpha1
    metrics: tuple[MetricRecordV1Alpha1, ...]
    aggregates: tuple[AggregateMetricRecordV1Alpha1, ...]
    crossovers: tuple[CrossoverRecordV1Alpha1, ...]
    analytic_validation: AnalyticValidationV1Alpha1
    payload_index: PayloadIndexV1Alpha1


@dataclass(frozen=True, slots=True)
class _ArtifactTreeSnapshot:
    root_stat: os.stat_result
    entries: Mapping[str, os.stat_result]


def _positive_zero(value: Any) -> Any:
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {key: _positive_zero(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_positive_zero(item) for item in value]
    return value


def canonical_json_bytes(value: BaseModel | Mapping[str, Any]) -> bytes:
    """Serialize one model or mapping using the exact artifact JSON contract."""

    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=True)
    else:
        payload = dict(value)
    normalized = _positive_zero(payload)
    rendered = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (rendered + "\n").encode("utf-8")


def canonical_ndjson_bytes(records: Sequence[BaseModel]) -> bytes:
    """Serialize canonical one-record-per-line NDJSON with bounded lines."""

    output = bytearray()
    for index, record in enumerate(records):
        line = canonical_json_bytes(record)
        if len(line) > MAX_LINE_BYTES:
            raise ArtifactValidationError(f"NDJSON record {index} exceeds the 1 MiB line cap")
        if len(output) + len(line) > MAX_SCIENTIFIC_MEMBER_BYTES:
            raise ArtifactValidationError("NDJSON member exceeds the 512 MiB cap")
        output.extend(line)
    if not output:
        raise ArtifactValidationError("NDJSON members must contain at least one record")
    return bytes(output)


def derive_run_id(
    *,
    manifest_sha256: str,
    git_revision: str,
    lockfile_sha256: str,
    package_version: str,
) -> str:
    """Derive the preregistered deterministic run identifier."""

    def framed(value: bytes) -> bytes:
        return len(value).to_bytes(4, "big") + value

    preimage = b"".join(
        (
            _RUN_ID_DOMAIN,
            framed(manifest_sha256.encode("utf-8")),
            framed(git_revision.encode("utf-8")),
            framed(lockfile_sha256.encode("utf-8")),
            framed(package_version.encode("utf-8")),
            framed(_ARTIFACT_CONTRACT),
        )
    )
    return f"run:{hashlib.sha256(preimage).hexdigest()}"


def compute_artifact_digest(payload_index_file_bytes: bytes) -> str:
    """Digest the exact canonical payload-index file bytes with length framing."""

    preimage = b"".join(
        (
            _ARTIFACT_DOMAIN,
            len(payload_index_file_bytes).to_bytes(8, "big"),
            payload_index_file_bytes,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def compute_run_record_digest(run_record_file_bytes: bytes) -> str:
    """Digest the exact finalized canonical run-record bytes with length framing."""

    preimage = b"".join(
        (
            _RUN_RECORD_DOMAIN,
            len(run_record_file_bytes).to_bytes(8, "big"),
            run_record_file_bytes,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _condition_key(
    *,
    fault_family: str,
    fault_axis: str,
    severity: SeverityCoordinate,
) -> tuple[str, str, int, float, str, str]:
    return (
        fault_family,
        fault_axis,
        severity.index,
        severity.magnitude,
        severity.direction,
        severity.unit,
    )


def _expected_condition_keys(
    manifest: AnalyticCrossoverManifest,
) -> tuple[tuple[str, str, int, float, str, str], ...]:
    return tuple(
        (
            condition.fault_family,
            condition.fault_axis,
            condition.severity_index,
            condition.magnitude,
            condition.direction,
            condition.unit,
        )
        for condition in expected_conditions(manifest)
    )


def _expected_directions(manifest: AnalyticCrossoverManifest) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            condition.direction
            for condition in expected_conditions(manifest)
            if condition.direction != "identity"
        )
    )


def _ordered_records(
    request: ArtifactWriteRequest,
) -> tuple[
    tuple[MetricRecordV1Alpha1, ...],
    tuple[AggregateMetricRecordV1Alpha1, ...],
    tuple[CrossoverRecordV1Alpha1, ...],
]:
    manifest = request.manifest
    sequence_ranks = {
        sequence_id: index for index, sequence_id in enumerate(expected_sequence_ids(manifest))
    }
    condition_ranks = {
        condition: index for index, condition in enumerate(_expected_condition_keys(manifest))
    }
    method_ranks = {method: index for index, method in enumerate(manifest.methods)}
    direction_ranks = {
        direction: index for index, direction in enumerate(_expected_directions(manifest))
    }

    def metric_key(record: MetricRecordV1Alpha1) -> tuple[int, int, int]:
        try:
            return (
                sequence_ranks[record.sequence_id],
                condition_ranks[
                    _condition_key(
                        fault_family=record.fault_family,
                        fault_axis=record.fault_axis,
                        severity=record.severity,
                    )
                ],
                method_ranks[record.method_id],
            )
        except KeyError as error:
            raise ArtifactValidationError(
                "sequence metric cannot be ordered from the manifest"
            ) from error

    def aggregate_key(record: AggregateMetricRecordV1Alpha1) -> tuple[int, int]:
        try:
            condition_rank = condition_ranks[
                _condition_key(
                    fault_family=record.fault_family,
                    fault_axis=record.fault_axis,
                    severity=record.severity,
                )
            ]
            if record.metric_name == "fused-minus-healthy":
                return condition_rank, len(method_ranks)
            return condition_rank, method_ranks[record.method_id]
        except KeyError as error:
            raise ArtifactValidationError(
                "aggregate metric cannot be ordered from the manifest"
            ) from error

    def crossover_key(record: CrossoverRecordV1Alpha1) -> int:
        try:
            return direction_ranks[record.direction]
        except KeyError as error:
            raise ArtifactValidationError(
                "crossover cannot be ordered from the manifest"
            ) from error

    return (
        tuple(sorted(request.metrics, key=metric_key)),
        tuple(sorted(request.aggregates, key=aggregate_key)),
        tuple(sorted(request.crossovers, key=crossover_key)),
    )


def _validate_execution_caps(
    manifest: AnalyticCrossoverManifest,
    *,
    sequence_rows: int | None = None,
) -> None:
    sequence_count = manifest.source.sequence_count
    bootstrap_replicates = manifest.evaluation.bootstrap.replicates
    implied_sequence_rows = (
        sequence_count * len(expected_conditions(manifest)) * len(manifest.methods)
    )
    if sequence_count > MAX_SEQUENCE_COUNT:
        raise ArtifactValidationError("sequence_count exceeds the M1 execution cap")
    if bootstrap_replicates > MAX_BOOTSTRAP_REPLICATES:
        raise ArtifactValidationError("bootstrap replicates exceed the M1 execution cap")
    if sequence_count * bootstrap_replicates > MAX_BOOTSTRAP_CELLS:
        raise ArtifactValidationError("bootstrap matrix exceeds the M1 execution cap")
    if implied_sequence_rows > MAX_SEQUENCE_ROWS:
        raise ArtifactValidationError(
            "manifest-implied sequence records exceed the M1 execution cap"
        )
    if sequence_rows is not None and sequence_rows > MAX_SEQUENCE_ROWS:
        raise ArtifactValidationError("sequence records exceed the M1 execution cap")


def _validate_run_identity(
    manifest: AnalyticCrossoverManifest,
    run: RunRecordV1Alpha1,
) -> str:
    manifest_digest = sha256_digest(manifest)
    expected_run_id = derive_run_id(
        manifest_sha256=manifest_digest,
        git_revision=run.git_revision,
        lockfile_sha256=run.lockfile_sha256,
        package_version=run.package_version,
    )
    if run.manifest_sha256 != manifest_digest:
        raise ArtifactValidationError("run manifest digest disagrees with the manifest")
    if run.run_id != expected_run_id:
        raise ArtifactValidationError("run_id disagrees with deterministic run identity")
    _validate_logical_command(manifest, run)
    return manifest_digest


def _validate_logical_command(
    manifest: AnalyticCrossoverManifest,
    run: RunRecordV1Alpha1,
) -> None:
    expected_output = f"reports/generated/{manifest.experiment}-{run.manifest_sha256[:12]}"
    if len(run.command) != 5:
        raise ArtifactValidationError("run command does not use the frozen logical structure")
    executable, subcommand, manifest_path, output_flag, output_path = run.command
    if (
        executable != "ffb"
        or subcommand != "run"
        or output_flag != "--output-dir"
        or output_path != expected_output
    ):
        raise ArtifactValidationError("run command disagrees with the frozen logical command")
    path_parts = manifest_path.split("/")
    if (
        not manifest_path
        or manifest_path == "manifest.json"
        or manifest_path.startswith("/")
        or "\\" in manifest_path
        or any(part in {"", ".", ".."} for part in path_parts)
        or any(_SAFE_REPOSITORY_SEGMENT.fullmatch(part) is None for part in path_parts)
        or not manifest_path.endswith(".json")
    ):
        raise ArtifactValidationError(
            "run command manifest must be a safe tracked POSIX repository-relative path"
        )


def _validate_analytic_evidence(
    manifest: AnalyticCrossoverManifest,
    run: RunRecordV1Alpha1,
    metrics: tuple[MetricRecordV1Alpha1, ...],
    analytic: AnalyticValidationV1Alpha1,
) -> None:
    if analytic.run_id != run.run_id:
        raise ArtifactValidationError("analytic validation has a different run_id")
    if analytic.manifest_sha256 != run.manifest_sha256:
        raise ArtifactValidationError("analytic validation has a different manifest digest")
    if (
        analytic.monte_carlo_standard_error_multiplier
        != manifest.analytic_validation.monte_carlo_standard_error_multiplier
    ):
        raise ArtifactValidationError("analytic validation multiplier disagrees with the manifest")

    expected = build_analytic_validation(
        manifest,
        run_id=run.run_id,
        metrics=metrics,
    )
    if analytic != expected:
        raise ArtifactValidationError(
            "analytic validation disagrees with independent population and empirical evidence"
        )


def _finalize_run(run: RunRecordV1Alpha1, artifact_digest: str) -> RunRecordV1Alpha1:
    value = run.model_dump(mode="python", by_alias=True)
    value["artifact_sha256"] = artifact_digest
    return RunRecordV1Alpha1.model_validate(value)


def _build_payload_index(
    *,
    run_id: str,
    manifest_sha256: str,
    payload_files: Mapping[str, bytes],
) -> PayloadIndexV1Alpha1:
    return PayloadIndexV1Alpha1(
        schema="ffb.payload-index/v1alpha1",
        artifact_contract="ffb.scientific-payload/v1",
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        files=tuple(
            PayloadFileEntryV1Alpha1(
                path=path,  # type: ignore[arg-type]
                byte_length=len(payload_files[path]),
                sha256=_sha256_bytes(payload_files[path]),
            )
            for path in INDEXED_PAYLOAD_PATHS
        ),
    )


def _prepare_artifact(request: ArtifactWriteRequest) -> _PreparedArtifact:
    manifest = request.manifest
    _validate_execution_caps(manifest, sequence_rows=len(request.metrics))
    manifest_digest = _validate_run_identity(manifest, request.run)
    metrics, aggregates, crossovers = _ordered_records(request)
    _validate_analytic_evidence(
        manifest,
        request.run,
        metrics,
        request.analytic_validation,
    )

    payload_files: dict[str, bytes] = {
        MANIFEST_FILE: canonical_json_bytes(manifest),
        SEQUENCE_METRICS_FILE: canonical_ndjson_bytes(metrics),
        AGGREGATE_METRICS_FILE: canonical_ndjson_bytes(aggregates),
        CROSSOVERS_FILE: canonical_ndjson_bytes(crossovers),
        ANALYTIC_VALIDATION_FILE: canonical_json_bytes(request.analytic_validation),
    }
    for path, value in payload_files.items():
        if len(value) > MAX_SCIENTIFIC_MEMBER_BYTES:
            raise ArtifactValidationError(f"{path} exceeds the 512 MiB member cap")
        if path in {MANIFEST_FILE, ANALYTIC_VALIDATION_FILE} and len(value) > MAX_LINE_BYTES:
            raise ArtifactValidationError(f"{path} exceeds the 1 MiB line cap")

    payload_index = _build_payload_index(
        run_id=request.run.run_id,
        manifest_sha256=manifest_digest,
        payload_files=payload_files,
    )
    payload_index_bytes = canonical_json_bytes(payload_index)
    artifact_digest = compute_artifact_digest(payload_index_bytes)
    run = _finalize_run(request.run, artifact_digest)
    run_bytes = canonical_json_bytes(run)
    run_digest = compute_run_record_digest(run_bytes)
    success_marker = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_digest,
        run_sha256=run_digest,
    )

    validate_result_bundle(manifest, run, metrics, aggregates, crossovers)
    _validate_analytic_evidence(
        manifest,
        run,
        metrics,
        request.analytic_validation,
    )

    files = {
        **payload_files,
        PAYLOAD_INDEX_FILE: payload_index_bytes,
        RUN_FILE: run_bytes,
        SUCCESS_FILE: canonical_json_bytes(success_marker),
    }
    if any(
        len(files[path]) > MAX_LINE_BYTES for path in (PAYLOAD_INDEX_FILE, RUN_FILE, SUCCESS_FILE)
    ):
        raise ArtifactValidationError("artifact metadata exceeds the 1 MiB line cap")
    if sum(map(len, files.values())) > MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("complete artifact exceeds the 1 GiB cap")
    return _PreparedArtifact(
        files=files,
        manifest=manifest,
        run=run,
        metrics=metrics,
        aggregates=aggregates,
        crossovers=crossovers,
        analytic_validation=request.analytic_validation,
        payload_index=payload_index,
    )


def _reject_nonstandard_number(value: str) -> NoReturn:
    raise ArtifactValidationError(f"non-standard JSON number is forbidden: {value}")


def _parse_json_int(value: str) -> int:
    if value == "-0":
        raise ArtifactValidationError("negative zero is forbidden in canonical JSON")
    return int(value)


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if parsed == 0.0 and value.startswith("-"):
        raise ArtifactValidationError("negative zero is forbidden in canonical JSON")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactValidationError(f"duplicate JSON object key is forbidden: {key}")
        value[key] = item
    return value


def _strict_json_body(data: bytes, *, label: str) -> bytes:
    if len(data) > MAX_LINE_BYTES:
        raise ArtifactValidationError(f"{label} exceeds the 1 MiB line cap")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ArtifactValidationError(f"{label} contains a UTF-8 BOM")
    if b"\r" in data:
        raise ArtifactValidationError(f"{label} contains a forbidden CR byte")
    if not data.endswith(b"\n"):
        raise ArtifactValidationError(f"{label} is missing its terminal LF")
    body = data[:-1]
    if not body or b"\n" in body:
        raise ArtifactValidationError(f"{label} is blank or has noncanonical extra LF bytes")
    try:
        raw = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactValidationError(f"{label} is not valid UTF-8") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_number,
            parse_float=_parse_json_float,
            parse_int=_parse_json_int,
        )
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must contain a top-level JSON object")
    return body


def _load_canonical_model[ModelT: BaseModel](
    data: bytes,
    *,
    label: str,
    validate: Callable[[bytes], ModelT],
) -> ModelT:
    body = _strict_json_body(data, label=label)
    try:
        model = validate(body)
    except (ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"{label} violates its schema: {error}") from error
    if canonical_json_bytes(model) != data:
        raise ArtifactValidationError(f"{label} is not canonical JSON")
    return model


def _load_ndjson[ModelT: BaseModel](
    path: Path,
    *,
    label: str,
    validate: Callable[[bytes], ModelT],
    record_cap: int | None = None,
    expected_stat: os.stat_result | None = None,
) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    descriptor = _open_regular_nofollow(path, expected_stat=expected_stat)
    with os.fdopen(descriptor, "rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            line_number += 1
            if len(line) > MAX_LINE_BYTES:
                raise ArtifactValidationError(f"{label} line {line_number} exceeds the 1 MiB cap")
            if record_cap is not None and line_number > record_cap:
                raise ArtifactValidationError(f"{label} exceeds its record-count cap")
            records.append(
                _load_canonical_model(
                    line,
                    label=f"{label} line {line_number}",
                    validate=validate,
                )
            )
    if not records:
        raise ArtifactValidationError(f"{label} must contain at least one record")
    return tuple(records)


def _validate_metric_json(data: bytes) -> MetricRecordV1Alpha1:
    return cast(MetricRecordV1Alpha1, METRIC_RECORD_ADAPTER.validate_json(data))


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_nofollow(
    path: Path,
    *,
    expected_stat: os.stat_result | None = None,
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactValidationError(f"{path.name} is not a regular file")
        if expected_stat is not None and _stat_fingerprint(file_stat) != _stat_fingerprint(
            expected_stat
        ):
            raise ArtifactValidationError(f"{path.name} changed after the artifact tree scan")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_small_file(
    path: Path,
    *,
    label: str,
    expected_stat: os.stat_result | None = None,
) -> bytes:
    descriptor = _open_regular_nofollow(path, expected_stat=expected_stat)
    with os.fdopen(descriptor, "rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_LINE_BYTES:
            raise ArtifactValidationError(f"{label} exceeds the 1 MiB line cap")
        data = stream.read(MAX_LINE_BYTES + 1)
        if len(data) > MAX_LINE_BYTES:
            raise ArtifactValidationError(f"{label} exceeds the 1 MiB line cap")
        return data


def _sha256_file(
    path: Path,
    *,
    expected_stat: os.stat_result | None = None,
) -> str:
    digest = hashlib.sha256()
    descriptor = _open_regular_nofollow(path, expected_stat=expected_stat)
    with os.fdopen(descriptor, "rb") as stream:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_artifact_root_symlink_components(root: Path) -> None:
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            component_stat = os.lstat(current)
        except OSError as error:
            raise ArtifactValidationError(
                f"cannot inspect artifact root path component: {current}"
            ) from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ArtifactValidationError(
                f"artifact root path contains a symlink component: {current}"
            )


def _require_safe_artifact_tree(root: Path) -> _ArtifactTreeSnapshot:
    absolute = Path(os.path.abspath(os.fspath(root)))
    _reject_artifact_root_symlink_components(absolute)
    try:
        root_stat = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError(f"cannot inspect artifact directory: {error}") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ArtifactValidationError("artifact root must be a real directory, not a symlink")

    entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(absolute) as iterator:
            for entry in iterator:
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ArtifactValidationError(f"artifact member {entry.name!r} is a symlink")
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ArtifactValidationError(
                        f"artifact member {entry.name!r} is not a regular file"
                    )
                entries[entry.name] = entry_stat
    except OSError as error:
        raise ArtifactValidationError(f"cannot inspect artifact members: {error}") from error

    actual_names = set(entries)
    expected_names = set(ARTIFACT_PATHS)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ArtifactValidationError(
            f"artifact file allowlist mismatch; missing={missing!r}, extra={extra!r}"
        )
    for path in INDEXED_PAYLOAD_PATHS:
        if entries[path].st_size > MAX_SCIENTIFIC_MEMBER_BYTES:
            raise ArtifactValidationError(f"{path} exceeds the 512 MiB member cap")
    if sum(entry.st_size for entry in entries.values()) > MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("complete artifact exceeds the 1 GiB cap")
    return _ArtifactTreeSnapshot(root_stat=root_stat, entries=entries)


def _verify_tree_snapshot(
    root: Path,
    snapshot: _ArtifactTreeSnapshot,
) -> None:
    _reject_artifact_root_symlink_components(root)
    try:
        current_root_stat = os.lstat(root)
    except OSError as error:
        raise ArtifactValidationError("artifact root disappeared during validation") from error
    if _stat_fingerprint(current_root_stat) != _stat_fingerprint(snapshot.root_stat):
        raise ArtifactValidationError("artifact root changed during validation")

    current_entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                current_entries[entry.name] = entry.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactValidationError("cannot recheck artifact members") from error
    if set(current_entries) != set(snapshot.entries):
        raise ArtifactValidationError("artifact file allowlist changed during validation")
    for name, expected_stat in snapshot.entries.items():
        try:
            current_stat = current_entries[name]
        except KeyError as error:
            raise ArtifactValidationError(f"artifact member disappeared: {name}") from error
        if _stat_fingerprint(current_stat) != _stat_fingerprint(expected_stat):
            raise ArtifactValidationError(f"artifact member changed during validation: {name}")


def _validate_loaded_order(
    manifest: AnalyticCrossoverManifest,
    metrics: Sequence[MetricRecordV1Alpha1],
    aggregates: Sequence[AggregateMetricRecordV1Alpha1],
    crossovers: Sequence[CrossoverRecordV1Alpha1],
) -> None:
    request = ArtifactWriteRequest(
        manifest=manifest,
        run=RunRecordV1Alpha1.model_construct(),
        metrics=metrics,
        aggregates=aggregates,
        crossovers=crossovers,
        analytic_validation=AnalyticValidationV1Alpha1.model_construct(),
    )
    ordered_metrics, ordered_aggregates, ordered_crossovers = _ordered_records(request)
    if tuple(metrics) != ordered_metrics:
        raise ArtifactValidationError("sequence metrics are in the wrong canonical order")
    if tuple(aggregates) != ordered_aggregates:
        raise ArtifactValidationError("aggregate metrics are in the wrong canonical order")
    if tuple(crossovers) != ordered_crossovers:
        raise ArtifactValidationError("crossovers are in the wrong canonical order")


def _validate_indexed_files(
    root: Path,
    index: PayloadIndexV1Alpha1,
    entries: Mapping[str, os.stat_result],
) -> None:
    for expected_path, indexed in zip(INDEXED_PAYLOAD_PATHS, index.files, strict=True):
        if indexed.path != expected_path:
            raise ArtifactValidationError("payload index path order is invalid")
        if indexed.byte_length != entries[expected_path].st_size:
            raise ArtifactValidationError(f"{expected_path} byte length disagrees with the index")
        if indexed.sha256 != _sha256_file(
            root / expected_path,
            expected_stat=entries[expected_path],
        ):
            raise ArtifactValidationError(f"{expected_path} digest disagrees with the index")


def _load_artifact(path: Path) -> LoadedArtifact:
    root = Path(os.path.abspath(os.fspath(path)))
    snapshot = _require_safe_artifact_tree(root)
    entries = snapshot.entries

    manifest_data = _read_small_file(
        root / MANIFEST_FILE,
        label=MANIFEST_FILE,
        expected_stat=entries[MANIFEST_FILE],
    )
    manifest_union = _load_canonical_model(
        manifest_data,
        label=MANIFEST_FILE,
        validate=EXPERIMENT_MANIFEST_ADAPTER.validate_json,
    )
    if not isinstance(manifest_union, AnalyticCrossoverManifest):
        raise ArtifactValidationError("M1 artifact manifest must be analytic-crossover")
    manifest = manifest_union
    _validate_execution_caps(manifest)

    run_bytes = _read_small_file(
        root / RUN_FILE,
        label=RUN_FILE,
        expected_stat=entries[RUN_FILE],
    )
    run = _load_canonical_model(
        run_bytes,
        label=RUN_FILE,
        validate=RunRecordV1Alpha1.model_validate_json,
    )
    _validate_run_identity(manifest, run)

    metrics = _load_ndjson(
        root / SEQUENCE_METRICS_FILE,
        label=SEQUENCE_METRICS_FILE,
        validate=_validate_metric_json,
        record_cap=MAX_SEQUENCE_ROWS,
        expected_stat=entries[SEQUENCE_METRICS_FILE],
    )
    aggregates = _load_ndjson(
        root / AGGREGATE_METRICS_FILE,
        label=AGGREGATE_METRICS_FILE,
        validate=AggregateMetricRecordV1Alpha1.model_validate_json,
        expected_stat=entries[AGGREGATE_METRICS_FILE],
    )
    crossovers = _load_ndjson(
        root / CROSSOVERS_FILE,
        label=CROSSOVERS_FILE,
        validate=CrossoverRecordV1Alpha1.model_validate_json,
        expected_stat=entries[CROSSOVERS_FILE],
    )
    analytic_validation = _load_canonical_model(
        _read_small_file(
            root / ANALYTIC_VALIDATION_FILE,
            label=ANALYTIC_VALIDATION_FILE,
            expected_stat=entries[ANALYTIC_VALIDATION_FILE],
        ),
        label=ANALYTIC_VALIDATION_FILE,
        validate=AnalyticValidationV1Alpha1.model_validate_json,
    )
    payload_index_bytes = _read_small_file(
        root / PAYLOAD_INDEX_FILE,
        label=PAYLOAD_INDEX_FILE,
        expected_stat=entries[PAYLOAD_INDEX_FILE],
    )
    payload_index = _load_canonical_model(
        payload_index_bytes,
        label=PAYLOAD_INDEX_FILE,
        validate=PayloadIndexV1Alpha1.model_validate_json,
    )

    _validate_execution_caps(manifest, sequence_rows=len(metrics))
    _validate_loaded_order(manifest, metrics, aggregates, crossovers)
    _validate_analytic_evidence(manifest, run, metrics, analytic_validation)
    _validate_indexed_files(root, payload_index, entries)

    manifest_digest = sha256_digest(manifest)
    if payload_index.run_id != run.run_id:
        raise ArtifactValidationError("payload index run_id disagrees with run.json")
    if payload_index.manifest_sha256 != manifest_digest:
        raise ArtifactValidationError("payload index manifest digest is invalid")
    artifact_digest = compute_artifact_digest(payload_index_bytes)
    if run.artifact_sha256 != artifact_digest:
        raise ArtifactValidationError("run artifact digest disagrees with payload-index.json")
    run_digest = compute_run_record_digest(run_bytes)
    marker = _load_canonical_model(
        _read_small_file(
            root / SUCCESS_FILE,
            label=SUCCESS_FILE,
            expected_stat=entries[SUCCESS_FILE],
        ),
        label=SUCCESS_FILE,
        validate=SuccessMarkerV1Alpha1.model_validate_json,
    )
    if marker.artifact_sha256 != artifact_digest:
        raise ArtifactValidationError("_SUCCESS artifact digest is invalid")
    if marker.run_sha256 != run_digest:
        raise ArtifactValidationError("_SUCCESS run digest is invalid")

    validate_result_bundle(manifest, run, metrics, aggregates, crossovers)
    _verify_tree_snapshot(root, snapshot)
    return LoadedArtifact(
        path=root,
        manifest=manifest,
        run=run,
        metrics=metrics,
        aggregates=aggregates,
        crossovers=crossovers,
        analytic_validation=analytic_validation,
        payload_index=payload_index,
        artifact_sha256=artifact_digest,
        run_sha256=run_digest,
    )


def load_artifact(path: Path) -> LoadedArtifact:
    """Strictly load and validate one complete artifact directory."""

    try:
        return _load_artifact(path)
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactValidationError(f"invalid artifact: {error}") from error


def discover_git_metadata_dirs(source_root: Path | None = None) -> tuple[Path, ...]:
    """Return absolute worktree and common Git metadata directories."""

    root = Path.cwd() if source_root is None else source_root
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(root),
                "rev-parse",
                "--path-format=absolute",
                "--git-dir",
                "--git-common-dir",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArtifactValidationError("Git metadata directories are unavailable") from error
    directories = tuple(
        Path(line).resolve(strict=True) for line in result.stdout.splitlines() if line.strip()
    )
    if not directories:
        raise ArtifactValidationError("Git metadata discovery returned no directories")
    return tuple(dict.fromkeys(directories))


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_equal_or_descendant(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _reject_git_metadata_destination(
    destination: Path,
    git_metadata_dirs: Sequence[Path],
) -> None:
    for directory in git_metadata_dirs:
        metadata = directory.resolve(strict=True)
        if _is_equal_or_descendant(destination, metadata):
            raise ArtifactValidationError("artifact destination is inside Git metadata")


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_or_create_real_directory(path: Path) -> int:
    """Traverse an absolute directory path without following symlink components."""

    if not path.is_absolute():
        raise ArtifactValidationError("destination parent must be absolute")
    descriptor = os.open(path.anchor, _directory_open_flags())
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current /= part
            try:
                child = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o777, dir_fd=descriptor)
                try:
                    child = os.open(part, _directory_open_flags(), dir_fd=descriptor)
                except OSError as error:
                    raise ArtifactValidationError(
                        f"destination parent component is not a real directory: {current}"
                    ) from error
            except OSError as error:
                raise ArtifactValidationError(
                    f"destination parent component is not a real directory: {current}"
                ) from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _assert_directory_fd_matches_path(
    descriptor: int,
    path: Path,
    *,
    label: str,
) -> None:
    descriptor_stat = os.fstat(descriptor)
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ArtifactValidationError(f"{label} changed during artifact publication") from error
    if (
        not stat.S_ISDIR(descriptor_stat.st_mode)
        or not stat.S_ISDIR(path_stat.st_mode)
        or _stat_identity(descriptor_stat) != _stat_identity(path_stat)
    ):
        raise ArtifactValidationError(f"{label} changed during artifact publication")


def _reject_directory_fd_in_git_metadata(
    descriptor: int,
    git_metadata_dirs: Sequence[Path],
) -> None:
    protected = {
        _stat_identity(os.stat(directory.resolve(strict=True), follow_symlinks=False))
        for directory in git_metadata_dirs
    }
    current = os.dup(descriptor)
    try:
        while True:
            current_stat = os.fstat(current)
            if _stat_identity(current_stat) in protected:
                raise ArtifactValidationError("artifact destination is inside Git metadata")
            parent = os.open("..", _directory_open_flags(), dir_fd=current)
            parent_stat = os.fstat(parent)
            if _stat_identity(parent_stat) == _stat_identity(current_stat):
                os.close(parent)
                break
            os.close(current)
            current = parent
    finally:
        os.close(current)


def _entry_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_at(directory_fd: int, name: str, *, byte_cap: int) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactValidationError(f"staged member is not a regular file: {name}")
        if file_stat.st_size > byte_cap:
            raise ArtifactValidationError(f"staged member exceeds its byte cap: {name}")
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
            raise ArtifactValidationError(f"staged member exceeds its byte cap: {name}")
        return value
    finally:
        os.close(descriptor)


def _write_exclusive_at(directory_fd: int, name: str, value: bytes) -> None:
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
    try:
        remaining = memoryview(value)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("short write while staging artifact member")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if _read_at(directory_fd, name, byte_cap=len(value)) != value:
        raise ArtifactValidationError(f"staging verification failed for {name}")


def _create_staging_directory_at(parent_fd: int) -> tuple[str, int]:
    for _ in range(128):
        name = f"{_STAGING_PREFIX}{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
        except BaseException:
            with suppress(OSError):
                os.rmdir(name, dir_fd=parent_fd)
            raise
        return name, descriptor
    raise ArtifactValidationError("could not allocate a unique artifact staging directory")


def _safe_cleanup_staging_at(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
) -> None:
    for name in ARTIFACT_PATHS:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=staging_fd)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _validate_relative_member_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ArtifactValidationError("artifact publication requires a single path segment")


def _atomic_rename_no_replace_at(
    source_dir_fd: int,
    source_name: str,
    destination_dir_fd: int,
    destination_name: str,
) -> None:
    """Atomically publish relative to pinned directories without replacing a target."""

    _validate_relative_member_name(source_name)
    _validate_relative_member_name(destination_name)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = 0x00000004
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            source_bytes,
            destination_dir_fd,
            destination_bytes,
            rename_exclusive,
        )
    elif sys.platform.startswith("linux"):
        rename_no_replace = 1
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            source_bytes,
            destination_dir_fd,
            destination_bytes,
            rename_no_replace,
        )
    else:
        raise ArtifactValidationError(
            f"atomic no-replace directory publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def write_artifact(
    request: ArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path | None = None,
    git_metadata_dirs: Sequence[Path] | None = None,
) -> LoadedArtifact:
    """Write, reload, and atomically publish one no-overwrite M1 artifact."""

    prepared = _prepare_artifact(request)
    target = _absolute_lexical(destination)
    metadata_dirs = (
        discover_git_metadata_dirs(source_root)
        if git_metadata_dirs is None
        else tuple(git_metadata_dirs)
    )
    _reject_git_metadata_destination(target, metadata_dirs)
    if os.path.lexists(target):
        raise FileExistsError(f"artifact destination already exists: {target}")

    parent = target.parent
    parent_fd = _open_or_create_real_directory(parent)
    try:
        _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
        _reject_directory_fd_in_git_metadata(parent_fd, metadata_dirs)
        if _entry_exists_at(parent_fd, target.name):
            raise FileExistsError(f"artifact destination already exists: {target}")
        staging_name, staging_fd = _create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        published = False
        try:
            for path in (*INDEXED_PAYLOAD_PATHS, PAYLOAD_INDEX_FILE, RUN_FILE):
                _write_exclusive_at(staging_fd, path, prepared.files[path])
            for path in (*INDEXED_PAYLOAD_PATHS, PAYLOAD_INDEX_FILE, RUN_FILE):
                if (
                    _read_at(staging_fd, path, byte_cap=len(prepared.files[path]))
                    != prepared.files[path]
                ):
                    raise ArtifactValidationError(f"staged member changed after write: {path}")

            _write_exclusive_at(staging_fd, SUCCESS_FILE, prepared.files[SUCCESS_FILE])
            os.fsync(staging_fd)
            _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
            _load_artifact(staging)
            _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
            _reject_directory_fd_in_git_metadata(parent_fd, metadata_dirs)
            if _entry_exists_at(parent_fd, target.name):
                raise FileExistsError(f"artifact destination already exists: {target}")
            _atomic_rename_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            published = True
            os.fsync(parent_fd)
            _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
            loaded = load_artifact(target)
            _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
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
