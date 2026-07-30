"""Secure local execution and repeat evidence for the frozen M5 replay.

This module is deliberately split from public curation.  It reads the dataset
root only from ``NUSCENES_ROOT``, permits the adapter to open only the fixed
nuScenes metadata-table allowlist, and writes ignored local scientific source
members.  It does not manufacture release validation, review, figure, or
software-verification attestations.
"""

from __future__ import annotations

import builtins
import fcntl
import hashlib
import io
import json
import math
import os
import re
import resource
import stat
import subprocess
import sys
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Never, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from fusion_fault_bench.adapters.nuscenes import (
    NuScenesAdapterError,
    NuScenesMiniMetadata,
    load_nuscenes_mini,
)
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    assert_directory_descriptor_matches_path,
    atomic_rename_directory_no_replace_at,
    canonical_json_bytes,
    compute_run_record_digest,
    create_staging_directory_at,
    derive_run_id,
    entry_exists_at,
    open_or_create_real_directory,
    read_file_at,
    reject_directory_descriptor_in_git_metadata,
    write_exclusive_file_at,
)
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    EXPECTED_KEYFRAME_BLOB_CHECK_COUNT,
)
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_FIT_RUN_SHA256,
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    M5_SCIENTIFIC_SOURCE_ROLES,
    M5_TRACKED_AGGREGATE_TERMS,
    REPLAY_CURATED_ARTIFACT_CONTRACT,
    ReplayDescriptorAggregateV1,
    ReplayExecutionResourceEvidenceV1,
    ReplayProfileSummaryV1,
    ReplayRepeatVerificationV1,
    ReplaySourceMemberCommitmentV1,
)
from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayHealthResultV1,
    ReplayHealthSequenceEventV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_PATH,
    M5_REPLAY_INTENT_SHA256,
    M5_SCENE_NAMES,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    ProvenanceError,
    collect_runtime_environment,
    discover_clean_source,
    verify_locked_execution,
)
from fusion_fault_bench.replay_artifacts import canonical_replay_ndjson_bytes
from fusion_fault_bench.replay_benchmark import (
    ReplayBenchmarkEvidence,
    run_replay_benchmark,
)
from fusion_fault_bench.replay_curation import (
    ReplayCuratedAggregateEvidence,
    ReplayLogGroupBinding,
    curate_replay_evidence,
)
from fusion_fault_bench.replay_descriptors import ReplayDescriptorAggregate
from fusion_fault_bench.replay_experiments import M5_DATA_MASTER_SEED
from fusion_fault_bench.replay_fit import M4_FROZEN_CALIBRATION_SHA256
from fusion_fault_bench.replay_health_population import ReplayHealthPopulationMetric
from fusion_fault_bench.replay_inference import ReplayHealthSequenceContrast
from fusion_fault_bench.replay_persistent import ReplayPersistentSceneEvaluation
from fusion_fault_bench.replay_persistent_inference import (
    ReplayPersistentCrossoverEstimate,
    ReplayPersistentPopulationMetric,
)
from fusion_fault_bench.replay_plan import LoadedReplayPlan, load_replay_plan
from fusion_fault_bench.replay_resources import (
    ReplayResourceRunBinding,
    import_replay_execution_resource_evidence,
)
from fusion_fault_bench.replay_source import (
    ReplayPopulation,
    ReplaySourceError,
    extract_m5_replay_source,
)

_DATASET_ROOT_ENV = "NUSCENES_ROOT"
_GENERATED_ROOT = Path("reports/generated")
_LOCAL_ARTIFACT_CONTRACT = "ffb.replay-local-source-payload/v1"
_LOCAL_INDEX_FILE = "source-index.json"
_LOCAL_RESOURCES_FILE = "resources.json"
_LOCAL_RESOURCES_ROLE = "execution-resource-diagnostics"
_LOCAL_RESOURCES_SCHEMA = "ffb.replay-local-resources/v1"
_LOCAL_RUN_FILE = "run.json"
_LOCAL_SUCCESS_FILE = "_SUCCESS"
_LOCAL_ARTIFACT_DOMAIN = b"fusion-fault-bench/replay-local-source-payload/v1\x00"
_LOCAL_MAX_RECORD_BYTES = 1024 * 1024
_LOCAL_MAX_MEMBER_BYTES = 128 * 1024 * 1024
_LOCAL_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_WALL_TIME_CAP_SECONDS = 1800.0
_PEAK_RSS_CAP_BYTES = 1024 * 1024 * 1024
_BOOTSTRAP_REPLICATES = 2_000

_NUSCENES_TABLES = (
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
)

_LOCAL_MEMBER_LAYOUT = (
    (
        "descriptor-aggregates.ndjson",
        "descriptor-aggregates",
        "ffb.replay-local-descriptor-aggregate/v1",
        "descriptor_aggregates",
    ),
    (
        "health-population-metrics.ndjson",
        "health-population-metrics",
        "ffb.replay-local-health-population-metric/v1",
        "health_metrics",
    ),
    (
        "health-sequence-contrasts.ndjson",
        "health-sequence-contrasts",
        "ffb.replay-local-health-sequence-contrast/v1",
        "health_contrasts",
    ),
    (
        "health-sequence-events.ndjson",
        "health-sequence-events",
        "ffb.replay-health-sequence-event/v1",
        "health_events",
    ),
    (
        "health-sequence-results.ndjson",
        "health-sequence-results",
        "ffb.replay-health-result/v1",
        "health_results",
    ),
    (
        "persistent-crossovers.ndjson",
        "persistent-crossovers",
        "ffb.replay-local-persistent-crossover/v1",
        "persistent_crossovers",
    ),
    (
        "persistent-population-metrics.ndjson",
        "persistent-population-metrics",
        "ffb.replay-local-persistent-population-metric/v1",
        "persistent_metrics",
    ),
    (
        "persistent-scene-evaluations.ndjson",
        "persistent-scene-evaluations",
        "ffb.replay-local-persistent-scene-evaluation/v1",
        "persistent_scene_evaluations",
    ),
)
if tuple(row[1] for row in _LOCAL_MEMBER_LAYOUT) != M5_SCIENTIFIC_SOURCE_ROLES:
    raise RuntimeError("local M5 source roles disagree with the release contract")
REPLAY_LOCAL_SCIENTIFIC_PATHS = tuple(row[0] for row in _LOCAL_MEMBER_LAYOUT)
_REPLAY_LOCAL_PENDING_PATHS = (
    *REPLAY_LOCAL_SCIENTIFIC_PATHS,
    _LOCAL_INDEX_FILE,
    _LOCAL_RESOURCES_FILE,
    _LOCAL_RUN_FILE,
)
REPLAY_LOCAL_ARTIFACT_PATHS = (
    *_REPLAY_LOCAL_PENDING_PATHS,
    _LOCAL_SUCCESS_FILE,
)
_ROLE_BY_PATH = {path: role for path, role, _, _ in _LOCAL_MEMBER_LAYOUT}

_FORBIDDEN_LOCAL_KEYS = frozenset(
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
        "object_id",
        "point_cloud_path",
        "pose",
        "rotation",
        "sample_token",
        "timestamp",
        "timestamp_us",
        "translation",
    }
)
_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?:file:(?://)?|/(?:Users|home|private|tmp|Volumes)/|[A-Z]:[\\/])"
)
_RAW_PAYLOAD_PATTERN = re.compile(
    r"(?i)[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|pcd|las|laz|bin|tar|tgz|zip)(?:\s|$)"
)
_RAW_PAYLOAD_SUFFIXES = frozenset(
    {".bin", ".jpeg", ".jpg", ".las", ".laz", ".pcd", ".png", ".tar", ".tgz", ".zip"}
)
_RAW_PAYLOAD_DIRECTORIES = frozenset({"maps", "samples", "sweeps"})
_SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    r"(?:[\"'\s]*[:=][\"'\s]*|_)[A-Za-z0-9+/=_-]{8,}"
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

type ReplayResourceMeasurementScope = Literal[
    "metadata-through-canonical-scientific-members-before-publication",
    "metadata-through-publication-and-final-source-verification-before-profile-binding",
]
type _OpenPath = int | str | bytes | os.PathLike[str] | os.PathLike[bytes]
type _OsOpenPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]


class ReplayRunnerError(ValueError):
    """A fixed-stage M5 execution failure that cannot expose local details."""


@dataclass(frozen=True, slots=True)
class ReplayRunResources:
    """Measured single-scientific-worker resources with an explicit scope."""

    elapsed_seconds: float
    peak_rss_bytes: int
    measurement_scope: ReplayResourceMeasurementScope
    raw_sensor_payload_reads: int = 0
    scientific_replay_worker_count: int = 1
    gpu_used: bool = False
    torch_imported: bool = False
    cuda_used: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.elapsed_seconds) is not float
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0.0
            or self.elapsed_seconds >= _WALL_TIME_CAP_SECONDS
            or type(self.peak_rss_bytes) is not int
            or not 0 < self.peak_rss_bytes < _PEAK_RSS_CAP_BYTES
            or self.raw_sensor_payload_reads != 0
            or self.scientific_replay_worker_count != 1
            or self.gpu_used
            or self.torch_imported
            or self.cuda_used
            or self.measurement_scope
            not in {
                "metadata-through-canonical-scientific-members-before-publication",
                (
                    "metadata-through-publication-and-final-source-verification-"
                    "before-profile-binding"
                ),
            }
            or type(self.measurement_scope) is not str
            or type(self.raw_sensor_payload_reads) is not int
            or type(self.scientific_replay_worker_count) is not int
            or type(self.gpu_used) is not bool
            or type(self.torch_imported) is not bool
            or type(self.cuda_used) is not bool
        ):
            raise ValueError("M5 replay resource evidence violates a frozen cap")


@dataclass(frozen=True, slots=True)
class _ReplayLocalSourceContext:
    """Non-private source context committed by the local source index."""

    run_id: str
    fit_calibration_sha256: str
    data_master_seed: int
    bootstrap_replicates: int
    scene_frame_counts: tuple[int, ...]
    log_group_ordinals: tuple[str, ...]

    def __post_init__(self) -> None:
        distinct_groups = tuple(
            sorted(set(self.log_group_ordinals), key=lambda value: value.encode("utf-8"))
        )
        if (
            not self.run_id.startswith("run:")
            or self.fit_calibration_sha256 != M4_FROZEN_CALIBRATION_SHA256
            or self.data_master_seed != M5_DATA_MASTER_SEED
            or type(self.data_master_seed) is not int
            or self.bootstrap_replicates != _BOOTSTRAP_REPLICATES
            or type(self.bootstrap_replicates) is not int
            or len(self.scene_frame_counts) != len(M5_SCENE_NAMES)
            or any(type(value) is not int or value < 16 for value in self.scene_frame_counts)
            or len(self.log_group_ordinals) != len(M5_SCENE_NAMES)
            or any(type(value) is not str for value in self.log_group_ordinals)
            or distinct_groups
            != tuple(f"log-group:{index:02d}" for index in range(len(distinct_groups)))
        ):
            raise ValueError("local replay source context is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class LoadedReplayLocalArtifact:
    """Strictly reloaded ignored local scientific source bundle."""

    path: Path
    run: RunRecordV1Alpha1
    resources: ReplayRunResources
    member_bytes: Mapping[str, bytes]
    member_record_counts: Mapping[str, int]
    artifact_sha256: str
    run_sha256: str
    source_root: Path
    benchmark: ReplayBenchmarkEvidence

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class _PendingReplayLocalArtifact:
    """Published local payload that is intentionally unusable without `_SUCCESS`."""

    path: Path
    source_root: Path
    run: RunRecordV1Alpha1
    resources: ReplayRunResources
    benchmark: ReplayBenchmarkEvidence
    member_bytes: Mapping[str, bytes]
    member_record_counts: Mapping[str, int]
    artifact_sha256: str
    run_sha256: str

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class ReplayLocalExecution:
    """One complete local run plus its in-memory curation handoff."""

    benchmark: ReplayBenchmarkEvidence
    artifact: LoadedReplayLocalArtifact
    run: RunRecordV1Alpha1
    resources: ReplayRunResources

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class ReplayRepeatExecution:
    """Two independent local runs and their exact scientific-byte comparison."""

    primary: ReplayLocalExecution
    repeat: ReplayLocalExecution
    source_commitments: tuple[ReplaySourceMemberCommitmentV1, ...]
    repeat_verification: ReplayRepeatVerificationV1

    def __post_init__(self) -> None:
        if not self.repeat_verification.all_checks_passed:
            raise ValueError("repeat execution cannot represent a failed release gate")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class ReplayLoadedRepeatEvidence:
    """Strictly reloaded primary/repeat source bundles and their exact comparison."""

    primary: LoadedReplayLocalArtifact
    repeat: LoadedReplayLocalArtifact
    source_commitments: tuple[ReplaySourceMemberCommitmentV1, ...]
    repeat_verification: ReplayRepeatVerificationV1

    def __post_init__(self) -> None:
        if not self.repeat_verification.all_checks_passed:
            raise ValueError("loaded repeat evidence cannot represent a failed release gate")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(slots=True)
class _MetadataReadEvidence:
    metadata_table_reads: int = 0
    blocked_dataset_reads: int = 0
    raw_sensor_payload_reads: int = 0


def _raise(message: str) -> Never:
    raise ReplayRunnerError(message) from None


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_equal_or_descendant(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current /= part
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                _raise("M5 dataset root validation failed")
    except ReplayRunnerError:
        raise
    except OSError:
        _raise("M5 dataset root validation failed")


def _resolve_dataset_root(*, source_root: Path) -> Path:
    raw_value = os.environ.get(_DATASET_ROOT_ENV)
    if raw_value is None or not raw_value:
        _raise("M5 dataset environment is unavailable")
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        _raise("M5 dataset root must be absolute")
    _reject_symlink_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            _raise("M5 dataset root validation failed")
    except ReplayRunnerError:
        raise
    except (OSError, RuntimeError):
        _raise("M5 dataset root validation failed")
    source = source_root.resolve(strict=True)
    if _is_equal_or_descendant(resolved, source) or _is_equal_or_descendant(source, resolved):
        _raise("M5 dataset root must remain disjoint from the source checkout")
    return resolved


def _is_raw_payload_candidate(path: Path, *, dataset_root: Path) -> bool:
    try:
        relative = path.relative_to(dataset_root)
    except ValueError:
        return False
    return (
        bool(relative.parts) and relative.parts[0].casefold() in _RAW_PAYLOAD_DIRECTORIES
    ) or path.suffix.casefold() in _RAW_PAYLOAD_SUFFIXES


@contextmanager
def _metadata_read_guard(dataset_root: Path) -> Generator[_MetadataReadEvidence]:
    """Permit only fixed metadata-table opens below the resolved dataset root."""

    version_root = dataset_root / "v1.0-mini"
    allowed = {_lexical_absolute(version_root / f"{table}.json") for table in _NUSCENES_TABLES}
    root = _lexical_absolute(dataset_root)
    evidence = _MetadataReadEvidence()
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def directory_descriptor_path(descriptor: int) -> Path:
        try:
            if sys.platform == "darwin":
                raw = fcntl.fcntl(descriptor, 50, b"\0" * 1024)
                encoded = raw.split(b"\0", maxsplit=1)[0]
                if not encoded:
                    raise OSError("empty descriptor path")
                return Path(os.fsdecode(encoded))
            return Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except (OSError, TypeError, ValueError):
            raise OSError("directory descriptor path unavailable") from None

    def candidate_path(
        raw_path: _OpenPath,
        *,
        directory_descriptor: int | None = None,
    ) -> Path | None:
        if isinstance(raw_path, int):
            return None
        try:
            candidate = Path(os.fsdecode(os.fspath(raw_path)))
        except TypeError:
            raise OSError("unsupported open path") from None
        if not candidate.is_absolute():
            base = (
                Path.cwd()
                if directory_descriptor is None
                else directory_descriptor_path(directory_descriptor)
            )
            candidate = base / candidate
        try:
            return candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return _lexical_absolute(candidate)

    def authorize(
        raw_path: _OpenPath,
        *,
        read_only: bool,
        directory_descriptor: int | None = None,
    ) -> None:
        candidate = candidate_path(
            raw_path,
            directory_descriptor=directory_descriptor,
        )
        if candidate is None or not _is_equal_or_descendant(candidate, root):
            return
        if candidate not in allowed or not read_only:
            evidence.blocked_dataset_reads += 1
            if read_only and _is_raw_payload_candidate(candidate, dataset_root=root):
                evidence.raw_sensor_payload_reads += 1
            raise OSError("dataset payload open blocked")
        evidence.metadata_table_reads += 1

    def guarded_builtin_open(
        file: _OpenPath,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Any | None = None,
    ) -> Any:
        authorize(
            file,
            read_only="r" in mode and not any(flag in mode for flag in ("+", "w", "a", "x")),
        )
        return original_builtin_open(
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )

    def guarded_io_open(
        file: _OpenPath,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Any | None = None,
    ) -> Any:
        authorize(
            file,
            read_only="r" in mode and not any(flag in mode for flag in ("+", "w", "a", "x")),
        )
        return original_io_open(
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )

    def guarded_os_open(
        path: _OsOpenPath,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        write_flags = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_TRUNC
            | os.O_APPEND
            | getattr(os, "O_EXCL", 0)
        )
        authorize(
            path,
            read_only=flags & write_flags == 0,
            directory_descriptor=dir_fd,
        )
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    builtins.open = guarded_builtin_open  # type: ignore[assignment]
    io.open = guarded_io_open  # type: ignore[assignment]
    os.open = guarded_os_open  # type: ignore[assignment]
    try:
        yield evidence
    finally:
        os.open = original_os_open  # type: ignore[assignment]
        io.open = original_io_open  # type: ignore[assignment]
        builtins.open = original_builtin_open  # type: ignore[assignment]


def _load_population(dataset_root: Path) -> ReplayPopulation:
    try:
        with _metadata_read_guard(dataset_root) as reads:
            metadata: NuScenesMiniMetadata = load_nuscenes_mini(dataset_root)
        validation = metadata.validation
        if (
            reads.metadata_table_reads != len(_NUSCENES_TABLES)
            or reads.blocked_dataset_reads != 0
            or reads.raw_sensor_payload_reads != 0
            or not validation.headline_profile_passed_attested
            or not validation.structural_integrity_passed_attested
            or validation.keyframe_blob_check_count != EXPECTED_KEYFRAME_BLOB_CHECK_COUNT
            or not validation.keyframe_blob_validation_passed_attested
        ):
            _raise("M5 local metadata acceptance gate failed")
        return extract_m5_replay_source(metadata)
    except ReplayRunnerError:
        raise
    except (NuScenesAdapterError, ReplaySourceError, OSError, TypeError, ValueError):
        _raise("M5 local metadata replay extraction failed")


def _initial_snapshot() -> CleanSourceSnapshot:
    try:
        snapshot = discover_clean_source(M5_REPLAY_INTENT_PATH)
        verify_locked_execution(snapshot)
    except (OSError, ProvenanceError, ValueError):
        _raise("M5 clean-source validation failed")
    if snapshot.manifest_relative_path != M5_REPLAY_INTENT_PATH.as_posix():
        _raise("M5 source snapshot does not identify the frozen replay intent")
    return snapshot


def _verify_unchanged_source(initial: CleanSourceSnapshot) -> None:
    try:
        final = discover_clean_source(M5_REPLAY_INTENT_PATH)
        verify_locked_execution(final)
    except (OSError, ProvenanceError, ValueError):
        _raise("M5 final clean-source validation failed")
    if final != initial:
        _raise("M5 source provenance changed during replay execution")


def _validated_output_path(
    output_dir: Path,
    *,
    source_root: Path,
) -> tuple[Path, str]:
    if output_dir.is_absolute() or any(part in {".", ".."} for part in output_dir.parts):
        _raise("M5 local output must use a normalized repository-relative path")
    try:
        output_dir.relative_to(_GENERATED_ROOT)
    except ValueError:
        _raise("M5 local output must remain under reports/generated")
    if output_dir == _GENERATED_ROOT:
        _raise("M5 local output requires a run-specific directory")
    target = _lexical_absolute(source_root / output_dir)
    if os.path.lexists(target):
        _raise("M5 local output destination already exists")
    return target, output_dir.as_posix()


def _ensure_cpu_only_import_boundary() -> None:
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        _raise("M5 replay CPU-only import boundary failed")


@contextmanager
def _single_process_guard() -> Generator[None]:
    """Block Python-visible process creation during scientific computation."""

    owners_and_names: list[tuple[object, str]] = [
        (subprocess, "Popen"),
        *(
            (os, name)
            for name in (
                "fork",
                "forkpty",
                "posix_spawn",
                "posix_spawnp",
                "spawnl",
                "spawnle",
                "spawnlp",
                "spawnlpe",
                "spawnv",
                "spawnve",
                "spawnvp",
                "spawnvpe",
                "system",
            )
            if hasattr(os, name)
        ),
    ]
    originals = tuple((owner, name, getattr(owner, name)) for owner, name in owners_and_names)

    def blocked(*_args: object, **_kwargs: object) -> Never:
        _raise("M5 replay single-process boundary failed")

    for owner, name, _ in originals:
        setattr(owner, name, blocked)
    try:
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw * 1024 if sys.platform.startswith("linux") else raw


def _json_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if is_dataclass(value) and not isinstance(value, type):
        dumped: dict[str, Any] = asdict(cast(Any, value))
        return dumped
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): item for key, item in mapping.items()}
    raise TypeError("local replay row is not serializable")


def _scan_local_value(value: object, *, forbidden_paths: Sequence[str]) -> None:
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        for raw_key, item in mapping.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_LOCAL_KEYS:
                raise ValueError("local replay row contains a forbidden private field")
            _scan_local_value(item, forbidden_paths=forbidden_paths)
        return
    if isinstance(value, (list, tuple)):
        for item in cast("Sequence[object]", value):
            _scan_local_value(item, forbidden_paths=forbidden_paths)
        return
    if isinstance(value, str) and (
        value.startswith("track:")
        or value.startswith("/")
        or _ABSOLUTE_PATH_PATTERN.search(value) is not None
        or _RAW_PAYLOAD_PATTERN.search(value) is not None
        or _SECRET_PATTERN.search(value) is not None
        or any(path and path in value for path in forbidden_paths)
    ):
        raise ValueError("local replay row contains private or payload material")


def _local_row_mapping(
    value: object,
    *,
    schema: str,
    forbidden_paths: Sequence[str],
) -> dict[str, Any]:
    payload = _json_mapping(value)
    existing_schema = payload.get("schema")
    if existing_schema is None:
        payload["schema"] = schema
    elif existing_schema != schema:
        raise ValueError("local replay row schema is inconsistent")
    existing_intent = payload.get("replay_intent_sha256")
    if existing_intent is None:
        payload["replay_intent_sha256"] = M5_REPLAY_INTENT_SHA256
    elif existing_intent != M5_REPLAY_INTENT_SHA256:
        raise ValueError("local replay row does not bind the frozen intent")
    _scan_local_value(payload, forbidden_paths=forbidden_paths)
    return payload


def _local_ndjson_bytes(
    records: Sequence[object],
    *,
    schema: str,
    forbidden_paths: Sequence[str],
) -> bytes:
    if not records:
        raise ValueError("local replay scientific members must be nonempty")
    output = bytearray()
    for record in records:
        line = canonical_json_bytes(
            _local_row_mapping(
                record,
                schema=schema,
                forbidden_paths=forbidden_paths,
            )
        )
        if len(line) > _LOCAL_MAX_RECORD_BYTES:
            raise ValueError("local replay record exceeds its byte cap")
        if len(output) + len(line) > _LOCAL_MAX_MEMBER_BYTES:
            raise ValueError("local replay member exceeds its byte cap")
        output.extend(line)
    return bytes(output)


def _scientific_members(
    benchmark: ReplayBenchmarkEvidence,
    *,
    forbidden_paths: Sequence[str],
) -> tuple[dict[str, bytes], dict[str, int]]:
    members: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    for path, _, schema, attribute in _LOCAL_MEMBER_LAYOUT:
        rows = cast("Sequence[object]", getattr(benchmark, attribute))
        members[path] = _local_ndjson_bytes(
            rows,
            schema=schema,
            forbidden_paths=forbidden_paths,
        )
        counts[path] = len(rows)
    if len(members["persistent-scene-evaluations.ndjson"].splitlines()) != 710:
        raise ValueError("local M5-A source rows do not cover all 710 scene cases")
    if sum(len(value) for value in members.values()) > _LOCAL_MAX_ARTIFACT_BYTES:
        raise ValueError("local replay scientific source exceeds its artifact cap")
    return members, counts


def _source_index_mapping(
    *,
    run_id: str,
    benchmark: ReplayBenchmarkEvidence,
    members: Mapping[str, bytes],
    record_counts: Mapping[str, int],
    resources_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema": "ffb.replay-local-source-index/v1",
        "artifact_contract": _LOCAL_ARTIFACT_CONTRACT,
        "run_id": run_id,
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "fit_calibration_sha256": benchmark.fit_calibration_sha256,
        "data_master_seed": M5_DATA_MASTER_SEED,
        "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
        "scene_frame_counts": list(benchmark.scene_frame_counts),
        "log_group_ordinals": list(benchmark.log_group_ordinals),
        "members": [
            {
                "path": path,
                "relative_role": _ROLE_BY_PATH[path],
                "byte_length": len(members[path]),
                "record_count": record_counts[path],
                "sha256": hashlib.sha256(members[path]).hexdigest(),
            }
            for path in REPLAY_LOCAL_SCIENTIFIC_PATHS
        ],
        "resources_member": {
            "path": _LOCAL_RESOURCES_FILE,
            "relative_role": _LOCAL_RESOURCES_ROLE,
            "schema": _LOCAL_RESOURCES_SCHEMA,
            "byte_length": len(resources_bytes),
            "record_count": 1,
            "sha256": hashlib.sha256(resources_bytes).hexdigest(),
        },
    }


def _local_artifact_digest(index_bytes: bytes) -> str:
    preimage = b"".join(
        (
            _LOCAL_ARTIFACT_DOMAIN,
            len(index_bytes).to_bytes(8, "big"),
            index_bytes,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def _resources_mapping(run_id: str, resources: ReplayRunResources) -> dict[str, Any]:
    return {
        "schema": _LOCAL_RESOURCES_SCHEMA,
        "run_id": run_id,
        "elapsed_seconds": resources.elapsed_seconds,
        "peak_rss_bytes": resources.peak_rss_bytes,
        "measurement_scope": resources.measurement_scope,
        "raw_sensor_payload_reads": resources.raw_sensor_payload_reads,
        "scientific_replay_worker_count": resources.scientific_replay_worker_count,
        "gpu_used": resources.gpu_used,
        "torch_imported": resources.torch_imported,
        "cuda_used": resources.cuda_used,
        "wall_time_cap_seconds": _WALL_TIME_CAP_SECONDS,
        "peak_rss_cap_bytes": _PEAK_RSS_CAP_BYTES,
    }


def _strict_json_mapping(value: bytes, *, label: str) -> dict[str, Any]:
    if (
        not value
        or len(value) > _LOCAL_MAX_MEMBER_BYTES
        or not value.endswith(b"\n")
        or b"\r" in value
        or b"\n" in value[:-1]
    ):
        raise ArtifactValidationError(f"{label} is not one canonical JSON record")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = item
        return output

    def reject_constant(_: str) -> Never:
        raise ValueError("non-finite JSON number")

    try:
        raw = json.loads(
            value[:-1].decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactValidationError(f"{label} is invalid JSON") from error
    if not isinstance(raw, dict):
        raise ArtifactValidationError(f"{label} must be a JSON object")
    mapping = cast("dict[str, Any]", raw)
    if canonical_json_bytes(mapping) != value:
        raise ArtifactValidationError(f"{label} is not canonical JSON")
    return mapping


def _strict_ndjson(
    value: bytes,
    *,
    label: str,
) -> tuple[tuple[dict[str, Any], ...], int]:
    if (
        not value
        or len(value) > _LOCAL_MAX_MEMBER_BYTES
        or not value.endswith(b"\n")
        or b"\r" in value
    ):
        raise ArtifactValidationError(f"{label} is not bounded canonical NDJSON")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(value.splitlines(keepends=True)):
        if len(line) > _LOCAL_MAX_RECORD_BYTES:
            raise ArtifactValidationError(f"{label} contains an oversized record")
        records.append(_strict_json_mapping(line, label=f"{label}:{index}"))
    return tuple(records), len(records)


def _validate_source_index(
    value: Mapping[str, Any],
    *,
    members: Mapping[str, bytes],
    record_counts: Mapping[str, int],
    resources_bytes: bytes,
) -> _ReplayLocalSourceContext:
    expected_keys = {
        "schema",
        "artifact_contract",
        "run_id",
        "replay_intent_sha256",
        "replay_identity_set_sha256",
        "fit_calibration_sha256",
        "data_master_seed",
        "bootstrap_replicates",
        "scene_frame_counts",
        "log_group_ordinals",
        "members",
        "resources_member",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != "ffb.replay-local-source-index/v1"
        or value.get("artifact_contract") != _LOCAL_ARTIFACT_CONTRACT
        or value.get("replay_intent_sha256") != M5_REPLAY_INTENT_SHA256
        or value.get("replay_identity_set_sha256") != M5_REPLAY_IDENTITY_SET_SHA256
        or value.get("fit_calibration_sha256") != M4_FROZEN_CALIBRATION_SHA256
        or value.get("data_master_seed") != M5_DATA_MASTER_SEED
        or value.get("bootstrap_replicates") != _BOOTSTRAP_REPLICATES
        or not isinstance(value.get("run_id"), str)
        or not isinstance(value.get("scene_frame_counts"), list)
        or not isinstance(value.get("log_group_ordinals"), list)
        or not isinstance(value.get("members"), list)
        or not isinstance(value.get("resources_member"), dict)
    ):
        raise ArtifactValidationError("local replay source index is invalid")
    entries = cast("list[object]", value["members"])
    if len(entries) != len(REPLAY_LOCAL_SCIENTIFIC_PATHS):
        raise ArtifactValidationError("local replay source index is incomplete")
    for expected_path, raw_entry in zip(
        REPLAY_LOCAL_SCIENTIFIC_PATHS,
        entries,
        strict=True,
    ):
        if not isinstance(raw_entry, dict):
            raise ArtifactValidationError("local replay source index entry is invalid")
        entry = cast("dict[str, Any]", raw_entry)
        expected_entry = {
            "path": expected_path,
            "relative_role": _ROLE_BY_PATH[expected_path],
            "byte_length": len(members[expected_path]),
            "record_count": record_counts[expected_path],
            "sha256": hashlib.sha256(members[expected_path]).hexdigest(),
        }
        if entry != expected_entry:
            raise ArtifactValidationError("local replay source index commitment is invalid")
    expected_resources_entry = {
        "path": _LOCAL_RESOURCES_FILE,
        "relative_role": _LOCAL_RESOURCES_ROLE,
        "schema": _LOCAL_RESOURCES_SCHEMA,
        "byte_length": len(resources_bytes),
        "record_count": 1,
        "sha256": hashlib.sha256(resources_bytes).hexdigest(),
    }
    if value["resources_member"] != expected_resources_entry:
        raise ArtifactValidationError("local replay resource commitment is invalid")
    try:
        return _ReplayLocalSourceContext(
            run_id=cast(str, value["run_id"]),
            fit_calibration_sha256=cast(str, value["fit_calibration_sha256"]),
            data_master_seed=cast(int, value["data_master_seed"]),
            bootstrap_replicates=cast(int, value["bootstrap_replicates"]),
            scene_frame_counts=tuple(cast("list[int]", value["scene_frame_counts"])),
            log_group_ordinals=tuple(cast("list[str]", value["log_group_ordinals"])),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError("local replay source context is invalid") from error


def _resources_from_mapping(value: Mapping[str, Any], *, run_id: str) -> ReplayRunResources:
    expected_keys = {
        "schema",
        "run_id",
        "elapsed_seconds",
        "peak_rss_bytes",
        "measurement_scope",
        "raw_sensor_payload_reads",
        "scientific_replay_worker_count",
        "gpu_used",
        "torch_imported",
        "cuda_used",
        "wall_time_cap_seconds",
        "peak_rss_cap_bytes",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != _LOCAL_RESOURCES_SCHEMA
        or value.get("run_id") != run_id
        or value.get("wall_time_cap_seconds") != _WALL_TIME_CAP_SECONDS
        or value.get("peak_rss_cap_bytes") != _PEAK_RSS_CAP_BYTES
        or type(value["elapsed_seconds"]) is not float
        or type(value["peak_rss_bytes"]) is not int
        or type(value["measurement_scope"]) is not str
        or type(value["raw_sensor_payload_reads"]) is not int
        or type(value["scientific_replay_worker_count"]) is not int
        or type(value["gpu_used"]) is not bool
        or type(value["torch_imported"]) is not bool
        or type(value["cuda_used"]) is not bool
        or type(value["wall_time_cap_seconds"]) is not float
        or type(value["peak_rss_cap_bytes"]) is not int
    ):
        raise ArtifactValidationError("local replay resource evidence is invalid")
    try:
        return ReplayRunResources(
            elapsed_seconds=value["elapsed_seconds"],
            peak_rss_bytes=value["peak_rss_bytes"],
            measurement_scope=cast(
                "ReplayResourceMeasurementScope",
                value["measurement_scope"],
            ),
            raw_sensor_payload_reads=value["raw_sensor_payload_reads"],
            scientific_replay_worker_count=value["scientific_replay_worker_count"],
            gpu_used=value["gpu_used"],
            torch_imported=value["torch_imported"],
            cuda_used=value["cuda_used"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError("local replay resource evidence is invalid") from error


def _require_safe_local_tree(
    root: Path,
    *,
    expected_paths: tuple[str, ...] = REPLAY_LOCAL_ARTIFACT_PATHS,
) -> dict[str, os.stat_result]:
    try:
        root_stat = root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ArtifactValidationError("local replay artifact root is not a real directory")
        names = tuple(sorted(os.listdir(root), key=lambda item: item.encode("utf-8")))
    except ArtifactValidationError:
        raise
    except OSError as error:
        raise ArtifactValidationError("local replay artifact cannot be inspected") from error
    if names != tuple(sorted(expected_paths, key=lambda item: item.encode("utf-8"))):
        raise ArtifactValidationError("local replay artifact member allowlist is invalid")
    entries: dict[str, os.stat_result] = {}
    total_bytes = 0
    for name in expected_paths:
        metadata = (root / name).lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactValidationError("local replay artifact member is not a regular file")
        if metadata.st_nlink != 1:
            raise ArtifactValidationError("local replay artifact member must not be hard-linked")
        total_bytes += metadata.st_size
        entries[name] = metadata
    if total_bytes > _LOCAL_MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("local replay artifact exceeds its byte cap")
    return entries


def _read_local_member(path: Path, *, expected: os.stat_result) -> bytes:
    def fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    expected_fingerprint = fingerprint(expected)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or fingerprint(observed) != expected_fingerprint
            or observed.st_size > _LOCAL_MAX_MEMBER_BYTES
        ):
            raise ArtifactValidationError("local replay artifact member changed during load")
        chunks: list[bytes] = []
        remaining = _LOCAL_MAX_MEMBER_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) > _LOCAL_MAX_MEMBER_BYTES:
            raise ArtifactValidationError("local replay artifact member exceeds its byte cap")
        final_descriptor = os.fstat(descriptor)
        final_path = path.lstat()
        if (
            fingerprint(final_descriptor) != expected_fingerprint
            or fingerprint(final_path) != expected_fingerprint
            or len(value) != expected.st_size
        ):
            raise ArtifactValidationError("local replay artifact member changed during load")
        return value
    finally:
        os.close(descriptor)


_DESCRIPTOR_ADAPTER = TypeAdapter(ReplayDescriptorAggregate)
_PERSISTENT_SCENE_ADAPTER = TypeAdapter(ReplayPersistentSceneEvaluation)
_PERSISTENT_METRIC_ADAPTER = TypeAdapter(ReplayPersistentPopulationMetric)
_PERSISTENT_CROSSOVER_ADAPTER = TypeAdapter(ReplayPersistentCrossoverEstimate)
_HEALTH_CONTRAST_ADAPTER = TypeAdapter(ReplayHealthSequenceContrast)
_HEALTH_METRIC_ADAPTER = TypeAdapter(ReplayHealthPopulationMetric)


def _decode_dataclass_row[RowT](
    record: Mapping[str, Any],
    *,
    adapter: TypeAdapter[RowT],
    label: str,
) -> RowT:
    payload = dict(record)
    payload.pop("schema", None)
    payload.pop("replay_intent_sha256", None)
    try:
        decoded = adapter.validate_python(payload)
    except (TypeError, ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"{label} contains an invalid typed row") from error
    if canonical_json_bytes(_json_mapping(decoded)) != canonical_json_bytes(payload):
        raise ArtifactValidationError(f"{label} typed row is not an exact canonical round trip")
    return decoded


def _decode_contract_row[RowT: BaseModel](
    record: Mapping[str, Any],
    *,
    model: type[RowT],
    label: str,
) -> RowT:
    payload = dict(record)
    payload.pop("replay_intent_sha256", None)
    try:
        decoded = model.model_validate(payload)
    except (TypeError, ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"{label} contains an invalid typed row") from error
    if canonical_json_bytes(decoded) != canonical_json_bytes(payload):
        raise ArtifactValidationError(f"{label} typed row is not an exact canonical round trip")
    return decoded


def _decode_local_benchmark(
    records_by_path: Mapping[str, tuple[dict[str, Any], ...]],
    *,
    context: _ReplayLocalSourceContext,
    plan: LoadedReplayPlan,
) -> ReplayBenchmarkEvidence:
    """Decode every source row and invoke the benchmark's authoritative validators."""

    try:
        descriptor_aggregates = tuple(
            _decode_dataclass_row(
                row,
                adapter=_DESCRIPTOR_ADAPTER,
                label="descriptor-aggregates.ndjson",
            )
            for row in records_by_path["descriptor-aggregates.ndjson"]
        )
        persistent_scene_evaluations = tuple(
            _decode_dataclass_row(
                row,
                adapter=_PERSISTENT_SCENE_ADAPTER,
                label="persistent-scene-evaluations.ndjson",
            )
            for row in records_by_path["persistent-scene-evaluations.ndjson"]
        )
        persistent_metrics = tuple(
            _decode_dataclass_row(
                row,
                adapter=_PERSISTENT_METRIC_ADAPTER,
                label="persistent-population-metrics.ndjson",
            )
            for row in records_by_path["persistent-population-metrics.ndjson"]
        )
        persistent_crossovers = tuple(
            _decode_dataclass_row(
                row,
                adapter=_PERSISTENT_CROSSOVER_ADAPTER,
                label="persistent-crossovers.ndjson",
            )
            for row in records_by_path["persistent-crossovers.ndjson"]
        )
        health_results = tuple(
            _decode_contract_row(
                row,
                model=ReplayHealthResultV1,
                label="health-sequence-results.ndjson",
            )
            for row in records_by_path["health-sequence-results.ndjson"]
        )
        health_contrasts = tuple(
            _decode_dataclass_row(
                row,
                adapter=_HEALTH_CONTRAST_ADAPTER,
                label="health-sequence-contrasts.ndjson",
            )
            for row in records_by_path["health-sequence-contrasts.ndjson"]
        )
        health_events = tuple(
            _decode_contract_row(
                row,
                model=ReplayHealthSequenceEventV1,
                label="health-sequence-events.ndjson",
            )
            for row in records_by_path["health-sequence-events.ndjson"]
        )
        health_metrics = tuple(
            _decode_dataclass_row(
                row,
                adapter=_HEALTH_METRIC_ADAPTER,
                label="health-population-metrics.ndjson",
            )
            for row in records_by_path["health-population-metrics.ndjson"]
        )
        return ReplayBenchmarkEvidence(
            plan=plan,
            fit_calibration_sha256=context.fit_calibration_sha256,
            data_master_seed=context.data_master_seed,
            bootstrap_replicates=context.bootstrap_replicates,
            log_group_ordinals=context.log_group_ordinals,
            scene_frame_counts=context.scene_frame_counts,
            descriptor_aggregates=descriptor_aggregates,
            persistent_scene_evaluations=persistent_scene_evaluations,
            persistent_metrics=persistent_metrics,
            persistent_crossovers=persistent_crossovers,
            health_results=health_results,
            health_contrasts=health_contrasts,
            health_events=health_events,
            health_metrics=health_metrics,
        )
    except ArtifactValidationError:
        raise
    except (KeyError, TypeError, ValidationError, ValueError) as error:
        raise ArtifactValidationError(
            "local replay scientific members fail authoritative validation"
        ) from error


def _load_replay_local_payload(
    path: Path,
    *,
    source_root: Path,
    plan: LoadedReplayPlan,
    require_success: bool,
) -> _PendingReplayLocalArtifact:
    lexical = _lexical_absolute(path)
    root_metadata = lexical.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ArtifactValidationError("local replay artifact root is not a real directory")
    root = path.resolve(strict=True)
    expected_paths = REPLAY_LOCAL_ARTIFACT_PATHS if require_success else _REPLAY_LOCAL_PENDING_PATHS
    entries = _require_safe_local_tree(root, expected_paths=expected_paths)
    values = {
        name: _read_local_member(root / name, expected=entries[name]) for name in expected_paths
    }
    member_bytes: dict[str, bytes] = {}
    record_counts: dict[str, int] = {}
    records_by_path: dict[str, tuple[dict[str, Any], ...]] = {}
    expected_schema = {path: schema for path, _, schema, _ in _LOCAL_MEMBER_LAYOUT}
    for name in REPLAY_LOCAL_SCIENTIFIC_PATHS:
        records, count = _strict_ndjson(values[name], label=name)
        if any(
            record.get("schema") != expected_schema[name]
            or record.get("replay_intent_sha256") != M5_REPLAY_INTENT_SHA256
            for record in records
        ):
            raise ArtifactValidationError("local replay scientific row binding is invalid")
        for record in records:
            _scan_local_value(record, forbidden_paths=())
        member_bytes[name] = values[name]
        record_counts[name] = count
        records_by_path[name] = records
    if record_counts["persistent-scene-evaluations.ndjson"] != 710:
        raise ArtifactValidationError("local replay M5-A scene source is incomplete")

    index_mapping = _strict_json_mapping(values[_LOCAL_INDEX_FILE], label=_LOCAL_INDEX_FILE)
    context = _validate_source_index(
        index_mapping,
        members=member_bytes,
        record_counts=record_counts,
        resources_bytes=values[_LOCAL_RESOURCES_FILE],
    )
    try:
        run = RunRecordV1Alpha1.model_validate_json(values[_LOCAL_RUN_FILE])
    except (ValidationError, ValueError) as error:
        raise ArtifactValidationError("local replay run record is invalid") from error
    if canonical_json_bytes(run) != values[_LOCAL_RUN_FILE]:
        raise ArtifactValidationError("local replay run record is not canonical")
    expected_run_id = derive_run_id(
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        git_revision=run.git_revision,
        lockfile_sha256=run.lockfile_sha256,
        package_version=run.package_version,
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
    )
    if (
        run.run_id != context.run_id
        or run.run_id != expected_run_id
        or run.manifest_sha256 != M5_REPLAY_INTENT_SHA256
        or run.source_dirty
        or run.status != "succeeded"
        or run.artifact_sha256 != "0" * 64
    ):
        raise ArtifactValidationError("local replay run identity is invalid")

    resources = _resources_from_mapping(
        _strict_json_mapping(values[_LOCAL_RESOURCES_FILE], label=_LOCAL_RESOURCES_FILE),
        run_id=run.run_id,
    )
    benchmark = _decode_local_benchmark(
        records_by_path,
        context=context,
        plan=plan,
    )
    index_digest = _local_artifact_digest(values[_LOCAL_INDEX_FILE])
    run_digest = compute_run_record_digest(values[_LOCAL_RUN_FILE])
    if require_success:
        success = _strict_json_mapping(values[_LOCAL_SUCCESS_FILE], label=_LOCAL_SUCCESS_FILE)
        if success != {
            "schema": "ffb.replay-local-success/v1",
            "artifact_sha256": index_digest,
            "run_sha256": run_digest,
            "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        }:
            raise ArtifactValidationError("local replay success marker is invalid")
    return _PendingReplayLocalArtifact(
        path=root,
        source_root=source_root.resolve(strict=True),
        run=run,
        resources=resources,
        benchmark=benchmark,
        member_bytes=MappingProxyType(member_bytes),
        member_record_counts=MappingProxyType(record_counts),
        artifact_sha256=index_digest,
        run_sha256=run_digest,
    )


def _load_replay_local_artifact(
    path: Path,
    *,
    source_root: Path,
    plan: LoadedReplayPlan,
) -> LoadedReplayLocalArtifact:
    payload = _load_replay_local_payload(
        path,
        source_root=source_root,
        plan=plan,
        require_success=True,
    )
    return LoadedReplayLocalArtifact(
        path=payload.path,
        run=payload.run,
        resources=payload.resources,
        member_bytes=payload.member_bytes,
        member_record_counts=payload.member_record_counts,
        artifact_sha256=payload.artifact_sha256,
        run_sha256=payload.run_sha256,
        source_root=payload.source_root,
        benchmark=payload.benchmark,
    )


def _runner_source_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[2]


def load_replay_local_artifact(path: Path) -> LoadedReplayLocalArtifact:
    """Strictly reload one ignored M5 source bundle without dataset access."""

    try:
        source_root = _runner_source_root()
        plan = load_replay_plan(source_root=source_root)
        return _load_replay_local_artifact(
            path,
            source_root=source_root,
            plan=plan,
        )
    except ArtifactValidationError:
        raise
    except (OSError, RuntimeError, TypeError, ValidationError, ValueError) as error:
        raise ArtifactValidationError("invalid M5 local replay artifact") from error


def _cleanup_local_staging(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
) -> None:
    for name in REPLAY_LOCAL_ARTIFACT_PATHS:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=staging_fd)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _write_local_artifact(
    *,
    destination: Path,
    snapshot: CleanSourceSnapshot,
    benchmark: ReplayBenchmarkEvidence,
    run: RunRecordV1Alpha1,
    resources: ReplayRunResources,
    members: Mapping[str, bytes],
    record_counts: Mapping[str, int],
) -> _PendingReplayLocalArtifact:
    resources_bytes = canonical_json_bytes(_resources_mapping(run.run_id, resources))
    index_bytes = canonical_json_bytes(
        _source_index_mapping(
            run_id=run.run_id,
            benchmark=benchmark,
            members=members,
            record_counts=record_counts,
            resources_bytes=resources_bytes,
        )
    )
    run_bytes = canonical_json_bytes(run)
    artifact_digest = _local_artifact_digest(index_bytes)
    run_digest = compute_run_record_digest(run_bytes)
    files = {
        **members,
        _LOCAL_INDEX_FILE: index_bytes,
        _LOCAL_RESOURCES_FILE: resources_bytes,
        _LOCAL_RUN_FILE: run_bytes,
    }
    if tuple(files) != _REPLAY_LOCAL_PENDING_PATHS:
        raise ArtifactValidationError("local replay prepared member order is invalid")

    parent = destination.parent
    parent_fd = open_or_create_real_directory(parent)
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            parent,
            label="M5 local destination parent",
        )
        reject_directory_descriptor_in_git_metadata(
            parent_fd,
            (snapshot.git_dir, snapshot.git_common_dir),
        )
        if entry_exists_at(parent_fd, destination.name):
            raise FileExistsError("M5 local output destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        published = False
        try:
            for name in _REPLAY_LOCAL_PENDING_PATHS:
                write_exclusive_file_at(staging_fd, name, files[name])
            for name in _REPLAY_LOCAL_PENDING_PATHS:
                if read_file_at(staging_fd, name, byte_cap=len(files[name])) != files[name]:
                    raise ArtifactValidationError("M5 local staging verification failed")
            os.fsync(staging_fd)
            staged = _load_replay_local_payload(
                staging,
                source_root=snapshot.source_root,
                plan=benchmark.plan,
                require_success=False,
            )
            if staged.artifact_sha256 != artifact_digest or staged.run_sha256 != run_digest:
                raise ArtifactValidationError("M5 local staged identity is invalid")
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="M5 local destination parent",
            )
            if entry_exists_at(parent_fd, destination.name):
                raise FileExistsError("M5 local output destination already exists")
            atomic_rename_directory_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                destination.name,
            )
            published = True
            os.fsync(parent_fd)
            loaded = _load_replay_local_payload(
                destination,
                source_root=snapshot.source_root,
                plan=benchmark.plan,
                require_success=False,
            )
            if loaded.artifact_sha256 != artifact_digest or loaded.run_sha256 != run_digest:
                raise ArtifactValidationError("M5 local published identity is invalid")
            return loaded
        except BaseException:
            if not published:
                _cleanup_local_staging(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=staging_name,
                )
            else:
                _cleanup_local_staging(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=destination.name,
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def _rollback_pending_local_artifact(
    pending: _PendingReplayLocalArtifact,
    *,
    snapshot: CleanSourceSnapshot,
) -> None:
    """Best-effort removal of a payload that can never carry a trusted success marker."""

    parent = pending.path.parent
    parent_fd = open_or_create_real_directory(parent)
    artifact_fd: int | None = None
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            parent,
            label="M5 local destination parent",
        )
        reject_directory_descriptor_in_git_metadata(
            parent_fd,
            (snapshot.git_dir, snapshot.git_common_dir),
        )
        try:
            artifact_fd = os.open(
                pending.path.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        with suppress(FileNotFoundError):
            os.unlink(_LOCAL_SUCCESS_FILE, dir_fd=artifact_fd)
        with suppress(OSError):
            os.fsync(artifact_fd)
        _cleanup_local_staging(
            parent_fd=parent_fd,
            staging_fd=artifact_fd,
            staging_name=pending.path.name,
        )
        with suppress(OSError):
            os.fsync(parent_fd)
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        os.close(parent_fd)


def _finalize_local_artifact(
    pending: _PendingReplayLocalArtifact,
    *,
    snapshot: CleanSourceSnapshot,
) -> LoadedReplayLocalArtifact:
    """Create `_SUCCESS` only after every post-publication gate has passed."""

    success_bytes = canonical_json_bytes(
        {
            "schema": "ffb.replay-local-success/v1",
            "artifact_sha256": pending.artifact_sha256,
            "run_sha256": pending.run_sha256,
            "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        }
    )
    parent = pending.path.parent
    parent_fd = open_or_create_real_directory(parent)
    artifact_fd: int | None = None
    success_written = False
    try:
        assert_directory_descriptor_matches_path(
            parent_fd,
            parent,
            label="M5 local destination parent",
        )
        reject_directory_descriptor_in_git_metadata(
            parent_fd,
            (snapshot.git_dir, snapshot.git_common_dir),
        )
        artifact_fd = os.open(
            pending.path.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        assert_directory_descriptor_matches_path(
            artifact_fd,
            pending.path,
            label="M5 local pending artifact",
        )
        reloaded_pending = _load_replay_local_payload(
            pending.path,
            source_root=pending.source_root,
            plan=pending.benchmark.plan,
            require_success=False,
        )
        if (
            reloaded_pending.artifact_sha256 != pending.artifact_sha256
            or reloaded_pending.run_sha256 != pending.run_sha256
        ):
            raise ArtifactValidationError("M5 local pending identity changed before finalization")
        write_exclusive_file_at(
            artifact_fd,
            _LOCAL_SUCCESS_FILE,
            success_bytes,
        )
        success_written = True
        os.fsync(artifact_fd)
        loaded = _load_replay_local_artifact(
            pending.path,
            source_root=pending.source_root,
            plan=pending.benchmark.plan,
        )
        if (
            loaded.artifact_sha256 != pending.artifact_sha256
            or loaded.run_sha256 != pending.run_sha256
        ):
            raise ArtifactValidationError("M5 local finalized identity is invalid")
        return loaded
    except BaseException:
        if artifact_fd is not None and success_written:
            with suppress(OSError):
                os.unlink(_LOCAL_SUCCESS_FILE, dir_fd=artifact_fd)
            with suppress(OSError):
                os.fsync(artifact_fd)
        raise
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        os.close(parent_fd)


def _run_record(
    *,
    snapshot: CleanSourceSnapshot,
    output_argument: str,
    environment: RuntimeEnvironment,
    started_at: datetime,
    ended_at: datetime,
) -> RunRecordV1Alpha1:
    run_id = derive_run_id(
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        git_revision=snapshot.git_revision,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version=snapshot.package_version,
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
    )
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        package_version=snapshot.package_version,
        git_revision=snapshot.git_revision,
        source_dirty=False,
        lockfile_sha256=snapshot.lockfile_sha256,
        command=(
            "ffb",
            "replay",
            "run",
            "--output-dir",
            output_argument,
        ),
        environment=environment,
        started_at=started_at,
        ended_at=ended_at,
        status="succeeded",
        artifact_sha256="0" * 64,
    )


def _descriptor_contracts(
    benchmark: ReplayBenchmarkEvidence,
    *,
    run_id: str,
) -> tuple[ReplayDescriptorAggregateV1, ...]:
    return tuple(
        ReplayDescriptorAggregateV1(
            schema="ffb.replay-descriptor-aggregate/v1",
            run_id=run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            descriptor_id=row.descriptor_id,
            population=row.population,
            population_count=row.population_count,
            statistic=row.statistic,
            category_label=row.category_label,
            status=row.status,
            value=row.value,
            unit=row.unit,
            tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        )
        for row in benchmark.descriptor_aggregates
    )


def _profile_summary(
    benchmark: ReplayBenchmarkEvidence,
    *,
    run_id: str,
    resource_evidence: tuple[
        ReplayExecutionResourceEvidenceV1,
        ReplayExecutionResourceEvidenceV1,
    ],
) -> ReplayProfileSummaryV1:
    return ReplayProfileSummaryV1(
        schema="ffb.replay-profile-summary/v1",
        run_id=run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        release_id=M5_REPLAY_RELEASE_ID,
        replay_intent_byte_sha256=M5_REPLAY_INTENT_BYTE_SHA256,
        persistent_source_matrix_sha256=M5_PERSISTENT_MATRIX_SHA256,
        health_fit_artifact_sha256=M5_HEALTH_FIT_SHA256,
        health_fit_run_sha256=M5_HEALTH_FIT_RUN_SHA256,
        dataset_profile="official-nuscenes-v1.0-mini",
        adapter_profile="nuscenes-mini-matched-centers-v1",
        scene_count=10,
        persistent_experiment_count=8,
        health_experiment_count=14,
        replay_experiment_count=22,
        distinct_log_group_count=len(set(benchmark.log_group_ordinals)),
        all_scenes_have_base_support=True,
        all_health_schedules_valid=True,
        raw_sensor_payload_reads=0,
        scientific_replay_worker_count=1,
        gpu_used=False,
        torch_imported=False,
        cuda_used=False,
        resource_evidence=resource_evidence,
        peak_rss_bytes=max(row.peak_rss_bytes for row in resource_evidence),
        elapsed_seconds=max(row.elapsed_seconds for row in resource_evidence),
        dataset_root_serialized=False,
        dataset_bytes_authenticated=False,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        attribution_and_non_endorsement_required=True,
    )


def _execute_replay_local(
    output_dir: Path,
    *,
    snapshot: CleanSourceSnapshot,
) -> ReplayLocalExecution:
    destination, output_argument = _validated_output_path(
        output_dir,
        source_root=snapshot.source_root,
    )
    dataset_root = _resolve_dataset_root(source_root=snapshot.source_root)
    _ensure_cpu_only_import_boundary()
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()
    try:
        with _single_process_guard():
            population = _load_population(dataset_root)
            benchmark = run_replay_benchmark(
                population,
                source_root=snapshot.source_root,
            )
            _ensure_cpu_only_import_boundary()
    except ReplayRunnerError:
        raise
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError):
        _raise("M5 replay benchmark computation failed")
    try:
        environment = collect_runtime_environment()
        members, record_counts = _scientific_members(
            benchmark,
            forbidden_paths=(
                os.fspath(dataset_root),
                os.fspath(snapshot.source_root),
            ),
        )
        scientific_resources = ReplayRunResources(
            elapsed_seconds=float(time.perf_counter() - started_clock),
            peak_rss_bytes=_peak_rss_bytes(),
            measurement_scope=("metadata-through-canonical-scientific-members-before-publication"),
        )
        ended_at = datetime.now(UTC)
        run = _run_record(
            snapshot=snapshot,
            output_argument=output_argument,
            environment=environment,
            started_at=started_at,
            ended_at=ended_at,
        )
    except (OSError, ProvenanceError, TypeError, ValidationError, ValueError):
        _raise("M5 replay evidence construction failed")

    _verify_unchanged_source(snapshot)
    try:
        pending = _write_local_artifact(
            destination=destination,
            snapshot=snapshot,
            benchmark=benchmark,
            run=run,
            resources=scientific_resources,
            members=members,
            record_counts=record_counts,
        )
    except (ArtifactValidationError, FileExistsError, OSError, RuntimeError, ValueError):
        _raise("M5 local replay artifact publication failed")

    try:
        _verify_unchanged_source(snapshot)
        completion_resources = ReplayRunResources(
            elapsed_seconds=float(time.perf_counter() - started_clock),
            peak_rss_bytes=_peak_rss_bytes(),
            measurement_scope=(
                "metadata-through-publication-and-final-source-verification-before-profile-binding"
            ),
        )
        # This final, deliberately unembedded check avoids a self-referential
        # resource member while still applying the local diagnostic caps.
        ReplayRunResources(
            elapsed_seconds=float(time.perf_counter() - started_clock),
            peak_rss_bytes=_peak_rss_bytes(),
            measurement_scope=(
                "metadata-through-publication-and-final-source-verification-before-profile-binding"
            ),
        )
    except ReplayRunnerError:
        with suppress(ArtifactValidationError, OSError, RuntimeError, ValueError):
            _rollback_pending_local_artifact(pending, snapshot=snapshot)
        raise
    except (TypeError, ValidationError, ValueError):
        with suppress(ArtifactValidationError, OSError, RuntimeError, ValueError):
            _rollback_pending_local_artifact(pending, snapshot=snapshot)
        _raise("M5 replay post-publication resource gate failed")

    try:
        artifact = _finalize_local_artifact(
            pending,
            snapshot=snapshot,
        )
    except (ArtifactValidationError, FileExistsError, OSError, RuntimeError, ValueError):
        with suppress(ArtifactValidationError, OSError, RuntimeError, ValueError):
            _rollback_pending_local_artifact(pending, snapshot=snapshot)
        _raise("M5 local replay artifact finalization failed")
    return ReplayLocalExecution(
        benchmark=artifact.benchmark,
        artifact=artifact,
        run=run,
        resources=completion_resources,
    )


def run_replay_local(output_dir: Path) -> ReplayLocalExecution:
    """Execute one clean, metadata-only M5 run into an ignored no-overwrite bundle."""

    snapshot = _initial_snapshot()
    return _execute_replay_local(output_dir, snapshot=snapshot)


def _independent_source_inodes(
    primary: LoadedReplayLocalArtifact,
    repeat: LoadedReplayLocalArtifact,
) -> bool:
    try:
        first = primary.path.resolve(strict=True)
        second = repeat.path.resolve(strict=True)
        if (
            _is_equal_or_descendant(first, second)
            or _is_equal_or_descendant(second, first)
            or (first.stat().st_dev, first.stat().st_ino)
            == (second.stat().st_dev, second.stat().st_ino)
        ):
            return False
        for name in REPLAY_LOCAL_ARTIFACT_PATHS:
            left = (first / name).stat(follow_symlinks=False)
            right = (second / name).stat(follow_symlinks=False)
            if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
                return False
    except OSError:
        return False
    return True


def build_replay_repeat_verification(
    primary: LoadedReplayLocalArtifact,
    repeat: LoadedReplayLocalArtifact,
) -> tuple[
    tuple[ReplaySourceMemberCommitmentV1, ...],
    ReplayRepeatVerificationV1,
]:
    """Compare every exact local scientific member from two independent runs."""

    if primary.run.run_id != repeat.run.run_id:
        raise ValueError("M5 repeat artifacts do not share one frozen run identity")
    commitment_paths = tuple(
        sorted(
            REPLAY_LOCAL_SCIENTIFIC_PATHS,
            key=lambda path: _ROLE_BY_PATH[path],
        )
    )
    commitments = tuple(
        ReplaySourceMemberCommitmentV1(
            schema="ffb.replay-source-member-commitment/v1",
            run_id=primary.run.run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            relative_role=_ROLE_BY_PATH[path],
            primary_byte_length=len(primary.member_bytes[path]),
            repeat_byte_length=len(repeat.member_bytes[path]),
            primary_record_count=primary.member_record_counts[path],
            repeat_record_count=repeat.member_record_counts[path],
            primary_sha256=hashlib.sha256(primary.member_bytes[path]).hexdigest(),
            repeat_sha256=hashlib.sha256(repeat.member_bytes[path]).hexdigest(),
            equal=(
                primary.member_bytes[path] == repeat.member_bytes[path]
                and primary.member_record_counts[path] == repeat.member_record_counts[path]
            ),
        )
        for path in commitment_paths
    )
    mismatch_count = sum(not row.equal for row in commitments)
    independent = _independent_source_inodes(primary, repeat)
    same_environment = primary.run.environment == repeat.run.environment
    commitment_bytes = canonical_replay_ndjson_bytes(commitments)
    verification = ReplayRepeatVerificationV1(
        schema="ffb.replay-repeat-verification/v1",
        run_id=primary.run.run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        primary_local_artifact_sha256=primary.artifact_sha256,
        repeat_local_artifact_sha256=repeat.artifact_sha256,
        primary_run_sha256=primary.run_sha256,
        repeat_run_sha256=repeat.run_sha256,
        source_member_commitments_sha256=hashlib.sha256(commitment_bytes).hexdigest(),
        scientific_member_count=len(commitments),
        mismatch_count=mismatch_count,
        scientific_members_all_equal=mismatch_count == 0,
        run_records_distinct=primary.run_sha256 != repeat.run_sha256,
        source_paths_and_inodes_independent=independent,
        same_named_cpu_environment=same_environment,
        evidence_scope=("distinct-path-inode-run-and-member-consistency-not-cryptographic-proof"),
        all_checks_passed=(
            mismatch_count == 0
            and primary.run_sha256 != repeat.run_sha256
            and independent
            and same_environment
        ),
    )
    return commitments, verification


def _require_current_replay_authority(
    artifact: LoadedReplayLocalArtifact,
    *,
    snapshot: CleanSourceSnapshot,
    environment: RuntimeEnvironment,
) -> None:
    try:
        output_argument = artifact.path.relative_to(snapshot.source_root).as_posix()
    except ValueError as error:
        raise ValueError("local replay artifact is outside the current source root") from error
    expected_run_id = derive_run_id(
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        git_revision=snapshot.git_revision,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version=snapshot.package_version,
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
    )
    run = artifact.run
    if (
        artifact.source_root != snapshot.source_root
        or run.run_id != expected_run_id
        or run.git_revision != snapshot.git_revision
        or run.lockfile_sha256 != snapshot.lockfile_sha256
        or run.package_version != snapshot.package_version
        or run.source_dirty
        or run.environment != environment
        or run.command
        != (
            "ffb",
            "replay",
            "run",
            "--output-dir",
            output_argument,
        )
    ):
        raise ValueError("local replay artifact does not match current curation authority")


def verify_replay_repeat_artifacts(
    *,
    primary_path: Path,
    repeat_path: Path,
) -> ReplayLoadedRepeatEvidence:
    """Strictly reload and compare two separately executed local replay artifacts."""

    try:
        snapshot = _initial_snapshot()
        environment = collect_runtime_environment()
        plan = load_replay_plan(source_root=snapshot.source_root)
        primary = _load_replay_local_artifact(
            primary_path,
            source_root=snapshot.source_root,
            plan=plan,
        )
        repeat = _load_replay_local_artifact(
            repeat_path,
            source_root=snapshot.source_root,
            plan=plan,
        )
        _require_current_replay_authority(
            primary,
            snapshot=snapshot,
            environment=environment,
        )
        _require_current_replay_authority(
            repeat,
            snapshot=snapshot,
            environment=environment,
        )
        commitments, verification = build_replay_repeat_verification(
            primary,
            repeat,
        )
        _verify_unchanged_source(snapshot)
    except (ArtifactValidationError, OSError, RuntimeError, TypeError, ValidationError, ValueError):
        _raise("M5 existing replay artifact verification failed")
    if not verification.all_checks_passed:
        _raise("M5 existing replay artifact repeat gate failed")
    return ReplayLoadedRepeatEvidence(
        primary=primary,
        repeat=repeat,
        source_commitments=commitments,
        repeat_verification=verification,
    )


def run_replay_repeat(
    *,
    primary_output_dir: Path,
    repeat_output_dir: Path,
) -> ReplayRepeatExecution:
    """Run two clean local executions and require exact scientific repeatability."""

    snapshot = _initial_snapshot()
    first, _ = _validated_output_path(
        primary_output_dir,
        source_root=snapshot.source_root,
    )
    second, _ = _validated_output_path(
        repeat_output_dir,
        source_root=snapshot.source_root,
    )
    if _is_equal_or_descendant(first, second) or _is_equal_or_descendant(second, first):
        _raise("M5 primary and repeat outputs must be disjoint")
    primary = _execute_replay_local(primary_output_dir, snapshot=snapshot)
    repeat = _execute_replay_local(repeat_output_dir, snapshot=snapshot)
    try:
        commitments, verification = build_replay_repeat_verification(
            primary.artifact,
            repeat.artifact,
        )
    except (OSError, TypeError, ValidationError, ValueError):
        _raise("M5 replay repeat verification failed")
    if not verification.all_checks_passed:
        _raise("M5 replay repeat gate failed")
    return ReplayRepeatExecution(
        primary=primary,
        repeat=repeat,
        source_commitments=commitments,
        repeat_verification=verification,
    )


def _curate_reloaded_local_artifact(
    artifact: LoadedReplayLocalArtifact,
    *,
    profile_summary: ReplayProfileSummaryV1,
) -> ReplayCuratedAggregateEvidence:
    benchmark = artifact.benchmark
    with _single_process_guard():
        descriptors = _descriptor_contracts(
            benchmark,
            run_id=artifact.run.run_id,
        )
        curated = curate_replay_evidence(
            plan=benchmark.plan,
            persistent_scene_evaluations=benchmark.persistent_scene_evaluations,
            persistent_metrics=benchmark.persistent_metrics,
            persistent_crossovers=benchmark.persistent_crossovers,
            health_results=benchmark.health_results,
            health_contrasts=benchmark.health_contrasts,
            health_events=benchmark.health_events,
            descriptor_aggregates=descriptors,
            log_group_bindings=tuple(
                ReplayLogGroupBinding(
                    sequence_id=f"nuscenes:{scene_name}",
                    log_group_ordinal=log_group_ordinal,
                )
                for scene_name, log_group_ordinal in zip(
                    M5_SCENE_NAMES,
                    benchmark.log_group_ordinals,
                    strict=True,
                )
            ),
            profile_summary=profile_summary,
            run=artifact.run,
        )
    _ensure_cpu_only_import_boundary()
    return curated


def _strictly_reload_curation_source(
    artifact: LoadedReplayLocalArtifact,
) -> LoadedReplayLocalArtifact:
    try:
        plan = load_replay_plan(source_root=artifact.source_root)
        reloaded = _load_replay_local_artifact(
            artifact.path,
            source_root=artifact.source_root,
            plan=plan,
        )
    except (ArtifactValidationError, OSError, RuntimeError, TypeError, ValidationError, ValueError):
        _raise("M5 replay curation source validation failed")
    if (
        reloaded.path != artifact.path
        or reloaded.source_root != artifact.source_root
        or reloaded.artifact_sha256 != artifact.artifact_sha256
        or reloaded.run_sha256 != artifact.run_sha256
        or reloaded.run != artifact.run
    ):
        _raise("M5 replay curation source identity changed")
    return reloaded


def curate_replay_local_artifact(
    artifact: LoadedReplayLocalArtifact,
    *,
    profile_summary: ReplayProfileSummaryV1,
) -> ReplayCuratedAggregateEvidence:
    """Curate only from freshly reloaded, typed local source bytes."""

    reloaded = _strictly_reload_curation_source(artifact)
    return _curate_reloaded_local_artifact(
        reloaded,
        profile_summary=profile_summary,
    )


def _resource_run_binding(
    artifact: LoadedReplayLocalArtifact,
) -> ReplayResourceRunBinding:
    if (
        artifact.resources.measurement_scope
        != "metadata-through-canonical-scientific-members-before-publication"
    ):
        _raise("M5 replay local resource scope is invalid")
    return ReplayResourceRunBinding(
        run=artifact.run,
        local_artifact_sha256=artifact.artifact_sha256,
        local_run_sha256=artifact.run_sha256,
        persisted_internal_elapsed_seconds=artifact.resources.elapsed_seconds,
        persisted_internal_peak_rss_bytes=artifact.resources.peak_rss_bytes,
        persisted_internal_measurement_scope=artifact.resources.measurement_scope,
    )


def curate_replay_verified_repeat(
    repeat_evidence: ReplayLoadedRepeatEvidence,
    *,
    primary_log_path: Path,
    repeat_log_path: Path,
) -> ReplayCuratedAggregateEvidence:
    """Import two complete-process logs and curate from a freshly verified primary."""

    curation_snapshot = _initial_snapshot()
    curation_environment = collect_runtime_environment()
    freshly_verified = verify_replay_repeat_artifacts(
        primary_path=repeat_evidence.primary.path,
        repeat_path=repeat_evidence.repeat.path,
    )
    try:
        _require_current_replay_authority(
            freshly_verified.primary,
            snapshot=curation_snapshot,
            environment=curation_environment,
        )
        _require_current_replay_authority(
            freshly_verified.repeat,
            snapshot=curation_snapshot,
            environment=curation_environment,
        )
    except (TypeError, ValueError):
        _raise("M5 replay curation repeat authority changed")
    if (
        freshly_verified.primary.artifact_sha256 != repeat_evidence.primary.artifact_sha256
        or freshly_verified.primary.run_sha256 != repeat_evidence.primary.run_sha256
        or freshly_verified.repeat.artifact_sha256 != repeat_evidence.repeat.artifact_sha256
        or freshly_verified.repeat.run_sha256 != repeat_evidence.repeat.run_sha256
        or freshly_verified.source_commitments != repeat_evidence.source_commitments
        or freshly_verified.repeat_verification != repeat_evidence.repeat_verification
    ):
        _raise("M5 replay curation repeat authority changed")
    try:
        resource_evidence = import_replay_execution_resource_evidence(
            primary_log_path=primary_log_path,
            repeat_log_path=repeat_log_path,
            primary=_resource_run_binding(freshly_verified.primary),
            repeat=_resource_run_binding(freshly_verified.repeat),
        )
        profile = _profile_summary(
            freshly_verified.primary.benchmark,
            run_id=freshly_verified.primary.run.run_id,
            resource_evidence=resource_evidence,
        )
    except (OSError, TypeError, ValidationError, ValueError):
        _raise("M5 replay external resource evidence validation failed")
    curated = _curate_reloaded_local_artifact(
        freshly_verified.primary,
        profile_summary=profile,
    )
    _verify_unchanged_source(curation_snapshot)
    return curated


def curate_replay_local_execution(
    execution: ReplayLocalExecution,
    *,
    repeat_evidence: ReplayLoadedRepeatEvidence,
    primary_log_path: Path,
    repeat_log_path: Path,
) -> ReplayCuratedAggregateEvidence:
    """Bind a just-completed primary execution to strict repeat/resource authority."""

    if (
        execution.artifact.path != repeat_evidence.primary.path
        or execution.artifact.artifact_sha256 != repeat_evidence.primary.artifact_sha256
        or execution.artifact.run_sha256 != repeat_evidence.primary.run_sha256
    ):
        _raise("M5 replay execution does not match verified primary evidence")
    return curate_replay_verified_repeat(
        repeat_evidence,
        primary_log_path=primary_log_path,
        repeat_log_path=repeat_log_path,
    )


def replay_runner_pending_validation_checks() -> tuple[str, ...]:
    """Name release checks that this runner intentionally cannot attest."""

    runner_evidenced = {
        "intent-freeze",
        "fixed-scene-population",
        "base-support",
        "health-schedules",
        "persistent-panel-completeness",
        "health-panel-completeness",
        "scene-bootstrap-and-cluster-sensitivity",
        "repeat-scientific-members",
        "cpu-and-memory-caps",
        "no-raw-payload-reads",
    }
    return tuple(
        check_id for check_id in M5_RELEASE_VALIDATION_CHECK_IDS if check_id not in runner_evidenced
    )
