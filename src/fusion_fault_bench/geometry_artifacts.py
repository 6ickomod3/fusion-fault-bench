"""Strict construction, loading, and publication of M2 geometry artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    compute_artifact_digest,
    compute_run_record_digest,
    derive_run_id,
    discover_git_metadata_dirs,
)
from fusion_fault_bench.artifacts import (
    absolute_artifact_path as _absolute_lexical,
)
from fusion_fault_bench.artifacts import (
    assert_directory_descriptor_matches_path as _assert_directory_fd_matches_path,
)
from fusion_fault_bench.artifacts import (
    atomic_rename_directory_no_replace_at as _atomic_rename_no_replace_at,
)
from fusion_fault_bench.artifacts import (
    create_staging_directory_at as _create_staging_directory_at,
)
from fusion_fault_bench.artifacts import (
    entry_exists_at as _entry_exists_at,
)
from fusion_fault_bench.artifacts import (
    open_or_create_real_directory as _open_or_create_real_directory,
)
from fusion_fault_bench.artifacts import (
    read_file_at as _read_at,
)
from fusion_fault_bench.artifacts import (
    reject_directory_descriptor_in_git_metadata as _reject_directory_fd_in_git_metadata,
)
from fusion_fault_bench.artifacts import (
    reject_git_metadata_destination as _reject_git_metadata_destination,
)
from fusion_fault_bench.artifacts import (
    strict_json_object_body as _strict_json_body,
)
from fusion_fault_bench.artifacts import (
    write_exclusive_file_at as _write_exclusive_at,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    FROZEN_GEOMETRY_MANIFEST_SHA256,
    GEOMETRY_ARTIFACT_CONTRACT,
    GEOMETRY_ARTIFACT_PATHS,
    GEOMETRY_INDEXED_PAYLOAD_PATHS,
    GEOMETRY_MANIFEST_FILE,
    GEOMETRY_MEMBER_BYTE_CAP,
    GEOMETRY_PAYLOAD_INDEX_FILE,
    GEOMETRY_RUN_FILE,
    GEOMETRY_SUCCESS_FILE,
    GEOMETRY_TREE_BYTE_CAP,
    GEOMETRY_VALIDATION_FILE,
    GeometryPayloadIndexV1,
    GeometryValidationManifestV1,
    GeometryValidationV1,
    PayloadFileEntryV1,
    validate_geometry_result_against_manifest,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RunRecordV1Alpha1


@dataclass(frozen=True, slots=True)
class GeometryArtifactWriteRequest:
    """Sanitized M2 records ready for deterministic artifact construction."""

    manifest: GeometryValidationManifestV1
    validation: GeometryValidationV1
    run: RunRecordV1Alpha1


@dataclass(frozen=True, slots=True)
class LoadedGeometryArtifact:
    """One strictly loaded and cross-validated M2 geometry artifact."""

    path: Path
    manifest: GeometryValidationManifestV1
    validation: GeometryValidationV1
    payload_index: GeometryPayloadIndexV1
    run: RunRecordV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedGeometryArtifact:
    files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class _GeometryTreeSnapshot:
    root_stat: os.stat_result
    entries: Mapping[str, os.stat_result]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
            raise ArtifactValidationError("geometry artifact path cannot be inspected") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ArtifactValidationError("geometry artifact path contains a symlink")


def _require_safe_tree(root: Path) -> _GeometryTreeSnapshot:
    absolute = _absolute_lexical(root)
    _reject_root_symlink_components(absolute)
    try:
        root_stat = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError("geometry artifact directory is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ArtifactValidationError("geometry artifact root must be a real directory")

    entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(absolute) as iterator:
            for entry in iterator:
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ArtifactValidationError("geometry artifact contains a symlink member")
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ArtifactValidationError("geometry artifact members must be regular files")
                entries[entry.name] = entry_stat
    except ArtifactValidationError:
        raise
    except OSError as error:
        raise ArtifactValidationError("geometry artifact members cannot be inspected") from error

    if set(entries) != set(GEOMETRY_ARTIFACT_PATHS):
        raise ArtifactValidationError("geometry artifact file allowlist mismatch")
    if any(entry.st_size > GEOMETRY_MEMBER_BYTE_CAP for entry in entries.values()):
        raise ArtifactValidationError("geometry artifact member exceeds the 1 MiB cap")
    if sum(entry.st_size for entry in entries.values()) > GEOMETRY_TREE_BYTE_CAP:
        raise ArtifactValidationError("geometry artifact exceeds the 5 MiB tree cap")
    return _GeometryTreeSnapshot(root_stat=root_stat, entries=entries)


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
            raise ArtifactValidationError("geometry artifact member is not a regular file")
        if _stat_fingerprint(file_stat) != _stat_fingerprint(expected_stat):
            raise ArtifactValidationError("geometry artifact member changed during validation")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_member(
    root: Path,
    name: str,
    *,
    expected_stat: os.stat_result,
) -> bytes:
    descriptor = _open_regular_member(root, name, expected_stat=expected_stat)
    try:
        chunks: list[bytes] = []
        remaining = GEOMETRY_MEMBER_BYTE_CAP + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) > GEOMETRY_MEMBER_BYTE_CAP:
            raise ArtifactValidationError("geometry artifact member exceeds the 1 MiB cap")
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
        body = _strict_json_body(data, label=label)
    except ArtifactValidationError as error:
        raise ArtifactValidationError(f"{label} is not strict canonical JSON") from error
    try:
        model = validate(body)
    except (ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"{label} violates its fixed schema") from error
    if canonical_json_bytes(model) != data:
        raise ArtifactValidationError(f"{label} is not canonical JSON")
    return model


def _verify_tree_snapshot(root: Path, snapshot: _GeometryTreeSnapshot) -> None:
    _reject_root_symlink_components(root)
    try:
        current_root_stat = os.lstat(root)
    except OSError as error:
        raise ArtifactValidationError(
            "geometry artifact root disappeared during validation"
        ) from error
    if _stat_fingerprint(current_root_stat) != _stat_fingerprint(snapshot.root_stat):
        raise ArtifactValidationError("geometry artifact root changed during validation")

    current_entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                current_entries[entry.name] = entry.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactValidationError("geometry artifact cannot be rechecked") from error
    if set(current_entries) != set(snapshot.entries):
        raise ArtifactValidationError("geometry artifact allowlist changed during validation")
    for name, expected_stat in snapshot.entries.items():
        current_stat = current_entries.get(name)
        if current_stat is None or _stat_fingerprint(current_stat) != _stat_fingerprint(
            expected_stat
        ):
            raise ArtifactValidationError("geometry artifact member changed during validation")


def _expected_run_id(manifest_sha256: str, run: RunRecordV1Alpha1) -> str:
    return derive_run_id(
        manifest_sha256=manifest_sha256,
        git_revision=run.git_revision,
        lockfile_sha256=run.lockfile_sha256,
        package_version=run.package_version,
        artifact_contract=GEOMETRY_ARTIFACT_CONTRACT,
    )


def _validate_run(
    manifest: GeometryValidationManifestV1,
    manifest_sha256: str,
    run: RunRecordV1Alpha1,
    *,
    artifact_sha256: str | None = None,
) -> None:
    if run.manifest_sha256 != manifest_sha256:
        raise ArtifactValidationError("geometry run manifest identity is invalid")
    if run.run_id != _expected_run_id(manifest_sha256, run):
        raise ArtifactValidationError("geometry run_id is invalid")
    if run.source_dirty or run.status != "succeeded":
        raise ArtifactValidationError("geometry artifact requires a clean successful run")
    if tuple(run.command) != tuple(manifest.artifact.logical_command):
        raise ArtifactValidationError("geometry run command is not the frozen logical command")
    cpu_model = run.environment.cpu_model
    if (
        cpu_model != cpu_model.strip()
        or "/" in cpu_model
        or "\\" in cpu_model
        or cpu_model.casefold().startswith("file:")
        or any(ord(character) < 32 for character in cpu_model)
    ):
        raise ArtifactValidationError("geometry run CPU model is not a sanitized hardware name")
    if artifact_sha256 is not None and run.artifact_sha256 != artifact_sha256:
        raise ArtifactValidationError("geometry run artifact identity is invalid")


def validate_geometry_validation_bundle(
    manifest: GeometryValidationManifestV1,
    validation: GeometryValidationV1,
    run: RunRecordV1Alpha1,
) -> None:
    """Cross-validate one sanitized M2 manifest, result, and run record."""

    manifest_sha256 = sha256_digest(manifest)
    if manifest_sha256 != FROZEN_GEOMETRY_MANIFEST_SHA256:
        raise ArtifactValidationError("geometry manifest is not the frozen M2 intent")
    _validate_run(manifest, manifest_sha256, run)
    if validation.run_id != run.run_id:
        raise ArtifactValidationError("geometry result run_id is invalid")
    if validation.manifest_sha256 != manifest_sha256:
        raise ArtifactValidationError("geometry result manifest identity is invalid")
    try:
        validate_geometry_result_against_manifest(manifest, validation)
    except ValueError as error:
        raise ArtifactValidationError("geometry result contradicts the frozen manifest") from error
    if not validation.all_checks_passed:
        raise ArtifactValidationError("geometry result did not pass every frozen M2 check")


def _finalize_run(run: RunRecordV1Alpha1, artifact_sha256: str) -> RunRecordV1Alpha1:
    value = run.model_dump(mode="python", by_alias=True)
    value["artifact_sha256"] = artifact_sha256
    return RunRecordV1Alpha1.model_validate(value)


def _prepare_geometry_artifact(
    request: GeometryArtifactWriteRequest,
) -> _PreparedGeometryArtifact:
    validate_geometry_validation_bundle(request.manifest, request.validation, request.run)
    manifest_sha256 = sha256_digest(request.manifest)
    indexed_files: dict[str, bytes] = {
        GEOMETRY_MANIFEST_FILE: canonical_json_bytes(request.manifest),
        GEOMETRY_VALIDATION_FILE: canonical_json_bytes(request.validation),
    }
    if any(len(value) > GEOMETRY_MEMBER_BYTE_CAP for value in indexed_files.values()):
        raise ArtifactValidationError("geometry artifact member exceeds the 1 MiB cap")

    payload_index = GeometryPayloadIndexV1(
        schema="ffb.geometry-payload-index/v1",
        artifact_contract=GEOMETRY_ARTIFACT_CONTRACT,
        run_id=request.run.run_id,
        manifest_sha256=manifest_sha256,
        files=tuple(
            PayloadFileEntryV1(
                path=path,
                byte_length=len(indexed_files[path]),
                sha256=_sha256_bytes(indexed_files[path]),
            )
            for path in GEOMETRY_INDEXED_PAYLOAD_PATHS
        ),
    )
    index_bytes = canonical_json_bytes(payload_index)
    artifact_sha256 = compute_artifact_digest(index_bytes)
    run = _finalize_run(request.run, artifact_sha256)
    _validate_run(
        request.manifest,
        manifest_sha256,
        run,
        artifact_sha256=artifact_sha256,
    )
    run_bytes = canonical_json_bytes(run)
    run_sha256 = compute_run_record_digest(run_bytes)
    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )
    files = {
        **indexed_files,
        GEOMETRY_PAYLOAD_INDEX_FILE: index_bytes,
        GEOMETRY_RUN_FILE: run_bytes,
        GEOMETRY_SUCCESS_FILE: canonical_json_bytes(success),
    }
    if any(len(value) > GEOMETRY_MEMBER_BYTE_CAP for value in files.values()):
        raise ArtifactValidationError("geometry artifact member exceeds the 1 MiB cap")
    if sum(map(len, files.values())) > GEOMETRY_TREE_BYTE_CAP:
        raise ArtifactValidationError("geometry artifact exceeds the 5 MiB tree cap")
    return _PreparedGeometryArtifact(files=files)


def _load_geometry_artifact(path: Path) -> LoadedGeometryArtifact:
    root = _absolute_lexical(path)
    snapshot = _require_safe_tree(root)
    member_bytes = {
        name: _read_member(root, name, expected_stat=snapshot.entries[name])
        for name in GEOMETRY_ARTIFACT_PATHS
    }

    manifest = _load_model(
        member_bytes[GEOMETRY_MANIFEST_FILE],
        label=GEOMETRY_MANIFEST_FILE,
        validate=GeometryValidationManifestV1.model_validate_json,
    )
    validation = _load_model(
        member_bytes[GEOMETRY_VALIDATION_FILE],
        label=GEOMETRY_VALIDATION_FILE,
        validate=GeometryValidationV1.model_validate_json,
    )
    payload_index = _load_model(
        member_bytes[GEOMETRY_PAYLOAD_INDEX_FILE],
        label=GEOMETRY_PAYLOAD_INDEX_FILE,
        validate=GeometryPayloadIndexV1.model_validate_json,
    )
    run = _load_model(
        member_bytes[GEOMETRY_RUN_FILE],
        label=GEOMETRY_RUN_FILE,
        validate=RunRecordV1Alpha1.model_validate_json,
    )
    success = _load_model(
        member_bytes[GEOMETRY_SUCCESS_FILE],
        label=GEOMETRY_SUCCESS_FILE,
        validate=SuccessMarkerV1Alpha1.model_validate_json,
    )

    validate_geometry_validation_bundle(manifest, validation, run)
    manifest_sha256 = sha256_digest(manifest)
    if payload_index.run_id != run.run_id:
        raise ArtifactValidationError("geometry payload index run_id is invalid")
    if payload_index.manifest_sha256 != manifest_sha256:
        raise ArtifactValidationError("geometry payload index manifest identity is invalid")
    for expected_path, entry in zip(
        GEOMETRY_INDEXED_PAYLOAD_PATHS,
        payload_index.files,
        strict=True,
    ):
        value = member_bytes[expected_path]
        if entry.path != expected_path:
            raise ArtifactValidationError("geometry payload index path order is invalid")
        if entry.byte_length != len(value):
            raise ArtifactValidationError("geometry payload byte length is invalid")
        if entry.sha256 != _sha256_bytes(value):
            raise ArtifactValidationError("geometry payload digest is invalid")

    index_bytes = member_bytes[GEOMETRY_PAYLOAD_INDEX_FILE]
    artifact_sha256 = compute_artifact_digest(index_bytes)
    _validate_run(
        manifest,
        manifest_sha256,
        run,
        artifact_sha256=artifact_sha256,
    )
    run_sha256 = compute_run_record_digest(member_bytes[GEOMETRY_RUN_FILE])
    if success.artifact_sha256 != artifact_sha256:
        raise ArtifactValidationError("geometry completion artifact identity is invalid")
    if success.run_sha256 != run_sha256:
        raise ArtifactValidationError("geometry completion run identity is invalid")

    _verify_tree_snapshot(root, snapshot)
    return LoadedGeometryArtifact(
        path=root,
        manifest=manifest,
        validation=validation,
        payload_index=payload_index,
        run=run,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )


def load_geometry_validation_artifact(path: Path) -> LoadedGeometryArtifact:
    """Strictly load one complete five-file M2 geometry artifact."""

    try:
        return _load_geometry_artifact(path)
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactValidationError("invalid M2 geometry artifact") from error


def _safe_cleanup_staging_at(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
) -> None:
    for name in GEOMETRY_ARTIFACT_PATHS:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=staging_fd)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _write_geometry_validation_artifact(
    request: GeometryArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path | None,
    git_metadata_dirs: Sequence[Path] | None,
) -> LoadedGeometryArtifact:
    prepared = _prepare_geometry_artifact(request)
    target = _absolute_lexical(destination)
    metadata_dirs = (
        discover_git_metadata_dirs(source_root)
        if git_metadata_dirs is None
        else tuple(git_metadata_dirs)
    )
    _reject_git_metadata_destination(target, metadata_dirs)
    if os.path.lexists(target):
        raise FileExistsError("geometry artifact destination already exists")

    parent = target.parent
    parent_fd = _open_or_create_real_directory(parent)
    try:
        _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
        _reject_directory_fd_in_git_metadata(parent_fd, metadata_dirs)
        if _entry_exists_at(parent_fd, target.name):
            raise FileExistsError("geometry artifact destination already exists")
        staging_name, staging_fd = _create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        published = False
        try:
            for name in GEOMETRY_ARTIFACT_PATHS[:-1]:
                _write_exclusive_at(staging_fd, name, prepared.files[name])
            for name in GEOMETRY_ARTIFACT_PATHS[:-1]:
                if (
                    _read_at(
                        staging_fd,
                        name,
                        byte_cap=len(prepared.files[name]),
                    )
                    != prepared.files[name]
                ):
                    raise ArtifactValidationError("geometry staging verification failed")

            _write_exclusive_at(
                staging_fd,
                GEOMETRY_SUCCESS_FILE,
                prepared.files[GEOMETRY_SUCCESS_FILE],
            )
            os.fsync(staging_fd)
            _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
            _load_geometry_artifact(staging)
            _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
            _reject_directory_fd_in_git_metadata(parent_fd, metadata_dirs)
            if _entry_exists_at(parent_fd, target.name):
                raise FileExistsError("geometry artifact destination already exists")
            _atomic_rename_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            published = True
            os.fsync(parent_fd)
            _assert_directory_fd_matches_path(parent_fd, parent, label="destination parent")
            loaded = load_geometry_validation_artifact(target)
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


def write_geometry_validation_artifact(
    request: GeometryArtifactWriteRequest,
    destination: Path,
    *,
    source_root: Path | None = None,
    git_metadata_dirs: Sequence[Path] | None = None,
) -> LoadedGeometryArtifact:
    """Validate, stage, and atomically publish one no-overwrite M2 artifact."""

    try:
        return _write_geometry_validation_artifact(
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
        raise ArtifactValidationError("M2 geometry artifact publication failed") from error
