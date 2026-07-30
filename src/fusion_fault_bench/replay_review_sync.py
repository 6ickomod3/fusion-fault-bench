"""Crash-recoverable synchronization of the two reviewed M5 public evidence files."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from fusion_fault_bench.artifacts import (
    absolute_artifact_path,
    assert_directory_descriptor_matches_path,
    atomic_rename_directory_no_replace_at,
    discover_git_metadata_dirs,
    entry_exists_at,
    open_or_create_real_directory,
    reject_directory_descriptor_in_git_metadata,
    reject_git_metadata_destination,
    write_exclusive_file_at,
)

_REPORT_PATH = Path("docs/reviews/m5-results-review.md")
_ATTESTATION_PATH = Path("docs/reviews/m5-results-review-attestation.json")
_STAGING_NAME = ".ffb-m5-reviewed-evidence-staging"
_SAFE_PUBLIC_FILE_MODES = frozenset({0o600, 0o644})


class _LoadedPackage(Protocol):
    @property
    def path(self) -> Path: ...

    @property
    def release_package_sha256(self) -> str: ...


class ReplayReviewSyncError(ValueError):
    """Reviewed evidence could not be synchronized as one safe pair."""


def _declared_output(path: Path, *, source_root: Path) -> Path:
    return absolute_artifact_path(path if path.is_absolute() else source_root / path)


def _file_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_exact_existing_file(
    directory_fd: int,
    name: str,
    expected: bytes,
    *,
    label: str,
) -> bool:
    flags = (
        os.O_RDWR
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ReplayReviewSyncError(f"M5 {label} is unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) not in _SAFE_PUBLIC_FILE_MODES
            or before.st_size != len(expected)
        ):
            raise ReplayReviewSyncError(f"M5 {label} is unsafe or differs from the release package")
        chunks: list[bytes] = []
        remaining = len(expected) + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        after = os.fstat(descriptor)
        try:
            reopened = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ReplayReviewSyncError(f"M5 {label} changed while it was read") from error
        if _file_fingerprint(before) != _file_fingerprint(after) or _file_fingerprint(
            before
        ) != _file_fingerprint(reopened):
            raise ReplayReviewSyncError(f"M5 {label} changed while it was read")
        if value != expected:
            raise ReplayReviewSyncError(f"M5 {label} differs from the release package")
        os.fsync(descriptor)
        return True
    finally:
        os.close(descriptor)


def _open_or_create_staging_directory(parent_fd: int) -> int:
    with suppress(FileExistsError):
        os.mkdir(_STAGING_NAME, mode=0o700, dir_fd=parent_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(_STAGING_NAME, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ReplayReviewSyncError("M5 reviewed-evidence staging directory is unsafe") from error
    try:
        observed = os.fstat(descriptor)
        reopened = os.stat(_STAGING_NAME, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.getuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
            or (observed.st_dev, observed.st_ino) != (reopened.st_dev, reopened.st_ino)
        ):
            raise ReplayReviewSyncError("M5 reviewed-evidence staging directory is unsafe")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _remove_staging_directory(parent_fd: int, staging_fd: int) -> None:
    if os.listdir(staging_fd):
        raise ReplayReviewSyncError("M5 reviewed-evidence staging directory is not empty")
    observed = os.fstat(staging_fd)
    try:
        reopened = os.stat(_STAGING_NAME, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ReplayReviewSyncError("M5 reviewed-evidence staging directory changed") from error
    if (observed.st_dev, observed.st_ino) != (reopened.st_dev, reopened.st_ino):
        raise ReplayReviewSyncError("M5 reviewed-evidence staging directory changed")
    os.rmdir(_STAGING_NAME, dir_fd=parent_fd)
    os.fsync(parent_fd)


def sync_reviewed_evidence_transaction(
    package: _LoadedPackage,
    *,
    report_output: Path,
    attestation_output: Path,
    source_root: Path,
) -> None:
    """Publish or resume the exact package review bytes at the two fixed paths."""

    from fusion_fault_bench.replay_release_package import validate_release_package

    root = absolute_artifact_path(source_root)
    expected = (root / _REPORT_PATH, root / _ATTESTATION_PATH)
    outputs = (
        _declared_output(report_output, source_root=root),
        _declared_output(attestation_output, source_root=root),
    )
    if outputs != expected or outputs[0].parent != outputs[1].parent:
        raise ReplayReviewSyncError("M5 reviewed evidence has only two fixed public paths")
    validated = validate_release_package(package.path)
    reloaded = validated.package
    if reloaded.release_package_sha256 != package.release_package_sha256:
        raise ReplayReviewSyncError("M5 release package changed before evidence synchronization")
    sources = (
        reloaded.files["evidence/results-review.md"],
        reloaded.files["evidence/results-review-attestation.json"],
    )
    git_metadata = discover_git_metadata_dirs(root)
    for output in outputs:
        reject_git_metadata_destination(output, git_metadata)
    parent = outputs[0].parent
    parent_fd = open_or_create_real_directory(parent)
    staging_fd: int | None = None
    try:
        assert_directory_descriptor_matches_path(parent_fd, parent, label="review output parent")
        reject_directory_descriptor_in_git_metadata(parent_fd, git_metadata)
        present = tuple(
            _read_exact_existing_file(
                parent_fd,
                output.name,
                value,
                label=f"reviewed-evidence output {output.name}",
            )
            for output, value in zip(outputs, sources, strict=True)
        )
        staging_exists = entry_exists_at(parent_fd, _STAGING_NAME)
        if all(present) and not staging_exists:
            return

        staging_fd = _open_or_create_staging_directory(parent_fd)
        expected_names = frozenset(output.name for output in outputs)
        observed_names = frozenset(os.listdir(staging_fd))
        if not observed_names <= expected_names:
            raise ReplayReviewSyncError(
                "M5 reviewed-evidence staging directory contains an unexpected member"
            )

        staged: dict[str, bool] = {}
        for output, value, output_present in zip(outputs, sources, present, strict=True):
            staged_present = _read_exact_existing_file(
                staging_fd,
                output.name,
                value,
                label=f"staged reviewed evidence {output.name}",
            )
            if output_present and staged_present:
                os.unlink(output.name, dir_fd=staging_fd)
                staged_present = False
            if not output_present and not staged_present:
                with suppress(FileExistsError):
                    write_exclusive_file_at(staging_fd, output.name, value)
                staged_present = _read_exact_existing_file(
                    staging_fd,
                    output.name,
                    value,
                    label=f"staged reviewed evidence {output.name}",
                )
                if not staged_present:
                    raise ReplayReviewSyncError("M5 reviewed evidence could not be staged")
            staged[output.name] = staged_present
        os.fsync(staging_fd)

        for output, value in zip(outputs, sources, strict=True):
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="review output parent",
            )
            if _read_exact_existing_file(
                parent_fd,
                output.name,
                value,
                label=f"reviewed-evidence output {output.name}",
            ):
                if staged[output.name]:
                    os.unlink(output.name, dir_fd=staging_fd)
                    staged[output.name] = False
                continue
            if not staged[output.name]:
                raise ReplayReviewSyncError("M5 reviewed evidence is missing its staged member")
            atomic_rename_directory_no_replace_at(
                staging_fd,
                output.name,
                parent_fd,
                output.name,
            )
            staged[output.name] = False
            if not _read_exact_existing_file(
                parent_fd,
                output.name,
                value,
                label=f"reviewed-evidence output {output.name}",
            ):
                raise ReplayReviewSyncError("M5 reviewed evidence disappeared after publication")

        os.fsync(staging_fd)
        os.fsync(parent_fd)
        for output, value in zip(outputs, sources, strict=True):
            if not _read_exact_existing_file(
                parent_fd,
                output.name,
                value,
                label=f"reviewed-evidence output {output.name}",
            ):
                raise ReplayReviewSyncError("M5 reviewed evidence is incomplete")
        _remove_staging_directory(parent_fd, staging_fd)
        assert_directory_descriptor_matches_path(parent_fd, parent, label="review output parent")
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


__all__ = ["ReplayReviewSyncError", "sync_reviewed_evidence_transaction"]
