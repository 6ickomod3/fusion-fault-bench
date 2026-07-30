"""Transactional synchronization of the two reviewed M5 public evidence files."""

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
    create_staging_directory_at,
    discover_git_metadata_dirs,
    entry_exists_at,
    open_or_create_real_directory,
    read_file_at,
    reject_directory_descriptor_in_git_metadata,
    reject_git_metadata_destination,
    write_exclusive_file_at,
)

_REPORT_PATH = Path("docs/reviews/m5-results-review.md")
_ATTESTATION_PATH = Path("docs/reviews/m5-results-review-attestation.json")


class _LoadedPackage(Protocol):
    @property
    def path(self) -> Path: ...

    @property
    def release_package_sha256(self) -> str: ...


class ReplayReviewSyncError(ValueError):
    """Reviewed evidence could not be synchronized as one safe pair."""


def _declared_output(path: Path, *, source_root: Path) -> Path:
    return absolute_artifact_path(path if path.is_absolute() else source_root / path)


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size


def sync_reviewed_evidence_transaction(
    package: _LoadedPackage,
    *,
    report_output: Path,
    attestation_output: Path,
    source_root: Path,
) -> None:
    """Publish exact package review bytes to the two fixed absent documentation paths."""

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
    staging_name: str | None = None
    published: list[tuple[str, tuple[int, int, int, int]]] = []
    try:
        assert_directory_descriptor_matches_path(parent_fd, parent, label="review output parent")
        reject_directory_descriptor_in_git_metadata(parent_fd, git_metadata)
        if any(entry_exists_at(parent_fd, output.name) for output in outputs):
            raise FileExistsError("M5 reviewed-evidence outputs must both be absent")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staged: dict[str, tuple[int, int, int, int]] = {}
        for output, value in zip(outputs, sources, strict=True):
            write_exclusive_file_at(staging_fd, output.name, value)
            if read_file_at(staging_fd, output.name, byte_cap=len(value)) != value:
                raise ReplayReviewSyncError("M5 staged reviewed evidence changed after write")
            metadata = os.stat(output.name, dir_fd=staging_fd, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ReplayReviewSyncError("M5 staged reviewed evidence is unsafe")
            staged[output.name] = _fingerprint(metadata)
        os.fsync(staging_fd)
        for output in outputs:
            assert_directory_descriptor_matches_path(
                parent_fd,
                parent,
                label="review output parent",
            )
            if entry_exists_at(parent_fd, output.name):
                raise FileExistsError("M5 reviewed-evidence output appeared during publication")
            atomic_rename_directory_no_replace_at(
                staging_fd,
                output.name,
                parent_fd,
                output.name,
            )
            observed = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            if _fingerprint(observed) != staged[output.name]:
                raise ReplayReviewSyncError("M5 reviewed-evidence identity changed at publication")
            published.append((output.name, staged[output.name]))
        os.rmdir(staging_name, dir_fd=parent_fd)
        staging_name = None
        os.fsync(parent_fd)
        assert_directory_descriptor_matches_path(parent_fd, parent, label="review output parent")
    except BaseException:
        if len(published) < len(outputs):
            for name, expected_fingerprint in reversed(published):
                with suppress(OSError):
                    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    if _fingerprint(observed) == expected_fingerprint:
                        os.unlink(name, dir_fd=parent_fd)
            if staging_fd is not None:
                for output in outputs:
                    with suppress(OSError):
                        os.unlink(output.name, dir_fd=staging_fd)
            if staging_name is not None:
                with suppress(OSError):
                    os.rmdir(staging_name, dir_fd=parent_fd)
            with suppress(OSError):
                os.fsync(parent_fd)
        raise
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


__all__ = ["ReplayReviewSyncError", "sync_reviewed_evidence_transaction"]
