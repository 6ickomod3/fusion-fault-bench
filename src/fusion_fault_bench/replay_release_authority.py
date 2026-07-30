"""Authenticated source, review, software, and validation authority for M5.

This module deliberately contains no dataset or outcome evaluation.  It turns
already-produced, bounded public evidence into content-addressed release
authority and independently reconstructs each validation digest.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fusion_fault_bench.contracts.matrix_v1 import (
    M3_PROCEDURAL_MATRIX_SHA256,
    load_experiment_matrix,
)
from fusion_fault_bench.health_release import (
    HEALTH_RELEASE_ARTIFACT_PATHS,
    load_health_release,
)
from fusion_fault_bench.replay_fit import M4_RELEASE_RELATIVE_PATH
from fusion_fault_bench.replay_plan import M5_PERSISTENT_MATRIX_PATH

M5_IMPLEMENTATION_SNAPSHOT_DOMAIN = b"fusion-fault-bench/m5-implementation-snapshot/v1\x00"
M5_IMPLEMENTATION_SNAPSHOT_MAX_FILES = 10_000
M5_IMPLEMENTATION_SNAPSHOT_MAX_FILE_BYTES = 64 * 1024 * 1024
M5_IMPLEMENTATION_SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024

_IMPLEMENTATION_TRACKED_PREFIXES = ("src/", "tools/", "tests/")
_IMPLEMENTATION_STATIC_PATHS = (
    "examples/replay/m5-nuscenes-mini-replay-v1.json",
    M5_PERSISTENT_MATRIX_PATH.as_posix(),
    "pyproject.toml",
    ".python-version",
    "uv.lock",
    "LICENSE",
    "DATA_AND_MODEL_TERMS.md",
    ".github/workflows/ci.yml",
    "docs/benchmark-contract-v0.1.md",
    "docs/m5-replay-plan.md",
    "docs/reviews/m5-plan-review.md",
    "docs/m5-resource-scope-amendment.md",
    "docs/m5-release-pipeline-plan.md",
    "docs/reviews/m5-release-pipeline-plan-review.md",
)


class ReplayReleaseAuthorityError(ValueError):
    """M5 release authority is incomplete, contradictory, or unauthenticated."""


@dataclass(frozen=True, slots=True)
class ImplementationSnapshotEntry:
    """One repository-relative member of the implementation snapshot."""

    path: str
    byte_length: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ImplementationSnapshot:
    """Domain-separated digest and audit table for the exact reviewed source."""

    scientific_git_revision: str
    entries: tuple[ImplementationSnapshotEntry, ...]
    sha256: str

    @property
    def file_count(self) -> int:
        return len(self.entries)


def _require_revision(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ReplayReleaseAuthorityError(
            "scientific Git revision must be a lowercase 40-character object ID"
        )
    return value


def _require_snapshot_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ReplayReleaseAuthorityError(
            "implementation snapshot paths must be normalized repository-relative POSIX paths"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReplayReleaseAuthorityError(
            "implementation snapshot paths must be normalized repository-relative POSIX paths"
        )
    return value


def implementation_snapshot_from_files(
    files: Mapping[str, bytes],
    *,
    scientific_git_revision: str,
) -> ImplementationSnapshot:
    """Digest an exact canonical path/byte table with unambiguous framing."""

    revision = _require_revision(scientific_git_revision)
    if not files or len(files) > M5_IMPLEMENTATION_SNAPSHOT_MAX_FILES:
        raise ReplayReleaseAuthorityError("implementation snapshot file count is invalid")

    normalized: list[tuple[str, bytes]] = []
    total_bytes = 0
    for raw_path, raw_value in files.items():
        path = _require_snapshot_path(raw_path)
        if len(raw_value) > M5_IMPLEMENTATION_SNAPSHOT_MAX_FILE_BYTES:
            raise ReplayReleaseAuthorityError("implementation snapshot member exceeds its byte cap")
        total_bytes += len(raw_value)
        if total_bytes > M5_IMPLEMENTATION_SNAPSHOT_MAX_BYTES:
            raise ReplayReleaseAuthorityError("implementation snapshot exceeds its total byte cap")
        normalized.append((path, raw_value))
    normalized.sort(key=lambda item: item[0].encode("utf-8"))

    digest = hashlib.sha256(M5_IMPLEMENTATION_SNAPSHOT_DOMAIN)
    digest.update(len(normalized).to_bytes(8, "big"))
    entries: list[ImplementationSnapshotEntry] = []
    for path, value in normalized:
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
        entries.append(
            ImplementationSnapshotEntry(
                path=path,
                byte_length=len(value),
                sha256=hashlib.sha256(value).hexdigest(),
            )
        )
    return ImplementationSnapshot(
        scientific_git_revision=revision,
        entries=tuple(entries),
        sha256=digest.hexdigest(),
    )


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", os.fspath(source_root), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ReplayReleaseAuthorityError("Git implementation-snapshot query failed")
    return result.stdout


def _parse_git_entries(raw: bytes, *, tree: bool) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\x00"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", maxsplit=1)
            if tree:
                mode, object_type, object_id = metadata.split(b" ", maxsplit=2)
                if object_type != b"blob":
                    raise ValueError
            else:
                mode, object_id, stage = metadata.split(b" ", maxsplit=2)
                if stage != b"0":
                    raise ValueError
            path = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ReplayReleaseAuthorityError(
                "Git contains an unsupported implementation-snapshot entry"
            ) from error
        _require_snapshot_path(path)
        if mode not in {b"100644", b"100755"} or path in result:
            raise ReplayReleaseAuthorityError(
                "Git contains an unsupported implementation-snapshot entry"
            )
        result[path] = (mode.decode("ascii"), object_id.decode("ascii"))
    return result


def _git_head_and_index(
    source_root: Path,
) -> tuple[str, dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    try:
        revision = _git_bytes(source_root, "rev-parse", "HEAD").decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ReplayReleaseAuthorityError("Git revision is not ASCII") from error
    _require_revision(revision)
    head = _parse_git_entries(
        _git_bytes(source_root, "ls-tree", "-r", "-z", "--full-tree", revision),
        tree=True,
    )
    index = _parse_git_entries(
        _git_bytes(source_root, "ls-files", "--stage", "-z"),
        tree=False,
    )
    return revision, head, index


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _read_regular_at(root_descriptor: int, path: str, *, executable: bool) -> bytes:
    parts = PurePosixPath(_require_snapshot_path(path)).parts
    directory_descriptor = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or bool(before.st_mode & stat.S_IXUSR) != executable
                or before.st_size > M5_IMPLEMENTATION_SNAPSHOT_MAX_FILE_BYTES
            ):
                raise ReplayReleaseAuthorityError(
                    "implementation snapshot member is not a supported tracked file"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ReplayReleaseAuthorityError(
                        "implementation snapshot member changed while reading"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ReplayReleaseAuthorityError(
                    "implementation snapshot member changed while reading"
                )
            after = os.fstat(descriptor)
            reopened = os.stat(parts[-1], dir_fd=directory_descriptor, follow_symlinks=False)
            stable_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
            if any(
                getattr(before, field) != getattr(after, field) for field in stable_fields
            ) or any(getattr(before, field) != getattr(reopened, field) for field in stable_fields):
                raise ReplayReleaseAuthorityError(
                    "implementation snapshot member changed while reading"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ReplayReleaseAuthorityError(
            "implementation snapshot member is missing or redirected"
        ) from error
    finally:
        os.close(directory_descriptor)


def _scoped(path: str) -> bool:
    return path.startswith(_IMPLEMENTATION_TRACKED_PREFIXES)


def build_implementation_snapshot(source_root: Path) -> ImplementationSnapshot:
    """Expand and authenticate the exact implementation content snapshot."""

    absolute_root = Path(os.path.abspath(os.fspath(source_root)))
    try:
        if (
            absolute_root.is_symlink()
            or not absolute_root.is_dir()
            or absolute_root.resolve(strict=True) != absolute_root
        ):
            raise ReplayReleaseAuthorityError("source root must be a real directory")
    except OSError as error:
        raise ReplayReleaseAuthorityError("source root must be a real directory") from error

    matrix = load_experiment_matrix(M5_PERSISTENT_MATRIX_PATH, source_root=absolute_root)
    if (
        matrix.matrix_sha256 != M3_PROCEDURAL_MATRIX_SHA256
        or len(matrix.matrix.execution_order) != 8
        or len(matrix.matrix.profiles) != 3
    ):
        raise ReplayReleaseAuthorityError("implementation snapshot requires the exact M3 matrix")
    health = load_health_release(absolute_root / M4_RELEASE_RELATIVE_PATH)

    revision, head, index = _git_head_and_index(absolute_root)
    selected = {path for path in head if _scoped(path)}
    selected.update(_IMPLEMENTATION_STATIC_PATHS)
    selected.update(entry.manifest for entry in matrix.matrix.execution_order)
    selected.update(entry.profile for entry in matrix.matrix.profiles)
    selected.update(
        f"{M4_RELEASE_RELATIVE_PATH.as_posix()}/{name}" for name in HEALTH_RELEASE_ARTIFACT_PATHS
    )
    if not selected or len(selected) > M5_IMPLEMENTATION_SNAPSHOT_MAX_FILES:
        raise ReplayReleaseAuthorityError("implementation snapshot expansion is invalid")

    head_scoped = {path for path in head if _scoped(path)}
    index_scoped = {path for path in index if _scoped(path)}
    if index_scoped != head_scoped:
        raise ReplayReleaseAuthorityError(
            "tracked implementation scope does not exactly match HEAD"
        )
    untracked = _git_bytes(
        absolute_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "src",
        "tools",
        "tests",
    )
    if untracked:
        raise ReplayReleaseAuthorityError(
            "implementation scope contains untracked non-ignored files"
        )
    for path in selected:
        if path not in head or index.get(path) != head[path]:
            raise ReplayReleaseAuthorityError(
                "implementation snapshot controlling input does not exactly match HEAD"
            )

    root_descriptor = os.open(
        absolute_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        root_before = os.fstat(root_descriptor)
        files: dict[str, bytes] = {}
        for path in sorted(selected, key=lambda value: value.encode("utf-8")):
            mode, object_id = head[path]
            value = _read_regular_at(root_descriptor, path, executable=mode == "100755")
            if _git_blob_sha1(value) != object_id:
                raise ReplayReleaseAuthorityError(
                    "implementation snapshot controlling input differs from HEAD"
                )
            files[path] = value
        root_after = os.fstat(root_descriptor)
        root_reopened = os.stat(absolute_root, follow_symlinks=False)
        root_fields = ("st_dev", "st_ino", "st_mode")
        if any(
            getattr(root_before, field) != getattr(root_after, field) for field in root_fields
        ) or any(
            getattr(root_before, field) != getattr(root_reopened, field) for field in root_fields
        ):
            raise ReplayReleaseAuthorityError("source root changed during snapshot expansion")
    finally:
        os.close(root_descriptor)

    final_revision, final_head, final_index = _git_head_and_index(absolute_root)
    if final_revision != revision or final_head != head or final_index != index:
        raise ReplayReleaseAuthorityError("Git authority changed during snapshot expansion")
    final_matrix = load_experiment_matrix(M5_PERSISTENT_MATRIX_PATH, source_root=absolute_root)
    final_health = load_health_release(absolute_root / M4_RELEASE_RELATIVE_PATH)
    if (
        final_matrix.matrix_sha256 != matrix.matrix_sha256
        or final_matrix.matrix != matrix.matrix
        or final_health.release_artifact_sha256 != health.release_artifact_sha256
    ):
        raise ReplayReleaseAuthorityError(
            "strict controlling input changed during snapshot expansion"
        )
    return implementation_snapshot_from_files(
        files,
        scientific_git_revision=revision,
    )
