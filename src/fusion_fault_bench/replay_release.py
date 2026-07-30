"""Strict review-candidate and outer-package publication for M5.

The scientific replay remains in :mod:`fusion_fault_bench.replay_runner`.
This module owns the two immutable publication envelopes defined by the M5
release-pipeline preregistration: a 34-file review candidate and, after an
independent review, a 41-file release package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Never, cast

from pydantic import ValidationError

from fusion_fault_bench.artifacts import (
    absolute_artifact_path,
    assert_directory_descriptor_matches_path,
    atomic_rename_directory_no_replace_at,
    canonical_json_bytes,
    create_staging_directory_at,
    discover_git_metadata_dirs,
    entry_exists_at,
    open_or_create_real_directory,
    reject_directory_descriptor_in_git_metadata,
    reject_git_metadata_destination,
    write_exclusive_file_at,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_FIGURE_PATHS,
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_PACKAGE_BYTE_CAP,
    M5_RELEASE_PACKAGE_PATHS,
    M5_RELEASE_SIDECAR_INDEXED_PATHS,
    M5_RELEASE_SIDECAR_ROLE_BY_PATH,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
    M5_REVIEW_CANDIDATE_PATHS,
    M5_REVIEW_CANDIDATE_ROLE_BY_PATH,
    ReplayReleaseSidecarFileEntryV1,
    ReplayReleaseSidecarIndexV1,
    ReplayResultsReviewAttestationV1,
    ReplayResultsReviewDecisionV1,
    ReplayReviewCandidateFileEntryV1,
    ReplayReviewCandidateIndexV1,
    compute_replay_release_package_sha256,
    compute_replay_release_sidecar_set_sha256,
    compute_replay_review_candidate_sha256,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
)

_READ_CHUNK_BYTES = 1024 * 1024
_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_NDJSON_RECORDS = 500_000
_METHODOLOGY_PATHS = frozenset(
    {
        "evidence/release-pipeline-plan.md",
        "evidence/release-pipeline-plan-review.md",
        "evidence/resource-scope-amendment.md",
        "evidence/implementation-review.md",
    }
)
_PRIVATE_PATH_PATTERN = re.compile(
    rb"(?:file:(?://)?|/(?:Users|home|private|tmp|Volumes)/|"
    rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/])",
    re.IGNORECASE,
)
_PRIVATE_CACHE_PATTERN = re.compile(
    rb"(?:\.cache|Library/(?:Caches|Application Support)|var/folders)",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    rb"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    rb"(?:[\"'\s]*[:=][\"'\s]*|_)[A-Za-z0-9+/=_-]{8,}",
    re.IGNORECASE,
)
_DATASET_PAYLOAD_PATTERN = re.compile(
    rb"(?:samples|sweeps|maps|v1\.0-mini)/|"
    rb"[A-Za-z0-9_.+-]+\.(?:jpg|jpeg|png|pcd|las|laz|bin|tar|tgz|zip)",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(rb"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", re.IGNORECASE)
_ALLOWED_METHODOLOGY_LITERALS = (
    b"/usr/bin/time",
    b"/dev/fd/<fd>",
)
_SAFE_METHODOLOGY_PLACEHOLDER_PATTERN = re.compile(rb"<[a-z][a-z0-9-]*(?: [a-z][a-z0-9/-]*){0,4}>")
_SAFE_PLACEHOLDER_RELATIVE_PATH_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9/])(?:[a-z0-9._-]+/)*[a-z0-9._-]*"
    rb"<[a-z][a-z0-9-]*(?: [a-z][a-z0-9/-]*){0,4}>"
    rb"[a-z0-9._/-]*"
)
_FROZEN_METHODOLOGY_SHA256 = {
    "evidence/release-pipeline-plan.md": (
        "0a645323d346668707442eb2e9cd76bac221f8a0c9ff48c4baad5bf078ce946d"
    ),
    "evidence/release-pipeline-plan-review.md": (
        "3a881c41de758d98a65e96a627499ad5cf1b4c4c5bf9a1d7fcece507d3e4c6af"
    ),
    "evidence/resource-scope-amendment.md": (
        "f7eb19e03661bec1663b1a2f6ce953e465e8a3a76d63277ece5bee4e59708aff"
    ),
}


class ReplayReleaseError(ValueError):
    """An M5 release candidate or package failed closed."""


@dataclass(frozen=True, slots=True)
class ReplayReviewCandidateContent:
    """The exact 33 already-generated members and their source bindings."""

    scientific_git_revision: str
    lockfile_sha256: str
    package_version: str
    run_id: str
    replay_identity_set_sha256: str
    primary_local_artifact_sha256: str
    repeat_local_artifact_sha256: str
    primary_local_run_sha256: str
    repeat_local_run_sha256: str
    files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class LoadedReplayReviewCandidate:
    """A strictly reloaded immutable M5 review candidate."""

    path: Path
    index: ReplayReviewCandidateIndexV1
    index_bytes: bytes
    files: Mapping[str, bytes]
    candidate_sha256: str
    candidate_index_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayReleasePackageContent:
    """All final machine files and reviewed sidecars before outer indexing."""

    reviewed_candidate_sha256: str
    results_review_attestation_sha256: str
    machine_artifact_sha256: str
    machine_run_sha256: str
    scientific_git_revision: str
    machine_files: Mapping[str, bytes]
    sidecar_files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class LoadedReplayReleasePackage:
    """A strictly reloaded complete M5 release package."""

    path: Path
    index: ReplayReleaseSidecarIndexV1
    index_bytes: bytes
    files: Mapping[str, bytes]
    release_package_sha256: str


def _raise(message: str) -> Never:
    raise ReplayReleaseError(message) from None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_count(path: str, value: bytes) -> int | None:
    if not path.endswith(".ndjson"):
        return None
    lines = value.splitlines(keepends=True)
    if not lines or len(lines) > _MAX_NDJSON_RECORDS:
        _raise("M5 release NDJSON record count is invalid")
    if any(not line.endswith(b"\n") or len(line) > _MAX_RECORD_BYTES for line in lines):
        _raise("M5 release NDJSON framing is invalid")
    return len(lines)


def _strict_json(path: str, value: bytes) -> dict[str, Any]:
    if not value.endswith(b"\n") or b"\r" in value:
        _raise("M5 release JSON does not use canonical LF framing")
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _raise("M5 release JSON is invalid")
    if not isinstance(parsed, dict):
        _raise("M5 release JSON is not a canonical object")
    mapping = cast(dict[str, Any], parsed)
    if canonical_json_bytes(mapping) != value:
        _raise("M5 release JSON is not canonical")
    return mapping


def _validate_member_encoding(path: str, value: bytes) -> None:
    if not value or len(value) > _MAX_MEMBER_BYTES or b"\x00" in value:
        _raise("M5 release member is empty, oversized, or contains NUL")
    if path in {"machine/intent.json", "artifact/intent.json"}:
        if _sha256(value) != M5_REPLAY_INTENT_BYTE_SHA256:
            _raise("M5 release intent bytes differ from the frozen authority")
        return
    if path.endswith(".json"):
        _strict_json(path, value)
    elif path.endswith(".ndjson"):
        for line in value.splitlines(keepends=True):
            _strict_json(path, line)
    else:
        try:
            value.decode("utf-8")
        except UnicodeDecodeError:
            _raise("M5 release text member is not UTF-8")
        if b"\r" in value or not value.endswith(b"\n"):
            _raise("M5 release text member must use LF and one final newline")


def _strip_allowed_methodology_literals(value: bytes) -> bytes:
    output = value
    for literal in _ALLOWED_METHODOLOGY_LITERALS:
        output = output.replace(literal, b"")
    output = _SAFE_PLACEHOLDER_RELATIVE_PATH_PATTERN.sub(b"", output)
    output = _SAFE_METHODOLOGY_PLACEHOLDER_PATTERN.sub(b"", output)
    return output


def _privacy_scan(path: str, value: bytes, *, methodology: bool) -> None:
    frozen_digest = _FROZEN_METHODOLOGY_SHA256.get(path)
    if methodology and frozen_digest is not None and _sha256(value) == frozen_digest:
        return
    scanned = _strip_allowed_methodology_literals(value) if methodology else value
    if (
        _PRIVATE_PATH_PATTERN.search(scanned)
        or _PRIVATE_CACHE_PATTERN.search(scanned)
        or _SECRET_PATTERN.search(scanned)
        or _DATASET_PAYLOAD_PATTERN.search(scanned)
        or _TOKEN_PATTERN.search(scanned)
        or b"reports/generated/" in scanned
        or b"interview/" in scanned.lower()
    ):
        _raise(f"M5 release privacy scan failed for {path}")


def validate_review_candidate_members(files: Mapping[str, bytes]) -> None:
    """Validate exact candidate member bytes before semantic index construction."""

    if tuple(files) != M5_REVIEW_CANDIDATE_INDEXED_PATHS:
        _raise("M5 review candidate member order or allowlist is invalid")
    if sum(len(value) for value in files.values()) >= M5_RELEASE_PACKAGE_BYTE_CAP:
        _raise("M5 review candidate exceeds the frozen byte cap")
    for path, value in files.items():
        _validate_member_encoding(path, value)
        _privacy_scan(path, value, methodology=path in _METHODOLOGY_PATHS)
    for path in (
        "presentation/README.md",
        "presentation/claim-evidence.md",
        "presentation/verification.md",
    ):
        value = files[path]
        for placeholder in M5_PRESENTATION_PLACEHOLDERS:
            if value.count(placeholder.encode("ascii")) != 1:
                _raise("M5 presentation placeholder count is invalid")


def build_review_candidate_files(
    content: ReplayReviewCandidateContent,
) -> dict[str, bytes]:
    """Build and self-validate the exact 34 candidate bytes in canonical order."""

    if tuple(content.files) != M5_REVIEW_CANDIDATE_INDEXED_PATHS:
        _raise("M5 review candidate member order or allowlist is invalid")
    files = {path: content.files[path] for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS}
    validate_review_candidate_members(files)
    entries = tuple(
        ReplayReviewCandidateFileEntryV1(
            path=path,
            role=M5_REVIEW_CANDIDATE_ROLE_BY_PATH[path],
            byte_length=len(files[path]),
            sha256=_sha256(files[path]),
            record_count=_record_count(path, files[path]),
        )
        for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS
    )
    core: dict[str, Any] = {
        "schema": "ffb.m5-release-review-candidate-index/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "scientific_git_revision": content.scientific_git_revision,
        "lockfile_sha256": content.lockfile_sha256,
        "package_version": content.package_version,
        "run_id": content.run_id,
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": content.replay_identity_set_sha256,
        "primary_local_artifact_sha256": content.primary_local_artifact_sha256,
        "repeat_local_artifact_sha256": content.repeat_local_artifact_sha256,
        "primary_local_run_sha256": content.primary_local_run_sha256,
        "repeat_local_run_sha256": content.repeat_local_run_sha256,
        "results_review_status": "pending",
        "files": tuple(entry.model_dump(mode="json", by_alias=True) for entry in entries),
    }
    index = ReplayReviewCandidateIndexV1.model_validate(
        {**core, "candidate_sha256": compute_replay_review_candidate_sha256(core)}
    )
    return {
        "candidate-index.json": canonical_json_bytes(index),
        **files,
    }


def _read_stable_file(descriptor: int, *, path: str, cap: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= cap:
        _raise(f"M5 release tree member is unsafe: {path}")
    output = bytearray()
    while len(output) <= cap:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, cap + 1 - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    after = os.fstat(descriptor)
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(output) != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in fields
    ):
        _raise(f"M5 release tree member changed while reading: {path}")
    return bytes(output)


def _expected_children(paths: Sequence[str]) -> dict[tuple[str, ...], set[str]]:
    result: dict[tuple[str, ...], set[str]] = {(): set()}
    for path in paths:
        parts = PurePosixPath(path).parts
        for index, part in enumerate(parts):
            parent = parts[:index]
            result.setdefault(parent, set()).add(part)
            if index < len(parts) - 1:
                result.setdefault(parts[: index + 1], set())
    return result


def _read_exact_tree(root: Path, expected_paths: Sequence[str]) -> dict[str, bytes]:
    absolute = absolute_artifact_path(root)
    expected = _expected_children(expected_paths)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(absolute, flags)
    except OSError:
        _raise("M5 release tree root cannot be opened safely")
    root_before = os.fstat(root_fd)
    files: dict[str, bytes] = {}

    def visit(directory_fd: int, prefix: tuple[str, ...]) -> None:
        directory_before = os.fstat(directory_fd)
        try:
            actual = set(os.listdir(directory_fd))
        except OSError:
            _raise("M5 release tree cannot be enumerated safely")
        if actual != expected[prefix]:
            _raise("M5 release tree allowlist mismatch")
        for name in sorted(actual, key=lambda value: value.encode("utf-8")):
            child = (*prefix, name)
            relative = "/".join(child)
            if child in expected:
                try:
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError:
                    _raise("M5 release tree directory is redirected")
                try:
                    visit(child_fd, child)
                    os.fsync(child_fd)
                finally:
                    os.close(child_fd)
            else:
                try:
                    file_fd = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                except OSError:
                    _raise("M5 release tree member is redirected")
                try:
                    files[relative] = _read_stable_file(
                        file_fd,
                        path=relative,
                        cap=_MAX_MEMBER_BYTES,
                    )
                finally:
                    os.close(file_fd)
        directory_after = os.fstat(directory_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(directory_before, field) != getattr(directory_after, field)
            for field in stable_fields
        ):
            _raise("M5 release tree directory changed while reading")

    try:
        visit(root_fd, ())
        root_after = os.fstat(root_fd)
        reopened = os.stat(absolute, follow_symlinks=False)
        identity_fields = ("st_dev", "st_ino", "st_mode", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(root_before, field) != getattr(root_after, field) for field in identity_fields
        ) or any(
            getattr(root_before, field) != getattr(reopened, field) for field in identity_fields
        ):
            _raise("M5 release tree changed while reading")
    finally:
        os.close(root_fd)
    return {path: files[path] for path in expected_paths}


def load_review_candidate(path: Path) -> LoadedReplayReviewCandidate:
    """Strictly load, hash, and cross-check one complete review candidate."""

    files = _read_exact_tree(path, M5_REVIEW_CANDIDATE_PATHS)
    try:
        index = ReplayReviewCandidateIndexV1.model_validate_json(files["candidate-index.json"])
    except ValidationError as error:
        raise ReplayReleaseError("M5 candidate index contract validation failed") from error
    if canonical_json_bytes(index) != files["candidate-index.json"]:
        _raise("M5 candidate index is not canonical")
    indexed = {name: files[name] for name in M5_REVIEW_CANDIDATE_INDEXED_PATHS}
    validate_review_candidate_members(indexed)
    for entry in index.files:
        value = indexed[entry.path]
        if (
            entry.byte_length != len(value)
            or entry.sha256 != _sha256(value)
            or entry.record_count != _record_count(entry.path, value)
        ):
            _raise("M5 candidate indexed member changed")
    return LoadedReplayReviewCandidate(
        path=absolute_artifact_path(path),
        index=index,
        index_bytes=files["candidate-index.json"],
        files=indexed,
        candidate_sha256=index.candidate_sha256,
        candidate_index_sha256=_sha256(files["candidate-index.json"]),
    )


def validate_review_candidate(path: Path) -> str:
    """Validate a review candidate and return its semantic digest."""

    return load_review_candidate(path).candidate_sha256


def _nested_directories(paths: Sequence[str]) -> tuple[str, ...]:
    values = {
        PurePosixPath(path).parent.as_posix()
        for path in paths
        if PurePosixPath(path).parent.as_posix() != "."
    }
    return tuple(sorted(values, key=lambda item: (item.count("/"), item.encode("utf-8"))))


def _open_directory_at(root_fd: int, relative: str) -> int:
    descriptor = os.dup(root_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in PurePosixPath(relative).parts:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _make_directories(root_fd: int, paths: Sequence[str]) -> None:
    for relative in _nested_directories(paths):
        parent = PurePosixPath(relative).parent.as_posix()
        parent_fd = os.dup(root_fd) if parent == "." else _open_directory_at(root_fd, parent)
        try:
            with suppress(FileExistsError):
                os.mkdir(PurePosixPath(relative).name, mode=0o700, dir_fd=parent_fd)
            child_fd = os.open(
                PurePosixPath(relative).name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            try:
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
        finally:
            os.close(parent_fd)


def _write_nested(root_fd: int, path: str, value: bytes) -> None:
    pure = PurePosixPath(path)
    parent = pure.parent.as_posix()
    parent_fd = os.dup(root_fd) if parent == "." else _open_directory_at(root_fd, parent)
    try:
        write_exclusive_file_at(parent_fd, pure.name, value)
    finally:
        os.close(parent_fd)


def _verify_nested(root_fd: int, path: str, expected: bytes) -> None:
    pure = PurePosixPath(path)
    parent = pure.parent.as_posix()
    parent_fd = os.dup(root_fd) if parent == "." else _open_directory_at(root_fd, parent)
    try:
        file_fd = os.open(
            pure.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            observed = _read_stable_file(file_fd, path=path, cap=len(expected))
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)
    if observed != expected:
        _raise(f"M5 staged release member changed after write: {path}")


def _cleanup_nested(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    paths: Sequence[str],
) -> None:
    for path in reversed(tuple(paths)):
        pure = PurePosixPath(path)
        parent = pure.parent.as_posix()
        with suppress(OSError):
            parent_descriptor = (
                os.dup(staging_fd) if parent == "." else _open_directory_at(staging_fd, parent)
            )
            try:
                os.unlink(pure.name, dir_fd=parent_descriptor)
            finally:
                os.close(parent_descriptor)
    for directory in reversed(_nested_directories(paths)):
        pure = PurePosixPath(directory)
        parent = pure.parent.as_posix()
        with suppress(OSError):
            parent_descriptor = (
                os.dup(staging_fd) if parent == "." else _open_directory_at(staging_fd, parent)
            )
            try:
                os.rmdir(pure.name, dir_fd=parent_descriptor)
            finally:
                os.close(parent_descriptor)
    with suppress(OSError):
        os.rmdir(staging_name, dir_fd=parent_fd)


def _publish_exact_tree[LoadedT](
    *,
    files: Mapping[str, bytes],
    paths: Sequence[str],
    destination: Path,
    loader: Callable[[Path], LoadedT],
    write_groups: Sequence[Sequence[str]],
) -> LoadedT:
    flattened = tuple(path for group in write_groups for path in group)
    if len(flattened) != len(paths) or set(flattened) != set(paths):
        _raise("M5 release staging groups do not match the exact path allowlist")
    target = absolute_artifact_path(destination)
    parent = target.parent
    git_metadata = discover_git_metadata_dirs(Path.cwd())
    reject_git_metadata_destination(target, git_metadata)
    parent_fd = open_or_create_real_directory(parent)
    published = False
    try:
        assert_directory_descriptor_matches_path(parent_fd, parent, label="destination parent")
        reject_directory_descriptor_in_git_metadata(parent_fd, git_metadata)
        if entry_exists_at(parent_fd, target.name):
            raise FileExistsError("M5 release destination already exists")
        staging_name, staging_fd = create_staging_directory_at(parent_fd)
        staging = parent / staging_name
        try:
            assert_directory_descriptor_matches_path(
                staging_fd,
                staging,
                label="M5 release staging directory",
            )
            _make_directories(staging_fd, paths)
            written: list[str] = []
            for group in write_groups:
                for path in group:
                    _write_nested(staging_fd, path, files[path])
                    written.append(path)
                os.fsync(staging_fd)
                for path in written:
                    _verify_nested(staging_fd, path, files[path])
            loader(staging)
            assert_directory_descriptor_matches_path(
                staging_fd,
                staging,
                label="M5 release staging directory",
            )
            assert_directory_descriptor_matches_path(parent_fd, parent, label="destination parent")
            reject_directory_descriptor_in_git_metadata(parent_fd, git_metadata)
            if entry_exists_at(parent_fd, target.name):
                raise FileExistsError("M5 release destination already exists")
            atomic_rename_directory_no_replace_at(
                parent_fd,
                staging_name,
                parent_fd,
                target.name,
            )
            published = True
            assert_directory_descriptor_matches_path(
                staging_fd,
                target,
                label="M5 published release directory",
            )
            os.fsync(parent_fd)
            assert_directory_descriptor_matches_path(parent_fd, parent, label="destination parent")
            loaded = loader(target)
            assert_directory_descriptor_matches_path(
                staging_fd,
                target,
                label="M5 published release directory",
            )
            return loaded
        except BaseException:
            if not published:
                _cleanup_nested(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=staging_name,
                    paths=paths,
                )
            raise
        finally:
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def publish_review_candidate(
    content: ReplayReviewCandidateContent,
    destination: Path,
) -> LoadedReplayReviewCandidate:
    """Atomically publish one no-overwrite M5 review candidate."""

    files = build_review_candidate_files(content)
    return _publish_exact_tree(
        files=files,
        paths=M5_REVIEW_CANDIDATE_PATHS,
        destination=destination,
        loader=load_review_candidate,
        write_groups=(M5_REVIEW_CANDIDATE_INDEXED_PATHS, ("candidate-index.json",)),
    )


def _set_digest(paths: Sequence[str], files: Mapping[str, bytes], *, schema: str) -> str:
    return sha256_digest(
        {
            "schema": schema,
            "files": [
                {"path": path, "byte_length": len(files[path]), "sha256": _sha256(files[path])}
                for path in paths
            ],
        }
    )


def attest_results_review(
    candidate: LoadedReplayReviewCandidate,
    *,
    review_report: bytes,
    decision: ReplayResultsReviewDecisionV1,
) -> ReplayResultsReviewAttestationV1:
    """Canonicalize—but never choose—one reviewer-authored candidate decision."""

    if not review_report or b"\x00" in review_report or not review_report.endswith(b"\n"):
        _raise("M5 results review report is not bounded LF-delimited text")
    _privacy_scan("results-review.md", review_report, methodology=False)
    if not (
        decision.negative_and_undefined_results_reviewed_and_retained
        and decision.limitations_reviewed_and_retained
    ):
        _raise("M5 results review decision does not retain required outcomes and limitations")
    findings = decision.findings
    unresolved = tuple(row.finding_id for row in findings if row.status == "unresolved")
    return ReplayResultsReviewAttestationV1(
        schema="ffb.m5-results-review-attestation/v1",
        release_id="m5-nuscenes-replay-v0.1.0",
        scientific_git_revision=candidate.index.scientific_git_revision,
        candidate_sha256=candidate.candidate_sha256,
        candidate_index_sha256=candidate.candidate_index_sha256,
        scientific_member_set_sha256=_set_digest(
            M5_REVIEW_CANDIDATE_INDEXED_PATHS[:10],
            candidate.files,
            schema="ffb.m5-reviewed-scientific-member-set/v1",
        ),
        claim_projection_sha256=_sha256(
            candidate.files["presentation/public-claim-projections.json"]
        ),
        figure_spec_set_sha256=_set_digest(
            M5_FIGURE_PATHS[::2],
            candidate.files,
            schema="ffb.m5-reviewed-figure-spec-set/v1",
        ),
        rendered_figure_set_sha256=_set_digest(
            M5_FIGURE_PATHS[1::2],
            candidate.files,
            schema="ffb.m5-reviewed-rendered-figure-set/v1",
        ),
        presentation_template_set_sha256=_set_digest(
            (
                "presentation/README.md",
                "presentation/claim-evidence.md",
                "presentation/verification.md",
                "presentation/release-summary.json",
            ),
            candidate.files,
            schema="ffb.m5-reviewed-presentation-template-set/v1",
        ),
        review_report_sha256=_sha256(review_report),
        reviewer_identity=decision.reviewer_identity,
        reviewer_identity_scope="operator-recorded-not-cryptographically-authenticated",
        findings=findings,
        p0_count=sum(row.severity == "p0" for row in findings),
        p1_count=sum(row.severity == "p1" for row in findings),
        p2_count=sum(row.severity == "p2" for row in findings),
        unresolved_finding_ids=unresolved,
        negative_and_undefined_results_reviewed_and_retained=True,
        limitations_reviewed_and_retained=True,
        disposition=decision.disposition,
    )


def build_release_package_files(content: ReplayReleasePackageContent) -> dict[str, bytes]:
    """Index and validate the exact 41 final package members."""

    machine_paths = tuple(path.removeprefix("artifact/") for path in M5_RELEASE_PACKAGE_PATHS[:14])
    if tuple(content.machine_files) != machine_paths:
        _raise("M5 machine artifact member allowlist or order is invalid")
    if tuple(content.sidecar_files) != M5_RELEASE_SIDECAR_INDEXED_PATHS:
        _raise("M5 release sidecar allowlist or order is invalid")
    files: dict[str, bytes] = {
        **{f"artifact/{path}": content.machine_files[path] for path in machine_paths},
        **{path: content.sidecar_files[path] for path in M5_RELEASE_SIDECAR_INDEXED_PATHS},
    }
    if sum(len(value) for value in files.values()) >= M5_RELEASE_PACKAGE_BYTE_CAP:
        _raise("M5 release package exceeds the frozen byte cap")
    for path, value in files.items():
        _validate_member_encoding(path, value)
        methodology = path in _METHODOLOGY_PATHS
        _privacy_scan(path, value, methodology=methodology)
    entries = tuple(
        ReplayReleaseSidecarFileEntryV1(
            path=path,
            role=M5_RELEASE_SIDECAR_ROLE_BY_PATH[path],
            byte_length=len(files[path]),
            sha256=_sha256(files[path]),
            record_count=_record_count(path, files[path]),
        )
        for path in M5_RELEASE_SIDECAR_INDEXED_PATHS
    )
    machine_bytes = sum(len(value) for value in content.machine_files.values())
    core: dict[str, Any] = {
        "schema": "ffb.m5-release-sidecar-index/v1",
        "release_id": "m5-nuscenes-replay-v0.1.0",
        "reviewed_candidate_sha256": content.reviewed_candidate_sha256,
        "results_review_attestation_sha256": content.results_review_attestation_sha256,
        "machine_artifact_sha256": content.machine_artifact_sha256,
        "machine_run_sha256": content.machine_run_sha256,
        "scientific_git_revision": content.scientific_git_revision,
        "files": tuple(entry.model_dump(mode="json", by_alias=True) for entry in entries),
        "machine_artifact_byte_length": machine_bytes,
        "indexed_sidecar_payload_byte_length": sum(entry.byte_length for entry in entries),
    }
    sidecar_digest = compute_replay_release_sidecar_set_sha256(core)
    index = ReplayReleaseSidecarIndexV1.model_validate(
        {
            **core,
            "sidecar_set_sha256": sidecar_digest,
            "release_package_sha256": compute_replay_release_package_sha256(
                content.machine_artifact_sha256,
                sidecar_digest,
            ),
        }
    )
    files["release-sidecar-index.json"] = canonical_json_bytes(index)
    return {path: files[path] for path in M5_RELEASE_PACKAGE_PATHS}


def load_release_package(path: Path) -> LoadedReplayReleasePackage:
    """Strictly load the complete outer package and reconstruct both digests."""

    files = _read_exact_tree(path, M5_RELEASE_PACKAGE_PATHS)
    try:
        index = ReplayReleaseSidecarIndexV1.model_validate_json(files["release-sidecar-index.json"])
    except ValidationError as error:
        raise ReplayReleaseError("M5 release sidecar index contract validation failed") from error
    if canonical_json_bytes(index) != files["release-sidecar-index.json"]:
        _raise("M5 release sidecar index is not canonical")
    for member_path, value in files.items():
        _validate_member_encoding(member_path, value)
        _privacy_scan(member_path, value, methodology=member_path in _METHODOLOGY_PATHS)
    for entry in index.files:
        value = files[entry.path]
        if (
            entry.byte_length != len(value)
            or entry.sha256 != _sha256(value)
            or entry.record_count != _record_count(entry.path, value)
        ):
            _raise("M5 release indexed sidecar changed")
    if sum(len(value) for value in files.values()) >= M5_RELEASE_PACKAGE_BYTE_CAP:
        _raise("M5 complete release package exceeds the frozen byte cap")
    return LoadedReplayReleasePackage(
        path=absolute_artifact_path(path),
        index=index,
        index_bytes=files["release-sidecar-index.json"],
        files=files,
        release_package_sha256=index.release_package_sha256,
    )


def validate_release_package(path: Path) -> str:
    """Validate a package and return its complete package digest."""

    return load_release_package(path).release_package_sha256


def publish_release_package(
    content: ReplayReleasePackageContent,
    destination: Path,
) -> LoadedReplayReleasePackage:
    """Atomically publish one no-overwrite complete M5 package."""

    files = build_release_package_files(content)
    return _publish_exact_tree(
        files=files,
        paths=M5_RELEASE_PACKAGE_PATHS,
        destination=destination,
        loader=load_release_package,
        write_groups=(
            M5_RELEASE_PACKAGE_PATHS[:13],
            ("artifact/_SUCCESS",),
            M5_RELEASE_SIDECAR_INDEXED_PATHS,
            ("release-sidecar-index.json",),
        ),
    )


def sync_reviewed_evidence(
    package: LoadedReplayReleasePackage,
    *,
    report_output: Path,
    attestation_output: Path,
    source_root: Path,
) -> None:
    """Transactionally copy exact review bytes to the two fixed public paths."""

    from fusion_fault_bench.replay_review_sync import (
        sync_reviewed_evidence_transaction,
    )

    sync_reviewed_evidence_transaction(
        package,
        report_output=report_output,
        attestation_output=attestation_output,
        source_root=source_root,
    )


__all__ = [
    "LoadedReplayReleasePackage",
    "LoadedReplayReviewCandidate",
    "ReplayReleaseError",
    "ReplayReleasePackageContent",
    "ReplayReviewCandidateContent",
    "attest_results_review",
    "build_release_package_files",
    "build_review_candidate_files",
    "load_release_package",
    "load_review_candidate",
    "publish_release_package",
    "publish_review_candidate",
    "sync_reviewed_evidence",
    "validate_release_package",
    "validate_review_candidate",
    "validate_review_candidate_members",
]
