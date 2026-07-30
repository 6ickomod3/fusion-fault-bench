"""Git and review authority for clean or exact pending M5 publication states."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Never, Protocol

from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_RELEASE_DESTINATION_PATH,
    M5_RELEASE_PACKAGE_PATHS,
    ReplayImplementationReviewAttestationV1,
)
from fusion_fault_bench.replay_publication import M5_PUBLICATION_DOCUMENT_PATHS
from fusion_fault_bench.replay_release_authority import ImplementationSnapshot
from fusion_fault_bench.replay_release_validation import (
    load_implementation_review_attestation,
)

_RESULTS_REVIEW_PUBLIC_PATHS = (
    "docs/reviews/m5-results-review.md",
    "docs/reviews/m5-results-review-attestation.json",
)
_MAX_PENDING_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_PENDING_TOTAL_BYTES = 512 * 1024 * 1024
_SAFE_PENDING_FILE_MODES = frozenset({0o600, 0o644})
_EXPECTED_MODIFIED_PATHS = frozenset(path.encode("utf-8") for path in M5_PUBLICATION_DOCUMENT_PATHS)
_EXPECTED_ADDED_PATHS = frozenset(
    {
        *(f"{M5_RELEASE_DESTINATION_PATH}/{path}".encode() for path in M5_RELEASE_PACKAGE_PATHS),
        *(path.encode("utf-8") for path in _RESULTS_REVIEW_PUBLIC_PATHS),
    }
)


class ReplayPublicationAuthorityError(ValueError):
    """Repository publication state is neither clean nor the exact pending projection."""


class _Package(Protocol):
    @property
    def files(self) -> Mapping[str, bytes]: ...


class ValidatedPublicationAuthority(Protocol):
    @property
    def package(self) -> _Package: ...

    @property
    def implementation_review_attestation(self) -> ReplayImplementationReviewAttestationV1: ...


@dataclass(frozen=True, slots=True)
class PendingPublicationState:
    """Exact Git state allowed between package closeout and staging."""

    source_root: Path
    scientific_git_revision: str
    state_digest_material: tuple[bytes, ...]


def _fail(message: str) -> Never:
    raise ReplayPublicationAuthorityError(message) from None


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", os.fspath(source_root), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _fail("publication Git authority query failed")
    return result.stdout


def _git_state(source_root: Path) -> tuple[bytes, ...]:
    return (
        _git_bytes(source_root, "rev-parse", "HEAD"),
        _git_bytes(source_root, "ls-files", "--unmerged", "-z"),
        _git_bytes(source_root, "ls-files", "-v", "-z"),
        _git_bytes(source_root, "diff", "--name-only", "-z", "--"),
        _git_bytes(source_root, "diff", "--cached", "--name-only", "-z", "--"),
        _git_bytes(
            source_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ),
    )


def _pending_content_digest(source_root: Path, paths: frozenset[bytes]) -> bytes:
    """Digest every allowlisted pending member after a stable, no-link read."""

    digest = hashlib.sha256(b"fusion-fault-bench/m5-pending-publication/v1\x00")
    total_bytes = 0
    for encoded_relative in sorted(paths):
        try:
            relative = encoded_relative.decode("utf-8")
        except UnicodeDecodeError:
            _fail("pending publication path is not UTF-8")
        target = source_root.joinpath(*relative.split("/"))
        descriptor: int | None = None
        try:
            before_path = os.lstat(target)
            descriptor = os.open(
                target,
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) not in _SAFE_PENDING_FILE_MODES
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > _MAX_PENDING_MEMBER_BYTES
                or before_path.st_dev != before.st_dev
                or before_path.st_ino != before.st_ino
            ):
                _fail("pending publication member is unavailable or unsafe")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    _fail("pending publication member changed while being read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                _fail("pending publication member changed while being read")
            value = b"".join(chunks)
            after = os.fstat(descriptor)
            reopened = os.lstat(target)
        except OSError:
            _fail("pending publication member is unavailable or unsafe")
        finally:
            if descriptor is not None:
                os.close(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or any(
            getattr(before, field) != getattr(reopened, field) for field in stable_fields
        ):
            _fail("pending publication member changed while being read")
        total_bytes += len(value)
        if total_bytes > _MAX_PENDING_TOTAL_BYTES:
            _fail("pending publication content exceeds its total byte cap")
        digest.update(len(encoded_relative).to_bytes(8, "big"))
        digest.update(encoded_relative)
        digest.update(stat.S_IMODE(before.st_mode).to_bytes(4, "big"))
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(hashlib.sha256(value).digest())
    return digest.digest()


def authenticate_pending_publication(
    source_root: Path,
    *,
    scientific_git_revision: str,
) -> PendingPublicationState:
    """Authenticate the exact unstaged package, review pair, and nine projections."""

    root = Path(os.path.abspath(os.fspath(source_root)))
    try:
        if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
            _fail("pending publication source root is unavailable or redirected")
    except OSError:
        _fail("pending publication source root is unavailable or redirected")
    if len(scientific_git_revision) != 40 or any(
        character not in "0123456789abcdef" for character in scientific_git_revision
    ):
        _fail("pending publication has an invalid scientific revision")
    state = _git_state(root)
    revision, unmerged, flags, worktree_diff, staged_diff, untracked = state
    if revision.strip() != scientific_git_revision.encode("ascii"):
        _fail("pending publication is not based on the scientific revision")
    if unmerged or staged_diff:
        _fail("pending publication index differs from HEAD")
    if any(record and not record.startswith(b"H ") for record in flags.split(b"\x00")):
        _fail("pending publication uses unsupported Git index flags")
    observed_modified = frozenset(record for record in worktree_diff.split(b"\x00") if record)
    if observed_modified != _EXPECTED_MODIFIED_PATHS:
        _fail("pending publication tracked changes differ from the nine closeout documents")
    observed_untracked = frozenset(record for record in untracked.split(b"\x00") if record)
    if observed_untracked != _EXPECTED_ADDED_PATHS:
        _fail("pending publication untracked paths differ from the exact package and review pair")
    content_digest = _pending_content_digest(
        root,
        _EXPECTED_MODIFIED_PATHS | _EXPECTED_ADDED_PATHS,
    )
    return PendingPublicationState(
        source_root=root,
        scientific_git_revision=scientific_git_revision,
        state_digest_material=(*state, content_digest),
    )


def verify_pending_publication_unchanged(token: PendingPublicationState) -> None:
    """Require the exact same pending Git state after validation."""

    observed = authenticate_pending_publication(
        token.source_root,
        scientific_git_revision=token.scientific_git_revision,
    )
    if observed != token:
        _fail("pending publication changed during validation")


def _tree_entries(source_root: Path, revision: str) -> dict[bytes, tuple[bytes, bytes, bytes]]:
    raw = _git_bytes(source_root, "ls-tree", "-rz", "--full-tree", revision)
    entries: dict[bytes, tuple[bytes, bytes, bytes]] = {}
    for record in (item for item in raw.split(b"\x00") if item):
        try:
            metadata, relative = record.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
        except ValueError:
            _fail("publication Git tree has an invalid entry")
        if relative in entries:
            _fail("publication Git tree repeats a path")
        entries[relative] = (mode, kind, object_id)
    return entries


def _clean_delta(source_root: Path, revision: str) -> dict[bytes, bytes]:
    raw = _git_bytes(
        source_root,
        "diff",
        "--name-status",
        "-z",
        "--no-renames",
        revision,
        "HEAD",
        "--",
    )
    fields = tuple(field for field in raw.split(b"\x00") if field)
    if len(fields) % 2:
        _fail("clean publication Git delta is malformed")
    observed: dict[bytes, bytes] = {}
    for index in range(0, len(fields), 2):
        status_value, relative = fields[index : index + 2]
        if status_value not in {b"A", b"M"} or relative in observed:
            _fail("clean publication Git delta has an unsupported status")
        observed[relative] = status_value
    return observed


def require_scientific_revision_ancestor(source_root: Path, revision: str) -> None:
    """Require the exact clean release-tree delta from the scientific revision."""

    root = Path(os.path.abspath(os.fspath(source_root)))
    try:
        if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
            _fail("clean publication source root is unavailable or redirected")
    except OSError:
        _fail("clean publication source root is unavailable or redirected")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        _fail("clean publication has an invalid scientific revision")

    state = _git_state(root)
    head, unmerged, flags, worktree_diff, staged_diff, untracked = state
    if len(head.strip()) != 40 or any(
        character not in b"0123456789abcdef" for character in head.strip()
    ):
        _fail("clean publication HEAD is invalid")
    if unmerged or worktree_diff or staged_diff or untracked:
        _fail("clean publication repository status is not empty")
    if any(record and not record.startswith(b"H ") for record in flags.split(b"\x00")):
        _fail("clean publication uses unsupported Git index flags")

    result = subprocess.run(
        ("git", "-C", os.fspath(root), "merge-base", "--is-ancestor", revision, "HEAD"),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _fail("packaged scientific revision is not an ancestor of current HEAD")

    expected_delta = {
        **{relative: b"M" for relative in _EXPECTED_MODIFIED_PATHS},
        **{relative: b"A" for relative in _EXPECTED_ADDED_PATHS},
    }
    if _clean_delta(root, revision) != expected_delta:
        _fail("clean publication differs from the exact reviewed release-tree delta")

    scientific_tree = _tree_entries(root, revision)
    release_tree = _tree_entries(root, "HEAD")
    expected_release_tree = dict(scientific_tree)
    for relative in _EXPECTED_MODIFIED_PATHS:
        before = scientific_tree.get(relative)
        after = release_tree.get(relative)
        if (
            before is None
            or after is None
            or before[:2] != (b"100644", b"blob")
            or after[:2] != (b"100644", b"blob")
            or before[2] == after[2]
        ):
            _fail("clean publication has an invalid modified document tree entry")
        expected_release_tree[relative] = after
    for relative in _EXPECTED_ADDED_PATHS:
        after = release_tree.get(relative)
        if relative in scientific_tree or after is None or after[:2] != (b"100644", b"blob"):
            _fail("clean publication has an invalid added release tree entry")
        expected_release_tree[relative] = after
    if release_tree != expected_release_tree:
        _fail("clean publication final Git tree differs beyond the reviewed release delta")


def validate_current_implementation_review(
    validated: ValidatedPublicationAuthority,
    *,
    snapshot: ImplementationSnapshot,
    tracked_report: bytes,
    tracked_attestation: bytes,
) -> None:
    """Bind package review bytes and attestation to the current implementation snapshot."""

    package_report = validated.package.files["evidence/implementation-review.md"]
    package_attestation = validated.package.files["evidence/implementation-review-attestation.json"]
    if package_report != tracked_report or package_attestation != tracked_attestation:
        _fail("packaged implementation review differs from current tracked authority")
    observed = load_implementation_review_attestation(
        package_attestation,
        review_report=package_report,
        snapshot=snapshot,
        require_release_permitting=True,
    )
    if observed != validated.implementation_review_attestation:
        _fail("packaged implementation review object differs from current authority")


__all__ = [
    "PendingPublicationState",
    "ReplayPublicationAuthorityError",
    "authenticate_pending_publication",
    "require_scientific_revision_ancestor",
    "validate_current_implementation_review",
    "verify_pending_publication_unchanged",
]
