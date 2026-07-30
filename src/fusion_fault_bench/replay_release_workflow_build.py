"""Clean-source orchestration around final M5 package publication."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from fusion_fault_bench.contracts.replay_release_v1 import M5_RELEASE_PACKAGE_PATHS
from fusion_fault_bench.provenance import CleanSourceSnapshot, verify_locked_execution
from fusion_fault_bench.replay_release_authority import (
    ImplementationSnapshot,
    build_implementation_snapshot,
)
from fusion_fault_bench.replay_release_build import (
    M5_RELEASE_DESTINATION,
    build_reviewed_release,
)

type CleanAuthority = Callable[
    [Path],
    tuple[CleanSourceSnapshot, ImplementationSnapshot],
]
type ReviewAuthority = Callable[
    [Path, ImplementationSnapshot],
    tuple[bytes, bytes],
]


class ReplayReleaseWorkflowBuildError(ValueError):
    """Final package orchestration observed source or publication drift."""


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", os.fspath(source_root), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ReplayReleaseWorkflowBuildError("M5 final publication Git authority is unavailable")
    return result.stdout


def _same_clean_authority(
    *,
    source_root: Path,
    clean: CleanSourceSnapshot,
    implementation: ImplementationSnapshot,
    report: bytes,
    attestation: bytes,
    clean_authority: CleanAuthority,
    review_authority: ReviewAuthority,
) -> None:
    observed_clean, observed_implementation = clean_authority(source_root)
    if observed_clean != clean or observed_implementation != implementation:
        raise ReplayReleaseWorkflowBuildError(
            "M5 source authority changed during final package construction"
        )
    observed_report, observed_attestation = review_authority(
        observed_clean.source_root,
        observed_implementation,
    )
    if observed_report != report or observed_attestation != attestation:
        raise ReplayReleaseWorkflowBuildError(
            "M5 implementation review authority changed during final package construction"
        )


def _release_git_state(source_root: Path) -> tuple[bytes, ...]:
    return (
        _git_bytes(source_root, "rev-parse", "HEAD"),
        _git_bytes(source_root, "ls-files", "--stage", "-z"),
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


def _validate_release_git_state(
    state: tuple[bytes, ...],
    *,
    clean: CleanSourceSnapshot,
) -> None:
    revision, _index, flags, worktree_diff, staged_diff, untracked = state
    if revision.strip() != clean.git_revision.encode("ascii"):
        raise ReplayReleaseWorkflowBuildError("M5 Git revision changed during final publication")
    if worktree_diff or staged_diff:
        raise ReplayReleaseWorkflowBuildError("M5 tracked source changed during final publication")
    if any(record and not record.startswith(b"H ") for record in flags.split(b"\x00")):
        raise ReplayReleaseWorkflowBuildError("M5 Git index flags changed during final publication")
    expected = frozenset(
        f"{M5_RELEASE_DESTINATION.as_posix()}/{path}".encode("ascii")
        for path in M5_RELEASE_PACKAGE_PATHS
    )
    observed = frozenset(record for record in untracked.split(b"\x00") if record)
    if observed != expected:
        raise ReplayReleaseWorkflowBuildError(
            "M5 final publication created an unexpected untracked path"
        )


def _postflight_release_authority(
    *,
    source_root: Path,
    output_dir: Path,
    clean: CleanSourceSnapshot,
    implementation: ImplementationSnapshot,
    report: bytes,
    attestation: bytes,
    review_authority: ReviewAuthority,
) -> None:
    expected_output = clean.source_root / M5_RELEASE_DESTINATION
    observed_output = Path(os.path.abspath(os.fspath(output_dir)))
    if observed_output != expected_output:
        raise ReplayReleaseWorkflowBuildError(
            "M5 final publication does not occupy the frozen release path"
        )
    before = _release_git_state(clean.source_root)
    _validate_release_git_state(before, clean=clean)
    try:
        verify_locked_execution(clean)
        observed_implementation = build_implementation_snapshot(clean.source_root)
        observed_report, observed_attestation = review_authority(
            clean.source_root,
            observed_implementation,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ReplayReleaseWorkflowBuildError(
            "M5 source authority is invalid after final publication"
        ) from error
    after = _release_git_state(clean.source_root)
    _validate_release_git_state(after, clean=clean)
    if (
        after != before
        or observed_implementation != implementation
        or observed_report != report
        or observed_attestation != attestation
    ):
        raise ReplayReleaseWorkflowBuildError(
            "M5 source or review authority changed during final publication postflight"
        )


def orchestrate_reviewed_release(
    *,
    candidate: Path,
    results_review: Path,
    results_review_attestation: Path,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
    output_dir: Path,
    source_root: Path,
    clean_authority: CleanAuthority,
    review_authority: ReviewAuthority,
) -> object:
    """Authenticate, rebuild, publish, and postflight the exact M5 package."""

    clean, implementation = clean_authority(source_root)
    report, attestation = review_authority(clean.source_root, implementation)

    def prepublish_authority() -> None:
        _same_clean_authority(
            source_root=clean.source_root,
            clean=clean,
            implementation=implementation,
            report=report,
            attestation=attestation,
            clean_authority=clean_authority,
            review_authority=review_authority,
        )

    result = build_reviewed_release(
        candidate=candidate,
        results_review=results_review,
        results_review_attestation=results_review_attestation,
        primary_artifact=primary_artifact,
        repeat_artifact=repeat_artifact,
        primary_time_l=primary_time_l,
        repeat_time_l=repeat_time_l,
        software_verification=software_verification,
        output_dir=output_dir,
        source_root=clean.source_root,
        clean_snapshot=clean,
        implementation_snapshot=implementation,
        implementation_report=report,
        implementation_attestation=attestation,
        prepublish_authority=prepublish_authority,
    )
    _postflight_release_authority(
        source_root=clean.source_root,
        output_dir=output_dir,
        clean=clean,
        implementation=implementation,
        report=report,
        attestation=attestation,
        review_authority=review_authority,
    )
    return result


__all__ = [
    "ReplayReleaseWorkflowBuildError",
    "orchestrate_reviewed_release",
]
