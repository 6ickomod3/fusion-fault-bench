from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from fusion_fault_bench import replay_release_workflow as workflow
from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_IMPLEMENTATION_REVIEW_AREAS,
    ReplayImplementationReviewDecisionV1,
)
from fusion_fault_bench.replay_release_authority import (
    ImplementationSnapshot,
    implementation_snapshot_from_files,
)
from fusion_fault_bench.replay_release_validation import (
    build_implementation_review_attestation,
)

_IMPLEMENTATION_REVIEW_PATH = Path("docs/reviews/m5-release-implementation-review.md")
_IMPLEMENTATION_ATTESTATION_PATH = Path(
    "docs/reviews/m5-release-implementation-review-attestation.json"
)
_READ_REVIEW_AUTHORITY = cast(
    Callable[[Path, ImplementationSnapshot], tuple[bytes, bytes]],
    vars(workflow)["_read_review_authority"],
)


def _git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(source_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _review_lifecycle(
    source_root: Path,
) -> tuple[ImplementationSnapshot, ImplementationSnapshot, bytes, bytes]:
    _git(source_root, "init")
    _git(source_root, "config", "user.email", "m5-review@example.invalid")
    _git(source_root, "config", "user.name", "M5 Review Test")
    _git(source_root, "config", "commit.gpgsign", "false")

    implementation_path = source_root / "src/example.py"
    implementation_path.parent.mkdir(parents=True)
    implementation_bytes = b"VALUE = 1\n"
    implementation_path.write_bytes(implementation_bytes)
    _git(source_root, "add", "--", "src/example.py")
    _git(source_root, "commit", "-m", "reviewed implementation")
    reviewed_revision = _git(source_root, "rev-parse", "HEAD")
    implementation_files = {"src/example.py": implementation_bytes}
    reviewed_snapshot = implementation_snapshot_from_files(
        implementation_files,
        scientific_git_revision=reviewed_revision,
    )

    decision = ReplayImplementationReviewDecisionV1(
        schema="ffb.m5-implementation-review-decision/v1",
        reviewer_identity="independent-reviewer",
        reviewed_areas=M5_IMPLEMENTATION_REVIEW_AREAS,
        findings=(),
        disposition="pass",
    )
    report = b"# M5 implementation review\n\nNo blocking findings.\n"
    attestation = canonical_json_bytes(
        build_implementation_review_attestation(
            decision,
            review_report=report,
            snapshot=reviewed_snapshot,
        )
    )
    report_path = source_root / _IMPLEMENTATION_REVIEW_PATH
    attestation_path = source_root / _IMPLEMENTATION_ATTESTATION_PATH
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(report)
    attestation_path.write_bytes(attestation)
    _git(
        source_root,
        "add",
        "--",
        _IMPLEMENTATION_REVIEW_PATH.as_posix(),
        _IMPLEMENTATION_ATTESTATION_PATH.as_posix(),
    )
    _git(source_root, "commit", "-m", "commit implementation review evidence")
    evidence_revision = _git(source_root, "rev-parse", "HEAD")
    evidence_snapshot = implementation_snapshot_from_files(
        implementation_files,
        scientific_git_revision=evidence_revision,
    )
    return reviewed_snapshot, evidence_snapshot, report, attestation


def test_review_authority_accepts_content_bound_evidence_commit(tmp_path: Path) -> None:
    reviewed, evidence, report, attestation = _review_lifecycle(tmp_path)

    assert reviewed.scientific_git_revision != evidence.scientific_git_revision
    assert reviewed.sha256 == evidence.sha256
    assert _READ_REVIEW_AUTHORITY(tmp_path, evidence) == (report, attestation)


def test_review_authority_rejects_index_blob_drift(tmp_path: Path) -> None:
    _reviewed, evidence, _report, attestation = _review_lifecycle(tmp_path)
    attestation_path = tmp_path / _IMPLEMENTATION_ATTESTATION_PATH
    attestation_path.write_bytes(attestation + b" ")
    _git(
        tmp_path,
        "add",
        "--",
        _IMPLEMENTATION_ATTESTATION_PATH.as_posix(),
    )

    with pytest.raises(
        workflow.ReplayReleaseWorkflowError,
        match="absent, stale, or blocking",
    ):
        _READ_REVIEW_AUTHORITY(tmp_path, evidence)


def test_review_authority_rejects_untracked_review_blob(tmp_path: Path) -> None:
    _reviewed, evidence, _report, _attestation = _review_lifecycle(tmp_path)
    _git(
        tmp_path,
        "rm",
        "--cached",
        "--",
        _IMPLEMENTATION_REVIEW_PATH.as_posix(),
    )

    with pytest.raises(
        workflow.ReplayReleaseWorkflowError,
        match="absent, stale, or blocking",
    ):
        _READ_REVIEW_AUTHORITY(tmp_path, evidence)
