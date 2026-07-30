from __future__ import annotations

import os
from pathlib import Path

import pytest

import fusion_fault_bench.replay_release as release_module
from fusion_fault_bench.artifacts import ArtifactValidationError, canonical_json_bytes
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_REPLAY_IDENTITY_SET_SHA256,
    REPLAY_ARTIFACT_PATHS,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_PACKAGE_PATHS,
    M5_RELEASE_SIDECAR_INDEXED_PATHS,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
)
from fusion_fault_bench.replay_release import (
    ReplayReleasePackageContent,
    ReplayReviewCandidateContent,
    publish_release_package,
    publish_review_candidate,
)

_FROZEN_INTENT = (
    Path(__file__).resolve().parents[1] / "examples/replay/m5-nuscenes-mini-replay-v1.json"
).read_bytes()


def _member(path: str) -> bytes:
    if path in {"machine/intent.json", "intent.json"}:
        return _FROZEN_INTENT
    if path.endswith(".ndjson"):
        return canonical_json_bytes({"path": path, "row": 1})
    if path.endswith(".json") or path.endswith("_SUCCESS"):
        return canonical_json_bytes({"path": path, "value": 1})
    if path.endswith(".svg"):
        return b'<svg xmlns="http://www.w3.org/2000/svg"><text>fixed</text></svg>\n'
    if path.startswith("presentation/"):
        return ("\n".join(M5_PRESENTATION_PLACEHOLDERS) + "\n").encode()
    return b"# Fixed M5 evidence\n"


def _candidate() -> ReplayReviewCandidateContent:
    return ReplayReviewCandidateContent(
        scientific_git_revision="1" * 40,
        lockfile_sha256="2" * 64,
        package_version="0.1.0",
        run_id="run:m5-publication-order",
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        primary_local_artifact_sha256="3" * 64,
        repeat_local_artifact_sha256="4" * 64,
        primary_local_run_sha256="5" * 64,
        repeat_local_run_sha256="6" * 64,
        files={path: _member(path) for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS},
    )


def _package() -> ReplayReleasePackageContent:
    return ReplayReleasePackageContent(
        reviewed_candidate_sha256="3" * 64,
        results_review_attestation_sha256="4" * 64,
        machine_artifact_sha256="5" * 64,
        machine_run_sha256="6" * 64,
        scientific_git_revision="1" * 40,
        machine_files={path: _member(path) for path in REPLAY_ARTIFACT_PATHS},
        sidecar_files={path: _member(path) for path in M5_RELEASE_SIDECAR_INDEXED_PATHS},
    )


def test_candidate_index_is_written_only_after_all_indexed_members(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: list[str] = []
    original = release_module._write_nested

    def record(root_fd: int, path: str, value: bytes) -> None:
        observed.append(path)
        original(root_fd, path, value)

    monkeypatch.setattr(release_module, "_write_nested", record)
    publish_review_candidate(_candidate(), tmp_path / "candidate")

    assert tuple(observed[:-1]) == M5_REVIEW_CANDIDATE_INDEXED_PATHS
    assert observed[-1] == "candidate-index.json"


def test_release_commit_markers_follow_their_complete_payloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: list[str] = []
    original = release_module._write_nested

    def record(root_fd: int, path: str, value: bytes) -> None:
        observed.append(path)
        original(root_fd, path, value)

    monkeypatch.setattr(release_module, "_write_nested", record)
    publish_release_package(_package(), tmp_path / "release")

    assert tuple(observed) == M5_RELEASE_PACKAGE_PATHS
    success_position = observed.index("artifact/_SUCCESS")
    assert success_position == 13
    assert all(path.startswith("artifact/") for path in observed[:success_position])
    assert observed[-1] == "release-sidecar-index.json"


def test_release_rejects_staging_path_swap_at_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = release_module.atomic_rename_directory_no_replace_at

    def swap_before_rename(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
    ) -> None:
        os.rename(
            source_name,
            ".displaced-owned-staging",
            src_dir_fd=source_dir_fd,
            dst_dir_fd=source_dir_fd,
        )
        os.mkdir(source_name, mode=0o700, dir_fd=source_dir_fd)
        original(
            source_dir_fd,
            source_name,
            destination_dir_fd,
            destination_name,
        )

    monkeypatch.setattr(
        release_module,
        "atomic_rename_directory_no_replace_at",
        swap_before_rename,
    )
    destination = tmp_path / "release"

    with pytest.raises(ArtifactValidationError, match="published release directory changed"):
        publish_release_package(_package(), destination)

    assert destination.is_dir()
