from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fusion_fault_bench.replay_release_workflow_build as workflow_build
from fusion_fault_bench.contracts.replay_release_v1 import M5_RELEASE_PACKAGE_PATHS
from fusion_fault_bench.replay_release_build import M5_RELEASE_DESTINATION

_REVISION = "a" * 40


def _valid_git_state() -> tuple[bytes, ...]:
    untracked = (
        b"\x00".join(
            f"{M5_RELEASE_DESTINATION.as_posix()}/{path}".encode("ascii")
            for path in M5_RELEASE_PACKAGE_PATHS
        )
        + b"\x00"
    )
    return (
        f"{_REVISION}\n".encode("ascii"),
        b"100644 object 0\ttracked\x00",
        b"H tracked\x00",
        b"",
        b"",
        untracked,
    )


def test_release_git_state_allows_only_the_exact_new_package() -> None:
    clean = SimpleNamespace(git_revision=_REVISION)
    workflow_build._validate_release_git_state(_valid_git_state(), clean=clean)

    state = list(_valid_git_state())
    state[-1] += b"unexpected.txt\x00"
    with pytest.raises(
        workflow_build.ReplayReleaseWorkflowBuildError,
        match="unexpected untracked",
    ):
        workflow_build._validate_release_git_state(tuple(state), clean=clean)


@pytest.mark.parametrize(
    ("position", "value", "message"),
    (
        (0, b"b" * 40 + b"\n", "revision changed"),
        (2, b"S tracked\x00", "index flags changed"),
        (3, b"tracked\x00", "tracked source changed"),
        (4, b"tracked\x00", "tracked source changed"),
    ),
)
def test_release_git_state_rejects_source_drift(
    position: int,
    value: bytes,
    message: str,
) -> None:
    state = list(_valid_git_state())
    state[position] = value
    with pytest.raises(workflow_build.ReplayReleaseWorkflowBuildError, match=message):
        workflow_build._validate_release_git_state(
            tuple(state),
            clean=SimpleNamespace(git_revision=_REVISION),
        )


def test_orchestration_reauthenticates_immediately_before_publish_and_postflights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean = SimpleNamespace(source_root=tmp_path, git_revision=_REVISION)
    implementation = object()
    report = b"review\n"
    attestation = b"{}\n"
    events: list[str] = []
    delegated: dict[str, Any] = {}
    result = object()

    def clean_authority(source_root: Path) -> tuple[object, object]:
        events.append("clean")
        assert source_root == tmp_path
        return clean, implementation

    def review_authority(source_root: Path, snapshot: object) -> tuple[bytes, bytes]:
        events.append("review")
        assert (source_root, snapshot) == (tmp_path, implementation)
        return report, attestation

    def build(**arguments: Any) -> object:
        events.append("build")
        delegated.update(arguments)
        arguments["prepublish_authority"]()
        events.append("publish")
        return result

    def postflight(**arguments: Any) -> None:
        events.append("postflight")
        assert arguments["clean"] is clean
        assert arguments["implementation"] is implementation
        assert arguments["report"] == report
        assert arguments["attestation"] == attestation

    monkeypatch.setattr(workflow_build, "build_reviewed_release", build)
    monkeypatch.setattr(workflow_build, "_postflight_release_authority", postflight)
    output = M5_RELEASE_DESTINATION
    observed = workflow_build.orchestrate_reviewed_release(
        candidate=Path("candidate"),
        results_review=Path("results-review"),
        results_review_attestation=Path("results-attestation"),
        primary_artifact=Path("primary"),
        repeat_artifact=Path("repeat"),
        primary_time_l=Path("primary-time"),
        repeat_time_l=Path("repeat-time"),
        software_verification=Path("software"),
        output_dir=output,
        source_root=tmp_path,
        clean_authority=clean_authority,
        review_authority=review_authority,
    )

    assert observed is result
    assert delegated["clean_snapshot"] is clean
    assert delegated["implementation_snapshot"] is implementation
    assert delegated["implementation_report"] == report
    assert delegated["implementation_attestation"] == attestation
    assert delegated["source_root"] == tmp_path
    assert events == [
        "clean",
        "review",
        "build",
        "clean",
        "review",
        "publish",
        "postflight",
    ]
