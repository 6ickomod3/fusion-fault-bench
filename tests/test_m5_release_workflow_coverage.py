from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from fusion_fault_bench import replay_release_candidate
from fusion_fault_bench import replay_release_workflow as workflow
from fusion_fault_bench import replay_release_workflow_build as workflow_build
from fusion_fault_bench.contracts.replay_release_v1 import M5_RELEASE_PACKAGE_PATHS
from fusion_fault_bench.replay_release_build import M5_RELEASE_DESTINATION

_REVISION = "a" * 40


def _git(source_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", "-C", os.fspath(source_root), *arguments),
        check=True,
        capture_output=True,
    ).stdout


def _tracked_repository(source_root: Path) -> Path:
    _git(source_root, "init")
    _git(source_root, "config", "user.email", "workflow@example.invalid")
    _git(source_root, "config", "user.name", "M5 Workflow Test")
    _git(source_root, "config", "commit.gpgsign", "false")
    tracked = source_root / "docs/review.md"
    tracked.parent.mkdir(parents=True)
    tracked.write_bytes(b"review authority\n")
    _git(source_root, "add", "--", "docs/review.md")
    _git(source_root, "commit", "-m", "tracked authority")
    return tracked


def test_git_helpers_read_exact_tracked_head_blob_and_reject_index_drift(
    tmp_path: Path,
) -> None:
    tracked = _tracked_repository(tmp_path)

    assert workflow._tracked_head_blob(tmp_path, Path("docs/review.md")) == tracked.read_bytes()
    assert (
        workflow._git_text(tmp_path, "rev-parse", "HEAD")
        == _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    )

    tracked.write_bytes(b"staged replacement\n")
    _git(tmp_path, "add", "--", "docs/review.md")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="exact tracked HEAD blob"):
        workflow._tracked_head_blob(tmp_path, Path("docs/review.md"))


def test_git_helpers_fail_closed_outside_a_repository(tmp_path: Path) -> None:
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="Git execution authority"):
        workflow._git_text(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="Git byte authority"):
        workflow._git_bytes(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(
        workflow_build.ReplayReleaseWorkflowBuildError,
        match="Git authority",
    ):
        workflow_build._git_bytes(tmp_path, "rev-parse", "HEAD")


def test_clean_authority_normalizes_and_cross_checks_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    clean = SimpleNamespace(source_root=tmp_path, git_revision=_REVISION)
    implementation = SimpleNamespace(scientific_git_revision=_REVISION)
    observed: list[object] = []
    monkeypatch.setattr(
        workflow,
        "discover_clean_source",
        lambda intent: observed.append(intent) or clean,
    )
    monkeypatch.setattr(
        workflow,
        "verify_locked_execution",
        lambda snapshot: observed.append(snapshot),
    )
    monkeypatch.setattr(
        workflow,
        "build_implementation_snapshot",
        lambda root: observed.append(root) or implementation,
    )

    assert workflow._clean_authority(tmp_path / ".") == (clean, implementation)
    assert observed[0] == tmp_path / "examples/replay/m5-nuscenes-mini-replay-v1.json"
    assert observed[1:] == [clean, tmp_path]

    clean.source_root = tmp_path / "other"
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="disagree"):
        workflow._clean_authority(tmp_path)


def test_clean_authority_rejects_a_different_working_tree(tmp_path: Path) -> None:
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="working directory"):
        workflow._clean_authority(tmp_path)


def test_clean_authority_wraps_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        workflow,
        "discover_clean_source",
        lambda _intent: (_ for _ in ()).throw(OSError("unavailable")),
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="clean locked"):
        workflow._clean_authority(tmp_path)


def test_normalized_replay_paths_accept_exact_attempt() -> None:
    output = Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1")
    timing = output.with_name(f"{output.name}.time-l.txt")
    assert workflow._normalized_run_paths(
        run_label="primary",
        revision=_REVISION,
        output_dir=output,
        time_l_output=timing,
    ) == (output.as_posix(), timing.as_posix())


@pytest.mark.parametrize(
    ("run_label", "output", "timing", "message"),
    (
        (
            "other",
            Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1"),
            Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1.time-l.txt"),
            "label",
        ),
        (
            "primary",
            Path(f"/tmp/m5-replay-primary-{_REVISION}-r1"),
            Path(f"/tmp/m5-replay-primary-{_REVISION}-r1.time-l.txt"),
            "not normalized",
        ),
        (
            "primary",
            Path(f"reports/other/m5-replay-primary-{_REVISION}-r1"),
            Path(f"reports/other/m5-replay-primary-{_REVISION}-r1.time-l.txt"),
            "outside reports/generated",
        ),
        (
            "primary",
            Path("reports/generated/not-an-attempt"),
            Path("reports/generated/not-an-attempt.time-l.txt"),
            "frozen r1",
        ),
        (
            "primary",
            Path(f"reports/generated/m5-replay-primary-{_REVISION}-r2"),
            Path(f"reports/generated/m5-replay-primary-{_REVISION}-r2.time-l.txt"),
            "frozen r1",
        ),
        (
            "repeat",
            Path(f"reports/generated/m5-replay-repeat-{_REVISION}-r1"),
            Path("reports/generated/wrong.time-l.txt"),
            "does not match",
        ),
    ),
)
def test_normalized_replay_paths_reject_wrong_authority(
    run_label: str,
    output: Path,
    timing: Path,
    message: str,
) -> None:
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match=message):
        workflow._normalized_run_paths(
            run_label=run_label,
            revision=_REVISION,
            output_dir=output,
            time_l_output=timing,
        )


def test_authenticated_executable_requires_executable_single_regular_file(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ffb"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    executable_path, fingerprint = workflow._authenticated_executable(executable, label="ffb")
    assert executable_path == os.fspath(executable.resolve())
    assert fingerprint.byte_length == executable.stat().st_size
    assert fingerprint.link_count == 1
    assert fingerprint.sha256 == "306c6ca7407560340797866e077e053627ad409277d1b9da58106fce4cf717cb"

    executable.chmod(0o600)
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="safe regular"):
        workflow._authenticated_executable(executable, label="ffb")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="unavailable"):
        workflow._authenticated_executable(tmp_path / "missing", label="ffb")


def test_authenticated_executable_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    executable = tmp_path / "ffb"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o700)
    redirected = tmp_path / "redirected"
    redirected.symlink_to(executable)
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="unavailable"):
        workflow._authenticated_executable(redirected, label="ffb")

    hardlink = tmp_path / "hardlink"
    os.link(executable, hardlink)
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="unavailable"):
        workflow._authenticated_executable(executable, label="ffb")


@pytest.mark.parametrize("raw", (None, "relative/cache", "/definitely/missing/m5-cache"))
def test_authenticated_input_rejects_absent_relative_or_missing(raw: str | None) -> None:
    message = (
        "absolute input" if raw is None else "normalized absolute|unavailable|safe real directory"
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match=message):
        workflow._authenticated_input_directory(
            raw,
            label="cache root",
            require_private=True,
        )


def _execution_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, Path, Path]:
    clean = SimpleNamespace(
        source_root=tmp_path,
        git_revision=_REVISION,
        lockfile_sha256="b" * 64,
        package_version="0.1.0",
    )
    implementation = SimpleNamespace(sha256="c" * 64)
    environment = SimpleNamespace(os_name="Darwin")
    dataset = tmp_path.parent / f"{tmp_path.name}-dataset"
    cache = tmp_path.parent / f"{tmp_path.name}-cache"
    expected_ffb = tmp_path / ".venv/bin/ffb"
    expected_ffb.parent.mkdir(parents=True)
    expected_ffb.write_bytes(b"#!/bin/sh\n")
    expected_ffb.chmod(0o700)
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: (clean, implementation))
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda _root, _implementation: (b"report\n", b"attestation\n"),
    )
    monkeypatch.setattr(
        workflow,
        "_authenticated_input_directory",
        lambda raw, **_kwargs: (dataset, (1, 2)) if raw == os.fspath(dataset) else (cache, (3, 4)),
    )
    monkeypatch.setattr(workflow, "collect_runtime_environment", lambda: environment)
    monkeypatch.setattr(workflow.shutil, "which", lambda _name: os.fspath(expected_ffb))
    fingerprint = workflow.ReplayExecutableFingerprint(
        device=1,
        inode=2,
        mode=stat.S_IFREG | 0o700,
        link_count=1,
        owner_uid=3,
        owner_gid=4,
        byte_length=10,
        modified_time_ns=5,
        changed_time_ns=6,
        sha256="d" * 64,
    )
    monkeypatch.setattr(
        workflow,
        "_authenticated_executable",
        lambda path, **_kwargs: (os.fspath(path.resolve()), fingerprint),
    )
    monkeypatch.setattr(
        workflow,
        "_software_verification_authority",
        lambda _clean, _implementation: (
            f"reports/generated/m5-software-verification-{_REVISION}.json",
            "e" * 64,
        ),
    )
    monkeypatch.setattr(workflow, "_require_attempt_lifecycle", lambda *_a, **_k: None)
    monkeypatch.setattr(workflow, "_require_upstream_sync", lambda _root, _revision: "origin/main")
    for name in workflow._THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("NUSCENES_ROOT", os.fspath(dataset))
    monkeypatch.setenv("UV_CACHE_DIR", os.fspath(cache))
    return clean, implementation, environment, dataset, cache


def test_execution_authority_builds_complete_outcome_blind_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, implementation, environment, _dataset, _cache = _execution_mocks(monkeypatch, tmp_path)
    output = Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1")
    timing = output.with_name(f"{output.name}.time-l.txt")

    token = workflow.authenticate_replay_execution(
        source_root=tmp_path,
        run_label="primary",
        output_dir=output,
        time_l_output=timing,
    )

    assert token.source_root == tmp_path
    assert token.dataset_root_identity == (1, 2)
    assert token.uv_cache_root_identity == (3, 4)
    assert token.scientific_git_revision == clean.git_revision
    assert token.implementation_snapshot_sha256 == implementation.sha256
    assert (
        token.implementation_attestation_sha256
        == __import__("hashlib").sha256(b"attestation\n").hexdigest()
    )
    assert token.environment is environment
    assert token.upstream_ref == "origin/main"
    assert token.software_verification_sha256 == "e" * 64
    assert token.ffb_executable_fingerprint.sha256 == "d" * 64


def test_execution_authority_rejects_thread_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _execution_mocks(monkeypatch, tmp_path)
    monkeypatch.setenv(workflow._THREAD_ENVIRONMENT_KEYS[0], "2")
    output = Path(f"reports/generated/m5-replay-repeat-{_REVISION}-r1")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="thread environment"):
        workflow._execution_authority(
            source_root=tmp_path,
            run_label="repeat",
            output_dir=output,
            time_l_output=output.with_name(f"{output.name}.time-l.txt"),
        )


def test_execution_authority_rejects_overlapping_inputs_and_non_darwin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, _implementation, environment, _dataset, _cache = _execution_mocks(monkeypatch, tmp_path)
    output = Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1")
    timing = output.with_name(f"{output.name}.time-l.txt")
    monkeypatch.setattr(
        workflow,
        "_authenticated_input_directory",
        lambda _raw, **_kwargs: (clean.source_root, (1, 1)),
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="must be isolated"):
        workflow._execution_authority(
            source_root=tmp_path,
            run_label="primary",
            output_dir=output,
            time_l_output=timing,
        )

    dataset = tmp_path.parent / "separate-dataset"
    cache = tmp_path.parent / "separate-cache"
    identities: Iterator[tuple[Path, tuple[int, int]]] = iter(((dataset, (1, 2)), (cache, (3, 4))))
    monkeypatch.setattr(
        workflow,
        "_authenticated_input_directory",
        lambda _raw, **_kwargs: next(identities),
    )
    environment.os_name = "Linux"
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="Darwin CPU"):
        workflow._execution_authority(
            source_root=tmp_path,
            run_label="primary",
            output_dir=output,
            time_l_output=timing,
        )


def test_execution_token_postflight_accepts_identity_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = SimpleNamespace(value="stable")
    monkeypatch.setattr(workflow, "_execution_authority", lambda **_kwargs: token)
    workflow.verify_replay_execution_unchanged(
        token=cast(Any, token),
        source_root=Path("."),
        run_label="primary",
        output_dir=Path("output"),
        time_l_output=Path("timing"),
    )

    monkeypatch.setattr(
        workflow,
        "_execution_authority",
        lambda **_kwargs: SimpleNamespace(value="changed"),
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="changed during the run"):
        workflow.verify_replay_execution_unchanged(
            token=cast(Any, token),
            source_root=Path("."),
            run_label="primary",
            output_dir=Path("output"),
            time_l_output=Path("timing"),
        )


def _candidate_authorities(tmp_path: Path) -> tuple[SimpleNamespace, object, bytes, bytes]:
    return (
        SimpleNamespace(source_root=tmp_path, git_revision=_REVISION),
        object(),
        b"implementation review\n",
        b"implementation attestation\n",
    )


def test_candidate_prepare_wrapper_delegates_and_reloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, implementation, report, attestation = _candidate_authorities(tmp_path)
    delegated: dict[str, dict[str, Any]] = {}
    candidate = object()
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: (clean, implementation))
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda _root, _implementation: (report, attestation),
    )
    monkeypatch.setattr(
        workflow,
        "_authenticate_completed_replays",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        replay_release_candidate,
        "prepare_review_candidate",
        lambda **kwargs: delegated.setdefault("prepare", kwargs),
    )
    monkeypatch.setattr(
        replay_release_candidate,
        "load_validated_review_candidate",
        lambda **kwargs: delegated.setdefault("load", kwargs) and candidate,
    )
    output = Path(f"reports/generated/m5-review-candidate-{_REVISION}")

    primary = Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1")
    repeat = Path(f"reports/generated/m5-replay-repeat-{_REVISION}-r1")
    observed = workflow.prepare_review_candidate(
        primary_artifact=primary,
        repeat_artifact=repeat,
        primary_time_l=primary.with_name(f"{primary.name}.time-l.txt"),
        repeat_time_l=repeat.with_name(f"{repeat.name}.time-l.txt"),
        software_verification=Path(f"reports/generated/m5-software-verification-{_REVISION}.json"),
        output_dir=output,
        source_root=tmp_path,
    )

    assert observed is candidate
    assert delegated["prepare"]["clean_snapshot"] is clean
    assert delegated["prepare"]["implementation_snapshot"] is implementation
    assert delegated["load"]["path"] == output
    assert delegated["load"]["implementation_report"] == report
    assert delegated["load"]["implementation_attestation"] == attestation


def test_candidate_prepare_rejects_wrong_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, implementation, report, attestation = _candidate_authorities(tmp_path)
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: (clean, implementation))
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda _root, _implementation: (report, attestation),
    )
    monkeypatch.setattr(
        workflow,
        "_authenticate_completed_replays",
        lambda **_kwargs: object(),
    )
    primary = Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1")
    repeat = Path(f"reports/generated/m5-replay-repeat-{_REVISION}-r1")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="destination"):
        workflow.prepare_review_candidate(
            primary_artifact=primary,
            repeat_artifact=repeat,
            primary_time_l=primary.with_name(f"{primary.name}.time-l.txt"),
            repeat_time_l=repeat.with_name(f"{repeat.name}.time-l.txt"),
            software_verification=Path(
                f"reports/generated/m5-software-verification-{_REVISION}.json"
            ),
            output_dir=Path("wrong"),
            source_root=tmp_path,
        )


def test_candidate_load_wrapper_postflights_and_validate_returns_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, implementation, report, attestation = _candidate_authorities(tmp_path)
    candidate = SimpleNamespace(candidate_sha256="d" * 64)
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: (clean, implementation))
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda _root, _implementation: (report, attestation),
    )
    monkeypatch.setattr(
        replay_release_candidate,
        "load_validated_review_candidate",
        lambda **_kwargs: candidate,
    )

    assert (
        workflow.load_validated_review_candidate(path=Path("candidate"), source_root=tmp_path)
        is candidate
    )
    assert (
        workflow.validate_review_candidate(path=Path("candidate"), source_root=tmp_path) == "d" * 64
    )

    candidate.candidate_sha256 = None
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="semantic digest"):
        workflow.validate_review_candidate(path=Path("candidate"), source_root=tmp_path)


def test_candidate_load_wrapper_rejects_postflight_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, implementation, report, attestation = _candidate_authorities(tmp_path)
    changed = SimpleNamespace(source_root=tmp_path, git_revision="b" * 40)
    authorities = iter(((clean, implementation), (changed, implementation)))
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: next(authorities))
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda _root, _implementation: (report, attestation),
    )
    monkeypatch.setattr(
        replay_release_candidate,
        "load_validated_review_candidate",
        lambda **_kwargs: object(),
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="changed"):
        workflow.load_validated_review_candidate(path=Path("candidate"), source_root=tmp_path)


def test_build_facade_delegates_authority_functions_and_wraps_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: dict[str, Any] = {}
    result = object()

    def orchestrate(**kwargs: Any) -> object:
        delegated.update(kwargs)
        return result

    monkeypatch.setattr(workflow_build, "orchestrate_reviewed_release", orchestrate)
    clean = SimpleNamespace(source_root=Path("."), git_revision=_REVISION)
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: (clean, object()))
    monkeypatch.setattr(
        workflow,
        "_authenticate_completed_replays",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        workflow,
        "_require_frozen_candidate_inputs",
        lambda **_kwargs: None,
    )
    arguments = {
        "candidate": Path("candidate"),
        "results_review": Path("review"),
        "results_review_attestation": Path("review-attestation"),
        "primary_artifact": Path("primary"),
        "repeat_artifact": Path("repeat"),
        "primary_time_l": Path("primary-time"),
        "repeat_time_l": Path("repeat-time"),
        "software_verification": Path("software"),
        "output_dir": M5_RELEASE_DESTINATION,
        "source_root": Path("."),
    }
    assert workflow.build_reviewed_release(**arguments) is result
    assert delegated["clean_authority"] is workflow._clean_authority
    assert delegated["review_authority"] is workflow._read_review_authority

    monkeypatch.setattr(
        workflow_build,
        "orchestrate_reviewed_release",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("blocked")),
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="failed closed"):
        workflow.build_reviewed_release(**arguments)


def test_real_release_git_state_accepts_only_exact_untracked_package(tmp_path: Path) -> None:
    tracked = _tracked_repository(tmp_path)
    revision = _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    for relative in M5_RELEASE_PACKAGE_PATHS:
        target = tmp_path / M5_RELEASE_DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"package member\n")

    state = workflow_build._release_git_state(tmp_path)
    workflow_build._validate_release_git_state(
        state,
        clean=SimpleNamespace(git_revision=revision),
    )
    assert tracked.read_bytes() == b"review authority\n"


def _valid_release_state() -> tuple[bytes, ...]:
    untracked = (
        b"\x00".join(
            f"{M5_RELEASE_DESTINATION.as_posix()}/{path}".encode("ascii")
            for path in M5_RELEASE_PACKAGE_PATHS
        )
        + b"\x00"
    )
    return (
        f"{_REVISION}\n".encode(),
        b"index-state",
        b"H tracked\x00",
        b"",
        b"",
        untracked,
    )


def test_postflight_revalidates_git_locked_source_and_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean = SimpleNamespace(source_root=tmp_path, git_revision=_REVISION)
    implementation = object()
    calls: list[object] = []
    state = _valid_release_state()
    monkeypatch.setattr(workflow_build, "_release_git_state", lambda _root: state)
    monkeypatch.setattr(
        workflow_build,
        "verify_locked_execution",
        lambda snapshot: calls.append(snapshot),
    )
    monkeypatch.setattr(
        workflow_build,
        "build_implementation_snapshot",
        lambda root: calls.append(root) or implementation,
    )
    review = cast(
        Callable[[Path, object], tuple[bytes, bytes]],
        lambda root, snapshot: calls.extend((root, snapshot)) or (b"report", b"attestation"),
    )

    workflow_build._postflight_release_authority(
        source_root=tmp_path,
        output_dir=tmp_path / M5_RELEASE_DESTINATION,
        clean=clean,
        implementation=cast(Any, implementation),
        report=b"report",
        attestation=b"attestation",
        review_authority=cast(Any, review),
    )

    assert calls == [clean, tmp_path, tmp_path, implementation]


def test_postflight_rejects_wrong_path_invalid_source_and_state_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean = SimpleNamespace(source_root=tmp_path, git_revision=_REVISION)
    implementation = object()
    arguments = {
        "source_root": tmp_path,
        "clean": clean,
        "implementation": implementation,
        "report": b"report",
        "attestation": b"attestation",
        "review_authority": lambda _root, _snapshot: (b"report", b"attestation"),
    }
    with pytest.raises(workflow_build.ReplayReleaseWorkflowBuildError, match="frozen"):
        workflow_build._postflight_release_authority(
            output_dir=tmp_path / "wrong",
            **cast(Any, arguments),
        )

    state = _valid_release_state()
    monkeypatch.setattr(workflow_build, "_release_git_state", lambda _root: state)
    monkeypatch.setattr(
        workflow_build,
        "verify_locked_execution",
        lambda _clean: (_ for _ in ()).throw(ValueError("stale")),
    )
    with pytest.raises(workflow_build.ReplayReleaseWorkflowBuildError, match="invalid after"):
        workflow_build._postflight_release_authority(
            output_dir=tmp_path / M5_RELEASE_DESTINATION,
            **cast(Any, arguments),
        )

    states = iter((state, (*state[:1], b"changed-index", *state[2:])))
    monkeypatch.setattr(workflow_build, "_release_git_state", lambda _root: next(states))
    monkeypatch.setattr(workflow_build, "verify_locked_execution", lambda _clean: None)
    monkeypatch.setattr(
        workflow_build,
        "build_implementation_snapshot",
        lambda _root: implementation,
    )
    with pytest.raises(workflow_build.ReplayReleaseWorkflowBuildError, match="changed during"):
        workflow_build._postflight_release_authority(
            output_dir=tmp_path / M5_RELEASE_DESTINATION,
            **cast(Any, arguments),
        )


def test_same_clean_authority_rejects_source_or_review_drift(tmp_path: Path) -> None:
    clean = SimpleNamespace(source_root=tmp_path)
    implementation = object()
    with pytest.raises(workflow_build.ReplayReleaseWorkflowBuildError, match="source authority"):
        workflow_build._same_clean_authority(
            source_root=tmp_path,
            clean=cast(Any, clean),
            implementation=cast(Any, implementation),
            report=b"report",
            attestation=b"attestation",
            clean_authority=cast(Any, lambda _root: (object(), implementation)),
            review_authority=cast(Any, lambda _root, _snapshot: (b"report", b"attestation")),
        )

    with pytest.raises(workflow_build.ReplayReleaseWorkflowBuildError, match="review authority"):
        workflow_build._same_clean_authority(
            source_root=tmp_path,
            clean=cast(Any, clean),
            implementation=cast(Any, implementation),
            report=b"report",
            attestation=b"attestation",
            clean_authority=cast(Any, lambda _root: (clean, implementation)),
            review_authority=cast(Any, lambda _root, _snapshot: (b"changed", b"attestation")),
        )
