from __future__ import annotations

import importlib
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
m5_release = importlib.import_module("tools.m5_release")


def _generated_root(root: Path) -> Path:
    generated = root / "reports" / "generated"
    generated.mkdir(parents=True)
    return generated


def test_timed_replay_uses_exclusive_private_fd_and_exact_logical_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_root(tmp_path)
    output = Path("reports/generated/m5-primary")
    log = Path("reports/generated/m5-primary.time-l.txt")
    captured: dict[str, object] = {}

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        captured["command"] = command
        captured.update(kwargs)
        descriptors = cast(tuple[int, ...], kwargs["pass_fds"])
        os.write(descriptors[0], b"1.25 real\n123456 maximum resident set size\n")
        (tmp_path / output).mkdir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(m5_release.subprocess, "run", fake_run)

    m5_release.run_replay(
        run_label="primary",
        output_dir=output,
        time_l_output=log,
        source_root=tmp_path,
    )

    command = cast(tuple[str, ...], captured["command"])
    descriptor = cast(tuple[int, ...], captured["pass_fds"])[0]
    assert command == (
        "/usr/bin/time",
        "-l",
        "-o",
        f"/dev/fd/{descriptor}",
        "ffb",
        "replay",
        "run",
        "--output-dir",
        output.as_posix(),
    )
    assert captured["cwd"] == tmp_path.resolve()
    assert captured["check"] is False
    assert captured["stdout"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.DEVNULL
    log_path = tmp_path / log
    assert log_path.read_bytes() == b"1.25 real\n123456 maximum resident set size\n"
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    assert generated.is_dir()


def test_timed_replay_rejects_existing_destinations_without_process_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_root(tmp_path)
    output = Path("reports/generated/m5-primary")
    log = Path("reports/generated/m5-primary.time-l.txt")
    calls = 0

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess((), 0)

    monkeypatch.setattr(m5_release.subprocess, "run", fake_run)
    (generated / "m5-primary").mkdir()
    with pytest.raises(m5_release.M5ReleaseToolError, match="must not already exist"):
        m5_release.run_replay(
            run_label="primary",
            output_dir=output,
            time_l_output=log,
            source_root=tmp_path,
        )
    assert calls == 0
    assert not (tmp_path / log).exists()

    (generated / "m5-primary").rmdir()
    existing = generated / "m5-primary.time-l.txt"
    existing.write_bytes(b"operator-owned")
    with pytest.raises(m5_release.M5ReleaseToolError, match="must not already exist"):
        m5_release.run_replay(
            run_label="primary",
            output_dir=output,
            time_l_output=log,
            source_root=tmp_path,
        )
    assert calls == 0
    assert existing.read_bytes() == b"operator-owned"


def test_failed_timed_replay_preserves_attempt_and_sanitizes_child_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _generated_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = Path("reports/generated/m5-primary")
    log = Path("reports/generated/m5-primary.time-l.txt")
    private_detail = "/Users/private-owner/datasets/nuScenes"

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        descriptor = cast(tuple[int, ...], kwargs["pass_fds"])[0]
        os.write(descriptor, b"0.50 real\n654321 maximum resident set size\n")
        (tmp_path / output).mkdir()
        (tmp_path / output / "partial").write_text(private_detail)
        return subprocess.CompletedProcess(
            command,
            7,
            stdout=private_detail.encode(),
            stderr=private_detail.encode(),
        )

    monkeypatch.setattr(m5_release.subprocess, "run", fake_run)

    assert (
        m5_release.main(
            [
                "run-replay",
                "--run-label",
                "primary",
                "--output-dir",
                output.as_posix(),
                "--time-l-output",
                log.as_posix(),
            ]
        )
        == 7
    )
    output_text = capsys.readouterr()
    assert output_text.out == ""
    assert output_text.err == "error: M5 release command failed closed\n"
    assert private_detail not in output_text.err
    assert (tmp_path / log).read_bytes().startswith(b"0.50 real")
    assert (tmp_path / output / "partial").read_text() == private_detail


def test_timed_replay_rejects_symlink_parent_and_non_generated_paths(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "generated").symlink_to(external, target_is_directory=True)

    with pytest.raises(m5_release.M5ReleaseToolError, match="real directory"):
        m5_release.run_replay(
            run_label="repeat",
            output_dir=Path("reports/generated/repeat"),
            time_l_output=Path("reports/generated/repeat.time-l.txt"),
            source_root=tmp_path,
        )
    with pytest.raises(m5_release.M5ReleaseToolError, match="reports/generated"):
        m5_release.run_replay(
            run_label="repeat",
            output_dir=Path("outside/repeat"),
            time_l_output=Path("reports/generated/repeat.time-l.txt"),
            source_root=tmp_path,
        )


def test_timed_replay_rejects_parent_replacement_and_preserves_displaced_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_root(tmp_path)
    output_parent = generated / "output-parent"
    log_parent = generated / "log-parent"
    output_parent.mkdir()
    log_parent.mkdir()
    output = Path("reports/generated/output-parent/replay")
    log = Path("reports/generated/log-parent/replay.time-l.txt")
    displaced = generated / "log-parent-displaced"

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        descriptor = cast(tuple[int, ...], kwargs["pass_fds"])[0]
        os.write(descriptor, b"1.00 real\n123456 maximum resident set size\n")
        (tmp_path / output).mkdir()
        log_parent.rename(displaced)
        log_parent.mkdir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(m5_release.subprocess, "run", fake_run)

    with pytest.raises(m5_release.M5ReleaseToolError, match="parent must remain"):
        m5_release.run_replay(
            run_label="repeat",
            output_dir=output,
            time_l_output=log,
            source_root=tmp_path,
        )
    assert (displaced / log.name).read_bytes().startswith(b"1.00 real")
    assert not (log_parent / log.name).exists()


def test_parser_exposes_every_frozen_release_command() -> None:
    parser = m5_release._build_parser()
    cases = (
        (
            [
                "attest-implementation-review",
                "--review-report",
                "review.md",
                "--decision",
                "decision.json",
                "--output",
                "attestation.json",
            ],
            "attest-implementation-review",
        ),
        (["verify-software", "--output", "software.json"], "verify-software"),
        (
            [
                "run-replay",
                "--run-label",
                "primary",
                "--output-dir",
                "reports/generated/primary",
                "--time-l-output",
                "reports/generated/primary.time-l.txt",
            ],
            "run-replay",
        ),
        (
            [
                "prepare-review",
                "--primary-artifact",
                "primary",
                "--repeat-artifact",
                "repeat",
                "--primary-time-l",
                "primary.time",
                "--repeat-time-l",
                "repeat.time",
                "--software-verification",
                "software.json",
                "--output-dir",
                "candidate",
            ],
            "prepare-review",
        ),
        (["validate-review-candidate", "candidate"], "validate-review-candidate"),
        (
            [
                "attest-results-review",
                "--candidate",
                "candidate",
                "--review-report",
                "review.md",
                "--decision",
                "decision.json",
                "--output",
                "attestation.json",
            ],
            "attest-results-review",
        ),
        (
            [
                "build-release",
                "--candidate",
                "candidate",
                "--results-review",
                "review.md",
                "--results-review-attestation",
                "attestation.json",
                "--primary-artifact",
                "primary",
                "--repeat-artifact",
                "repeat",
                "--primary-time-l",
                "primary.time",
                "--repeat-time-l",
                "repeat.time",
                "--software-verification",
                "software.json",
                "--output-dir",
                "release",
            ],
            "build-release",
        ),
        (
            [
                "sync-reviewed-evidence",
                "--release",
                "release",
                "--review-report-output",
                "docs/review.md",
                "--review-attestation-output",
                "docs/attestation.json",
            ],
            "sync-reviewed-evidence",
        ),
        (["validate-release", "release"], "validate-release"),
        (
            [
                "validate-publication",
                "--release",
                "release",
                "--source-root",
                ".",
            ],
            "validate-publication",
        ),
    )

    assert tuple(parser.parse_args(argv).command for argv, _ in cases) == tuple(
        expected for _, expected in cases
    )


def test_tool_never_echoes_invalid_arguments_or_core_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_detail = "/Users/private-owner/datasets/nuScenes"
    assert m5_release.main(["validate-release", private_detail, "--private", private_detail]) == 2
    invalid = capsys.readouterr()
    assert invalid.out == ""
    assert invalid.err == "error: M5 release command failed closed\n"
    assert private_detail not in invalid.err

    def fail(_path: Path) -> None:
        raise RuntimeError(private_detail)

    monkeypatch.setattr(m5_release.release_api, "validate_release", fail, raising=False)
    assert m5_release.main(["validate-release", private_detail]) == 2
    failed = capsys.readouterr()
    assert failed.out == ""
    assert failed.err == "error: M5 release command failed closed\n"
    assert private_detail not in failed.err


def test_tool_delegates_candidate_release_sync_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, object]] = []

    def inputs(**values: Path) -> SimpleNamespace:
        return SimpleNamespace(**values)

    monkeypatch.setattr(m5_release.release_api, "M5ReleaseLocalInputs", inputs)
    monkeypatch.setattr(
        m5_release.release_api,
        "prepare_review_candidate",
        lambda value, **kwargs: calls.append(("prepare", (value, kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        m5_release.release_api,
        "validate_review_candidate",
        lambda path: calls.append(("candidate", path)),
        raising=False,
    )
    monkeypatch.setattr(
        m5_release.release_api,
        "build_release",
        lambda **kwargs: calls.append(("build", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        m5_release.release_api,
        "sync_reviewed_evidence",
        lambda **kwargs: calls.append(("sync", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        m5_release.release_api,
        "validate_release",
        lambda path: calls.append(("release", path)),
        raising=False,
    )
    monkeypatch.setattr(
        m5_release.release_api,
        "validate_publication",
        lambda **kwargs: calls.append(("publication", kwargs)),
        raising=False,
    )

    local_flags = [
        "--primary-artifact",
        "primary",
        "--repeat-artifact",
        "repeat",
        "--primary-time-l",
        "primary.time",
        "--repeat-time-l",
        "repeat.time",
        "--software-verification",
        "software.json",
    ]
    commands = (
        ["prepare-review", *local_flags, "--output-dir", "candidate"],
        ["validate-review-candidate", "candidate"],
        [
            "build-release",
            "--candidate",
            "candidate",
            "--results-review",
            "review.md",
            "--results-review-attestation",
            "attestation.json",
            *local_flags,
            "--output-dir",
            "release",
        ],
        [
            "sync-reviewed-evidence",
            "--release",
            "release",
            "--review-report-output",
            "docs/review.md",
            "--review-attestation-output",
            "docs/attestation.json",
        ],
        ["validate-release", "release"],
        [
            "validate-publication",
            "--release",
            "release",
            "--source-root",
            ".",
        ],
    )
    for command in commands:
        assert m5_release.main(command) == 0
        assert capsys.readouterr().out == "M5 release command completed\n"

    assert tuple(name for name, _ in calls) == (
        "prepare",
        "candidate",
        "build",
        "sync",
        "release",
        "publication",
    )


def test_tool_delegates_review_attestations_and_software_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        m5_release.release_api,
        "attest_implementation_review",
        lambda **kwargs: calls.append(("implementation", kwargs)),
    )
    monkeypatch.setattr(
        m5_release.release_api,
        "run_software_verification_checks",
        lambda **kwargs: calls.append(("software", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        m5_release.release_api,
        "attest_results_review",
        lambda **kwargs: calls.append(("results", kwargs)),
        raising=False,
    )

    commands = (
        [
            "attest-implementation-review",
            "--review-report",
            "review.md",
            "--decision",
            "decision.json",
            "--output",
            "attestation.json",
        ],
        ["verify-software", "--output", "software.json"],
        [
            "attest-results-review",
            "--candidate",
            "candidate",
            "--review-report",
            "results.md",
            "--decision",
            "results.json",
            "--output",
            "results-attestation.json",
        ],
    )
    for command in commands:
        assert m5_release.main(command) == 0
        assert capsys.readouterr().out == "M5 release command completed\n"
    assert tuple(name for name, _ in calls) == (
        "implementation",
        "software",
        "results",
    )
    for _, arguments in calls:
        assert cast(dict[str, object], arguments).get("source_root", tmp_path) == tmp_path
