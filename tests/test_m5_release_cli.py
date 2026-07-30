from __future__ import annotations

import argparse
import importlib.util
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/m5_release.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("m5_release_tool", _TOOL_PATH)
if _TOOL_SPEC is None or _TOOL_SPEC.loader is None:
    raise RuntimeError("M5 release tool cannot be imported for tests")
m5_release = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(m5_release)


def _subcommand_names() -> tuple[str, ...]:
    parser = m5_release._build_parser()
    action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
    return tuple(action.choices)


def _run_arguments(*, label: str, output: str, log: str) -> tuple[str, ...]:
    return (
        "run-replay",
        "--run-label",
        label,
        "--output-dir",
        output,
        "--time-l-output",
        log,
        "--source-root",
        ".",
    )


def _execution_workflow(
    events: list[str],
    observed: dict[str, object],
    *,
    fail_preflight: bool = False,
    fail_postflight: bool = False,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    fingerprint = SimpleNamespace(
        device=1,
        inode=2,
        mode=stat.S_IFREG | 0o755,
        link_count=1,
        owner_uid=3,
        owner_gid=4,
        byte_length=5,
        modified_time_ns=6,
        changed_time_ns=7,
        sha256="a" * 64,
    )
    token = SimpleNamespace(
        time_executable="/usr/bin/time",
        ffb_executable="/locked/environment/bin/ffb",
        time_executable_fingerprint=fingerprint,
        ffb_executable_fingerprint=fingerprint,
    )

    def authenticate_replay_execution(**arguments: object) -> object:
        events.append("authenticate")
        observed["authenticate"] = arguments
        if fail_preflight:
            raise ValueError("rejected preflight")
        output_argument = str(arguments["output_dir"])
        token.source_root = Path(os.path.abspath(arguments["source_root"]))
        token.output_argument = output_argument
        token.time_l_argument = str(arguments["time_l_output"])
        token.success_argument = f"{output_argument}.success.json"
        return token

    def verify_replay_execution_unchanged(**arguments: object) -> object:
        events.append("verify")
        observed["verify"] = arguments
        if fail_postflight:
            raise ValueError("rejected postflight")
        return token

    def build_replay_execution_success_receipt(**arguments: object) -> object:
        events.append("build-receipt")
        observed["build-receipt"] = arguments
        return SimpleNamespace(
            path=Path(token.success_argument),
            value=b'{"schema":"test-success"}\n',
        )

    def verify_replay_execution_success_receipt(**arguments: object) -> None:
        events.append("verify-receipt")
        observed["verify-receipt"] = arguments

    return (
        SimpleNamespace(
            authenticate_replay_execution=authenticate_replay_execution,
            verify_replay_execution_unchanged=verify_replay_execution_unchanged,
            build_replay_execution_success_receipt=(build_replay_execution_success_receipt),
            verify_replay_execution_success_receipt=(verify_replay_execution_success_receipt),
        ),
        token,
    )


def test_parser_exposes_the_exact_frozen_command_set() -> None:
    assert _subcommand_names() == (
        "attest-implementation-review",
        "verify-software",
        "run-replay",
        "prepare-review",
        "validate-review-candidate",
        "attest-results-review",
        "build-release",
        "sync-reviewed-evidence",
        "validate-release",
        "validate-publication",
    )


def test_bounded_child_captures_stdout_and_stderr_without_reemitting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        m5_release._run_bounded_child(
            ("/bin/sh", "-c", "printf bounded-out; printf bounded-err >&2"),
            pass_fds=(),
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_bounded_child_fails_closed_on_output_overflow() -> None:
    with pytest.raises(m5_release.M5ReleaseDriverError, match="exceeded its byte cap"):
        m5_release._run_bounded_child(
            (sys.executable, "-c", "import os; os.write(1, b'x' * 70000)"),
            pass_fds=(),
        )


def test_run_replay_authenticates_reserves_and_postflights_exact_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "reports/generated").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    observed: dict[str, Any] = {}
    events: list[str] = []
    workflow, token = _execution_workflow(events, observed)
    monkeypatch.setattr(m5_release, "_workflow_api", lambda: workflow)

    def fake_child(command: tuple[str, ...], *, pass_fds: tuple[int, ...]) -> int:
        events.append("child")
        observed["command"] = command
        observed["pass_fds"] = pass_fds
        os.write(pass_fds[0], b"1.00 real 0.50 user 0.25 sys\n100 maximum resident set size\n")
        return 0

    monkeypatch.setattr(m5_release, "_run_bounded_child", fake_child)
    output = "reports/generated/m5-replay-primary-deadbeef-r1"
    log = "reports/generated/m5-replay-primary-deadbeef-r1.time-l.txt"
    assert m5_release.main(_run_arguments(label="primary", output=output, log=log)) == 0

    authority_arguments = {
        "source_root": Path("."),
        "run_label": "primary",
        "output_dir": Path(output),
        "time_l_output": Path(log),
    }
    assert events == [
        "authenticate",
        "child",
        "verify",
        "build-receipt",
        "verify-receipt",
    ]
    assert observed["authenticate"] == authority_arguments
    assert observed["verify"] == {"token": token, **authority_arguments}
    assert observed["command"] == (
        "/usr/bin/time",
        "-l",
        "-o",
        f"/dev/fd/{observed['pass_fds'][0]}",
        "/locked/environment/bin/ffb",
        "replay",
        "run",
        "--output-dir",
        output,
    )
    log_path = tmp_path / log
    assert log_path.read_bytes().startswith(b"1.00 real")
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    receipt_path = tmp_path / f"{output}.success.json"
    assert receipt_path.read_bytes() == b'{"schema":"test-success"}\n'
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert observed["build-receipt"] == {"token": token}
    assert observed["verify-receipt"] == {"token": token}


def test_run_replay_rejects_execution_authority_from_another_working_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "reports/generated").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    observed: dict[str, object] = {}
    facade, token = _execution_workflow(events, observed)
    authenticate = facade.authenticate_replay_execution

    def foreign_authority(**arguments: object) -> object:
        result = authenticate(**arguments)
        token.source_root = tmp_path.parent
        return result

    facade.authenticate_replay_execution = foreign_authority
    monkeypatch.setattr(m5_release, "_workflow_api", lambda: facade)
    monkeypatch.setattr(
        m5_release,
        "_run_bounded_child",
        lambda *_args, **_kwargs: pytest.fail("foreign authority launched replay"),
    )
    log = Path("reports/generated/primary.time-l.txt")
    assert (
        m5_release.main(
            _run_arguments(
                label="primary",
                output="reports/generated/primary-r1",
                log=log.as_posix(),
            )
        )
        == 2
    )
    assert events == ["authenticate"]
    assert not log.exists()


def test_run_replay_propagates_failure_once_and_preserves_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "reports/generated").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    observed: dict[str, object] = {}
    workflow, _token = _execution_workflow(events, observed)
    monkeypatch.setattr(m5_release, "_workflow_api", lambda: workflow)
    calls = 0

    def fake_child(command: tuple[str, ...], *, pass_fds: tuple[int, ...]) -> int:
        del command
        nonlocal calls
        calls += 1
        events.append("child")
        os.write(pass_fds[0], b"failed attempt resource block\n")
        return 7

    monkeypatch.setattr(m5_release, "_run_bounded_child", fake_child)
    log = Path("reports/generated/repeat.time-l.txt")
    assert (
        m5_release.main(
            _run_arguments(
                label="repeat",
                output="reports/generated/repeat-r1",
                log=log.as_posix(),
            )
        )
        == 7
    )
    assert calls == 1
    assert events == ["authenticate", "child", "verify"]
    assert log.read_bytes() == b"failed attempt resource block\n"


def test_run_replay_failed_preflight_never_launches_or_reserves_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "reports/generated").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    observed: dict[str, object] = {}
    workflow, _token = _execution_workflow(events, observed, fail_preflight=True)
    monkeypatch.setattr(m5_release, "_workflow_api", lambda: workflow)

    def forbidden_child(*args: object, **kwargs: object) -> int:
        raise AssertionError("the replay process must not launch")

    monkeypatch.setattr(m5_release, "_run_bounded_child", forbidden_child)
    log = Path("reports/generated/primary.time-l.txt")
    assert (
        m5_release.main(
            _run_arguments(
                label="primary",
                output="reports/generated/primary-r1",
                log=log.as_posix(),
            )
        )
        == 2
    )
    assert events == ["authenticate"]
    assert not log.exists()


def test_run_replay_postflight_failure_preserves_attempt_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "reports/generated").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    observed: dict[str, object] = {}
    workflow, _token = _execution_workflow(events, observed, fail_postflight=True)
    monkeypatch.setattr(m5_release, "_workflow_api", lambda: workflow)

    def fake_child(command: tuple[str, ...], *, pass_fds: tuple[int, ...]) -> int:
        del command
        events.append("child")
        os.write(pass_fds[0], b"postflight-failed attempt resource block\n")
        return 0

    monkeypatch.setattr(m5_release, "_run_bounded_child", fake_child)
    log = Path("reports/generated/primary.time-l.txt")
    assert (
        m5_release.main(
            _run_arguments(
                label="primary",
                output="reports/generated/primary-r1",
                log=log.as_posix(),
            )
        )
        == 2
    )
    assert events == ["authenticate", "child", "verify"]
    assert log.read_bytes() == b"postflight-failed attempt resource block\n"


def test_run_replay_refuses_existing_log_without_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generated = tmp_path / "reports/generated"
    generated.mkdir(parents=True)
    (generated / "primary.time-l.txt").write_bytes(b"keep\n")
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    observed: dict[str, object] = {}
    workflow, _token = _execution_workflow(events, observed)
    monkeypatch.setattr(m5_release, "_workflow_api", lambda: workflow)

    def forbidden_child(*args: object, **kwargs: object) -> int:
        raise AssertionError("the replay process must not launch")

    monkeypatch.setattr(m5_release, "_run_bounded_child", forbidden_child)
    assert (
        m5_release.main(
            _run_arguments(
                label="primary",
                output="reports/generated/primary-r1",
                log="reports/generated/primary.time-l.txt",
            )
        )
        == 2
    )
    assert events == ["authenticate"]
    assert (generated / "primary.time-l.txt").read_bytes() == b"keep\n"
    assert capsys.readouterr().err == "error: run-replay failed closed\n"


def test_prepare_review_is_a_thin_workflow_facade_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def prepare_review_candidate(**arguments: object) -> str:
        observed.update(arguments)
        return "a" * 64

    monkeypatch.setattr(
        m5_release,
        "_workflow_api",
        lambda: SimpleNamespace(prepare_review_candidate=prepare_review_candidate),
    )
    assert (
        m5_release.main(
            (
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
            )
        )
        == 0
    )
    assert observed == {
        "primary_artifact": Path("primary"),
        "repeat_artifact": Path("repeat"),
        "primary_time_l": Path("primary.time"),
        "repeat_time_l": Path("repeat.time"),
        "software_verification": Path("software.json"),
        "output_dir": Path("candidate"),
        "source_root": Path("."),
    }
    assert capsys.readouterr().out == f"ok: prepare-review sha256={'a' * 64}\n"


@pytest.mark.parametrize(
    ("command", "arguments", "operation", "expected"),
    (
        (
            "validate-review-candidate",
            ("candidate",),
            "validate_review_candidate",
            {"path": Path("candidate"), "source_root": Path(".")},
        ),
        (
            "validate-release",
            ("release",),
            "validate_release_package",
            {"path": Path("release")},
        ),
        (
            "validate-publication",
            ("--release", "release", "--source-root", "."),
            "validate_publication",
            {"release": Path("release"), "source_root": Path(".")},
        ),
    ),
)
def test_validation_commands_delegate_without_dataset_access(
    command: str,
    arguments: tuple[str, ...],
    operation: str,
    expected: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def validate(**values: object) -> str:
        observed.update(values)
        return "b" * 64

    monkeypatch.setattr(
        m5_release,
        "_workflow_api",
        lambda: SimpleNamespace(**{operation: validate}),
    )
    assert m5_release.main((command, *arguments)) == 0
    assert observed == expected


def test_attest_implementation_review_canonicalizes_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "review.md"
    decision = tmp_path / "decision.json"
    output = tmp_path / "attestation.json"
    report.write_bytes(b"review\n")
    decision.write_bytes(b"decision\n")
    snapshot = object()
    parsed_decision = object()
    attestation = object()
    monkeypatch.setattr(m5_release, "build_implementation_snapshot", lambda _root: snapshot)
    monkeypatch.setattr(
        m5_release,
        "load_implementation_review_decision",
        lambda value: parsed_decision if value == b"decision\n" else None,
    )

    def build(
        value: object,
        *,
        review_report: bytes,
        snapshot: object,
    ) -> object:
        assert value is parsed_decision
        assert review_report == b"review\n"
        assert snapshot is not None
        return attestation

    monkeypatch.setattr(m5_release, "build_implementation_review_attestation", build)
    monkeypatch.setattr(
        m5_release,
        "canonical_json_bytes",
        lambda value: b'{"schema":"test"}\n' if value is attestation else b"",
    )
    arguments = (
        "attest-implementation-review",
        "--review-report",
        os.fspath(report),
        "--decision",
        os.fspath(decision),
        "--output",
        os.fspath(output),
        "--source-root",
        os.fspath(tmp_path),
    )
    assert m5_release.main(arguments) == 0
    assert output.read_bytes() == b'{"schema":"test"}\n'
    assert m5_release.main(arguments) == 2
    assert output.read_bytes() == b'{"schema":"test"}\n'


def test_attest_results_review_uses_validated_candidate_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "review.md"
    decision = tmp_path / "decision.json"
    output = tmp_path / "attestation.json"
    report.write_bytes(b"review\n")
    decision.write_bytes(b"decision\n")
    candidate = object()
    parsed_decision = object()
    attestation = object()
    observed: dict[str, object] = {}

    def load_validated_review_candidate(**arguments: object) -> object:
        observed.update(arguments)
        return candidate

    monkeypatch.setattr(
        m5_release,
        "_workflow_api",
        lambda: SimpleNamespace(
            load_validated_review_candidate=load_validated_review_candidate,
        ),
    )
    monkeypatch.setattr(
        m5_release,
        "load_results_review_decision",
        lambda value: parsed_decision if value == b"decision\n" else None,
    )

    def attest(value: object, *, review_report: bytes, decision: object) -> object:
        assert value is candidate
        assert review_report == b"review\n"
        assert decision is parsed_decision
        return attestation

    monkeypatch.setattr(m5_release, "attest_results_review", attest)
    monkeypatch.setattr(
        m5_release,
        "canonical_json_bytes",
        lambda value: b'{"schema":"results"}\n' if value is attestation else b"",
    )
    assert (
        m5_release.main(
            (
                "attest-results-review",
                "--candidate",
                "candidate",
                "--review-report",
                os.fspath(report),
                "--decision",
                os.fspath(decision),
                "--output",
                os.fspath(output),
                "--source-root",
                os.fspath(tmp_path),
            )
        )
        == 0
    )
    assert observed == {"path": Path("candidate"), "source_root": tmp_path}
    assert output.read_bytes() == b'{"schema":"results"}\n'


def test_sync_reviewed_evidence_passes_exact_source_bound_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = object()
    observed: dict[str, object] = {}
    report = tmp_path / "docs/reviews/m5-results-review.md"
    attestation = tmp_path / "docs/reviews/m5-results-review-attestation.json"
    monkeypatch.setattr(m5_release, "load_release_package", lambda path: package)

    def sync(value: object, **arguments: object) -> None:
        assert value is package
        observed.update(arguments)

    monkeypatch.setattr(m5_release, "sync_reviewed_evidence", sync)
    assert (
        m5_release.main(
            (
                "sync-reviewed-evidence",
                "--release",
                "release",
                "--review-report-output",
                os.fspath(report),
                "--review-attestation-output",
                os.fspath(attestation),
                "--source-root",
                os.fspath(tmp_path),
            )
        )
        == 0
    )
    assert observed == {
        "report_output": report,
        "attestation_output": attestation,
        "source_root": tmp_path,
    }
