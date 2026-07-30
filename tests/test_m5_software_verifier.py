from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fusion_fault_bench import replay_release_software as software
from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.provenance import CleanSourceSnapshot
from fusion_fault_bench.replay_release_authority import (
    ImplementationSnapshot,
    ImplementationSnapshotEntry,
)
from fusion_fault_bench.replay_release_software import (
    M5_SOFTWARE_COMMAND_BY_CHECK,
    ReplaySoftwareVerificationError,
    _run_built_wheel_smoke,
    normalize_command_output,
    verify_software,
)
from fusion_fault_bench.replay_release_validation import (
    M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK,
    M5_SOFTWARE_VERIFICATION_CHECK_IDS,
)

REVISION = "a" * 40
LOCKFILE_SHA256 = "b" * 64
IMPLEMENTATION_SHA256 = "c" * 64


def _authorities(root: Path) -> tuple[CleanSourceSnapshot, ImplementationSnapshot]:
    clean = CleanSourceSnapshot(
        source_root=root,
        git_revision=REVISION,
        git_dir=root / ".git",
        git_common_dir=root / ".git",
        lockfile_sha256=LOCKFILE_SHA256,
        package_version="0.1.0",
        manifest_relative_path="examples/replay/m5-nuscenes-mini-replay-v1.json",
    )
    implementation = ImplementationSnapshot(
        scientific_git_revision=REVISION,
        entries=(
            ImplementationSnapshotEntry(
                path="src/fusion_fault_bench/__init__.py",
                byte_length=1,
                sha256="d" * 64,
            ),
        ),
        sha256=IMPLEMENTATION_SHA256,
    )
    return clean, implementation


def _output() -> Path:
    return Path(f"reports/generated/m5-software-verification-{REVISION}.json")


def _completed(
    command: tuple[str, ...],
    *,
    returncode: int = 0,
    stdout: bytes = b"ok\n",
    stderr: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def _mock_tool_authority(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> software._SoftwareToolAuthority:
    def fingerprint(name: str) -> software._ExecutableFingerprint:
        return software._ExecutableFingerprint(
            path=Path(f"/trusted/{name}"),
            device=1,
            inode=len(name),
            byte_length=10,
            mtime_ns=1,
            sha256=hashlib.sha256(name.encode()).hexdigest(),
        )

    authority = software._SoftwareToolAuthority(
        python=fingerprint("python"),
        ruff=fingerprint("ruff"),
        uv=fingerprint("uv"),
        site_packages=root / ".venv/lib/python3.12/site-packages",
        installed_tools_sha256="f" * 64,
    )
    monkeypatch.setattr(software, "_build_tool_authority", lambda *_args: authority)
    monkeypatch.setattr(software, "_require_tool_authority_unchanged", lambda _value: None)
    monkeypatch.setattr(
        software,
        "_runtime_command",
        lambda logical, **_keywords: logical,
    )
    return authority


def test_verify_software_runs_exact_order_binds_outputs_and_publishes_mode_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    (root / "reports").mkdir()
    clean, implementation = _authorities(root)
    _mock_tool_authority(root, monkeypatch)
    authority_calls: list[tuple[Path, CleanSourceSnapshot, ImplementationSnapshot]] = []

    def authority(**arguments: Any) -> None:
        authority_calls.append(
            (
                arguments["source_root"],
                arguments["clean_snapshot"],
                arguments["implementation_snapshot"],
            )
        )

    observed: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(
        command: tuple[str, ...],
        **arguments: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.append((command, arguments["env"]))
        return _completed(
            command,
            stdout=f"1 passed at {root}/work in 1.25s\n".encode(),
        )

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software._require_source_authority",
        authority,
    )
    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software.subprocess.run",
        run,
    )

    verification = verify_software(
        source_root=root,
        output=_output(),
        clean_snapshot=clean,
        implementation_snapshot=implementation,
    )

    assert tuple(command for command, _environment in observed) == tuple(
        M5_SOFTWARE_COMMAND_BY_CHECK[check_id] for check_id in M5_SOFTWARE_VERIFICATION_CHECK_IDS
    )
    assert len(authority_calls) == 2
    assert all(environment["UV_OFFLINE"] == "1" for _command, environment in observed)
    assert all("NUSCENES_ROOT" not in environment for _command, environment in observed)
    assert tuple(row.check_id for row in verification.checks) == (
        M5_SOFTWARE_VERIFICATION_CHECK_IDS
    )
    assert all(
        row.required_test_ids == M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[row.check_id]
        for row in verification.checks
    )
    expected_normalized = normalize_command_output(
        f"1 passed at {root}/work in 1.25s\n".encode(),
        b"",
        runtime_paths=(root,),
    )
    assert {row.output_sha256 for row in verification.checks} == {
        hashlib.sha256(expected_normalized).hexdigest()
    }

    output = root / _output()
    assert output.read_bytes() == canonical_json_bytes(verification)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_verify_software_fails_immediately_and_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    (root / "reports").mkdir()
    clean, implementation = _authorities(root)
    _mock_tool_authority(root, monkeypatch)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software._require_source_authority",
        lambda **_arguments: None,
    )

    def run(
        command: tuple[str, ...],
        **_arguments: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return _completed(command, returncode=1 if len(calls) == 3 else 0)

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software.subprocess.run",
        run,
    )

    with pytest.raises(ReplaySoftwareVerificationError, match="failure status"):
        verify_software(
            source_root=root,
            output=_output(),
            clean_snapshot=clean,
            implementation_snapshot=implementation,
        )

    assert calls == [
        M5_SOFTWARE_COMMAND_BY_CHECK[check_id]
        for check_id in M5_SOFTWARE_VERIFICATION_CHECK_IDS[:3]
    ]
    assert not (root / _output()).exists()


def test_verify_software_refuses_overwrite_before_starting_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    target = root / _output()
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing\n")
    clean, implementation = _authorities(root)
    _mock_tool_authority(root, monkeypatch)
    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software._require_source_authority",
        lambda **_arguments: None,
    )

    def unexpected(*_arguments: Any, **_keywords: Any) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("software check unexpectedly started")

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software.subprocess.run",
        unexpected,
    )
    with pytest.raises(ReplaySoftwareVerificationError, match="already exists"):
        verify_software(
            source_root=root,
            output=_output(),
            clean_snapshot=clean,
            implementation_snapshot=implementation,
        )
    assert target.read_bytes() == b"existing\n"


def test_postflight_source_drift_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    (root / "reports").mkdir()
    clean, implementation = _authorities(root)
    _mock_tool_authority(root, monkeypatch)
    authority_calls = 0

    def authority(**_arguments: Any) -> None:
        nonlocal authority_calls
        authority_calls += 1
        if authority_calls == 2:
            raise ReplaySoftwareVerificationError("source authority changed")

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software._require_source_authority",
        authority,
    )
    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software.subprocess.run",
        lambda command, **_arguments: _completed(command, stdout=b"1 passed\n"),
    )

    with pytest.raises(ReplaySoftwareVerificationError, match="changed"):
        verify_software(
            source_root=root,
            output=_output(),
            clean_snapshot=clean,
            implementation_snapshot=implementation,
        )
    assert authority_calls == 2
    assert not (root / _output()).exists()


def test_output_normalization_removes_runtime_paths_durations_and_terminal_codes(
    tmp_path: Path,
) -> None:
    normalized = normalize_command_output(
        (
            f"\x1b[31mfailed {tmp_path}/private.py after 3.25 seconds "
            "at 2026-07-30T12:34:56Z\x1b[0m\n"
        ).encode(),
        b"cache /private/var/folders/aa/tool in 250ms\n",
        runtime_paths=(tmp_path,),
    )
    assert os.fspath(tmp_path).encode() not in normalized
    assert b"/private/" not in normalized
    assert b"3.25" not in normalized
    assert b"250ms" not in normalized
    assert b"2026-07-30" not in normalized
    assert b"\x1b" not in normalized


def test_built_wheel_smoke_installs_exact_wheel_offline_and_cleans_owned_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    (root / "reports").mkdir()
    (root / "dist").mkdir()
    (root / ".python-version").write_text("3.12.13\n", encoding="ascii")
    wheel = root / "dist/fusion_fault_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"exact-wheel")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
    uv_authority = software._ExecutableFingerprint(
        path=Path("/trusted/uv"),
        device=1,
        inode=1,
        byte_length=10,
        mtime_ns=1,
        sha256="e" * 64,
    )
    monkeypatch.setattr(software, "_trusted_uv_authority", lambda _environment: uv_authority)
    monkeypatch.setattr(
        software,
        "_require_locked_tool_install",
        lambda _root: (root / ".venv/lib/python3.12/site-packages", "f" * 64),
    )
    monkeypatch.setattr(software, "_fingerprint_executable", lambda *_args, **_kwargs: uv_authority)

    def run(
        command: tuple[str, ...],
        **arguments: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, arguments["env"]))
        if len(calls) == 2:
            installed_ffb = Path(command[4]).with_name("ffb")
            installed_ffb.parent.mkdir(parents=True, exist_ok=True)
            installed_ffb.write_bytes(b"#!/trusted/python\n")
        return _completed(command, stdout=b"smoke-ok\n")

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software.subprocess.run",
        run,
    )
    environment = {"UV_OFFLINE": "1", "PIP_NO_INDEX": "1"}
    output = _run_built_wheel_smoke(
        source_root=root,
        package_version="0.1.0",
        environment=environment,
    )

    assert len(calls) == 8
    install = calls[1][0]
    assert install[:4] == ("/trusted/uv", "pip", "install", "--python")
    assert "--offline" in install
    assert Path(install[-1]) == wheel
    assert all(call_environment["UV_OFFLINE"] == "1" for _command, call_environment in calls)
    assert all("NUSCENES_ROOT" not in call_environment for _command, call_environment in calls)
    assert b"wheel_sha256=" in output
    assert list((root / "reports/generated").iterdir()) == []


def test_built_wheel_smoke_failure_still_cleans_owned_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    (root / "reports").mkdir()
    (root / "dist").mkdir()
    (root / ".python-version").write_text("3.12.13\n", encoding="ascii")
    (root / "dist/fusion_fault_bench-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    calls = 0
    uv_authority = software._ExecutableFingerprint(
        path=Path("/trusted/uv"),
        device=1,
        inode=1,
        byte_length=10,
        mtime_ns=1,
        sha256="e" * 64,
    )
    monkeypatch.setattr(software, "_trusted_uv_authority", lambda _environment: uv_authority)
    monkeypatch.setattr(
        software,
        "_require_locked_tool_install",
        lambda _root: (root / ".venv/lib/python3.12/site-packages", "f" * 64),
    )
    monkeypatch.setattr(software, "_fingerprint_executable", lambda *_args, **_kwargs: uv_authority)

    def run(
        command: tuple[str, ...],
        **_arguments: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        return _completed(command, returncode=1 if calls == 2 else 0)

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_software.subprocess.run",
        run,
    )
    with pytest.raises(ReplaySoftwareVerificationError, match="failure status"):
        _run_built_wheel_smoke(
            source_root=root,
            package_version="0.1.0",
            environment={"UV_OFFLINE": "1"},
        )
    assert list((root / "reports/generated").iterdir()) == []
