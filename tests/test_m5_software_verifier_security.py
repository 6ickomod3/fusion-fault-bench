from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from fusion_fault_bench import replay_release_software as software
from fusion_fault_bench.contracts.replay_release_v1 import ReplaySoftwareVerificationV1
from fusion_fault_bench.provenance import CleanSourceSnapshot
from fusion_fault_bench.replay_release_authority import (
    ImplementationSnapshot,
    ImplementationSnapshotEntry,
)

_REVISION = "a" * 40


def _authorities(root: Path) -> tuple[CleanSourceSnapshot, ImplementationSnapshot]:
    return (
        CleanSourceSnapshot(
            source_root=root,
            git_revision=_REVISION,
            git_dir=root / ".git",
            git_common_dir=root / ".git",
            lockfile_sha256="b" * 64,
            package_version="0.1.0",
            manifest_relative_path="examples/replay/m5-nuscenes-mini-replay-v1.json",
        ),
        ImplementationSnapshot(
            scientific_git_revision=_REVISION,
            entries=(
                ImplementationSnapshotEntry(
                    path="src/fusion_fault_bench/__init__.py",
                    byte_length=1,
                    sha256="c" * 64,
                ),
            ),
            sha256="d" * 64,
        ),
    )


def _wheel_fixture(root: Path, *, python_pin: bytes = b"3.12.13\n") -> Path:
    (root / "reports").mkdir()
    (root / "dist").mkdir()
    (root / ".python-version").write_bytes(python_pin)
    wheel = root / "dist/fusion_fault_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    return wheel


def test_normalization_rejects_oversized_and_non_utf8_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(software, "_COMMAND_OUTPUT_BYTE_CAP", 3)
    with pytest.raises(software.ReplaySoftwareVerificationError, match="byte cap"):
        software.normalize_command_output(b"1234", b"")
    with pytest.raises(software.ReplaySoftwareVerificationError, match="UTF-8"):
        software.normalize_command_output(b"\xff", b"")


def test_run_command_translates_process_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_arguments: object, **_keywords: object) -> subprocess.CompletedProcess[bytes]:
        raise OSError("private process detail")

    monkeypatch.setattr(software.subprocess, "run", fail)
    with pytest.raises(software.ReplaySoftwareVerificationError, match="could not be started"):
        software._run_command(
            ("tool", "check"),
            source_root=tmp_path,
            environment={},
        )


def test_bounded_regular_file_accepts_stable_bytes_and_rejects_invalid_shapes(
    tmp_path: Path,
) -> None:
    stable = tmp_path / "stable"
    stable.write_bytes(b"stable")
    assert software._regular_file_bytes(stable, byte_cap=6, label="fixture") == b"stable"

    empty = tmp_path / "empty"
    empty.touch()
    with pytest.raises(software.ReplaySoftwareVerificationError, match="bounded regular"):
        software._regular_file_bytes(empty, byte_cap=6, label="fixture")

    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"1234567")
    with pytest.raises(software.ReplaySoftwareVerificationError, match="bounded regular"):
        software._regular_file_bytes(oversized, byte_cap=6, label="fixture")

    linked = tmp_path / "linked"
    os.link(stable, linked)
    with pytest.raises(software.ReplaySoftwareVerificationError, match="bounded regular"):
        software._regular_file_bytes(linked, byte_cap=6, label="fixture")


def test_bounded_regular_file_detects_mutation_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "changing"
    target.write_bytes(b"before")
    original_read = Path.read_bytes

    def mutate(path: Path) -> bytes:
        value = original_read(path)
        path.write_bytes(value + b"!")
        return value

    monkeypatch.setattr(Path, "read_bytes", mutate)
    with pytest.raises(software.ReplaySoftwareVerificationError, match="changed while"):
        software._regular_file_bytes(target, byte_cap=32, label="fixture")


def test_wheel_discovery_fails_closed_for_missing_or_ambiguous_wheels(
    tmp_path: Path,
) -> None:
    with pytest.raises(software.ReplaySoftwareVerificationError, match="no wheel"):
        software._built_wheel(tmp_path, "0.1.0")

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "fusion_fault_bench-0.1.0-py3-none-any.whl").write_bytes(b"expected")
    (dist / "fusion_fault_bench-0.0.9-py3-none-any.whl").write_bytes(b"stale")
    with pytest.raises(software.ReplaySoftwareVerificationError, match="one exact wheel"):
        software._built_wheel(tmp_path, "0.1.0")


def test_generated_root_creates_private_directory_and_rejects_aliases(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    generated = software._generated_root(tmp_path)
    assert generated == reports / "generated"
    assert generated.is_dir()

    unsafe_root = tmp_path / "unsafe"
    unsafe_root.mkdir()
    (unsafe_root / "reports").write_bytes(b"not-a-directory")
    with pytest.raises(software.ReplaySoftwareVerificationError, match="safe directory"):
        software._generated_root(unsafe_root)

    alias_root = tmp_path / "alias"
    (alias_root / "reports").mkdir(parents=True)
    target = tmp_path / "generated-target"
    target.mkdir()
    (alias_root / "reports/generated").symlink_to(target, target_is_directory=True)
    with pytest.raises(software.ReplaySoftwareVerificationError, match="safe directory"):
        software._generated_root(alias_root)


def test_owned_temp_cleanup_rejects_identity_mismatch_without_deleting(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    metadata = owned.stat()
    with pytest.raises(software.ReplaySoftwareVerificationError, match="cleaned safely"):
        software._remove_owned_temp(
            owned,
            device=metadata.st_dev,
            inode=metadata.st_ino + 1,
        )
    assert owned.is_dir()


def test_wheel_smoke_rejects_invalid_python_pin_before_creating_temp(
    tmp_path: Path,
) -> None:
    _wheel_fixture(tmp_path, python_pin=b"3.11.9\n")
    with pytest.raises(software.ReplaySoftwareVerificationError, match="version pin"):
        software._run_built_wheel_smoke(
            source_root=tmp_path,
            package_version="0.1.0",
            environment={"UV_OFFLINE": "1"},
        )
    assert not (tmp_path / "reports/generated").exists()


def test_wheel_smoke_detects_built_wheel_mutation_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = _wheel_fixture(tmp_path)
    original_regular_file_bytes = software._regular_file_bytes
    wheel_reads = 0

    def regular_file_bytes(path: Path, *, byte_cap: int, label: str) -> bytes:
        nonlocal wheel_reads
        value = original_regular_file_bytes(path, byte_cap=byte_cap, label=label)
        if path == wheel:
            wheel_reads += 1
            return b"initial" if wheel_reads == 1 else b"mutated"
        return value

    monkeypatch.setattr(software, "_regular_file_bytes", regular_file_bytes)
    monkeypatch.setattr(software, "_run_command", lambda *_args, **_kwargs: b"ok\n")
    with pytest.raises(software.ReplaySoftwareVerificationError, match="changed"):
        software._run_built_wheel_smoke(
            source_root=tmp_path,
            package_version="0.1.0",
            environment={"UV_OFFLINE": "1"},
        )
    assert wheel_reads == 2
    assert list((tmp_path / "reports/generated").iterdir()) == []


def test_source_authority_accepts_exact_observation_and_translates_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean, implementation = _authorities(tmp_path)
    verified: list[CleanSourceSnapshot] = []
    monkeypatch.setattr(software, "discover_clean_source", lambda _path: clean)
    monkeypatch.setattr(software, "verify_locked_execution", verified.append)
    monkeypatch.setattr(software, "build_implementation_snapshot", lambda _root: implementation)
    software._require_source_authority(
        source_root=tmp_path,
        clean_snapshot=clean,
        implementation_snapshot=implementation,
    )
    assert verified == [clean]

    monkeypatch.setattr(
        software,
        "discover_clean_source",
        lambda _path: (_ for _ in ()).throw(OSError("private source failure")),
    )
    with pytest.raises(software.ReplaySoftwareVerificationError, match="clean locked"):
        software._require_source_authority(
            source_root=tmp_path,
            clean_snapshot=clean,
            implementation_snapshot=implementation,
        )


@pytest.mark.parametrize(
    "mode",
    (
        "observed-clean",
        "observed-implementation",
        "source-root",
        "revision-pair",
    ),
)
def test_source_authority_rejects_each_stale_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    clean, implementation = _authorities(tmp_path)
    supplied_clean = clean
    supplied_implementation = implementation
    observed_clean = clean
    observed_implementation = implementation
    if mode == "observed-clean":
        observed_clean = replace(clean, git_revision="e" * 40)
    elif mode == "observed-implementation":
        observed_implementation = replace(implementation, sha256="f" * 64)
    elif mode == "source-root":
        supplied_clean = replace(clean, source_root=Path("/different/source"))
        observed_clean = supplied_clean
    else:
        supplied_implementation = replace(implementation, scientific_git_revision="e" * 40)
        observed_implementation = supplied_implementation
    monkeypatch.setattr(software, "discover_clean_source", lambda _path: observed_clean)
    monkeypatch.setattr(software, "verify_locked_execution", lambda _snapshot: None)
    monkeypatch.setattr(
        software,
        "build_implementation_snapshot",
        lambda _root: observed_implementation,
    )
    with pytest.raises(software.ReplaySoftwareVerificationError, match="authority changed"):
        software._require_source_authority(
            source_root=tmp_path,
            clean_snapshot=supplied_clean,
            implementation_snapshot=supplied_implementation,
        )


def test_output_path_must_exactly_bind_revision(tmp_path: Path) -> None:
    relative = Path(f"reports/generated/m5-software-verification-{_REVISION}.json")
    expected = tmp_path / relative
    assert software._expected_output(tmp_path, relative, revision=_REVISION) == (
        expected,
        expected.name,
    )
    assert software._expected_output(tmp_path, expected, revision=_REVISION) == (
        expected,
        expected.name,
    )
    with pytest.raises(software.ReplaySoftwareVerificationError, match="does not bind"):
        software._expected_output(
            tmp_path, Path("reports/generated/wrong.json"), revision=_REVISION
        )
    with pytest.raises(software.ReplaySoftwareVerificationError, match="does not bind"):
        software._expected_output(tmp_path, tmp_path / "wrong.json", revision=_REVISION)


def test_require_absent_translates_unavailable_path_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "lstat", lambda _path: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(software.ReplaySoftwareVerificationError, match="state is unavailable"):
        software._require_absent(tmp_path / "output")


def test_exclusive_publish_refuses_overwrite_and_cleans_short_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.write_bytes(b"original")
    with pytest.raises(software.ReplaySoftwareVerificationError, match="already exists"):
        software._publish_exclusive(existing, b"replacement")
    assert existing.read_bytes() == b"original"

    target = tmp_path / "short-write"
    monkeypatch.setattr(software.os, "write", lambda _descriptor, _value: 0)
    with pytest.raises(software.ReplaySoftwareVerificationError, match="published durably"):
        software._publish_exclusive(target, b"value")
    assert not target.exists()


def test_software_verification_digest_hashes_canonical_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification = cast(ReplaySoftwareVerificationV1, object())
    monkeypatch.setattr(software, "canonical_json_bytes", lambda _value: b"canonical\n")
    assert (
        software.software_verification_sha256(verification)
        == hashlib.sha256(b"canonical\n").hexdigest()
    )


def test_built_wheel_smoke_entrypoint_success_failure_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(software, "_run_built_wheel_smoke", lambda **_arguments: b"proof\n")
    assert software._built_wheel_smoke_main() == 0
    assert capfd.readouterr().out == "proof\n"

    def fail(**_arguments: object) -> bytes:
        raise ValueError("private failure")

    monkeypatch.setattr(software, "_run_built_wheel_smoke", fail)
    assert software._built_wheel_smoke_main() == 2
    assert "failed closed" in capfd.readouterr().err

    assert software._main(("unknown",)) == 2
    assert "invalid" in capfd.readouterr().err
    monkeypatch.setattr(software, "_built_wheel_smoke_main", lambda: 7)
    assert software._main(("built-wheel-smoke",)) == 7
