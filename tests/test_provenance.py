from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

import pytest

from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    ProvenanceError,
    collect_runtime_environment,
    discover_clean_source,
    logical_reproduction_command,
    verify_locked_execution,
)


def _snapshot() -> CleanSourceSnapshot:
    return CleanSourceSnapshot(
        source_root=Path("/source"),
        git_revision="a" * 40,
        git_dir=Path("/source/.git"),
        git_common_dir=Path("/source/.git"),
        lockfile_sha256="b" * 64,
        package_version="0.1.0",
        manifest_relative_path="examples/manifests/analytic-bias-v1alpha1.json",
    )


def test_logical_command_is_source_relative_and_deterministic() -> None:
    assert logical_reproduction_command(
        snapshot=_snapshot(),
        experiment="analytic-camera-x-bias",
        manifest_sha256="c" * 64,
    ) == (
        "ffb",
        "run",
        "examples/manifests/analytic-bias-v1alpha1.json",
        "--output-dir",
        "reports/generated/analytic-camera-x-bias-cccccccccccc",
    )


def test_dirty_checkout_is_rejected_before_execution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "test@example.invalid"),
        ("git", "config", "user.name", "Test"),
        ("git", "add", "manifest.json", "uv.lock"),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=source, check=True)

    snapshot = discover_clean_source(manifest)
    assert snapshot.manifest_relative_path == "manifest.json"

    manifest.write_text('{"dirty":true}\n', encoding="utf-8")
    with pytest.raises(ProvenanceError, match="content differs from HEAD"):
        discover_clean_source(manifest)


def test_assume_unchanged_cannot_hide_modified_tracked_content(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "test@example.invalid"),
        ("git", "config", "user.name", "Test"),
        ("git", "add", "manifest.json", "uv.lock"),
        ("git", "commit", "-qm", "fixture"),
        ("git", "update-index", "--assume-unchanged", "manifest.json"),
    ):
        subprocess.run(command, cwd=source, check=True)
    manifest.write_text('{"hidden":"change"}\n', encoding="utf-8")
    assert (
        subprocess.run(
            ("git", "status", "--porcelain=v1"),
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )

    with pytest.raises(ProvenanceError, match="Git index uses"):
        discover_clean_source(manifest)


def test_skip_worktree_index_flag_is_rejected_even_before_content_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "test@example.invalid"),
        ("git", "config", "user.name", "Test"),
        ("git", "add", "manifest.json", "uv.lock"),
        ("git", "commit", "-qm", "fixture"),
        ("git", "update-index", "--skip-worktree", "manifest.json"),
    ):
        subprocess.run(command, cwd=source, check=True)

    with pytest.raises(ProvenanceError, match="Git index uses"):
        discover_clean_source(manifest)


def test_manifest_reproduction_path_must_be_safe_posix_segments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "bad name.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "test@example.invalid"),
        ("git", "config", "user.name", "Test"),
        ("git", "add", "--", "bad name.json", "uv.lock"),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=source, check=True)

    with pytest.raises(ProvenanceError, match="conservative POSIX"):
        discover_clean_source(manifest)


def test_runtime_environment_has_no_hostname_and_positive_resources() -> None:
    environment = collect_runtime_environment()

    assert environment.logical_cpu_count > 0
    assert environment.memory_bytes > 0
    assert environment.cpu_model


def _real_snapshot() -> CleanSourceSnapshot:
    source_root = Path.cwd().resolve()
    lockfile = source_root / "uv.lock"
    git_dir = (source_root / ".git").resolve()
    return CleanSourceSnapshot(
        source_root=source_root,
        git_revision="a" * 40,
        git_dir=git_dir,
        git_common_dir=git_dir,
        lockfile_sha256=hashlib.sha256(lockfile.read_bytes()).hexdigest(),
        package_version="0.1.0",
        manifest_relative_path="examples/manifests/analytic-bias-v1alpha1.json",
    )


def test_locked_execution_matches_source_environment_and_runtime_closure() -> None:
    verify_locked_execution(_real_snapshot())


def test_execution_package_must_come_from_snapshot_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "different-checkout"
    (source / "src" / "fusion_fault_bench").mkdir(parents=True)
    (source / ".venv").mkdir()
    (source / "uv.lock").write_text(
        "version = 1\npackage = []\n",
        encoding="utf-8",
    )
    (source / ".python-version").write_text("3.12.13\n", encoding="utf-8")
    monkeypatch.chdir(source)
    snapshot = CleanSourceSnapshot(
        source_root=source,
        git_revision="a" * 40,
        git_dir=source / ".git",
        git_common_dir=source / ".git",
        lockfile_sha256=hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest(),
        package_version="0.1.0",
        manifest_relative_path="manifest.json",
    )

    with pytest.raises(ProvenanceError, match="does not come from"):
        verify_locked_execution(snapshot)


def test_locked_execution_rejects_installed_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_version = importlib.metadata.version

    def drift_numpy(distribution: str) -> str:
        if distribution == "numpy":
            return "0.0.0"
        return installed_version(distribution)

    monkeypatch.setattr(importlib.metadata, "version", drift_numpy)
    with pytest.raises(ProvenanceError, match="does not match locked version"):
        verify_locked_execution(_real_snapshot())


def test_locked_execution_rejects_snapshot_package_version_drift() -> None:
    snapshot = _real_snapshot()
    changed = CleanSourceSnapshot(
        source_root=snapshot.source_root,
        git_revision=snapshot.git_revision,
        git_dir=snapshot.git_dir,
        git_common_dir=snapshot.git_common_dir,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version="9.9.9",
        manifest_relative_path=snapshot.manifest_relative_path,
    )

    with pytest.raises(ProvenanceError, match="versions do not agree"):
        verify_locked_execution(changed)
