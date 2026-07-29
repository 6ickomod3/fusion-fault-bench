from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import fusion_fault_bench.artifacts as artifact_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "m1-analytic-v0.1.0"
RELEASE_ROOT = REPOSITORY_ROOT / "reports" / "releases" / RELEASE_ID
RELEASE_TOOL = REPOSITORY_ROOT / "tools" / "m1_release.py"
sys.path.insert(0, str(REPOSITORY_ROOT))
m1_release = importlib.import_module("tools.m1_release")


def _run_tool(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RELEASE_TOOL), *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _load_index(candidate: Path) -> dict[str, Any]:
    return json.loads((candidate / "release-index.json").read_bytes())


def _write_index(candidate: Path, index: dict[str, Any]) -> None:
    value = json.dumps(
        index,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (candidate / "release-index.json").write_bytes(value + b"\n")


def test_curated_m1_release_validates() -> None:
    result = _run_tool("validate", str(RELEASE_ROOT))

    assert result.returncode == 0, result.stderr
    assert f"release_id={RELEASE_ID}" in result.stdout
    assert "source_revision=524c8f70ece3eca2e61796165b23ffe51baadfbc" in result.stdout


def test_release_validator_rejects_mutated_indexed_record(tmp_path: Path) -> None:
    candidate = tmp_path / RELEASE_ID
    shutil.copytree(RELEASE_ROOT, candidate)
    aggregate = next(candidate.glob("records/*/aggregate-metrics.ndjson"))
    aggregate.write_bytes(aggregate.read_bytes() + b"\n")

    result = _run_tool("validate", str(candidate))

    assert result.returncode == 2
    assert "hash or length mismatch" in result.stderr


def test_release_validator_rejects_unexpected_file(tmp_path: Path) -> None:
    candidate = tmp_path / RELEASE_ID
    shutil.copytree(RELEASE_ROOT, candidate)
    (candidate / "unindexed.txt").write_text("not indexed\n", encoding="utf-8")

    result = _run_tool("validate", str(candidate))

    assert result.returncode == 2
    assert "unexpected files" in result.stderr


def test_release_validator_rejects_rewritten_common_provenance(tmp_path: Path) -> None:
    candidate = tmp_path / RELEASE_ID
    shutil.copytree(RELEASE_ROOT, candidate)
    index = _load_index(candidate)
    index["lockfile_sha256"] = "0" * 64
    _write_index(candidate, index)

    result = _run_tool("validate", str(candidate))

    assert result.returncode == 2
    assert "common provenance is invalid" in result.stderr


def test_release_validator_recomputes_figure_from_aggregates(tmp_path: Path) -> None:
    candidate = tmp_path / RELEASE_ID
    shutil.copytree(RELEASE_ROOT, candidate)
    index = _load_index(candidate)
    figure_entry = index["figures"][0]
    figure = candidate / figure_entry["path"]
    mutated = figure.read_bytes().replace(b'cx="105.000"', b'cx="106.000"', 1)
    assert mutated != figure.read_bytes()
    figure.write_bytes(mutated)
    figure_entry["byte_length"] = len(mutated)
    figure_entry["sha256"] = hashlib.sha256(mutated).hexdigest()
    _write_index(candidate, index)

    result = _run_tool("validate", str(candidate))

    assert result.returncode == 2
    assert "figure bytes disagree with curated aggregates" in result.stderr


def test_release_validator_rejects_rewritten_logical_command(tmp_path: Path) -> None:
    candidate = tmp_path / RELEASE_ID
    shutil.copytree(RELEASE_ROOT, candidate)
    index = _load_index(candidate)
    index["experiments"][0]["logical_command"][2] = "examples/manifests/other.json"
    _write_index(candidate, index)

    result = _run_tool("validate", str(candidate))

    assert result.returncode == 2
    assert "not cross-bound" in result.stderr


def test_release_validator_rejects_rehashed_document_edit(tmp_path: Path) -> None:
    candidate = tmp_path / RELEASE_ID
    shutil.copytree(RELEASE_ROOT, candidate)
    index = _load_index(candidate)
    document_entry = index["documents"][0]
    document = candidate / document_entry["path"]
    mutated = document.read_bytes() + b"\n"
    document.write_bytes(mutated)
    document_entry["byte_length"] = len(mutated)
    document_entry["sha256"] = hashlib.sha256(mutated).hexdigest()
    _write_index(candidate, index)

    result = _run_tool("validate", str(candidate))

    assert result.returncode == 2
    assert "official release document changed" in result.stderr


def test_release_build_refuses_overwrite_before_loading_sources(tmp_path: Path) -> None:
    existing = tmp_path / RELEASE_ID
    existing.mkdir()

    result = _run_tool(
        "build",
        "--primary-root",
        str(tmp_path / "missing-primary"),
        "--repeat-root",
        str(tmp_path / "missing-repeat"),
        "--withheld-root",
        str(tmp_path / "missing-withheld"),
        "--documents-root",
        str(tmp_path / "missing-documents"),
        "--output-dir",
        str(existing),
    )

    assert result.returncode == 2
    assert "already exists" in result.stderr


def test_release_publication_does_not_replace_raced_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    staging = parent / ".m1-release-test"
    staging.mkdir()
    (staging / "sentinel").write_text("source\n", encoding="utf-8")
    original_publish = artifact_module._atomic_rename_no_replace_at

    def create_destination_then_publish(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, dir_fd=destination_dir_fd)
        original_publish(
            source_dir_fd,
            source_name,
            destination_dir_fd,
            destination_name,
        )

    monkeypatch.setattr(
        artifact_module,
        "_atomic_rename_no_replace_at",
        create_destination_then_publish,
    )
    with pytest.raises(FileExistsError):
        m1_release.publish_directory_no_replace(staging, parent / RELEASE_ID)

    assert staging.is_dir()
    assert (staging / "sentinel").read_text(encoding="utf-8") == "source\n"
    assert (parent / RELEASE_ID).is_dir()
    assert not any((parent / RELEASE_ID).iterdir())


def test_release_publication_requires_sibling_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    other_parent = tmp_path / "other"
    other_parent.mkdir()

    with pytest.raises(
        artifact_module.ArtifactValidationError,
        match="requires source and destination siblings",
    ):
        m1_release.publish_directory_no_replace(source, other_parent / RELEASE_ID)


def test_release_publication_rejects_non_directory_source(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(
        artifact_module.ArtifactValidationError,
        match="source must be a real directory",
    ):
        m1_release.publish_directory_no_replace(source, tmp_path / RELEASE_ID)


def test_release_publication_preserves_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / RELEASE_ID
    destination.mkdir()
    (destination / "sentinel").write_text("destination\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination already exists"):
        m1_release.publish_directory_no_replace(source, destination)

    assert source.is_dir()
    assert (destination / "sentinel").read_text(encoding="utf-8") == "destination\n"
