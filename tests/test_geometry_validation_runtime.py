from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    GEOMETRY_MANIFEST_SHA256,
    GeometryValidationManifestV1,
)
from fusion_fault_bench.geometry_validation import (
    GeometryValidationComputationError,
    build_covariance_validation,
    build_synthetic_geometry_validation,
)


def _source_root() -> Path:
    return Path(__file__).parents[1]


def _manifest() -> GeometryValidationManifestV1:
    path = _source_root() / "examples" / "validation" / "m2-geometry-v1.json"
    return GeometryValidationManifestV1.model_validate_json(path.read_bytes())


def test_frozen_manifest_builds_passing_deterministic_synthetic_evidence() -> None:
    manifest = _manifest()

    first = build_synthetic_geometry_validation(manifest, source_root=_source_root())
    second = build_synthetic_geometry_validation(manifest, source_root=_source_root())

    assert sha256_digest(manifest) == GEOMETRY_MANIFEST_SHA256
    assert first == second
    assert first.all_checks_passed


def test_frozen_manifest_builds_passing_deterministic_covariance_evidence() -> None:
    manifest = _manifest()

    first = build_covariance_validation(manifest, source_root=_source_root())
    second = build_covariance_validation(manifest, source_root=_source_root())

    assert first == second
    assert first.all_checks_passed
    assert tuple(entry.entry for entry in first.covariance_entries) == (
        "xx",
        "xy",
        "yy",
    )
    assert first.covariance_entry_max_gate_ratio < 1.0


def test_fixture_identity_failure_is_sanitized(tmp_path: Path) -> None:
    manifest = _manifest()
    destination = tmp_path / manifest.synthetic_fixture.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"not-the-frozen-fixture\n")

    with pytest.raises(
        GeometryValidationComputationError,
        match="synthetic geometry fixture identity is invalid",
    ) as captured:
        build_synthetic_geometry_validation(manifest, source_root=tmp_path)

    assert str(tmp_path) not in str(captured.value)


def test_runtime_does_not_depend_on_dataset_adapter() -> None:
    source_path = _source_root() / "src" / "fusion_fault_bench" / "geometry_validation.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))

    assert not any(
        name.startswith("fusion_fault_bench.adapters")
        or name.startswith("fusion_fault_bench.reference.nuscenes_projection")
        for name in imports
    )
