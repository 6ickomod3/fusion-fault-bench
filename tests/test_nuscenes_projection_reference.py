from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from fusion_fault_bench.reference.nuscenes_projection import (
    ScalarProjectionReferenceError,
    build_scalar_projection_diagnostic,
    render_scalar_diagnostic_svg,
)


def _write_table(root: Path, name: str, rows: list[dict[str, object]]) -> None:
    table_root = root / "v1.0-mini"
    table_root.mkdir(parents=True, exist_ok=True)
    (table_root / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def _write_reference_fixture(root: Path) -> None:
    _write_table(
        root,
        "scene",
        [
            {
                "token": "private-scene-token",
                "name": "scene-synthetic",
                "first_sample_token": "private-sample-token",
            }
        ],
    )
    _write_table(
        root,
        "sample",
        [{"token": "private-sample-token"}],
    )
    _write_table(
        root,
        "sensor",
        [{"token": "private-sensor-token", "channel": "CAM_FRONT"}],
    )
    _write_table(
        root,
        "calibrated_sensor",
        [
            {
                "token": "private-calibration-token",
                "sensor_token": "private-sensor-token",
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [
                    [10.0, 0.0, 50.0],
                    [0.0, 10.0, 50.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        ],
    )
    _write_table(
        root,
        "ego_pose",
        [
            {
                "token": "private-pose-token",
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        ],
    )
    _write_table(
        root,
        "sample_data",
        [
            {
                "token": "private-data-token",
                "sample_token": "private-sample-token",
                "calibrated_sensor_token": "private-calibration-token",
                "ego_pose_token": "private-pose-token",
                "is_key_frame": True,
                "width": 100,
                "height": 100,
            }
        ],
    )
    _write_table(
        root,
        "sample_annotation",
        [
            {
                "token": "private-annotation-token-b",
                "sample_token": "private-sample-token",
                "translation": [0.0, 0.0, -10.0],
                "size": [2.0, 2.0, 2.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "token": "private-annotation-token-a",
                "sample_token": "private-sample-token",
                "translation": [0.0, 0.0, 10.0],
                "size": [2.0, 2.0, 2.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        ],
    )


def test_scalar_reference_projects_all_annotations_without_identifiers(tmp_path: Path) -> None:
    _write_reference_fixture(tmp_path)

    diagnostic = build_scalar_projection_diagnostic(tmp_path)

    assert diagnostic.annotation_count == 2
    assert diagnostic.finite_positive_depth_center_count == 1
    assert diagnostic.annotations[0].center_camera_m == (0.0, 0.0, 10.0)
    assert diagnostic.annotations[0].center_uv_px == (50.0, 50.0)
    assert diagnostic.annotations[0].center_strict_image_inside
    assert diagnostic.annotations[0].box_any_visible
    assert diagnostic.annotations[1].center_uv_px is None
    assert not diagnostic.annotations[1].center_projection_valid

    serialized = repr(diagnostic)
    assert "private-" not in serialized
    assert str(tmp_path) not in serialized


def test_scalar_reference_svg_is_token_free(tmp_path: Path) -> None:
    _write_reference_fixture(tmp_path)

    svg = render_scalar_diagnostic_svg(build_scalar_projection_diagnostic(tmp_path))

    assert svg.startswith('<?xml version="1.0"')
    assert "<line " in svg
    assert "<circle " in svg
    assert "private-" not in svg
    assert str(tmp_path) not in svg


def test_scalar_reference_errors_are_sanitized(tmp_path: Path) -> None:
    with pytest.raises(
        ScalarProjectionReferenceError,
        match="local scalar reference could not load required metadata",
    ) as captured:
        build_scalar_projection_diagnostic(tmp_path)

    assert str(tmp_path) not in str(captured.value)


def test_scalar_reference_has_no_production_imports() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "fusion_fault_bench"
        / "reference"
        / "nuscenes_projection.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))

    assert not any(
        name.startswith("fusion_fault_bench.geometry")
        or name.startswith("fusion_fault_bench.adapters")
        for name in imports
    )
