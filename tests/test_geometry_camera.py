from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fusion_fault_bench.geometry import (
    SE3,
    FrameId,
    PinholeCamera,
    Projection,
    box_corners,
    devkit_box_any_visible,
    project_point,
)

FIXTURE_PATH = Path("tests/fixtures/m2_geometry_reference_v1.json")
MANIFEST_PATH = Path("examples/validation/m2-geometry-v1.json")


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _fixture_transform(record: dict[str, Any]) -> SE3:
    return SE3.from_quaternion_wxyz(
        target_frame=FrameId.parse(record["target_frame"]),
        source_frame=FrameId.parse(record["source_frame"]),
        translation_m=record["translation_m"],
        quaternion_wxyz=record["quaternion_wxyz"],
    )


def _fixture_camera(fixture: dict[str, Any]) -> PinholeCamera:
    model = fixture["camera_model"]
    return PinholeCamera(
        intrinsic=np.asarray(model["intrinsic"], dtype=np.float64),
        width_px=model["width_px"],
        height_px=model["height_px"],
    )


def test_fixture_global_to_camera_projection_matches_independent_goldens() -> None:
    fixture = _mapping(FIXTURE_PATH)
    manifest = _mapping(MANIFEST_PATH)["property_validation"]
    poses = fixture["poses"]
    global_from_camera = _fixture_transform(poses["global_from_camera_ego"]).compose(
        _fixture_transform(poses["camera_ego_from_camera"])
    )
    camera_from_global = global_from_camera.inverse()
    camera = _fixture_camera(fixture)

    for record in fixture["projection_points"]:
        point_camera = camera_from_global.apply(
            record["point_global_m"],
            source_frame=camera_from_global.source_frame,
        )
        assert np.max(np.abs(point_camera - record["expected_point_camera_m"])) <= 1e-12
        projection = project_point(point_camera, camera)
        assert projection.valid is record["expected_projection_valid"]
        assert (
            abs(projection.depth_m - record["expected_depth_m"])
            <= manifest["synthetic_depth_max_abs_tolerance_m"]
        )
        if record["expected_uv_px"] is None:
            assert projection.uv_px is None
            inside = False
        else:
            assert projection.uv_px is not None
            assert (
                np.max(np.abs(projection.uv_px - record["expected_uv_px"]))
                <= manifest["synthetic_projection_max_abs_tolerance_px"]
            )
            inside = camera.contains_strict(
                projection.uv_px,
                depth_m=projection.depth_m,
                minimum_depth_m=0.1,
            )
        assert inside is record["expected_strict_image_inside"]


def test_reversing_each_nuscenes_pose_direction_fails_the_projection_oracle() -> None:
    fixture = _mapping(FIXTURE_PATH)
    poses = fixture["poses"]
    global_from_camera_ego = _fixture_transform(poses["global_from_camera_ego"])
    camera_ego_from_camera = _fixture_transform(poses["camera_ego_from_camera"])
    first_point = fixture["projection_points"][0]
    expected = np.asarray(first_point["expected_point_camera_m"], dtype=np.float64)

    reversed_pose = global_from_camera_ego.inverse()
    mislabeled_global_from_camera_ego = SE3(
        target_frame=global_from_camera_ego.target_frame,
        source_frame=global_from_camera_ego.source_frame,
        rotation=reversed_pose.rotation,
        translation_m=reversed_pose.translation_m,
    )
    reversed_extrinsic = camera_ego_from_camera.inverse()
    mislabeled_camera_ego_from_camera = SE3(
        target_frame=camera_ego_from_camera.target_frame,
        source_frame=camera_ego_from_camera.source_frame,
        rotation=reversed_extrinsic.rotation,
        translation_m=reversed_extrinsic.translation_m,
    )

    for wrong_global_from_camera in (
        mislabeled_global_from_camera_ego.compose(camera_ego_from_camera),
        global_from_camera_ego.compose(mislabeled_camera_ego_from_camera),
    ):
        wrong_camera_from_global = wrong_global_from_camera.inverse()
        wrong_point = wrong_camera_from_global.apply(
            first_point["point_global_m"],
            source_frame=wrong_camera_from_global.source_frame,
        )
        assert np.max(np.abs(wrong_point - expected)) > 0.1


def test_official_order_box_corners_match_fixture() -> None:
    fixture = _mapping(FIXTURE_PATH)
    manifest = _mapping(MANIFEST_PATH)["property_validation"]
    record = fixture["box_corner_fixture"]

    corners = box_corners(
        center_m=record["center_global_m"],
        size_width_length_height_m=record["size_width_length_height_m"],
        orientation_wxyz=record["orientation_global_wxyz"],
    )

    assert corners.shape == (3, 8)
    assert (
        np.max(np.abs(corners - record["expected_corners_global_m_columns"]))
        <= manifest["box_corner_max_abs_tolerance_m"]
    )
    assert not corners.flags.writeable


def test_projection_and_center_bounds_are_strict() -> None:
    camera = PinholeCamera(
        intrinsic=np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        width_px=100,
        height_px=100,
    )

    for depth in (0.0, -1.0):
        result = project_point((1.0, 2.0, depth), camera)
        assert not result.valid
        assert result.uv_px is None

    assert not camera.contains_strict((0.0, 50.0), depth_m=2.0, minimum_depth_m=0.1)
    assert not camera.contains_strict((100.0, 50.0), depth_m=2.0, minimum_depth_m=0.1)
    assert not camera.contains_strict((50.0, 0.0), depth_m=2.0, minimum_depth_m=0.1)
    assert not camera.contains_strict((50.0, 100.0), depth_m=2.0, minimum_depth_m=0.1)
    assert not camera.contains_strict((50.0, 50.0), depth_m=0.1, minimum_depth_m=0.1)
    assert camera.contains_strict(
        (np.nextafter(0.0, 1.0), 50.0),
        depth_m=np.nextafter(0.1, 1.0),
        minimum_depth_m=0.1,
    )


def test_center_visibility_and_devkit_any_are_not_interchangeable() -> None:
    camera = PinholeCamera(
        intrinsic=np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        width_px=100,
        height_px=100,
    )
    center_inside = project_point((0.0, 0.0, 5.0), camera)
    assert center_inside.uv_px is not None
    assert camera.contains_strict(
        center_inside.uv_px,
        depth_m=center_inside.depth_m,
        minimum_depth_m=0.1,
    )

    partly_behind = np.asarray(
        [
            [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0, 4.0, -0.1, -0.1, -0.1, -0.1],
        ],
        dtype=np.float64,
    )
    assert not devkit_box_any_visible(partly_behind, camera)

    center_outside = project_point((3.0, 0.0, 5.0), camera)
    assert center_outside.uv_px is not None
    assert not camera.contains_strict(
        center_outside.uv_px,
        depth_m=center_outside.depth_m,
        minimum_depth_m=0.1,
    )
    one_corner_inside = np.asarray(
        [
            [2.0, 4.0, 4.0, 4.0, 2.0, 4.0, 4.0, 4.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0],
        ],
        dtype=np.float64,
    )
    assert devkit_box_any_visible(one_corner_inside, camera)


def test_devkit_any_depth_thresholds_are_strict() -> None:
    camera = PinholeCamera(
        intrinsic=np.asarray([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]),
        width_px=10,
        height_px=10,
    )
    corners = np.zeros((3, 8), dtype=np.float64)
    corners[2] = 1.0
    assert not devkit_box_any_visible(corners, camera)

    corners[2] = 1.1
    assert devkit_box_any_visible(corners, camera)

    corners[2] = 1.1
    corners[2, 7] = 0.1
    assert not devkit_box_any_visible(corners, camera)
    with pytest.raises(ValueError, match="thresholds"):
        devkit_box_any_visible(
            corners,
            camera,
            visible_corner_min_depth_m=-1.0,
        )


def test_camera_predicates_reject_nonfinite_values_and_nonfinite_projection() -> None:
    camera = PinholeCamera(
        intrinsic=np.asarray([[1e308, 0.0, 0.0], [0.0, 1e308, 0.0], [0.0, 0.0, 1.0]]),
        width_px=10,
        height_px=10,
    )
    with pytest.raises(ValueError, match="projection produced"):
        project_point((1e308, 1e308, 1.0), camera)
    with pytest.raises(ValueError, match="depth_m"):
        camera.contains_strict((1.0, 1.0), depth_m=float("nan"), minimum_depth_m=0.1)
    with pytest.raises(ValueError, match="minimum_depth"):
        camera.contains_strict((1.0, 1.0), depth_m=1.0, minimum_depth_m=-0.1)


@pytest.mark.parametrize(
    ("intrinsic", "width", "height"),
    [
        (np.eye(2), 100, 100),
        (np.diag([-1.0, 1.0, 1.0]), 100, 100),
        (np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.1, 1.0]]), 100, 100),
        (np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 0, 100),
        (np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 100, True),
    ],
)
def test_camera_rejects_invalid_calibration(intrinsic, width, height) -> None:
    with pytest.raises(ValueError):
        PinholeCamera(intrinsic=intrinsic, width_px=width, height_px=height)


def test_box_rejects_invalid_size_and_projection_rejects_nonfinite_points() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        box_corners(
            center_m=(0.0, 0.0, 0.0),
            size_width_length_height_m=(1.0, 0.0, 1.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    camera = PinholeCamera(intrinsic=np.eye(3), width_px=10, height_px=10)
    with pytest.raises(ValueError, match="finite"):
        project_point((0.0, float("nan"), 1.0), camera)


def test_camera_storage_and_geometry_outputs_cannot_be_made_writeable() -> None:
    intrinsic = np.eye(3, dtype=np.float64)
    camera = PinholeCamera(intrinsic=intrinsic, width_px=10, height_px=10)
    intrinsic[0, 0] = 99.0
    projection = project_point((1.0, 1.0, 2.0), camera)
    assert projection.uv_px is not None
    corners = box_corners(
        center_m=(0.0, 0.0, 5.0),
        size_width_length_height_m=(1.0, 2.0, 1.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    assert camera.intrinsic[0, 0] == 1.0
    for array in (camera.intrinsic, projection.uv_px, corners):
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)

    caller_owned_pixels = np.asarray([2.0, 3.0], dtype=np.float64)
    direct = Projection(depth_m=1.0, uv_px=caller_owned_pixels)
    caller_owned_pixels[0] = 99.0
    assert direct.uv_px == pytest.approx((2.0, 3.0))
    with pytest.raises(ValueError, match="strictly positive"):
        Projection(depth_m=0.0, uv_px=np.zeros(2))
