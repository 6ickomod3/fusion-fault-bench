from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fusion_fault_bench.geometry import (
    SE3,
    BearingDepthCovariance,
    EligibilityTransform,
    FrameId,
    PropagatedCovariance,
    bearing_depth_jacobian_camera,
    bearing_depth_jacobian_reference_ego_bev,
    bearing_depth_point_camera,
    bearing_depth_point_reference_ego_bev,
    propagate_bearing_depth_covariance,
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


def _reference_ego_from_camera() -> EligibilityTransform:
    fixture = _mapping(FIXTURE_PATH)
    poses = fixture["poses"]
    global_from_camera_ego = _fixture_transform(poses["global_from_camera_ego"])
    camera_ego_from_camera = _fixture_transform(poses["camera_ego_from_camera"])
    global_from_reference_ego = _fixture_transform(poses["global_from_reference_ego"])
    return EligibilityTransform(
        global_from_reference_ego.inverse().compose(
            global_from_camera_ego.compose(camera_ego_from_camera)
        )
    )


def _covariance_from_manifest(role: str) -> BearingDepthCovariance:
    spec = _mapping(MANIFEST_PATH)["bearing_depth_covariance_validation"]
    key = "actual_sampling_covariance" if role == "actual" else "reported_estimator_covariance"
    record = spec[key]
    return BearingDepthCovariance(
        role=record["role"],
        matrix=np.asarray(record["matrix_rad2_rad_m_m2"], dtype=np.float64),
    )


def test_camera_bearing_depth_map_and_jacobian() -> None:
    point = bearing_depth_point_camera(
        bearing_rad=0.0,
        optical_depth_m=10.0,
        camera_vertical_coordinate_m=0.5,
    )
    jacobian = bearing_depth_jacobian_camera(
        bearing_rad=0.0,
        optical_depth_m=10.0,
    )

    assert point == pytest.approx((0.0, 0.5, 10.0))
    assert jacobian == pytest.approx(np.asarray([[10.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))
    assert not point.flags.writeable
    assert not jacobian.flags.writeable


def test_full_reference_ego_jacobian_matches_independent_finite_difference() -> None:
    spec = _mapping(MANIFEST_PATH)["bearing_depth_covariance_validation"]
    finite_difference = spec["finite_difference"]
    bearing, depth = spec["monte_carlo"]["mean_bearing_rad_depth_m"]
    vertical = spec["camera_vertical_coordinate_m"]
    transform = _reference_ego_from_camera()

    analytic = bearing_depth_jacobian_reference_ego_bev(
        bearing_rad=bearing,
        optical_depth_m=depth,
        reference_ego_from_camera=transform,
    )

    def independent_xy(beta: float, optical_depth: float) -> np.ndarray:
        point_camera = np.asarray(
            [optical_depth * math.tan(beta), vertical, optical_depth],
            dtype=np.float64,
        )
        point_ego = transform.transform.rotation @ point_camera + transform.transform.translation_m
        return point_ego[:2]

    bearing_step = finite_difference["bearing_step_rad"]
    depth_step = finite_difference["depth_step_m"]
    finite = np.column_stack(
        (
            (
                independent_xy(bearing + bearing_step, depth)
                - independent_xy(bearing - bearing_step, depth)
            )
            / (2.0 * bearing_step),
            (
                independent_xy(bearing, depth + depth_step)
                - independent_xy(bearing, depth - depth_step)
            )
            / (2.0 * depth_step),
        )
    )
    maximum_error = float(np.max(np.abs(analytic - finite)))

    assert maximum_error <= finite_difference["max_abs_tolerance"]
    assert np.linalg.matrix_rank(analytic) == 2

    mapped = bearing_depth_point_reference_ego_bev(
        bearing_rad=bearing,
        optical_depth_m=depth,
        camera_vertical_coordinate_m=vertical,
        reference_ego_from_camera=transform,
    )
    assert mapped == pytest.approx(independent_xy(bearing, depth))
    assert not mapped.flags.writeable


def test_actual_and_reported_covariances_remain_separate_and_full() -> None:
    spec = _mapping(MANIFEST_PATH)["bearing_depth_covariance_validation"]
    bearing, depth = spec["monte_carlo"]["mean_bearing_rad_depth_m"]
    transform = _reference_ego_from_camera()
    actual = propagate_bearing_depth_covariance(
        bearing_rad=bearing,
        optical_depth_m=depth,
        covariance=_covariance_from_manifest("actual"),
        reference_ego_from_camera=transform,
    )
    reported = propagate_bearing_depth_covariance(
        bearing_rad=bearing,
        optical_depth_m=depth,
        covariance=_covariance_from_manifest("reported"),
        reference_ego_from_camera=transform,
    )

    assert actual.role == "actual"
    assert reported.role == "reported"
    assert actual.matrix_xy_m2.shape == (2, 2)
    assert actual.matrix_xy_m2[0, 1] != 0.0
    assert actual.matrix_xy_m2[0, 1] == pytest.approx(actual.matrix_xy_m2[1, 0])
    assert not np.array_equal(actual.matrix_xy_m2, reported.matrix_xy_m2)
    assert not actual.matrix_xy_m2.flags.writeable
    assert not actual.jacobian_xy.flags.writeable


def test_covariance_and_bearing_depth_outputs_cannot_be_made_writeable() -> None:
    transform = _reference_ego_from_camera()
    covariance = _covariance_from_manifest("actual")
    propagated = propagate_bearing_depth_covariance(
        bearing_rad=0.2,
        optical_depth_m=25.0,
        covariance=covariance,
        reference_ego_from_camera=transform,
    )
    outputs = (
        covariance.matrix,
        propagated.matrix_xy_m2,
        propagated.jacobian_xy,
        bearing_depth_point_camera(
            bearing_rad=0.2,
            optical_depth_m=25.0,
            camera_vertical_coordinate_m=0.5,
        ),
        bearing_depth_jacobian_camera(
            bearing_rad=0.2,
            optical_depth_m=25.0,
        ),
    )

    for array in outputs:
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)

    with pytest.raises(ValueError, match="symmetric"):
        PropagatedCovariance(
            role="actual",
            matrix_xy_m2=np.asarray([[1.0, 0.1], [0.0, 1.0]]),
            jacobian_xy=np.eye(2),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        PropagatedCovariance(
            role="actual",
            matrix_xy_m2=np.asarray([[1.0, 2.0], [2.0, 1.0]]),
            jacobian_xy=np.eye(2),
        )


def test_frozen_monte_carlo_covariance_gate() -> None:
    spec = _mapping(MANIFEST_PATH)["bearing_depth_covariance_validation"]
    monte_carlo = spec["monte_carlo"]
    bearing, depth = monte_carlo["mean_bearing_rad_depth_m"]
    vertical = spec["camera_vertical_coordinate_m"]
    transform = _reference_ego_from_camera()
    actual_covariance = _covariance_from_manifest("actual")
    analytic = propagate_bearing_depth_covariance(
        bearing_rad=bearing,
        optical_depth_m=depth,
        covariance=actual_covariance,
        reference_ego_from_camera=transform,
    ).matrix_xy_m2

    generator = np.random.Generator(np.random.PCG64DXSM(monte_carlo["seed"]))
    standard_normal = generator.standard_normal(
        size=(monte_carlo["sample_count"], 2),
        dtype=np.float64,
    )
    lower = np.linalg.cholesky(actual_covariance.matrix)
    observations = standard_normal @ lower.T
    observations += np.asarray([bearing, depth], dtype=np.float64)
    assert np.all(observations[:, 1] > 0.0)

    points_camera = np.column_stack(
        (
            observations[:, 1] * np.tan(observations[:, 0]),
            np.full(monte_carlo["sample_count"], vertical, dtype=np.float64),
            observations[:, 1],
        )
    )
    points_reference = (
        transform.transform.rotation @ points_camera.T
        + transform.transform.translation_m[:, np.newaxis]
    ).T
    empirical = np.cov(points_reference[:, :2], rowvar=False, ddof=1)

    for row, column in ((0, 0), (0, 1), (1, 1)):
        absolute_error = abs(float(empirical[row, column] - analytic[row, column]))
        standard_error = math.sqrt(
            (analytic[row, column] ** 2 + analytic[row, row] * analytic[column, column])
            / (monte_carlo["sample_count"] - 1)
        )
        allowed = (
            monte_carlo["standard_error_multiplier"] * standard_error
            + monte_carlo["nonlinear_roundoff_allowance_m2"]
        )
        assert absolute_error <= allowed


@pytest.mark.parametrize(
    ("role", "matrix", "parameter_order"),
    [
        ("unknown", np.eye(2), ("bearing_rad", "optical_depth_m")),
        ("actual", np.eye(3), ("bearing_rad", "optical_depth_m")),
        ("actual", np.asarray([[1.0, 0.1], [0.2, 1.0]]), ("bearing_rad", "optical_depth_m")),
        ("actual", np.asarray([[1.0, 2.0], [2.0, 1.0]]), ("bearing_rad", "optical_depth_m")),
        (
            "reported",
            np.eye(2),
            ("optical_depth_m", "bearing_rad"),
        ),
        (
            "reported",
            np.asarray([[float("nan"), 0.0], [0.0, 1.0]]),
            ("bearing_rad", "optical_depth_m"),
        ),
    ],
)
def test_covariance_rejects_invalid_role_shape_symmetry_psd_or_units(
    role,
    matrix,
    parameter_order,
) -> None:
    with pytest.raises(ValueError):
        BearingDepthCovariance(
            role=role,
            matrix=matrix,
            parameter_order=parameter_order,
        )


@pytest.mark.parametrize(
    ("bearing", "depth", "vertical"),
    [
        (math.pi / 2.0, 1.0, 0.0),
        (-math.pi / 2.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (float("nan"), 1.0, 0.0),
        (0.0, 1.0, float("inf")),
    ],
)
def test_bearing_depth_map_rejects_invalid_observations(
    bearing: float,
    depth: float,
    vertical: float,
) -> None:
    with pytest.raises(ValueError):
        bearing_depth_point_camera(
            bearing_rad=bearing,
            optical_depth_m=depth,
            camera_vertical_coordinate_m=vertical,
        )


def test_bearing_depth_mapping_rejects_reported_transform_role() -> None:
    transform = _reference_ego_from_camera()
    from fusion_fault_bench.geometry import ReportedReconstructionTransform

    reported = ReportedReconstructionTransform(transform.transform)
    with pytest.raises(TypeError, match="EligibilityTransform"):
        bearing_depth_point_reference_ego_bev(
            bearing_rad=0.2,
            optical_depth_m=25.0,
            camera_vertical_coordinate_m=0.5,
            reference_ego_from_camera=reported,  # type: ignore[arg-type]
        )


def test_bearing_depth_mapping_rejects_wrong_named_frame_direction() -> None:
    transform = _reference_ego_from_camera()
    wrong = EligibilityTransform(transform.transform.inverse())
    with pytest.raises(ValueError, match="camera into reference ego"):
        bearing_depth_jacobian_reference_ego_bev(
            bearing_rad=0.2,
            optical_depth_m=25.0,
            reference_ego_from_camera=wrong,
        )
