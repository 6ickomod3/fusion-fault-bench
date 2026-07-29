"""Deterministic synthetic geometry and covariance validation for M2."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, FiniteFloat, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    CovarianceEntryV1,
    CovarianceValidationV1,
    GeometryValidationManifestV1,
    SyntheticGeometryValidationV1,
)
from fusion_fault_bench.geometry.camera import (
    PinholeCamera,
    box_corners,
    project_point,
)
from fusion_fault_bench.geometry.covariance import (
    BearingDepthCovariance,
    bearing_depth_jacobian_reference_ego_bev,
    bearing_depth_point_reference_ego_bev,
    propagate_bearing_depth_covariance,
)
from fusion_fault_bench.geometry.frames import FrameId
from fusion_fault_bench.geometry.roi import EligibilityTransform
from fusion_fault_bench.geometry.se3 import SE3, quaternion_wxyz_to_rotation

type Vec2 = tuple[FiniteFloat, FiniteFloat]
type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type Vec4 = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
type Mat3 = tuple[Vec3, Vec3, Vec3]
type FloatArray = npt.NDArray[np.float64]


class GeometryValidationComputationError(ValueError):
    """A frozen synthetic validation could not be evaluated safely."""


class _FixtureFrames(ContractModel):
    global_frame: str = Field(alias="global")
    camera_ego: str
    reference_ego: str
    camera: str


class _FixturePose(ContractModel):
    target_frame: str
    source_frame: str
    translation_m: Vec3
    quaternion_wxyz: Vec4


class _FixturePoses(ContractModel):
    global_from_camera_ego: _FixturePose
    camera_ego_from_camera: _FixturePose
    global_from_reference_ego: _FixturePose


class _ExpectedTransform(ContractModel):
    rotation: Mat3
    translation_m: Vec3


class _FixtureCamera(ContractModel):
    intrinsic: Mat3
    width_px: Annotated[int, Field(gt=0)]
    height_px: Annotated[int, Field(gt=0)]


class _FixtureProjectionPoint(ContractModel):
    point_id: str
    point_global_m: Vec3
    expected_point_camera_m: Vec3
    expected_depth_m: FiniteFloat
    expected_uv_px: Vec2 | None
    expected_projection_valid: bool
    expected_strict_image_inside: bool

    @model_validator(mode="after")
    def require_projection_consistency(self) -> Self:
        if self.expected_projection_valid != (self.expected_uv_px is not None):
            raise ValueError("fixture projection validity disagrees with expected pixels")
        return self


class _FixtureBox(ContractModel):
    center_global_m: Vec3
    size_width_length_height_m: Vec3
    orientation_global_wxyz: Vec4
    expected_corners_global_m_columns: tuple[
        tuple[
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
        ],
        tuple[
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
        ],
        tuple[
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
            FiniteFloat,
        ],
    ]


class _GeometryFixture(ContractModel):
    schema_id: Literal["ffb.synthetic-geometry-fixture/v1"] = Field(alias="schema")
    fixture_id: Literal["m2-nuscenes-convention-independent-v1"]
    construction: Literal["hand-computed-column-vector-rigid-transform-v1"]
    frames: _FixtureFrames
    poses: _FixturePoses
    expected_reference_ego_from_camera: _ExpectedTransform
    camera_model: _FixtureCamera
    projection_points: tuple[
        _FixtureProjectionPoint,
        _FixtureProjectionPoint,
        _FixtureProjectionPoint,
        _FixtureProjectionPoint,
        _FixtureProjectionPoint,
    ]
    box_corner_fixture: _FixtureBox


def _load_fixture(
    manifest: GeometryValidationManifestV1,
    *,
    source_root: Path,
) -> _GeometryFixture:
    fixture_path = source_root / manifest.synthetic_fixture.path
    try:
        raw = fixture_path.read_bytes()
    except OSError:
        raise GeometryValidationComputationError(
            "synthetic geometry fixture is unavailable"
        ) from None
    if hashlib.sha256(raw).hexdigest() != manifest.synthetic_fixture.file_sha256:
        raise GeometryValidationComputationError("synthetic geometry fixture identity is invalid")
    try:
        return _GeometryFixture.model_validate_json(raw)
    except ValueError:
        raise GeometryValidationComputationError(
            "synthetic geometry fixture contract is invalid"
        ) from None


def _pose_transform(pose: _FixturePose) -> SE3:
    return SE3.from_quaternion_wxyz(
        target_frame=FrameId.parse(pose.target_frame),
        source_frame=FrameId.parse(pose.source_frame),
        translation_m=pose.translation_m,
        quaternion_wxyz=pose.quaternion_wxyz,
    )


def _fixture_transforms(
    fixture: _GeometryFixture,
) -> tuple[SE3, SE3, SE3, SE3]:
    global_from_camera_ego = _pose_transform(fixture.poses.global_from_camera_ego)
    camera_ego_from_camera = _pose_transform(fixture.poses.camera_ego_from_camera)
    global_from_reference_ego = _pose_transform(fixture.poses.global_from_reference_ego)
    reference_ego_from_camera = (
        global_from_reference_ego.inverse()
        .compose(global_from_camera_ego)
        .compose(camera_ego_from_camera)
    )
    return (
        global_from_camera_ego,
        camera_ego_from_camera,
        global_from_reference_ego,
        reference_ego_from_camera,
    )


def _normalized_quaternions(draws: FloatArray) -> FloatArray:
    norms = np.linalg.norm(draws, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise GeometryValidationComputationError("property-validation quaternion draw is invalid")
    return np.asarray(draws / norms[:, np.newaxis], dtype=np.float64)


def _property_errors(
    manifest: GeometryValidationManifestV1,
) -> tuple[float, float, float, float]:
    spec = manifest.property_validation
    rng = np.random.Generator(np.random.PCG64DXSM(spec.seed))
    count = spec.transform_count
    left_quaternions = _normalized_quaternions(
        rng.standard_normal(size=(count, 4), dtype=np.float64)
    )
    right_quaternions = _normalized_quaternions(
        rng.standard_normal(size=(count, 4), dtype=np.float64)
    )
    left_translations = rng.uniform(-100.0, 100.0, size=(count, 3))
    right_translations = rng.uniform(-100.0, 100.0, size=(count, 3))
    points = rng.standard_normal(size=(count, 3), dtype=np.float64) * 25.0

    rotation_error = 0.0
    translation_error = 0.0
    point_error = 0.0
    quaternion_sign_error = 0.0
    identity = np.eye(3, dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    for index in range(count):
        namespace = f"property-{index:06d}"
        frame_a = FrameId.global_frame(log_namespace=namespace)
        frame_b = FrameId.ego(
            log_namespace=namespace,
            timestamp_qualifier="middle",
        )
        frame_c = FrameId.camera(
            channel="CAM_FRONT",
            calibration_instance=namespace,
            timestamp_qualifier="source",
        )
        left = SE3.from_quaternion_wxyz(
            target_frame=frame_a,
            source_frame=frame_b,
            translation_m=left_translations[index],
            quaternion_wxyz=left_quaternions[index],
        )
        right = SE3.from_quaternion_wxyz(
            target_frame=frame_b,
            source_frame=frame_c,
            translation_m=right_translations[index],
            quaternion_wxyz=right_quaternions[index],
        )
        composed = left.compose(right)
        composed_identity = composed.compose(composed.inverse())
        rotation_error = max(
            rotation_error,
            float(np.max(np.abs(composed_identity.rotation - identity))),
            float(np.max(np.abs(composed.rotation - left.rotation @ right.rotation))),
        )
        translation_error = max(
            translation_error,
            float(np.max(np.abs(composed_identity.translation_m - zero))),
            float(
                np.max(
                    np.abs(
                        composed.translation_m
                        - (left.rotation @ right.translation_m + left.translation_m)
                    )
                )
            ),
        )
        point = points[index]
        transformed = composed.apply(
            point,
            source_frame=composed.source_frame,
        )
        round_trip = composed.inverse().apply(
            transformed,
            source_frame=composed.target_frame,
        )
        point_error = max(
            point_error,
            float(np.max(np.abs(round_trip - point))),
        )
        positive_rotation = quaternion_wxyz_to_rotation(left_quaternions[index])
        negative_rotation = quaternion_wxyz_to_rotation(-left_quaternions[index])
        quaternion_sign_error = max(
            quaternion_sign_error,
            float(np.max(np.abs(positive_rotation - negative_rotation))),
        )
    return rotation_error, translation_error, point_error, quaternion_sign_error


def build_synthetic_geometry_validation(
    manifest: GeometryValidationManifestV1,
    *,
    source_root: Path,
) -> SyntheticGeometryValidationV1:
    """Evaluate the frozen fixture and property stream into sanitized evidence."""

    fixture = _load_fixture(manifest, source_root=source_root)
    (
        global_from_camera_ego,
        camera_ego_from_camera,
        global_from_reference_ego,
        reference_ego_from_camera,
    ) = _fixture_transforms(fixture)
    del global_from_reference_ego
    expected_transform = fixture.expected_reference_ego_from_camera
    rotation_error, translation_error, point_error, quaternion_sign_error = _property_errors(
        manifest
    )
    rotation_error = max(
        rotation_error,
        float(
            np.max(
                np.abs(
                    reference_ego_from_camera.rotation
                    - np.asarray(expected_transform.rotation, dtype=np.float64)
                )
            )
        ),
    )
    translation_error = max(
        translation_error,
        float(
            np.max(
                np.abs(
                    reference_ego_from_camera.translation_m
                    - np.asarray(expected_transform.translation_m, dtype=np.float64)
                )
            )
        ),
    )

    camera_from_global = camera_ego_from_camera.inverse().compose(global_from_camera_ego.inverse())
    camera_spec = fixture.camera_model
    camera = PinholeCamera(
        intrinsic=np.asarray(camera_spec.intrinsic, dtype=np.float64),
        width_px=camera_spec.width_px,
        height_px=camera_spec.height_px,
    )
    projection_error = 0.0
    depth_error = 0.0
    for point in fixture.projection_points:
        point_camera = camera_from_global.apply(
            point.point_global_m,
            source_frame=camera_from_global.source_frame,
        )
        expected_camera = np.asarray(point.expected_point_camera_m, dtype=np.float64)
        fixture_point_error = float(np.max(np.abs(point_camera - expected_camera)))
        if fixture_point_error > manifest.property_validation.point_round_trip_max_abs_tolerance_m:
            raise GeometryValidationComputationError(
                "synthetic transform direction disagrees with the frozen fixture"
            )
        projection = project_point(point_camera, camera)
        if projection.valid != point.expected_projection_valid:
            raise GeometryValidationComputationError(
                "synthetic projection validity disagrees with the frozen fixture"
            )
        depth_error = max(
            depth_error,
            abs(projection.depth_m - float(point.expected_depth_m)),
        )
        if point.expected_uv_px is None:
            if projection.uv_px is not None:
                raise GeometryValidationComputationError(
                    "synthetic invalid projection unexpectedly produced pixels"
                )
            inside = False
        else:
            if projection.uv_px is None:
                raise GeometryValidationComputationError(
                    "synthetic valid projection did not produce pixels"
                )
            expected_uv = np.asarray(point.expected_uv_px, dtype=np.float64)
            projection_error = max(
                projection_error,
                float(np.max(np.abs(projection.uv_px - expected_uv))),
            )
            inside = camera.contains_strict(
                projection.uv_px,
                depth_m=projection.depth_m,
                minimum_depth_m=manifest.geometry.center_roi_min_depth_m,
            )
        if inside != point.expected_strict_image_inside:
            raise GeometryValidationComputationError(
                "synthetic image-bound decision disagrees with the frozen fixture"
            )

    box = fixture.box_corner_fixture
    observed_corners = box_corners(
        center_m=box.center_global_m,
        size_width_length_height_m=box.size_width_length_height_m,
        orientation_wxyz=box.orientation_global_wxyz,
    )
    expected_corners = np.asarray(
        box.expected_corners_global_m_columns,
        dtype=np.float64,
    )
    box_error = float(np.max(np.abs(observed_corners - expected_corners)))

    spec = manifest.property_validation
    all_passed = (
        rotation_error <= spec.rotation_identity_composition_max_abs_tolerance
        and translation_error <= spec.translation_inverse_composition_max_abs_tolerance_m
        and point_error <= spec.point_round_trip_max_abs_tolerance_m
        and quaternion_sign_error <= spec.quaternion_sign_rotation_max_abs_tolerance
        and projection_error <= spec.synthetic_projection_max_abs_tolerance_px
        and depth_error <= spec.synthetic_depth_max_abs_tolerance_m
        and box_error <= spec.box_corner_max_abs_tolerance_m
    )
    return SyntheticGeometryValidationV1(
        fixture_id=fixture.fixture_id,
        fixture_file_sha256=manifest.synthetic_fixture.file_sha256,
        rotation_max_abs_error=rotation_error,
        translation_max_abs_error_m=translation_error,
        point_round_trip_max_abs_error_m=point_error,
        quaternion_sign_max_abs_error=quaternion_sign_error,
        projection_max_abs_error_px=projection_error,
        depth_max_abs_error_m=depth_error,
        box_corner_max_abs_error_m=box_error,
        all_checks_passed=all_passed,
    )


def _reference_ego_from_camera(
    manifest: GeometryValidationManifestV1,
    *,
    source_root: Path,
) -> EligibilityTransform:
    fixture = _load_fixture(manifest, source_root=source_root)
    return EligibilityTransform(_fixture_transforms(fixture)[3])


def _covariance_entry(
    *,
    name: Literal["xx", "xy", "yy"],
    row: int,
    column: int,
    empirical: FloatArray,
    analytic: FloatArray,
    sample_count: int,
    multiplier: float,
    allowance_m2: float,
) -> CovarianceEntryV1:
    absolute_error = abs(float(empirical[row, column] - analytic[row, column]))
    sampling_variance = (
        analytic[row, column] ** 2 + analytic[row, row] * analytic[column, column]
    ) / (sample_count - 1)
    allowed_error = multiplier * math.sqrt(float(sampling_variance)) + allowance_m2
    ratio = absolute_error / allowed_error
    return CovarianceEntryV1(
        entry=name,
        absolute_error_m2=absolute_error,
        allowed_error_m2=allowed_error,
        gate_ratio=ratio,
        passed=ratio <= 1.0,
    )


def build_covariance_validation(
    manifest: GeometryValidationManifestV1,
    *,
    source_root: Path,
) -> CovarianceValidationV1:
    """Run the frozen finite-difference and nonlinear Monte Carlo gates."""

    spec = manifest.bearing_depth_covariance_validation
    transform = _reference_ego_from_camera(manifest, source_root=source_root)
    bearing_rad, optical_depth_m = spec.monte_carlo.mean_bearing_rad_depth_m
    vertical_m = spec.camera_vertical_coordinate_m

    analytic_jacobian = bearing_depth_jacobian_reference_ego_bev(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
        reference_ego_from_camera=transform,
    )
    beta_step = spec.finite_difference.bearing_step_rad
    depth_step = spec.finite_difference.depth_step_m

    def point(beta: float, depth: float) -> FloatArray:
        return bearing_depth_point_reference_ego_bev(
            bearing_rad=beta,
            optical_depth_m=depth,
            camera_vertical_coordinate_m=vertical_m,
            reference_ego_from_camera=transform,
        )

    finite_difference = np.column_stack(
        (
            (
                point(bearing_rad + beta_step, optical_depth_m)
                - point(bearing_rad - beta_step, optical_depth_m)
            )
            / (2.0 * beta_step),
            (
                point(bearing_rad, optical_depth_m + depth_step)
                - point(bearing_rad, optical_depth_m - depth_step)
            )
            / (2.0 * depth_step),
        )
    )
    finite_difference_error = float(np.max(np.abs(finite_difference - analytic_jacobian)))

    actual_spec = spec.actual_sampling_covariance
    reported_spec = spec.reported_estimator_covariance
    actual = BearingDepthCovariance(
        role=actual_spec.role,
        matrix=np.asarray(actual_spec.matrix_rad2_rad_m_m2, dtype=np.float64),
    )
    reported = BearingDepthCovariance(
        role=reported_spec.role,
        matrix=np.asarray(reported_spec.matrix_rad2_rad_m_m2, dtype=np.float64),
    )
    actual_propagated = propagate_bearing_depth_covariance(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
        covariance=actual,
        reference_ego_from_camera=transform,
    )
    reported_propagated = propagate_bearing_depth_covariance(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
        covariance=reported,
        reference_ego_from_camera=transform,
    )

    monte_carlo = spec.monte_carlo
    rng = np.random.Generator(np.random.PCG64DXSM(monte_carlo.seed))
    standard = rng.standard_normal(
        size=(monte_carlo.sample_count, 2),
        dtype=np.float64,
    )
    lower = np.linalg.cholesky(actual.matrix)
    mean = np.asarray(monte_carlo.mean_bearing_rad_depth_m, dtype=np.float64)
    samples = mean + (lower @ standard.T).T
    if np.any(samples[:, 1] <= 0.0):
        raise GeometryValidationComputationError(
            "covariance validation sampled a non-positive depth"
        )
    camera_points = np.vstack(
        (
            samples[:, 1] * np.tan(samples[:, 0]),
            np.full(monte_carlo.sample_count, vertical_m, dtype=np.float64),
            samples[:, 1],
        )
    )
    ego_points = transform.transform.apply(
        camera_points,
        source_frame=transform.transform.source_frame,
    )
    empirical = np.asarray(
        np.cov(ego_points[:2].T, rowvar=False, ddof=monte_carlo.sample_covariance_ddof),
        dtype=np.float64,
    )
    analytic = actual_propagated.matrix_xy_m2
    entries = (
        _covariance_entry(
            name="xx",
            row=0,
            column=0,
            empirical=empirical,
            analytic=analytic,
            sample_count=monte_carlo.sample_count,
            multiplier=monte_carlo.standard_error_multiplier,
            allowance_m2=monte_carlo.nonlinear_roundoff_allowance_m2,
        ),
        _covariance_entry(
            name="xy",
            row=0,
            column=1,
            empirical=empirical,
            analytic=analytic,
            sample_count=monte_carlo.sample_count,
            multiplier=monte_carlo.standard_error_multiplier,
            allowance_m2=monte_carlo.nonlinear_roundoff_allowance_m2,
        ),
        _covariance_entry(
            name="yy",
            row=1,
            column=1,
            empirical=empirical,
            analytic=analytic,
            sample_count=monte_carlo.sample_count,
            multiplier=monte_carlo.standard_error_multiplier,
            allowance_m2=monte_carlo.nonlinear_roundoff_allowance_m2,
        ),
    )
    role_separation = (
        actual_propagated.role == "actual"
        and reported_propagated.role == "reported"
        and not np.array_equal(
            actual_propagated.matrix_xy_m2,
            reported_propagated.matrix_xy_m2,
        )
    )
    actual_passed = all(entry.passed for entry in entries)
    all_passed = (
        finite_difference_error <= spec.finite_difference.max_abs_tolerance
        and actual_passed
        and role_separation
    )
    return CovarianceValidationV1(
        finite_difference_max_abs_error=finite_difference_error,
        monte_carlo_sample_count=monte_carlo.sample_count,
        covariance_entries=entries,
        covariance_entry_max_abs_error_m2=max(entry.absolute_error_m2 for entry in entries),
        covariance_entry_max_allowed_error_m2=max(entry.allowed_error_m2 for entry in entries),
        covariance_entry_max_gate_ratio=max(entry.gate_ratio for entry in entries),
        actual_sampling_gate_passed=actual_passed,
        reported_role_separation_passed_attested=role_separation,
        all_checks_passed=all_passed,
    )
