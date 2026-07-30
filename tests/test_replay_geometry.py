from __future__ import annotations

import math

import numpy as np
import pytest

from fusion_fault_bench.geometry.camera import PinholeCamera
from fusion_fault_bench.replay_geometry import (
    FullReconstruction,
    NominalEligibility,
    RigidTransform3,
    SceneBevAnchor,
    calibration_perturbation,
    evaluate_nominal_eligibility,
    generate_camera_proxy,
    monitoring_scene_projection,
    persistent_panel_projection,
    reconstruct_camera,
    reconstruct_camera_proxy,
    reconstruct_lidar,
    reported_camera_extrinsic,
)


def _rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _transform(
    rotation: np.ndarray | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RigidTransform3:
    return RigidTransform3(
        rotation=np.eye(3, dtype=np.float64) if rotation is None else rotation,
        translation_m=np.asarray(translation, dtype=np.float64),
    )


def _camera() -> PinholeCamera:
    return PinholeCamera(
        intrinsic=np.asarray(
            ((10.0, 0.0, 50.0), (0.0, 10.0, 50.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        width_px=100,
        height_px=100,
    )


def _identity_eligibility(
    center_global_m: tuple[float, float, float],
    *,
    num_lidar_points: int = 1,
) -> NominalEligibility:
    return evaluate_nominal_eligibility(
        center_global_m=center_global_m,
        global_from_reference_ego=RigidTransform3.identity(),
        global_from_camera_ego=RigidTransform3.identity(),
        true_camera_ego_from_camera=RigidTransform3.identity(),
        camera=_camera(),
        num_lidar_points=num_lidar_points,
    )


def test_nominal_eligibility_uses_inclusive_bev_and_strict_camera_bounds() -> None:
    lower = _identity_eligibility((5.0, 0.0, 10.0))
    upper = _identity_eligibility((60.0, 40.0, 100.0))
    pixel_edge = _identity_eligibility((50.0, 0.0, 10.0))
    zero_points = _identity_eligibility(
        (5.0, 0.0, 10.0),
        num_lidar_points=0,
    )

    assert lower.eligible
    assert upper.eligible
    assert lower.center_reference_ego_m.tolist() == [5.0, 0.0, 10.0]
    assert not pixel_edge.camera_center_pass
    assert not pixel_edge.eligible
    assert zero_points.roi_pass and zero_points.camera_center_pass
    assert not zero_points.lidar_points_pass


def test_true_generation_is_unchanged_when_reported_calibration_changes() -> None:
    reference_pose = _transform(_rotation_z(0.2), (4.0, -2.0, 0.5))
    camera_pose = _transform(_rotation_z(-0.1), (4.2, -1.7, 0.7))
    true_extrinsic = _transform(_rotation_y(0.05), (1.0, 0.1, 1.4))
    truth = np.asarray((12.0, 4.0, 2.0), dtype=np.float64)
    velocity = np.asarray((3.0, -0.5, 0.2), dtype=np.float64)

    proxy = generate_camera_proxy(
        truth_global_at_reference_m=truth,
        velocity_global_mps=velocity,
        reference_time_s=5.0,
        camera_time_s=5.08,
        base_error_reference_bev_m=(0.4, -0.2),
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        true_camera_ego_from_camera=true_extrinsic,
    )
    reported = reported_camera_extrinsic(
        true_camera_ego_from_camera=true_extrinsic,
        perturbation_camera_ego=calibration_perturbation(
            translation_camera_ego_m=(0.5, 0.0, 0.0),
            yaw_camera_ego_rad=0.04,
        ),
    )
    nominal = reconstruct_camera_proxy(
        proxy,
        velocity_global_mps=velocity,
        reference_time_s=5.0,
        camera_time_s=5.08,
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        reported_camera_ego_from_camera=true_extrinsic,
    )
    faulted = reconstruct_camera_proxy(
        proxy,
        velocity_global_mps=velocity,
        reference_time_s=5.0,
        camera_time_s=5.08,
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        reported_camera_ego_from_camera=reported,
    )

    expected_nominal = reference_pose.inverse().apply(truth) + np.asarray((0.4, -0.2, 0.0))
    assert np.allclose(nominal.point_current_ego_m, expected_nominal, atol=2e-14)
    assert not np.allclose(faulted.point_current_ego_m, nominal.point_current_ego_m)
    assert np.allclose(
        proxy.physical_point_global_m,
        truth + velocity * 0.08 + reference_pose.rotation @ np.asarray((0.4, -0.2, 0.0)),
        atol=2e-14,
    )


@pytest.mark.parametrize("yaw_rad", (-0.2, 0.2))
def test_calibration_yaw_matches_independent_scalar_oracle(yaw_rad: float) -> None:
    truth = np.asarray((12.0, -3.0, 2.0), dtype=np.float64)
    identity = RigidTransform3.identity()
    proxy = generate_camera_proxy(
        truth_global_at_reference_m=truth,
        velocity_global_mps=(0.0, 0.0, 0.0),
        reference_time_s=4.0,
        camera_time_s=4.1,
        base_error_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=identity,
        global_from_camera_ego=identity,
        true_camera_ego_from_camera=identity,
    )
    reported = reported_camera_extrinsic(
        true_camera_ego_from_camera=identity,
        perturbation_camera_ego=calibration_perturbation(
            yaw_camera_ego_rad=yaw_rad,
        ),
    )
    observed = reconstruct_camera_proxy(
        proxy,
        velocity_global_mps=(0.0, 0.0, 0.0),
        reference_time_s=4.0,
        camera_time_s=4.1,
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=identity,
        global_from_camera_ego=identity,
        reported_camera_ego_from_camera=reported,
    )

    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    expected = np.asarray(
        (
            cosine * truth[0] - sine * truth[1],
            sine * truth[0] + cosine * truth[1],
            truth[2],
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        proxy.physical_point_global_m,
        truth,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        observed.point_current_ego_m,
        expected,
        rtol=0.0,
        atol=1e-14,
    )


def test_corrupted_generation_and_reconstruction_cancellation_mutation_is_detected() -> None:
    reference_pose = _transform(
        _rotation_z(0.3) @ _rotation_y(-0.08) @ _rotation_x(0.04),
        (5.0, -3.0, 1.2),
    )
    camera_pose = _transform(
        _rotation_z(0.28) @ _rotation_y(-0.04) @ _rotation_x(0.02),
        (5.3, -2.7, 1.4),
    )
    true_extrinsic = _transform(
        _rotation_y(0.06) @ _rotation_x(-0.03),
        (1.1, 0.0, 1.5),
    )
    reported = reported_camera_extrinsic(
        true_camera_ego_from_camera=true_extrinsic,
        perturbation_camera_ego=calibration_perturbation(
            translation_camera_ego_m=(0.6, -0.2, 0.1),
            yaw_camera_ego_rad=0.12,
        ),
    )
    truth = np.asarray((18.0, 2.0, 1.0), dtype=np.float64)
    velocity = np.asarray((1.5, -0.4, 0.2), dtype=np.float64)
    base_error = np.asarray((0.3, -0.2), dtype=np.float64)

    observed = reconstruct_camera(
        truth_global_at_reference_m=truth,
        velocity_global_mps=velocity,
        reference_time_s=10.0,
        camera_time_s=10.09,
        base_error_reference_bev_m=base_error,
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        true_camera_ego_from_camera=true_extrinsic,
        reported_camera_ego_from_camera=reported,
    )
    corrupted_proxy = generate_camera_proxy(
        truth_global_at_reference_m=truth,
        velocity_global_mps=velocity,
        reference_time_s=10.0,
        camera_time_s=10.09,
        base_error_reference_bev_m=base_error,
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        true_camera_ego_from_camera=reported,
    )
    cancelling_mutation = reconstruct_camera_proxy(
        corrupted_proxy,
        velocity_global_mps=velocity,
        reference_time_s=10.0,
        camera_time_s=10.09,
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        reported_camera_ego_from_camera=reported,
    )
    nominal = reference_pose.inverse().apply(truth) + np.asarray(
        (base_error[0], base_error[1], 0.0),
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        cancelling_mutation.point_current_ego_m,
        nominal,
        rtol=0.0,
        atol=2e-14,
    )
    assert not np.allclose(
        observed.point_current_ego_m,
        cancelling_mutation.point_current_ego_m,
        rtol=0.0,
        atol=1e-3,
    )


def test_camera_full_reconstruction_jacobian_matches_central_difference() -> None:
    reference_pose = _transform(
        _rotation_z(0.3) @ _rotation_y(-0.08) @ _rotation_x(0.04),
        (5.0, -3.0, 1.2),
    )
    camera_pose = _transform(
        _rotation_z(0.28) @ _rotation_y(-0.04) @ _rotation_x(0.02),
        (5.3, -2.7, 1.4),
    )
    true_extrinsic = _transform(
        _rotation_y(0.06) @ _rotation_x(-0.03),
        (1.1, 0.0, 1.5),
    )
    reported = reported_camera_extrinsic(
        true_camera_ego_from_camera=true_extrinsic,
        perturbation_camera_ego=calibration_perturbation(
            translation_camera_ego_m=(0.7, -0.2, 0.1),
            yaw_camera_ego_rad=0.07,
        ),
    )

    def reconstruct(error: np.ndarray | tuple[float, float]) -> FullReconstruction:
        return reconstruct_camera(
            truth_global_at_reference_m=(18.0, 2.0, 1.0),
            velocity_global_mps=(4.0, -1.0, 0.3),
            reference_time_s=10.0,
            camera_time_s=9.94,
            base_error_reference_bev_m=error,
            timestamp_fault_s=0.15,
            additive_bias_reference_bev_m=(0.2, -0.1),
            global_from_reference_ego=reference_pose,
            global_from_camera_ego=camera_pose,
            true_camera_ego_from_camera=true_extrinsic,
            reported_camera_ego_from_camera=reported,
        )

    observed = reconstruct((0.3, -0.4))
    finite_difference = np.zeros((3, 2), dtype=np.float64)
    epsilon = 1e-6
    for axis in range(2):
        positive = np.asarray((0.3, -0.4), dtype=np.float64)
        negative = positive.copy()
        positive[axis] += epsilon
        negative[axis] -= epsilon
        plus = reconstruct(positive)
        minus = reconstruct(negative)
        finite_difference[:, axis] = (plus.point_current_ego_m - minus.point_current_ego_m) / (
            2.0 * epsilon
        )

    assert np.allclose(
        observed.base_error_jacobian,
        finite_difference,
        rtol=0.0,
        atol=1e-8,
    )
    assert abs(float(observed.base_error_jacobian[2, 0])) > 1e-4


def test_natural_asynchrony_cancels_and_timestamp_fault_has_declared_sign() -> None:
    identity = RigidTransform3.identity()

    def reconstruct(timestamp_fault_s: float) -> FullReconstruction:
        return reconstruct_camera(
            truth_global_at_reference_m=(10.0, 2.0, 4.0),
            velocity_global_mps=(2.0, -1.0, 0.5),
            reference_time_s=1.0,
            camera_time_s=1.2,
            base_error_reference_bev_m=(0.0, 0.0),
            timestamp_fault_s=timestamp_fault_s,
            additive_bias_reference_bev_m=(0.0, 0.0),
            global_from_reference_ego=identity,
            global_from_camera_ego=identity,
            true_camera_ego_from_camera=identity,
            reported_camera_ego_from_camera=identity,
        )

    nominal = reconstruct(0.0)
    delayed = reconstruct(0.4)

    assert np.allclose(nominal.point_current_ego_m, (10.0, 2.0, 4.0), atol=1e-14)
    assert np.allclose(
        delayed.point_current_ego_m,
        np.asarray((10.0, 2.0, 4.0)) - np.asarray((2.0, -1.0, 0.5)) * 0.4,
        atol=1e-14,
    )
    assert delayed.reference_state_time_s == 1.0
    assert delayed.reported_state_time_s == 1.4

    lidar = reconstruct_lidar(
        truth_global_at_reference_m=(10.0, 2.0, 4.0),
        velocity_global_mps=(2.0, -1.0, 0.5),
        reference_time_s=1.0,
        base_error_reference_bev_m=(0.0, 0.0),
        timestamp_fault_s=0.4,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=RigidTransform3.identity(),
    )
    assert np.allclose(lidar.point_current_ego_m, delayed.point_current_ego_m)


def test_stationary_track_has_zero_timestamp_displacement_for_both_modalities() -> None:
    reference_pose = _transform(
        _rotation_z(0.4) @ _rotation_y(-0.12) @ _rotation_x(0.07),
        (4.0, -2.0, 1.1),
    )
    camera_pose = _transform(
        _rotation_z(0.46) @ _rotation_y(-0.08) @ _rotation_x(0.03),
        (4.4, -1.7, 1.3),
    )
    true_extrinsic = _transform(
        _rotation_y(0.05) @ _rotation_x(-0.02),
        (1.0, 0.1, 1.4),
    )
    truth = np.asarray((20.0, 3.0, 1.5), dtype=np.float64)

    camera_nominal = reconstruct_camera(
        truth_global_at_reference_m=truth,
        velocity_global_mps=(0.0, 0.0, 0.0),
        reference_time_s=4.0,
        camera_time_s=4.12,
        base_error_reference_bev_m=(0.2, -0.1),
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        true_camera_ego_from_camera=true_extrinsic,
        reported_camera_ego_from_camera=true_extrinsic,
    )
    camera_delayed = reconstruct_camera(
        truth_global_at_reference_m=truth,
        velocity_global_mps=(0.0, 0.0, 0.0),
        reference_time_s=4.0,
        camera_time_s=4.12,
        base_error_reference_bev_m=(0.2, -0.1),
        timestamp_fault_s=0.6,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
        global_from_camera_ego=camera_pose,
        true_camera_ego_from_camera=true_extrinsic,
        reported_camera_ego_from_camera=true_extrinsic,
    )
    lidar_nominal = reconstruct_lidar(
        truth_global_at_reference_m=truth,
        velocity_global_mps=(0.0, 0.0, 0.0),
        reference_time_s=4.0,
        base_error_reference_bev_m=(0.2, -0.1),
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
    )
    lidar_delayed = reconstruct_lidar(
        truth_global_at_reference_m=truth,
        velocity_global_mps=(0.0, 0.0, 0.0),
        reference_time_s=4.0,
        base_error_reference_bev_m=(0.2, -0.1),
        timestamp_fault_s=0.6,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=reference_pose,
    )

    np.testing.assert_allclose(
        camera_delayed.point_current_ego_m,
        camera_nominal.point_current_ego_m,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        lidar_delayed.point_current_ego_m,
        lidar_nominal.point_current_ego_m,
        rtol=0.0,
        atol=2e-14,
    )
    assert camera_nominal.reported_state_time_s == 4.0
    assert camera_delayed.reported_state_time_s == 4.6
    assert lidar_nominal.reported_state_time_s == 4.0
    assert lidar_delayed.reported_state_time_s == 4.6


def test_m5_a_and_m5_b_covariances_use_exact_distinct_jacobians() -> None:
    reconstruction = FullReconstruction(
        point_current_ego_m=np.asarray((4.0, -2.0, 8.0)),
        base_error_jacobian=np.asarray(
            ((1.0, 0.2), (0.1, 0.9), (0.4, -0.3)),
            dtype=np.float64,
        ),
        reference_state_time_s=2.0,
        reported_state_time_s=2.0,
    )
    covariance = np.asarray(((2.0, 0.3), (0.3, 1.0)), dtype=np.float64)
    current_pose = _transform(
        _rotation_z(0.4) @ _rotation_y(0.2) @ _rotation_x(-0.1),
        (10.0, -5.0, 2.0),
    )
    first_pose = _transform(_rotation_z(0.15), (8.0, -4.0, 1.0))
    anchor = SceneBevAnchor.from_first_reference_pose(first_pose)

    persistent = persistent_panel_projection(
        reconstruction,
        reported_base_covariance_m2=covariance,
    )
    monitoring = monitoring_scene_projection(
        reconstruction,
        reported_base_covariance_m2=covariance,
        global_from_current_ego=current_pose,
        scene_anchor=anchor,
    )
    expected_a = reconstruction.base_error_jacobian[:2]
    scene_rotation = _rotation_z(-0.15)[:2, :2]
    expected_j = scene_rotation @ current_pose.rotation[:2, :] @ reconstruction.base_error_jacobian
    expected_point = scene_rotation @ (
        current_pose.apply(reconstruction.point_current_ego_m)[:2] - first_pose.translation_m[:2]
    )

    assert np.allclose(persistent.point_m, (4.0, -2.0))
    assert np.allclose(persistent.jacobian, expected_a)
    assert np.allclose(
        persistent.reported_covariance_m2,
        expected_a @ covariance @ expected_a.T,
    )
    assert np.allclose(monitoring.point_m, expected_point)
    assert np.allclose(monitoring.jacobian, expected_j)
    assert np.allclose(
        monitoring.reported_covariance_m2,
        expected_j @ covariance @ expected_j.T,
    )
    assert not np.allclose(monitoring.jacobian, persistent.jacobian)
    assert not np.allclose(
        monitoring.point_m,
        scene_rotation
        @ (
            current_pose.rotation[:2, :2] @ reconstruction.point_current_ego_m[:2]
            + current_pose.translation_m[:2]
            - first_pose.translation_m[:2]
        ),
    )


def test_monitoring_projection_is_invariant_to_planar_global_origin_and_yaw() -> None:
    reconstruction = reconstruct_lidar(
        truth_global_at_reference_m=(4.0, 2.0, 1.0),
        velocity_global_mps=(0.0, 0.0, 0.0),
        reference_time_s=0.0,
        base_error_reference_bev_m=(0.2, -0.1),
        timestamp_fault_s=0.0,
        additive_bias_reference_bev_m=(0.0, 0.0),
        global_from_reference_ego=RigidTransform3.identity(),
    )
    first = _transform(_rotation_z(0.2), (1.0, 3.0, 0.5))
    current = _transform(
        _rotation_z(0.5) @ _rotation_y(0.1),
        (5.0, 7.0, 1.0),
    )
    global_relabeling = _transform(_rotation_z(-0.7), (20.0, -9.0, 3.0))
    original = monitoring_scene_projection(
        reconstruction,
        reported_base_covariance_m2=np.diag((0.09, 0.16)),
        global_from_current_ego=current,
        scene_anchor=SceneBevAnchor.from_first_reference_pose(first),
    )
    relabeled = monitoring_scene_projection(
        reconstruction,
        reported_base_covariance_m2=np.diag((0.09, 0.16)),
        global_from_current_ego=global_relabeling.compose(current),
        scene_anchor=SceneBevAnchor.from_first_reference_pose(global_relabeling.compose(first)),
    )

    assert np.allclose(original.point_m, relabeled.point_m, atol=2e-14)
    assert np.allclose(original.jacobian, relabeled.jacobian, atol=2e-14)
    assert np.allclose(
        original.reported_covariance_m2,
        relabeled.reported_covariance_m2,
        atol=2e-14,
    )


def test_stationary_global_object_is_stationary_in_scene_frame_under_full_3d_ego_motion() -> None:
    truth_global = np.asarray((25.0, -4.0, 3.0), dtype=np.float64)
    reference_poses = (
        _transform(
            _rotation_z(0.25) @ _rotation_y(0.12) @ _rotation_x(-0.08),
            (3.0, -2.0, 1.0),
        ),
        _transform(
            _rotation_z(0.7) @ _rotation_y(-0.15) @ _rotation_x(0.12),
            (8.0, 1.0, 1.5),
        ),
    )
    camera_poses = (
        _transform(
            _rotation_z(0.29) @ _rotation_y(0.08) @ _rotation_x(-0.03),
            (3.5, -1.8, 1.3),
        ),
        _transform(
            _rotation_z(0.76) @ _rotation_y(-0.1) @ _rotation_x(0.06),
            (8.4, 1.3, 1.8),
        ),
    )
    true_extrinsic = _transform(
        _rotation_y(0.04) @ _rotation_x(-0.02),
        (1.1, 0.0, 1.5),
    )
    scene_anchor = SceneBevAnchor.from_first_reference_pose(reference_poses[0])
    covariance = np.diag((1.0, 0.36))
    observed_scene_points: list[np.ndarray] = []
    camera_reconstructions: list[FullReconstruction] = []

    for frame_index, (reference_pose, camera_pose) in enumerate(
        zip(reference_poses, camera_poses, strict=True)
    ):
        reference_time = float(frame_index)
        camera_reconstruction = reconstruct_camera(
            truth_global_at_reference_m=truth_global,
            velocity_global_mps=(0.0, 0.0, 0.0),
            reference_time_s=reference_time,
            camera_time_s=reference_time + 0.07,
            base_error_reference_bev_m=(0.0, 0.0),
            timestamp_fault_s=0.0,
            additive_bias_reference_bev_m=(0.0, 0.0),
            global_from_reference_ego=reference_pose,
            global_from_camera_ego=camera_pose,
            true_camera_ego_from_camera=true_extrinsic,
            reported_camera_ego_from_camera=true_extrinsic,
        )
        lidar_reconstruction = reconstruct_lidar(
            truth_global_at_reference_m=truth_global,
            velocity_global_mps=(0.0, 0.0, 0.0),
            reference_time_s=reference_time,
            base_error_reference_bev_m=(0.0, 0.0),
            timestamp_fault_s=0.0,
            additive_bias_reference_bev_m=(0.0, 0.0),
            global_from_reference_ego=reference_pose,
        )
        camera_reconstructions.append(camera_reconstruction)
        for reconstruction in (camera_reconstruction, lidar_reconstruction):
            observed_scene_points.append(
                monitoring_scene_projection(
                    reconstruction,
                    reported_base_covariance_m2=covariance,
                    global_from_current_ego=reference_pose,
                    scene_anchor=scene_anchor,
                ).point_m
            )

    inverse_yaw = -scene_anchor.yaw_global_from_scene_rad
    scene_rotation = np.asarray(
        (
            (math.cos(inverse_yaw), -math.sin(inverse_yaw)),
            (math.sin(inverse_yaw), math.cos(inverse_yaw)),
        ),
        dtype=np.float64,
    )
    expected = scene_rotation @ (truth_global[:2] - scene_anchor.origin_global_xy_m)
    for observed in observed_scene_points:
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=3e-14)

    wrong_sensor_pose = monitoring_scene_projection(
        camera_reconstructions[1],
        reported_base_covariance_m2=covariance,
        global_from_current_ego=camera_poses[1],
        scene_anchor=scene_anchor,
    ).point_m
    assert not np.allclose(wrong_sensor_pose, expected, rtol=0.0, atol=1e-3)

    current_reference = reference_poses[1]
    current_reconstruction = camera_reconstructions[1]
    prematurely_planar_global = (
        current_reference.rotation[:2, :2] @ current_reconstruction.point_current_ego_m[:2]
        + current_reference.translation_m[:2]
    )
    prematurely_planar_scene = scene_rotation @ (
        prematurely_planar_global - scene_anchor.origin_global_xy_m
    )
    assert not np.allclose(prematurely_planar_scene, expected, rtol=0.0, atol=1e-3)


def test_projection_rejects_non_spd_pushforward_and_arrays_are_immutable() -> None:
    reconstruction = FullReconstruction(
        point_current_ego_m=np.zeros(3),
        base_error_jacobian=np.zeros((3, 2)),
        reference_state_time_s=0.0,
        reported_state_time_s=0.0,
    )

    with pytest.raises(ValueError, match="positive definite"):
        persistent_panel_projection(
            reconstruction,
            reported_base_covariance_m2=np.eye(2),
        )
    with pytest.raises(ValueError):
        reconstruction.point_current_ego_m[0] = 1.0
