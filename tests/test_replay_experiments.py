from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import fusion_fault_bench.replay_experiments as replay_experiments
from fusion_fault_bench.geometry.camera import PinholeCamera
from fusion_fault_bench.replay_experiments import (
    FaultFamily,
    FaultTarget,
    ReplayFaultCondition,
    ReplaySceneDraws,
    draw_replay_scene_randomness,
    generate_replay_condition_sequence,
)
from fusion_fault_bench.replay_geometry import NominalEligibility, RigidTransform3
from fusion_fault_bench.replay_source import (
    ReplayFrame,
    ReplayObjectFrame,
    ReplayScene,
    ReplaySensorSnapshot,
)


def _transform(
    *,
    rotation: np.ndarray | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RigidTransform3:
    return RigidTransform3(
        rotation=np.eye(3, dtype=np.float64) if rotation is None else rotation,
        translation_m=np.asarray(translation, dtype=np.float64),
    )


def _scene(
    *,
    frame_count: int = 4,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    global_from_ego: RigidTransform3 | None = None,
) -> ReplayScene:
    pose = _transform() if global_from_ego is None else global_from_ego
    camera_model = PinholeCamera(
        intrinsic=np.asarray(
            ((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        width_px=100,
        height_px=100,
    )
    frames: list[ReplayFrame] = []
    for frame_index in range(frame_count):
        timestamp_us = 1_000_000 + frame_index * 500_000
        lidar = ReplaySensorSnapshot(
            timestamp_us=timestamp_us,
            global_from_ego=pose,
            ego_from_sensor=_transform(),
            camera=None,
        )
        camera = ReplaySensorSnapshot(
            timestamp_us=timestamp_us + 20_000,
            global_from_ego=pose,
            ego_from_sensor=_transform(),
            camera=camera_model,
        )
        center_global = np.asarray((10.0, 2.0, 1.0), dtype=np.float64)
        support = NominalEligibility(
            center_reference_ego_m=pose.inverse().apply(center_global),
            center_camera_m=center_global,
            roi_pass=True,
            camera_center_pass=True,
            lidar_points_pass=True,
            camera_estimator_available=True,
            lidar_estimator_available=True,
            eligible=True,
        )
        object_row = ReplayObjectFrame(
            object_id="track:0000",
            center_global_m=center_global,
            size_width_length_height_m=np.asarray((2.0, 4.0, 1.5)),
            orientation_global_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
            velocity_global_mps=np.asarray(velocity),
            velocity_method="centered",
            acceleration_global_mps2=np.zeros(3, dtype=np.float64),
            acceleration_method="centered",
            category_name="vehicle.car",
            visibility_level="v80-100",
            num_lidar_points=5,
            support=support,
        )
        frames.append(
            ReplayFrame(
                frame_index=frame_index,
                sample_timestamp_us=timestamp_us,
                reference_time_s=frame_index * 0.5,
                lidar=lidar,
                camera=camera,
                objects=(object_row,),
            )
        )
    return ReplayScene(
        scene_name="scene-test",
        sequence_id="nuscenes:scene-test",
        log_group_id="log-group:00",
        frames=tuple(frames),
    )


def _draws(
    scene: ReplayScene,
    *,
    camera: tuple[float, float] = (0.0, 0.0),
    lidar: tuple[float, float] = (0.0, 0.0),
    uniforms: tuple[float, ...] | None = None,
) -> ReplaySceneDraws:
    keys = tuple(
        (frame.frame_index, item.object_id)
        for frame in scene.frames
        for item in frame.eligible_objects
    )
    return ReplaySceneDraws(
        data_master_seed=1729,
        row_keys=keys,
        camera_standard_normal_xy=np.tile(np.asarray(camera), (len(keys), 1)),
        lidar_standard_normal_xy=np.tile(np.asarray(lidar), (len(keys), 1)),
        dropout_uniform_by_frame=np.asarray(
            uniforms if uniforms is not None else (0.5,) * len(scene.frames)
        ),
    )


def _condition(
    *,
    family: FaultFamily = "additive-position-bias",
    target: FaultTarget = "camera",
    axis: str = "y",
    value: float = 0.0,
    identity: bool = True,
    active_frames: tuple[int, int] | None = None,
) -> ReplayFaultCondition:
    return ReplayFaultCondition(
        experiment_id="replay-test",
        family=family,
        target=target,
        axis=axis,
        unit="m",
        value=value,
        identity=identity,
        active_frames=active_frames,
    )


def test_replay_draws_reject_nonpreregistered_master_seed() -> None:
    draws = _draws(_scene())
    with pytest.raises(ValueError, match="frozen M5 data master seed"):
        replace(draws, data_master_seed=1730)


def test_identity_reconstructs_truth_and_keeps_monitoring_separate() -> None:
    scene = _scene()
    result = generate_replay_condition_sequence(
        scene,
        condition=_condition(),
        draws=_draws(scene),
    )

    for frame in result.frames:
        estimate = frame.objects[0]
        np.testing.assert_allclose(
            estimate.camera_current_ego.point_m,
            estimate.truth_current_ego_xy_m,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            estimate.lidar_current_ego.point_m,
            estimate.truth_current_ego_xy_m,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            estimate.camera_monitoring_scene.point_m,
            estimate.camera_current_ego.point_m,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            estimate.camera_current_ego.reported_covariance_m2,
            np.eye(2),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            estimate.lidar_current_ego.reported_covariance_m2,
            np.eye(2) * 0.09,
            rtol=0.0,
            atol=1e-12,
        )


def test_metadata_calibration_translation_does_not_move_truth_or_lidar() -> None:
    scene = _scene()
    result = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family="calibration-translation",
            target="camera",
            axis="x",
            value=1.0,
            identity=False,
        ),
        draws=_draws(scene),
    )

    estimate = result.frames[0].objects[0]
    np.testing.assert_allclose(
        estimate.camera_current_ego.point_m - estimate.truth_current_ego_xy_m,
        (1.0, 0.0),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        estimate.lidar_current_ego.point_m,
        estimate.truth_current_ego_xy_m,
        rtol=0.0,
        atol=1e-12,
    )


def test_positive_timestamp_fault_matches_negative_velocity_displacement() -> None:
    scene = _scene(velocity=(2.0, -1.0, 0.0))
    result = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family="timestamp-offset",
            target="camera",
            axis="time",
            value=0.5,
            identity=False,
        ),
        draws=_draws(scene),
    )

    estimate = result.frames[0].objects[0]
    np.testing.assert_allclose(
        estimate.camera_current_ego.point_m - estimate.truth_current_ego_xy_m,
        (-1.0, 0.5),
        rtol=0.0,
        atol=1e-12,
    )
    assert estimate.camera_reported_state_time_s == pytest.approx(0.5)
    assert estimate.lidar_reported_state_time_s == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("family", "expected_variance"),
    [
        ("increased-noise-underreported", 1.0),
        ("increased-noise-correctly-reported", 9.0),
    ],
)
def test_noise_scale_changes_actual_draw_but_only_correct_reporting_changes_covariance(
    family: FaultFamily,
    expected_variance: float,
) -> None:
    scene = _scene()
    result = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family=family,
            target="camera",
            axis="xy",
            value=3.0,
            identity=False,
        ),
        draws=_draws(scene, camera=(1.0, -1.0)),
    )

    estimate = result.frames[0].objects[0]
    np.testing.assert_allclose(
        estimate.camera_current_ego.point_m - estimate.truth_current_ego_xy_m,
        (3.0, -3.0),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        estimate.camera_current_ego.reported_covariance_m2,
        np.eye(2) * expected_variance,
        rtol=0.0,
        atol=1e-12,
    )


def test_dropout_masks_are_frame_shared_and_nested() -> None:
    scene = _scene()
    draws = _draws(scene, uniforms=(0.05, 0.2, 0.6, 0.99))
    low = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family="dropout",
            target="camera",
            axis="availability",
            value=0.1,
            identity=False,
        ),
        draws=draws,
    )
    high = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family="dropout",
            target="camera",
            axis="availability",
            value=0.5,
            identity=False,
        ),
        draws=draws,
    )

    low_available = tuple(frame.camera_available for frame in low.frames)
    high_available = tuple(frame.camera_available for frame in high.frames)
    assert low_available == (False, True, True, True)
    assert high_available == (False, False, True, True)
    assert all(
        (not high_value) or low_value
        for low_value, high_value in zip(low_available, high_available, strict=True)
    )


def test_health_window_limits_fault_application() -> None:
    scene = _scene(frame_count=6)
    result = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family="additive-position-bias",
            target="lidar",
            axis="y",
            value=2.0,
            identity=False,
            active_frames=(2, 4),
        ),
        draws=_draws(scene),
    )

    errors = tuple(
        tuple(frame.objects[0].lidar_current_ego.point_m - frame.objects[0].truth_current_ego_xy_m)
        for frame in result.frames
    )
    assert errors == (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )


def test_reported_metadata_faults_cannot_change_frozen_eligibility() -> None:
    scene = _scene(velocity=(2.0, -1.0, 0.0))
    draws = _draws(scene)
    identity = generate_replay_condition_sequence(
        scene,
        condition=_condition(),
        draws=draws,
    )
    metadata_faults = (
        ReplayFaultCondition(
            experiment_id="replay-test-calibration-x",
            family="calibration-translation",
            target="camera",
            axis="x",
            unit="m",
            value=20.0,
            identity=False,
            active_frames=None,
        ),
        ReplayFaultCondition(
            experiment_id="replay-test-calibration-yaw",
            family="calibration-yaw",
            target="camera",
            axis="yaw",
            unit="rad",
            value=0.8,
            identity=False,
            active_frames=None,
        ),
        ReplayFaultCondition(
            experiment_id="replay-test-timestamp",
            family="timestamp-offset",
            target="camera",
            axis="time",
            unit="s",
            value=2.0,
            identity=False,
            active_frames=None,
        ),
    )

    expected_rows = tuple(
        (frame.frame_index, tuple(item.object_id for item in frame.objects))
        for frame in identity.frames
    )
    expected_truth = tuple(
        item.truth_current_ego_xy_m for frame in identity.frames for item in frame.objects
    )
    expected_lidar = tuple(
        item.lidar_current_ego.point_m for frame in identity.frames for item in frame.objects
    )
    for condition in metadata_faults:
        result = generate_replay_condition_sequence(
            scene,
            condition=condition,
            draws=draws,
        )
        observed_rows = tuple(
            (frame.frame_index, tuple(item.object_id for item in frame.objects))
            for frame in result.frames
        )
        assert observed_rows == expected_rows
        assert result.eligible_object_frame_count == identity.eligible_object_frame_count
        for observed, expected in zip(
            (item.truth_current_ego_xy_m for frame in result.frames for item in frame.objects),
            expected_truth,
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected)
        for observed, expected in zip(
            (item.lidar_current_ego.point_m for frame in result.frames for item in frame.objects),
            expected_lidar,
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected)
        assert any(
            not np.allclose(
                observed.camera_current_ego.point_m,
                nominal.camera_current_ego.point_m,
                rtol=0.0,
                atol=1e-12,
            )
            for observed_frame, nominal_frame in zip(
                result.frames,
                identity.frames,
                strict=True,
            )
            for observed, nominal in zip(
                observed_frame.objects,
                nominal_frame.objects,
                strict=True,
            )
        )


def test_fault_condition_contract_covers_identity_targets_ranges_and_windows() -> None:
    identity = ReplayFaultCondition(
        experiment_id="replay-identity",
        family="identity",
        target="none",
        axis="none",
        unit="identity",
        value=0.0,
        identity=True,
        active_frames=None,
    )
    assert identity.selector == "replay-identity:0"
    assert not identity.fault_is_active(0)
    full = _condition(value=1.0, identity=False)
    assert full.selector == "replay-test:+1"
    assert full.fault_is_active(0)
    windowed = replace(full, active_frames=(1, 3))
    assert not windowed.fault_is_active(0)
    assert windowed.fault_is_active(1)
    assert not windowed.fault_is_active(3)
    noise_identity = ReplayFaultCondition(
        experiment_id="replay-noise",
        family="increased-noise-underreported",
        target="camera",
        axis="xy",
        unit="std-scale",
        value=1.0,
        identity=True,
        active_frames=None,
    )
    assert noise_identity.selector == "replay-noise:1"

    valid = {
        "experiment_id": "replay-test",
        "family": "additive-position-bias",
        "target": "camera",
        "axis": "y",
        "unit": "m",
        "value": 1.0,
        "identity": False,
        "active_frames": None,
    }
    invalid = (
        {"experiment_id": ""},
        {"value": float("nan")},
        {"family": "identity", "target": "none", "value": 0.0, "identity": False},
        {"family": "identity", "target": "camera", "value": 0.0, "identity": True},
        {"family": "identity", "target": "none", "value": 1.0, "identity": True},
        {"identity": True, "value": 1.0},
        {
            "family": "increased-noise-underreported",
            "axis": "xy",
            "unit": "std-scale",
            "identity": True,
            "value": 0.0,
        },
        {"target": "none"},
        {"family": "common-mode-position-bias", "target": "camera"},
        {"target": "both"},
        {
            "family": "increased-noise-correctly-reported",
            "axis": "xy",
            "unit": "std-scale",
            "value": 0.5,
        },
        {
            "family": "dropout",
            "axis": "availability",
            "unit": "probability",
            "value": -0.1,
        },
        {
            "family": "dropout",
            "axis": "availability",
            "unit": "probability",
            "value": 1.1,
        },
        {"active_frames": (True, 2)},
        {"active_frames": (0, True)},
        {"active_frames": (-1, 2)},
        {"active_frames": (2, 2)},
    )
    for update in invalid:
        with pytest.raises(ValueError):
            ReplayFaultCondition(**{**valid, **update})  # type: ignore[arg-type]


def test_scene_draw_contract_rejects_order_shape_and_uniform_corruption() -> None:
    scene = _scene()
    draws = _draws(scene)
    assert "track:0000" not in repr(draws)
    for update in (
        {"row_keys": ()},
        {"row_keys": tuple(reversed(draws.row_keys))},
        {"row_keys": (draws.row_keys[0],) * len(draws.row_keys)},
        {"camera_standard_normal_xy": np.zeros((len(draws.row_keys), 3))},
        {
            "camera_standard_normal_xy": np.full(
                (len(draws.row_keys), 2),
                float("nan"),
            )
        },
        {"lidar_standard_normal_xy": np.zeros((len(draws.row_keys), 3))},
        {"dropout_uniform_by_frame": np.zeros((2, 2))},
        {"dropout_uniform_by_frame": np.asarray(())},
        {"dropout_uniform_by_frame": np.asarray((float("nan"),) * 4)},
        {"dropout_uniform_by_frame": np.asarray((-0.1,) * 4)},
        {"dropout_uniform_by_frame": np.asarray((1.0,) * 4)},
    ):
        with pytest.raises(ValueError):
            replace(draws, **update)


def test_estimate_contracts_reject_nonfinite_duplicate_and_noncontiguous_rows() -> None:
    scene = _scene()
    sequence = generate_replay_condition_sequence(
        scene,
        condition=_condition(),
        draws=_draws(scene),
    )
    item = sequence.frames[0].objects[0]
    assert item.object_id not in repr(item)
    for update in (
        {"object_id": ""},
        {"truth_current_ego_xy_m": np.zeros(3)},
        {"truth_current_ego_xy_m": np.asarray((float("nan"), 0.0))},
        {"fixed_current_ego_xy_m": np.zeros(3)},
        {"fixed_reported_covariance_m2": np.zeros((3, 3))},
        {"fixed_reported_covariance_m2": np.full((2, 2), float("nan"))},
        {"camera_reported_state_time_s": float("nan")},
        {"lidar_reported_state_time_s": float("nan")},
    ):
        with pytest.raises(ValueError):
            replace(item, **update)

    frame = sequence.frames[0]
    second = replace(item, object_id="track:0001")
    assert item.object_id not in repr(frame)
    for update in (
        {"frame_index": True},
        {"frame_index": -1},
        {"reference_time_s": float("nan")},
        {"objects": (second, item)},
        {"objects": (item, item)},
    ):
        with pytest.raises(ValueError):
            replace(frame, **update)

    assert sequence.sequence_id not in repr(sequence)
    for update in (
        {"sequence_id": ""},
        {"frames": ()},
        {
            "frames": (
                sequence.frames[0],
                replace(sequence.frames[1], frame_index=3),
                *sequence.frames[2:],
            )
        },
        {
            "frames": (
                replace(sequence.frames[0], reference_time_s=1.0),
                *sequence.frames[1:],
            )
        },
        {
            "frames": (
                sequence.frames[0],
                replace(sequence.frames[1], reference_time_s=0.0),
                *sequence.frames[2:],
            )
        },
    ):
        with pytest.raises(ValueError):
            replace(sequence, **update)


def test_draw_and_generation_bind_exact_scene_support_and_frame_count() -> None:
    scene = _scene()
    draws = draw_replay_scene_randomness(scene)
    assert draws.row_keys == _draws(scene).row_keys
    empty_scene = replace(
        scene,
        frames=tuple(replace(frame, objects=()) for frame in scene.frames),
    )
    with pytest.raises(ValueError, match="no eligible"):
        draw_replay_scene_randomness(empty_scene)
    with pytest.raises(ValueError, match="frame count"):
        generate_replay_condition_sequence(
            scene,
            condition=_condition(),
            draws=replace(
                draws,
                dropout_uniform_by_frame=draws.dropout_uniform_by_frame[:-1],
            ),
        )
    changed_keys = tuple(
        (frame_index, f"{object_id}-other") for frame_index, object_id in draws.row_keys
    )
    with pytest.raises(ValueError, match="frozen scene support"):
        generate_replay_condition_sequence(
            scene,
            condition=_condition(),
            draws=replace(draws, row_keys=changed_keys),
        )
    with pytest.raises(ValueError, match="active window"):
        generate_replay_condition_sequence(
            scene,
            condition=replace(_condition(value=1.0, identity=False), active_frames=(0, 5)),
            draws=draws,
        )
    unsupported_axis = replace(_condition(value=1.0, identity=False), axis="z")
    with pytest.raises(ValueError, match="unsupported axis"):
        replay_experiments._fault_axis_vector(unsupported_axis)


def test_lidar_noise_and_common_mode_exercise_complementary_fault_targets() -> None:
    scene = _scene()
    lidar_noise = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family="increased-noise-correctly-reported",
            target="lidar",
            axis="xy",
            value=2.0,
            identity=False,
        ),
        draws=_draws(scene, lidar=(1.0, -1.0)),
    )
    estimate = lidar_noise.frames[0].objects[0]
    np.testing.assert_allclose(
        estimate.lidar_current_ego.reported_covariance_m2,
        np.eye(2) * 0.36,
        rtol=0.0,
        atol=1e-12,
    )

    common = generate_replay_condition_sequence(
        scene,
        condition=_condition(
            family="common-mode-position-bias",
            target="both",
            axis="x",
            value=2.0,
            identity=False,
        ),
        draws=_draws(scene),
    )
    common_estimate = common.frames[0].objects[0]
    np.testing.assert_allclose(
        common_estimate.camera_current_ego.point_m - common_estimate.lidar_current_ego.point_m,
        (0.0, 0.0),
        rtol=0.0,
        atol=1e-12,
    )
