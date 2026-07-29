from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fusion_fault_bench.geometry import SE3, FrameId, quaternion_wxyz_to_rotation

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


def test_fixture_cross_time_chain_matches_independent_reference() -> None:
    fixture = _mapping(FIXTURE_PATH)
    poses = fixture["poses"]
    global_from_camera_ego = _fixture_transform(poses["global_from_camera_ego"])
    camera_ego_from_camera = _fixture_transform(poses["camera_ego_from_camera"])
    global_from_reference_ego = _fixture_transform(poses["global_from_reference_ego"])

    reference_ego_from_camera = global_from_reference_ego.inverse().compose(
        global_from_camera_ego.compose(camera_ego_from_camera)
    )
    expected = fixture["expected_reference_ego_from_camera"]

    assert np.max(np.abs(reference_ego_from_camera.rotation - expected["rotation"])) <= 1e-12
    assert (
        np.max(np.abs(reference_ego_from_camera.translation_m - expected["translation_m"])) <= 1e-10
    )
    assert reference_ego_from_camera.target_frame == FrameId.parse(
        fixture["frames"]["reference_ego"]
    )
    assert reference_ego_from_camera.source_frame == FrameId.parse(fixture["frames"]["camera"])


def test_lidar_time_pose_cannot_silently_replace_camera_time_pose() -> None:
    fixture = _mapping(FIXTURE_PATH)
    poses = fixture["poses"]
    global_from_reference_ego = _fixture_transform(poses["global_from_reference_ego"])
    camera_ego_from_camera = _fixture_transform(poses["camera_ego_from_camera"])

    with pytest.raises(ValueError, match="mismatched intermediate"):
        global_from_reference_ego.compose(camera_ego_from_camera)

    relabeled_extrinsic = SE3(
        target_frame=global_from_reference_ego.source_frame,
        source_frame=camera_ego_from_camera.source_frame,
        rotation=camera_ego_from_camera.rotation,
        translation_m=camera_ego_from_camera.translation_m,
    )
    wrong_camera_from_global = global_from_reference_ego.compose(relabeled_extrinsic).inverse()
    first_point = fixture["projection_points"][0]
    wrong_point = wrong_camera_from_global.apply(
        first_point["point_global_m"],
        source_frame=wrong_camera_from_global.source_frame,
    )
    assert np.max(np.abs(wrong_point - first_point["expected_point_camera_m"])) > 0.1


def test_frozen_random_transform_properties() -> None:
    manifest = _mapping(MANIFEST_PATH)["property_validation"]
    count = manifest["transform_count"]
    generator = np.random.Generator(np.random.PCG64DXSM(manifest["seed"]))
    left_quaternions = generator.standard_normal(size=(count, 4), dtype=np.float64)
    left_quaternions /= np.linalg.norm(left_quaternions, axis=1, keepdims=True)
    right_quaternions = generator.standard_normal(size=(count, 4), dtype=np.float64)
    right_quaternions /= np.linalg.norm(right_quaternions, axis=1, keepdims=True)
    left_translations = generator.uniform(-100.0, 100.0, size=(count, 3))
    right_translations = generator.uniform(-100.0, 100.0, size=(count, 3))
    points = generator.standard_normal(size=(count, 3), dtype=np.float64) * 25.0

    frame_a = FrameId.global_frame(log_namespace="property-log")
    frame_b = FrameId.ego(
        log_namespace="property-log",
        timestamp_qualifier="lidar-time",
    )
    frame_c = FrameId.camera(
        channel="CAM_FRONT",
        calibration_instance="property-calibration",
        timestamp_qualifier="camera-time",
    )
    identity_rotation_error = 0.0
    composition_rotation_error = 0.0
    translation_error = 0.0
    round_trip_error = 0.0
    quaternion_sign_error = 0.0
    for index in range(count):
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
        left_identity = left.compose(left.inverse())
        right_identity = left.inverse().compose(left)
        identity_rotation_error = max(
            identity_rotation_error,
            float(np.max(np.abs(left_identity.rotation - np.eye(3)))),
            float(np.max(np.abs(right_identity.rotation - np.eye(3)))),
        )
        translation_error = max(
            translation_error,
            float(np.max(np.abs(left_identity.translation_m))),
            float(np.max(np.abs(right_identity.translation_m))),
        )

        combined = left.compose(right)
        expected_homogeneous = left.homogeneous_matrix() @ right.homogeneous_matrix()
        composition_rotation_error = max(
            composition_rotation_error,
            float(
                np.max(
                    np.abs(combined.rotation - expected_homogeneous[:3, :3]),
                )
            ),
        )
        translation_error = max(
            translation_error,
            float(
                np.max(
                    np.abs(combined.translation_m - expected_homogeneous[:3, 3]),
                )
            ),
        )
        transformed = combined.apply(
            points[index],
            source_frame=combined.source_frame,
        )
        recovered = combined.inverse().apply(
            transformed,
            source_frame=combined.target_frame,
        )
        round_trip_error = max(
            round_trip_error,
            float(np.max(np.abs(recovered - points[index]))),
        )
        positive_rotation = quaternion_wxyz_to_rotation(left_quaternions[index])
        negative_rotation = quaternion_wxyz_to_rotation(-left_quaternions[index])
        quaternion_sign_error = max(
            quaternion_sign_error,
            float(np.max(np.abs(positive_rotation - negative_rotation))),
        )

    assert (
        max(identity_rotation_error, composition_rotation_error)
        <= manifest["rotation_identity_composition_max_abs_tolerance"]
    )
    assert translation_error <= manifest["translation_inverse_composition_max_abs_tolerance_m"]
    assert round_trip_error <= manifest["point_round_trip_max_abs_tolerance_m"]
    assert quaternion_sign_error <= manifest["quaternion_sign_rotation_max_abs_tolerance"]


def test_quaternion_order_is_explicit_and_never_guessed() -> None:
    root_half = np.sqrt(0.5)
    explicit_wxyz = quaternion_wxyz_to_rotation((root_half, 0.0, 0.0, root_half))
    scalar_last_coefficients = quaternion_wxyz_to_rotation((0.0, 0.0, root_half, root_half))
    expected_yaw = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    assert explicit_wxyz == pytest.approx(expected_yaw, abs=1e-15)
    assert not np.allclose(scalar_last_coefficients, expected_yaw)
    assert quaternion_wxyz_to_rotation((-root_half, -0.0, -0.0, -root_half)) == pytest.approx(
        expected_yaw, abs=1e-15
    )


@pytest.mark.parametrize(
    "quaternion",
    [
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
)
def test_quaternion_rejects_invalid_inputs(quaternion) -> None:
    with pytest.raises(ValueError):
        quaternion_wxyz_to_rotation(quaternion)


def test_reflection_and_frame_mismatches_fail_closed() -> None:
    global_a = FrameId.global_frame(log_namespace="log-a")
    global_b = FrameId.global_frame(log_namespace="log-b")
    ego_a = FrameId.ego(log_namespace="log-a", timestamp_qualifier="time-a")
    ego_b = FrameId.ego(log_namespace="log-b", timestamp_qualifier="time-b")

    with pytest.raises(ValueError, match="right-handed"):
        SE3(
            target_frame=global_a,
            source_frame=ego_a,
            rotation=np.diag([1.0, 1.0, -1.0]),
            translation_m=np.zeros(3),
        )

    global_a_from_ego_a = SE3(
        target_frame=global_a,
        source_frame=ego_a,
        rotation=np.eye(3),
        translation_m=np.zeros(3),
    )
    ego_b_from_global_b = SE3(
        target_frame=ego_b,
        source_frame=global_b,
        rotation=np.eye(3),
        translation_m=np.zeros(3),
    )
    with pytest.raises(ValueError, match="mismatched intermediate"):
        global_a_from_ego_a.compose(ego_b_from_global_b)
    with pytest.raises(ValueError, match="point source frame"):
        global_a_from_ego_a.apply((1.0, 2.0, 3.0), source_frame=ego_b)


def test_frame_constructors_and_qualifiers_fail_closed() -> None:
    lidar = FrameId.lidar(
        channel="LIDAR_TOP",
        calibration_instance="calibration-l",
        timestamp_qualifier="lidar-time",
    )
    assert lidar.qualified_name() == "lidar:LIDAR_TOP:calibration-l:lidar-time"
    assert FrameId.parse(lidar.qualified_name()) == lidar
    caller_owned_qualifiers = ["immutable-log"]
    global_from_list = FrameId(
        kind="global",
        qualifiers=caller_owned_qualifiers,  # type: ignore[arg-type]
    )
    caller_owned_qualifiers[0] = "mutated-log"
    assert global_from_list.qualifiers == ("immutable-log",)

    with pytest.raises(ValueError, match="unsupported"):
        FrameId.parse("radar:RADAR_FRONT:calibration-r:time")
    with pytest.raises(ValueError, match="exactly"):
        FrameId(kind="ego", qualifiers=("log-only",))
    with pytest.raises(ValueError, match="printable"):
        FrameId.global_frame(log_namespace="")
    with pytest.raises(ValueError, match="printable"):
        FrameId.global_frame(log_namespace="bad:namespace")
    with pytest.raises(ValueError, match="unsupported"):
        FrameId(kind="thermal", qualifiers=("x",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cross-log"):
        SE3(
            target_frame=FrameId.global_frame(log_namespace="log-a"),
            source_frame=FrameId.ego(
                log_namespace="log-b",
                timestamp_qualifier="time-b",
            ),
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        )


def test_se3_validates_rotations_translations_points_and_comparisons() -> None:
    global_frame = FrameId.global_frame(log_namespace="validation")
    ego_frame = FrameId.ego(
        log_namespace="validation",
        timestamp_qualifier="time",
    )
    with pytest.raises(ValueError, match="orthogonal"):
        SE3(
            target_frame=global_frame,
            source_frame=ego_frame,
            rotation=np.diag([1.0, 1.0, 2.0]),
            translation_m=np.zeros(3),
        )
    with pytest.raises(ValueError, match="shape"):
        SE3(
            target_frame=global_frame,
            source_frame=ego_frame,
            rotation=np.eye(3),
            translation_m=np.zeros(2),
        )
    transform = SE3(
        target_frame=global_frame,
        source_frame=ego_frame,
        rotation=np.eye(3),
        translation_m=np.ones(3),
    )
    points = transform.apply(
        np.zeros((3, 2)),
        source_frame=transform.source_frame,
    )
    assert points == pytest.approx(np.ones((3, 2)))
    with pytest.raises(ValueError, match="shape"):
        transform.apply(
            np.zeros((2, 3)),
            source_frame=transform.source_frame,
        )
    with pytest.raises(ValueError, match="one vector"):
        transform.apply(
            np.zeros((1, 1, 3)),
            source_frame=transform.source_frame,
        )
    with pytest.raises(ValueError, match="shape"):
        transform.apply(
            np.zeros(2),
            source_frame=transform.source_frame,
        )
    with pytest.raises(ValueError, match="finite"):
        transform.apply(
            (0.0, 0.0, float("inf")),
            source_frame=transform.source_frame,
        )
    with pytest.raises(TypeError):
        transform.apply((0.0, 0.0, 0.0))  # type: ignore[call-arg]

    same = SE3(
        target_frame=global_frame,
        source_frame=ego_frame,
        rotation=np.eye(3),
        translation_m=np.ones(3),
    )
    assert transform.is_close(same, rotation_atol=0.0, translation_atol_m=0.0)
    shifted = SE3(
        target_frame=global_frame,
        source_frame=ego_frame,
        rotation=np.eye(3),
        translation_m=np.asarray([1.0, 1.0, 1.1]),
    )
    assert not transform.is_close(shifted, rotation_atol=0.0, translation_atol_m=0.01)
    with pytest.raises(ValueError, match="different named mappings"):
        transform.is_close(
            SE3.identity(global_frame),
            rotation_atol=0.0,
            translation_atol_m=0.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        transform.is_close(same, rotation_atol=-1.0, translation_atol_m=0.0)


def test_se3_storage_and_exports_are_immutable() -> None:
    frame = FrameId.global_frame(log_namespace="immutable")
    transform = SE3.identity(frame)

    assert not transform.rotation.flags.writeable
    assert not transform.translation_m.flags.writeable
    homogeneous = transform.homogeneous_matrix()
    applied = transform.apply(
        (0.0, 0.0, 0.0),
        source_frame=transform.source_frame,
    )
    assert not homogeneous.flags.writeable
    assert not applied.flags.writeable
    for array in (
        transform.rotation,
        transform.translation_m,
        homogeneous,
        applied,
    ):
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(ValueError):
        transform.rotation[0, 0] = 2.0
